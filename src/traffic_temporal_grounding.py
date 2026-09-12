from __future__ import annotations

"""Question-guided temporal frame selection for RoadBuddy.

The module is deliberately independent from a particular video/question encoder.
It consumes pre-computed frame, question, and optional traffic features so the
grounder can be trained and evaluated without changing the pinned Vintern model.

Support-frame annotations are weak supervision and may only be converted into
targets for the training split. Validation and test selection must use model
scores alone.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TemporalGroundingConfig:
    frame_feature_dim: int
    question_feature_dim: int
    traffic_feature_dim: int = 0
    hidden_dim: int = 256
    dropout: float = 0.10
    candidate_count: int = 32
    selected_count: int = 3
    min_temporal_gap: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "frame_feature_dim": self.frame_feature_dim,
            "question_feature_dim": self.question_feature_dim,
            "hidden_dim": self.hidden_dim,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"Grounding dimensions/counts must be positive: {invalid}")
        if self.traffic_feature_dim < 0:
            raise ValueError("traffic_feature_dim cannot be negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.selected_count > self.candidate_count:
            raise ValueError("selected_count cannot exceed candidate_count")
        if self.min_temporal_gap < 0:
            raise ValueError("min_temporal_gap cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def uniform_candidate_indices(total_frames: int, candidate_count: int) -> list[int]:
    """Return deterministic temporal-bin midpoints without duplicate indices."""
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if candidate_count <= 0:
        raise ValueError(f"candidate_count must be positive, got {candidate_count}")
    count = min(total_frames, candidate_count)
    indices = [
        min(total_frames - 1, max(0, int((index + 0.5) * total_frames / count)))
        for index in range(count)
    ]
    if len(indices) != len(set(indices)):
        raise RuntimeError(f"Candidate generation produced duplicate indices: {indices}")
    return indices


def indices_to_timestamps(indices: list[int], fps: float) -> list[float]:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return [float(index) / float(fps) for index in indices]


def weak_targets_from_support_times(
    candidate_timestamps: torch.Tensor,
    support_times: torch.Tensor,
    *,
    split_name: str,
    sigma_seconds: float = 0.5,
) -> torch.Tensor:
    """Create Gaussian weak targets from train-only support timestamps.

    This guard prevents annotation-derived support frames from silently entering
    validation/test frame selection. The returned distribution sums to one.
    """
    if str(split_name).strip().lower() != "train":
        raise ValueError("Support-frame supervision is allowed only for the train split")
    if sigma_seconds <= 0:
        raise ValueError("sigma_seconds must be positive")
    if candidate_timestamps.ndim != 1 or candidate_timestamps.numel() == 0:
        raise ValueError("candidate_timestamps must be a non-empty 1D tensor")
    support_times = support_times.flatten()
    if support_times.numel() == 0:
        raise ValueError("support_times must contain at least one timestamp")
    distance = candidate_timestamps[:, None] - support_times[None, :]
    logits = -0.5 * (distance / sigma_seconds).pow(2)
    target = logits.logsumexp(dim=1).softmax(dim=0)
    if not torch.isfinite(target).all():
        raise RuntimeError("Weak temporal targets contain non-finite values")
    return target


def temporal_grounding_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy between predicted and target relevance distributions."""
    if scores.shape != targets.shape:
        raise ValueError(f"scores/targets shape mismatch: {scores.shape} vs {targets.shape}")
    if scores.ndim != 2:
        raise ValueError("scores and targets must have shape [batch, candidates]")
    if valid_mask is None:
        valid_mask = torch.ones_like(scores, dtype=torch.bool)
    if valid_mask.shape != scores.shape:
        raise ValueError("valid_mask must match scores")
    if not valid_mask.any(dim=1).all():
        raise ValueError("Every sample must contain at least one valid candidate")
    masked_scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
    masked_targets = targets.masked_fill(~valid_mask, 0.0)
    target_mass = masked_targets.sum(dim=1, keepdim=True)
    if (target_mass <= 0).any():
        raise ValueError("Every sample must assign positive target mass to valid candidates")
    normalized_targets = masked_targets / target_mass
    return -(normalized_targets * F.log_softmax(masked_scores, dim=1)).sum(dim=1).mean()


def select_diverse_topk(
    scores: torch.Tensor,
    timestamps: torch.Tensor,
    *,
    k: int,
    valid_mask: Optional[torch.Tensor] = None,
    min_temporal_gap: float = 0.0,
) -> torch.Tensor:
    """Greedily select high-scoring candidates and return them in time order."""
    if scores.ndim != 2 or timestamps.shape != scores.shape:
        raise ValueError("scores and timestamps must have shape [batch, candidates]")
    if k <= 0:
        raise ValueError("k must be positive")
    if min_temporal_gap < 0:
        raise ValueError("min_temporal_gap cannot be negative")
    if valid_mask is None:
        valid_mask = torch.ones_like(scores, dtype=torch.bool)
    if valid_mask.shape != scores.shape:
        raise ValueError("valid_mask must match scores")

    selections = []
    for row_scores, row_times, row_mask in zip(scores, timestamps, valid_mask):
        valid_indices = torch.nonzero(row_mask, as_tuple=False).flatten()
        if valid_indices.numel() < k:
            raise ValueError(f"Requested k={k}, but only {valid_indices.numel()} candidates are valid")
        ranked = valid_indices[torch.argsort(row_scores[valid_indices], descending=True, stable=True)]
        chosen: list[int] = []
        for raw_index in ranked.tolist():
            timestamp = float(row_times[raw_index])
            if all(abs(timestamp - float(row_times[other])) >= min_temporal_gap for other in chosen):
                chosen.append(raw_index)
            if len(chosen) == k:
                break
        if len(chosen) < k:
            # The diversity constraint can be infeasible for short videos. Fill
            # deterministically by score and expose the selected timestamps in
            # provenance rather than failing or changing k silently.
            chosen.extend(index for index in ranked.tolist() if index not in chosen)
            chosen = chosen[:k]
        chosen.sort(key=lambda index: (float(row_times[index]), index))
        selections.append(torch.tensor(chosen, dtype=torch.long, device=scores.device))
    return torch.stack(selections)


class TrafficAwareTemporalGrounder(nn.Module):
    """A lightweight question-conditioned traffic-aware frame scorer."""

    def __init__(self, config: TemporalGroundingConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.frame_projection = nn.Sequential(
            nn.Linear(config.frame_feature_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.question_projection = nn.Sequential(
            nn.Linear(config.question_feature_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.time_projection = nn.Sequential(nn.Linear(3, hidden), nn.GELU())
        self.traffic_projection = (
            nn.Sequential(
                nn.Linear(config.traffic_feature_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
            )
            if config.traffic_feature_dim
            else None
        )
        fusion_parts = 5 + int(self.traffic_projection is not None)
        self.scorer = nn.Sequential(
            nn.Linear(fusion_parts * hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        frame_features: torch.Tensor,
        question_features: torch.Tensor,
        normalized_timestamps: torch.Tensor,
        traffic_features: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if frame_features.ndim != 3:
            raise ValueError("frame_features must have shape [batch, candidates, dim]")
        batch, candidates, _ = frame_features.shape
        if question_features.shape[:1] != (batch,) or question_features.ndim != 2:
            raise ValueError("question_features must have shape [batch, dim]")
        if normalized_timestamps.shape != (batch, candidates):
            raise ValueError("normalized_timestamps must have shape [batch, candidates]")
        if (normalized_timestamps < 0).any() or (normalized_timestamps > 1).any():
            raise ValueError("normalized_timestamps must lie in [0, 1]")

        frame_hidden = self.frame_projection(frame_features)
        question_hidden = self.question_projection(question_features)[:, None, :].expand(-1, candidates, -1)
        time_basis = torch.stack(
            [
                normalized_timestamps,
                torch.sin(torch.pi * normalized_timestamps),
                torch.cos(torch.pi * normalized_timestamps),
            ],
            dim=-1,
        )
        parts = [
            frame_hidden,
            question_hidden,
            frame_hidden * question_hidden,
            torch.abs(frame_hidden - question_hidden),
            self.time_projection(time_basis),
        ]
        if self.traffic_projection is not None:
            expected = (batch, candidates, self.config.traffic_feature_dim)
            if traffic_features is None or traffic_features.shape != expected:
                raise ValueError(f"traffic_features must have shape {expected}")
            parts.append(self.traffic_projection(traffic_features))
        elif traffic_features is not None:
            raise ValueError("traffic_features were provided but traffic_feature_dim=0")

        scores = self.scorer(torch.cat(parts, dim=-1)).squeeze(-1)
        if valid_mask is not None:
            if valid_mask.shape != scores.shape:
                raise ValueError("valid_mask must match [batch, candidates]")
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        return scores

    @torch.no_grad()
    def select(
        self,
        frame_features: torch.Tensor,
        question_features: torch.Tensor,
        normalized_timestamps: torch.Tensor,
        timestamps_seconds: torch.Tensor,
        traffic_features: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        selected_count: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.forward(
            frame_features,
            question_features,
            normalized_timestamps,
            traffic_features=traffic_features,
            valid_mask=valid_mask,
        )
        selected = select_diverse_topk(
            scores,
            timestamps_seconds,
            k=selected_count or self.config.selected_count,
            valid_mask=valid_mask,
            min_temporal_gap=self.config.min_temporal_gap,
        )
        return selected, scores



def _selector_record_loss(
    model: TrafficAwareTemporalGrounder, record: Mapping[str, Any], device: torch.device,
) -> torch.Tensor:
    frame = torch.as_tensor(record["frame_features"], dtype=torch.float32, device=device).unsqueeze(0)
    question = torch.as_tensor(record["question_features"], dtype=torch.float32, device=device).unsqueeze(0)
    normalized = torch.as_tensor(record["normalized_timestamps"], dtype=torch.float32, device=device).unsqueeze(0)
    targets = torch.as_tensor(record["relevance_targets"], dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.as_tensor(record["valid_mask"], dtype=torch.bool, device=device).unsqueeze(0)
    traffic = record.get("traffic_features")
    traffic_tensor = None if traffic is None else torch.as_tensor(traffic, dtype=torch.float32, device=device).unsqueeze(0)
    scores = model(frame, question, normalized, traffic_features=traffic_tensor, valid_mask=mask)
    return temporal_grounding_loss(scores, targets, mask)


def validate_selector_records(records: Sequence[Mapping[str, Any]], config: TemporalGroundingConfig) -> None:
    if not records:
        raise ValueError("Selector training requires at least one feature record")
    required = {"sample_id", "group_id", "frame_features", "question_features", "normalized_timestamps", "valid_mask", "relevance_targets"}
    for index, record in enumerate(records):
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"Selector record {index} is missing fields: {missing}")
        frame = torch.as_tensor(record["frame_features"])
        question = torch.as_tensor(record["question_features"])
        mask = torch.as_tensor(record["valid_mask"])
        target = torch.as_tensor(record["relevance_targets"])
        if frame.shape != (config.candidate_count, config.frame_feature_dim):
            raise ValueError(f"Selector record {index} frame feature shape mismatch")
        if question.shape != (config.question_feature_dim,) or mask.shape != (config.candidate_count,) or target.shape != (config.candidate_count,):
            raise ValueError(f"Selector record {index} question/mask/target shape mismatch")
        traffic = record.get("traffic_features")
        if config.traffic_feature_dim:
            if traffic is None or torch.as_tensor(traffic).shape != (config.candidate_count, config.traffic_feature_dim):
                raise ValueError(f"Selector record {index} traffic feature shape mismatch")
        elif traffic is not None:
            raise ValueError("QTG records cannot inject traffic features")


def run_selector_training_stage(
    model: TrafficAwareTemporalGrounder,
    train_records: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    epochs: int,
    gradient_accumulation: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    max_optimizer_steps: int | None = None,
    evaluation_records: Sequence[Mapping[str, Any]] | None = None,
    evaluation_steps: int = 1,
    patience_evaluations: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Train only selector parameters with correct final partial accumulation."""
    if epochs <= 0 or gradient_accumulation <= 0 or evaluation_steps <= 0 or patience_evaluations <= 0:
        raise ValueError("Training counts must be positive")
    validate_selector_records(train_records, model.config)
    if evaluation_records is not None:
        validate_selector_records(evaluation_records, model.config)
        train_groups = {str(record["group_id"]) for record in train_records}
        dev_groups = {str(record["group_id"]) for record in evaluation_records}
        if train_groups & dev_groups:
            raise ValueError("Selector train and inner-dev groups overlap")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.train()
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable or len(trainable) != len(list(model.parameters())):
        raise ValueError("The selector optimizer may contain selector parameters only")
    initial = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=learning_rate, weight_decay=weight_decay)
    available_steps = math.ceil(len(train_records) / gradient_accumulation) * epochs
    horizon = min(available_steps, max_optimizer_steps) if max_optimizer_steps is not None else available_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, (horizon-step)/max(1,horizon)))
    generator = torch.Generator().manual_seed(seed)
    history=[]; optimizer_step=0; best_step=None; best_loss=float("inf"); best_state=None; stale=0; started=time.perf_counter(); stop=False
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        order=torch.randperm(len(train_records),generator=generator).tolist()
        for offset in range(0,len(order),gradient_accumulation):
            window=order[offset:offset+gradient_accumulation]
            optimizer.zero_grad(set_to_none=True); raw_losses=[]; sample_ids=[]
            for index in window:
                loss=_selector_record_loss(model,train_records[index],device)
                raw_losses.append(float(loss.detach())); sample_ids.append(str(train_records[index]["sample_id"]))
                (loss/len(window)).backward()
            gradient_norm=torch.nn.utils.clip_grad_norm_(trainable.values(),max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer_step+=1
            row={"epoch":epoch+1,"optimizer_step":optimizer_step,"unscaled_loss":sum(raw_losses)/len(raw_losses),"gradient_norm":float(gradient_norm),"learning_rate":float(scheduler.get_last_lr()[0]),"accumulated_sample_ids":sample_ids,"accumulated_micro_batches":len(window)}
            if evaluation_records is not None and optimizer_step % evaluation_steps == 0:
                model.eval()
                with torch.no_grad(): dev_loss=sum(float(_selector_record_loss(model,r,device)) for r in evaluation_records)/len(evaluation_records)
                model.train(); row["inner_dev_loss"]=dev_loss
                if dev_loss < best_loss:
                    best_loss=dev_loss; best_step=optimizer_step; stale=0; best_state={name:p.detach().cpu().clone() for name,p in model.state_dict().items()}
                    checkpoint=output_dir/'checkpoints'/f'step-{optimizer_step}'; checkpoint.mkdir(parents=True,exist_ok=True)
                    torch.save({"model":best_state,"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"torch_rng_state":torch.get_rng_state(),"cuda_rng_state":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,"optimizer_step":optimizer_step,"accumulated_sample_ids":sample_ids},checkpoint/'selector_training_state.pt')
                else:
                    stale+=1
                    if stale>=patience_evaluations: stop=True
            history.append(row)
            if optimizer_step>=horizon or stop: break
        if optimizer_step>=horizon or stop: break
    if evaluation_records is None:
        best_step=optimizer_step; best_loss=None; best_state={name:p.detach().cpu().clone() for name,p in model.state_dict().items()}
    if best_state is None or best_step is None:
        raise RuntimeError("No selector checkpoint was selected")
    model.load_state_dict(best_state)
    changed=[name for name,p in trainable.items() if not torch.equal(initial[name],p.detach().cpu())]
    if not changed:
        raise RuntimeError("Selector parameters did not change")
    final_path=output_dir/'selector_final.pt'
    torch.save({"model":best_state,"config":model.config.to_dict(),"locked_optimizer_step":best_step},final_path)
    (output_dir/'training_history.json').write_text(json.dumps(history,indent=2)+'\n')
    result={"optimizer_steps":optimizer_step,"best_step":best_step,"best_inner_dev_loss":best_loss,"changed_parameters":changed,"trainable_parameters":sum(p.numel() for p in trainable.values()),"history":history,"runtime_seconds":time.perf_counter()-started,"peak_vram_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else 0,"final_checkpoint":str(final_path)}
    (output_dir/'training_result.json').write_text(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2)+'\n')
    return result
