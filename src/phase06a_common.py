from __future__ import annotations

"""Shared integrity, evaluation, and statistics helpers for RoadBuddy Phase 06A.

The module contains no public-test workflow.  Full-scope functions enforce the
canonical 1,192/298 group-safe split and never select checkpoints from frozen
validation labels.
"""

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from roadbuddy_common import (
    CHOICES,
    IMG_CONTEXT_TOKEN,
    build_training_tokens,
    build_transform,
    dynamic_preprocess,
    format_mcq,
    load_model_and_tokenizer,
    make_collator,
    native_chat_multiframe,
    parse_choice,
    read_video_frames_at_indices,
    save_json,
    seed_everything,
)
from traffic_temporal_grounding import uniform_candidate_indices


EXPECTED_RAW_ROWS = 1490
EXPECTED_TRAIN_ROWS = 1192
EXPECTED_VALIDATION_ROWS = 298
EXPECTED_TRAIN_GROUPS = 439
EXPECTED_VALIDATION_GROUPS = 110
EXPECTED_VALIDATION_IDS_SHA256 = "dbfa2d337f56bd681df70206fa8cee83d743c6743a3493619083d03b804145c5"
PHASE06A_SCHEMA_VERSION = 1
REQUIRED_PREDICTION_COLUMNS = {
    "sample_id",
    "group_id",
    "answer",
    "prediction",
    "raw_response",
    "parse_status",
    "correct",
    "frame_count",
    "frame_indices",
    "frame_timestamps_sec",
    "num_patches_list",
    "realized_tile_count",
    "latency_seconds",
}


@dataclass(frozen=True)
class Phase06AProtocol:
    seed: int = 42
    run_scope: str = "smoke"
    attention_implementation: str = "eager"
    dtype: str = "bfloat16"
    total_tile_budget: int = 8
    use_thumbnail: bool = False
    frame_policy: str = "equal_temporal_bin_midpoints"
    max_new_tokens: int = 16
    do_sample: bool = False
    num_beams: int = 1
    primary_metric: str = "accuracy"
    secondary_metric: str = "macro_f1"
    bootstrap_resamples: int = 10_000
    noninferiority_margin: float = 0.02

    def __post_init__(self) -> None:
        require_run_scope(self.run_scope)
        if self.total_tile_budget < 8:
            raise ValueError("The common tile budget must support the F8 arm")
        if self.use_thumbnail:
            raise ValueError("Phase 06A compute-matched protocol requires use_thumbnail=False")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": PHASE06A_SCHEMA_VERSION, **asdict(self)}


def require_run_scope(run_scope: str) -> str:
    normalized = str(run_scope).strip().lower()
    if normalized not in {"smoke", "full"}:
        raise ValueError("RUN_SCOPE must be explicitly set to 'smoke' or 'full'")
    return normalized


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sorted_id_hash(values: Iterable[Any]) -> str:
    return sha256_json(sorted(str(value) for value in values))


def artifact_record(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_canonical_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    frozen_ids: Iterable[Any],
    *,
    run_scope: str,
    expected_ids_file_sha256: Optional[str] = None,
) -> dict[str, Any]:
    scope = require_run_scope(run_scope)
    required = {"sample_id", "group_id", "video_path", "answer"}
    for name, frame in (("train", train), ("validation", validation)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing canonical columns: {missing}")
        if frame.sample_id.astype(str).duplicated().any():
            raise ValueError(f"{name} contains duplicate sample IDs")

    train_ids = set(train.sample_id.astype(str))
    validation_ids = set(validation.sample_id.astype(str))
    frozen = sorted(str(value) for value in frozen_ids)
    if sorted(validation_ids) != frozen:
        raise ValueError("validation.csv membership differs from frozen validation_sample_ids.json")
    sample_overlap = train_ids & validation_ids
    group_overlap = set(train.group_id.astype(str)) & set(validation.group_id.astype(str))
    if sample_overlap or group_overlap:
        raise ValueError(
            f"Leakage detected: sample_overlap={len(sample_overlap)}, group_overlap={len(group_overlap)}"
        )
    if expected_ids_file_sha256 and expected_ids_file_sha256 != EXPECTED_VALIDATION_IDS_SHA256:
        raise ValueError(
            "Frozen validation file checksum differs from the preregistered checksum; "
            "do not replace the split automatically"
        )

    report = {
        "run_scope": scope,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_groups": int(train.group_id.astype(str).nunique()),
        "validation_groups": int(validation.group_id.astype(str).nunique()),
        "sample_overlap": 0,
        "group_overlap": 0,
        "validation_membership_hash": sorted_id_hash(frozen),
        "validation_ids_file_sha256": expected_ids_file_sha256,
    }
    if scope == "full":
        expected = {
            "train_rows": EXPECTED_TRAIN_ROWS,
            "validation_rows": EXPECTED_VALIDATION_ROWS,
            "train_groups": EXPECTED_TRAIN_GROUPS,
            "validation_groups": EXPECTED_VALIDATION_GROUPS,
        }
        mismatches = {key: (report[key], value) for key, value in expected.items() if report[key] != value}
        if mismatches:
            raise ValueError(f"Canonical full split contract failed: {mismatches}")
        if expected_ids_file_sha256 != EXPECTED_VALIDATION_IDS_SHA256:
            raise ValueError("Full scope requires the exact preregistered validation IDs file checksum")
    return report


def fixed_tile_allocation(frame_count: int, total_budget: int = 8) -> list[int]:
    """Allocate a common maximum tile budget symmetrically over time."""
    if frame_count <= 0 or total_budget < frame_count:
        raise ValueError("total_budget must provide at least one tile per frame")
    base, remainder = divmod(total_budget, frame_count)
    allocation = [base] * frame_count
    if remainder:
        positions = np.linspace(0, frame_count - 1, remainder, dtype=int).tolist()
        if len(set(positions)) != remainder:
            positions = list(range(remainder))
        for position in positions:
            allocation[position] += 1
    if sum(allocation) != total_budget or min(allocation) <= 0:
        raise RuntimeError(f"Invalid tile allocation: {allocation}")
    return allocation


def probe_video(video_path: str) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if fps <= 0 or total_frames <= 0:
        raise RuntimeError(f"Invalid video metadata: fps={fps}, total_frames={total_frames}, path={video_path}")
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": total_frames / fps,
    }


def load_fixed_budget_visual_tiles(
    row: pd.Series,
    frame_count: int,
    total_budget: int = 8,
    dtype=torch.bfloat16,
) -> tuple[torch.Tensor, list[int], dict[str, Any]]:
    metadata = probe_video(str(row["video_path"]))
    indices = uniform_candidate_indices(metadata["total_frames"], frame_count)
    allocation = fixed_tile_allocation(frame_count, total_budget)
    frames = read_video_frames_at_indices(str(row["video_path"]), indices)
    transform = build_transform()
    all_tiles = []
    num_patches_list = []
    for image, per_frame_budget in zip(frames, allocation):
        tiles = dynamic_preprocess(image, max_num=per_frame_budget, use_thumbnail=False)
        if len(tiles) > per_frame_budget:
            raise RuntimeError("Dynamic preprocessing exceeded its per-frame tile budget")
        num_patches_list.append(len(tiles))
        all_tiles.extend(transform(tile) for tile in tiles)
    if not all_tiles or sum(num_patches_list) > total_budget:
        raise RuntimeError(
            f"Fixed-budget expansion failed: allocation={allocation}, realized={num_patches_list}"
        )
    timestamps = [index / metadata["fps"] for index in indices]
    evidence = {
        **metadata,
        "frame_indices": indices,
        "frame_timestamps_sec": timestamps,
        "tile_allocation": allocation,
        "realized_tile_count": len(all_tiles),
    }
    return torch.stack(all_tiles).to(dtype=dtype), num_patches_list, evidence


def compute_classification_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    required = {"answer", "prediction", "parse_status"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing metric columns: {missing}")
    y_true = predictions.answer.astype(str)
    y_pred = predictions.prediction.fillna("INVALID").astype(str)
    precision, recall, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CHOICES), zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(per_f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(CHOICES)
    }
    matrix = confusion_matrix(y_true, y_pred, labels=list(CHOICES))
    return {
        "rows": len(predictions),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CHOICES), average="macro", zero_division=0)),
        "parse_rate": float((predictions.parse_status == "parsed").mean()),
        "per_class": per_class,
        "confusion_matrix": {"labels": list(CHOICES), "values": matrix.tolist()},
    }


def evaluate_fixed_budget_rows(
    model,
    tokenizer,
    frame: pd.DataFrame,
    *,
    frame_count: int,
    total_tile_budget: int,
    limit: Optional[int] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = []
    selected = frame.head(limit) if limit else frame
    for _, row in selected.iterrows():
        started = time.perf_counter()
        pixels, patches, evidence = load_fixed_budget_visual_tiles(
            row, frame_count, total_tile_budget, dtype=torch.bfloat16
        )
        pixels = pixels.cuda(non_blocking=True)
        response = native_chat_multiframe(
            model, tokenizer, pixels, format_mcq(row), patches, max_new_tokens=16
        )
        prediction = parse_choice(response)
        records.append(
            {
                "sample_id": str(row["sample_id"]),
                "group_id": str(row["group_id"]),
                "question_type": str(row.get("question_type", "unknown")),
                "answer": str(row["answer"]),
                "prediction": prediction,
                # Keep an explicit sentinel because an empty CSV field is parsed back as NaN.
                "raw_response": str(response) if str(response) else "<EMPTY>",
                "parse_status": "parsed" if prediction is not None else "invalid",
                "correct": bool(prediction == row["answer"]),
                "frame_count": frame_count,
                "frame_indices": json.dumps(evidence["frame_indices"]),
                "frame_timestamps_sec": json.dumps(evidence["frame_timestamps_sec"]),
                "fps": evidence["fps"],
                "duration_seconds": evidence["duration_seconds"],
                "tile_allocation": json.dumps(evidence["tile_allocation"]),
                "num_patches_list": json.dumps(patches),
                "realized_tile_count": evidence["realized_tile_count"],
                "latency_seconds": time.perf_counter() - started,
            }
        )
    predictions = pd.DataFrame(records)
    return predictions, compute_classification_metrics(predictions)


class FixedBudgetF1Dataset(torch.utils.data.Dataset):
    """Canonical midpoint-F1 training dataset; support timestamps are ignored."""

    def __init__(self, frame: pd.DataFrame, model, tokenizer, total_tile_budget: int = 8):
        self.frame = frame.reset_index(drop=True)
        self.model = model
        self.tokenizer = tokenizer
        self.total_tile_budget = total_tile_budget

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        pixels, _, evidence = load_fixed_budget_visual_tiles(
            row, 1, self.total_tile_budget, dtype=torch.bfloat16
        )
        input_ids, labels = build_training_tokens(
            self.model, self.tokenizer, format_mcq(row), str(row["answer"]), len(pixels)
        )
        return {
            "sample_id": str(row["sample_id"]),
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixels,
            "image_flags": torch.ones(len(pixels), dtype=torch.long),
            "frame_indices": evidence["frame_indices"],
        }


def create_lora_model(rank: int, alpha: int, *, training: bool = True):
    """Reload the pinned base and attach a fresh LoRA adapter."""
    from peft import LoraConfig, get_peft_model

    if rank <= 0 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    model, tokenizer = load_model_and_tokenizer(training=training, attn_implementation="eager")
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Only LoRA parameters may be trainable")
    return model, tokenizer


def run_lora_training_stage(
    model,
    tokenizer,
    train_frame: pd.DataFrame,
    *,
    output_dir: Path,
    total_tile_budget: int,
    epochs: int,
    gradient_accumulation: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    max_optimizer_steps: Optional[int],
    scheduler_total_steps: Optional[int] = None,
    evaluation_frame: Optional[pd.DataFrame] = None,
    evaluation_limit: Optional[int] = None,
    evaluation_steps: int = 8,
    patience_evaluations: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Train with correct partial accumulation and optional inner-dev selection."""
    from torch.utils.data import DataLoader

    if gradient_accumulation <= 0 or epochs <= 0 or evaluation_steps <= 0:
        raise ValueError("Training counts must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_started = time.perf_counter()
    seed_everything(seed)
    dataset = FixedBudgetF1Dataset(train_frame, model, tokenizer, total_tile_budget)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=make_collator(tokenizer),
        generator=torch.Generator().manual_seed(seed),
    )
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    initial = {name: parameter.detach().float().cpu().clone() for name, parameter in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=learning_rate, weight_decay=weight_decay)
    available_steps = max(1, math.ceil(len(loader) / gradient_accumulation) * epochs)
    execution_steps = available_steps
    if max_optimizer_steps is not None:
        execution_steps = min(execution_steps, max_optimizer_steps)
    schedule_source = "explicit" if scheduler_total_steps is not None else "execution_horizon"
    if scheduler_total_steps is None and evaluation_frame is None:
        selection_result = output_dir.parent / "stage_a_inner_selection" / "training_result.json"
        if selection_result.is_file():
            selected = json.loads(selection_result.read_text(encoding="utf-8"))
            scheduler_total_steps = int(selected["scheduler_total_steps"])
            schedule_source = str(selection_result)
    schedule_steps = int(scheduler_total_steps or execution_steps)
    if schedule_steps < execution_steps:
        raise ValueError("scheduler_total_steps cannot be shorter than the executed training horizon")
    warmup_steps = max(1, round(schedule_steps * 0.05))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        return max(0.0, (schedule_steps - step) / max(1, schedule_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    history: list[dict[str, Any]] = []
    optimizer_step = 0
    best_step: Optional[int] = None
    best_key: Optional[tuple[float, float, int]] = None
    best_metrics: Optional[dict[str, Any]] = None
    stale_evaluations = 0
    accumulated_ids: list[str] = []
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    stop = False

    for epoch in range(epochs):
        model.train()
        for micro_index, batch in enumerate(loader, start=1):
            sample_ids = list(batch.pop("sample_ids"))
            accumulated_ids.extend(sample_ids)
            window_start = ((micro_index - 1) // gradient_accumulation) * gradient_accumulation + 1
            window_size = min(gradient_accumulation, len(loader) - window_start + 1)
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            if batch["image_flags"].numel() != batch["pixel_values"].shape[0]:
                raise RuntimeError("image_flags do not match visual tiles")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                unscaled_loss = model(**batch).loss
                loss = unscaled_loss / window_size
            loss.backward()
            should_step = micro_index % gradient_accumulation == 0 or micro_index == len(loader)
            if not should_step:
                continue
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable.values(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            record = {
                "epoch": epoch + 1,
                "optimizer_step": optimizer_step,
                "unscaled_loss": float(unscaled_loss.detach()),
                "gradient_norm": float(gradient_norm),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "accumulated_sample_ids": json.dumps(accumulated_ids),
                "accumulated_micro_batches": len(accumulated_ids),
            }
            accumulated_ids = []
            should_evaluate = evaluation_frame is not None and (
                optimizer_step % evaluation_steps == 0 or optimizer_step == execution_steps
            )
            if should_evaluate:
                model.eval()
                predictions, metrics = evaluate_fixed_budget_rows(
                    model,
                    tokenizer,
                    evaluation_frame,
                    frame_count=1,
                    total_tile_budget=total_tile_budget,
                    limit=evaluation_limit,
                )
                record.update({"inner_dev_accuracy": metrics["accuracy"], "inner_dev_macro_f1": metrics["macro_f1"]})
                checkpoint_dir = output_dir / "checkpoints" / f"step-{optimizer_step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_dir / "adapter")
                predictions.to_csv(checkpoint_dir / "inner_dev_predictions.csv", index=False)
                save_json(checkpoint_dir / "inner_dev_metrics.json", metrics)
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                        "optimizer_step": optimizer_step,
                    },
                    checkpoint_dir / "training_state.pt",
                )
                key = (float(metrics["accuracy"]), float(metrics["macro_f1"]), -optimizer_step)
                if best_key is None or key > best_key:
                    best_key, best_step, best_metrics = key, optimizer_step, metrics
                    stale_evaluations = 0
                else:
                    stale_evaluations += 1
                model.train()
                if stale_evaluations >= patience_evaluations:
                    stop = True
            history.append(record)
            if optimizer_step >= execution_steps or stop:
                break
        if optimizer_step >= execution_steps or stop:
            break

    changed = sum(
        not torch.equal(initial[name], parameter.detach().float().cpu())
        for name, parameter in trainable.items()
    )
    if changed != len(trainable):
        raise RuntimeError(f"Only {changed}/{len(trainable)} LoRA tensors changed")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    result = {
        "optimizer_steps": optimizer_step,
        "planned_optimizer_steps": execution_steps,
        "scheduler_total_steps": schedule_steps,
        "scheduler_total_steps_source": schedule_source,
        "best_step": best_step,
        "best_metrics": best_metrics,
        "early_stopped": stop,
        "changed_lora_tensors": changed,
        "total_lora_tensors": len(trainable),
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable.values())),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "elapsed_seconds": time.perf_counter() - stage_started,
    }
    save_json(output_dir / "training_result.json", result)
    return result


def validate_prediction_artifact(
    predictions: pd.DataFrame,
    expected_ids: Iterable[Any],
    *,
    run_scope: str,
) -> dict[str, Any]:
    scope = require_run_scope(run_scope)
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction artifact is missing columns: {missing}")
    actual = predictions.sample_id.astype(str)
    expected = sorted(str(value) for value in expected_ids)
    if actual.duplicated().any():
        raise ValueError("Prediction artifact contains duplicate sample IDs")
    if sorted(actual) != expected:
        raise ValueError("Prediction artifact membership differs from the expected validation IDs")
    if predictions.raw_response.isna().any():
        raise ValueError("Every prediction, including parse failures, must retain raw_response")
    invalid_status = set(predictions.parse_status.astype(str)) - {"parsed", "invalid"}
    if invalid_status:
        raise ValueError(f"Unknown parse statuses: {sorted(invalid_status)}")
    invalid_rows = predictions.parse_status.astype(str).eq("invalid")
    if predictions.loc[invalid_rows, "correct"].astype(bool).any():
        raise ValueError("Parse failures must be counted as incorrect")
    if scope == "full" and len(predictions) != EXPECTED_VALIDATION_ROWS:
        raise ValueError("Full prediction artifacts require exactly 298 rows")
    return {
        "run_scope": scope,
        "rows": len(predictions),
        "unique_ids": int(actual.nunique()),
        "membership_hash": sorted_id_hash(actual),
        "parse_failures": int(invalid_rows.sum()),
    }


def group_safe_inner_split(
    train: pd.DataFrame,
    *,
    dev_fraction: float = 0.2,
    seed: int = 42,
    search_trials: int = 512,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {"sample_id", "group_id", "answer"}
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"Training table is missing columns: {missing}")
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be in (0, 1)")
    groups = sorted(train.group_id.astype(str).unique())
    dev_group_count = max(1, min(len(groups) - 1, round(len(groups) * dev_fraction)))
    overall = train.answer.value_counts(normalize=True).reindex(CHOICES, fill_value=0.0)
    best: Optional[tuple[float, set[str]]] = None
    for trial in range(search_trials):
        shuffled = groups.copy()
        random.Random(seed + trial).shuffle(shuffled)
        candidate = set(shuffled[:dev_group_count])
        dev = train[train.group_id.astype(str).isin(candidate)]
        distribution = dev.answer.value_counts(normalize=True).reindex(CHOICES, fill_value=0.0)
        row_error = abs(len(dev) / len(train) - dev_fraction)
        class_error = float(np.abs(distribution - overall).sum())
        score = row_error + class_error
        if best is None or score < best[0]:
            best = (score, candidate)
    assert best is not None
    dev_groups = best[1]
    inner_dev = train[train.group_id.astype(str).isin(dev_groups)].copy().reset_index(drop=True)
    train_fit = train[~train.group_id.astype(str).isin(dev_groups)].copy().reset_index(drop=True)
    if set(train_fit.group_id.astype(str)) & set(inner_dev.group_id.astype(str)):
        raise RuntimeError("Inner split leaks group IDs")
    report = {
        "seed": seed,
        "dev_fraction_target": dev_fraction,
        "search_trials": search_trials,
        "train_fit_rows": len(train_fit),
        "inner_dev_rows": len(inner_dev),
        "train_fit_groups": int(train_fit.group_id.nunique()),
        "inner_dev_groups": int(inner_dev.group_id.nunique()),
        "group_overlap": 0,
        "train_fit_ids_hash": sorted_id_hash(train_fit.sample_id),
        "inner_dev_ids_hash": sorted_id_hash(inner_dev.sample_id),
        "label_distribution": {
            "all_train": train.answer.value_counts(normalize=True).reindex(CHOICES, fill_value=0.0).to_dict(),
            "train_fit": train_fit.answer.value_counts(normalize=True).reindex(CHOICES, fill_value=0.0).to_dict(),
            "inner_dev": inner_dev.answer.value_counts(normalize=True).reindex(CHOICES, fill_value=0.0).to_dict(),
        },
    }
    return train_fit, inner_dev, report


def paired_group_cluster_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    resamples: int = 10_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = ["sample_id", "group_id", "answer", "prediction"]
    columns.extend(
        column
        for column in ("parse_status", "latency_seconds", "realized_tile_count")
        if column in baseline.columns and column in candidate.columns
    )
    left = baseline[columns].copy()
    right = candidate[columns].copy()
    paired = left.merge(right, on=["sample_id", "group_id", "answer"], suffixes=("_baseline", "_candidate"), validate="one_to_one")
    if len(paired) != len(left) or len(left) != len(right):
        raise ValueError("Paired bootstrap requires identical sample membership")
    groups = sorted(paired.group_id.astype(str).unique())
    group_rows = {group: paired[paired.group_id.astype(str) == group] for group in groups}
    rng = np.random.default_rng(seed)
    records = []
    for index in range(resamples):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled = pd.concat([group_rows[group] for group in sampled_groups], ignore_index=True)
        y = sampled.answer.astype(str)
        base_pred = sampled.prediction_baseline.fillna("INVALID").astype(str)
        candidate_pred = sampled.prediction_candidate.fillna("INVALID").astype(str)
        record = {
            "resample": index,
            "accuracy_delta": accuracy_score(y, candidate_pred) - accuracy_score(y, base_pred),
            "macro_f1_delta": f1_score(y, candidate_pred, labels=list(CHOICES), average="macro", zero_division=0)
            - f1_score(y, base_pred, labels=list(CHOICES), average="macro", zero_division=0),
        }
        base_class = f1_score(y, base_pred, labels=list(CHOICES), average=None, zero_division=0)
        candidate_class = f1_score(y, candidate_pred, labels=list(CHOICES), average=None, zero_division=0)
        record.update(
            {f"class_f1_{label}_delta": float(candidate_class[i] - base_class[i]) for i, label in enumerate(CHOICES)}
        )
        if "parse_status_baseline" in sampled:
            record["parse_rate_delta"] = float(
                (sampled.parse_status_candidate == "parsed").mean()
                - (sampled.parse_status_baseline == "parsed").mean()
            )
        if "latency_seconds_baseline" in sampled:
            record["mean_latency_seconds_delta"] = float(
                sampled.latency_seconds_candidate.mean() - sampled.latency_seconds_baseline.mean()
            )
        if "realized_tile_count_baseline" in sampled:
            record["mean_realized_tiles_delta"] = float(
                sampled.realized_tile_count_candidate.mean() - sampled.realized_tile_count_baseline.mean()
            )
        records.append(record)
    distribution = pd.DataFrame(records)
    summary = {
        metric: {
            "mean": float(distribution[metric].mean()),
            "ci95_low": float(distribution[metric].quantile(0.025)),
            "ci95_high": float(distribution[metric].quantile(0.975)),
        }
        for metric in distribution.columns
        if metric != "resample"
    }
    return distribution, summary


def group_cluster_metric_ci(
    predictions: pd.DataFrame,
    *,
    resamples: int = 10_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Estimate sample-level endpoint uncertainty by resampling whole video groups."""
    required = {"group_id", "answer", "prediction"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Cluster CI table is missing columns: {missing}")
    groups = sorted(predictions.group_id.astype(str).unique())
    if not groups:
        raise ValueError("Cluster CI requires at least one group")
    grouped = {
        group: predictions[predictions.group_id.astype(str) == group]
        for group in groups
    }
    rng = np.random.default_rng(seed)
    records = []
    for index in range(resamples):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled = pd.concat([grouped[group] for group in sampled_groups], ignore_index=True)
        y_true = sampled.answer.astype(str)
        y_pred = sampled.prediction.fillna("INVALID").astype(str)
        record = {
            "resample": index,
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(
                y_true,
                y_pred,
                labels=list(CHOICES),
                average="macro",
                zero_division=0,
            ),
        }
        per_class = f1_score(y_true, y_pred, labels=list(CHOICES), average=None, zero_division=0)
        record.update({f"class_f1_{label}": float(per_class[i]) for i, label in enumerate(CHOICES)})
        if "parse_status" in sampled:
            record["parse_rate"] = float((sampled.parse_status == "parsed").mean())
        if "latency_seconds" in sampled:
            record["mean_latency_seconds"] = float(sampled.latency_seconds.mean())
        if "realized_tile_count" in sampled:
            record["mean_realized_tiles"] = float(sampled.realized_tile_count.mean())
        records.append(record)
    distribution = pd.DataFrame(records)
    summary = {
        metric: {
            "mean": float(distribution[metric].mean()),
            "ci95_low": float(distribution[metric].quantile(0.025)),
            "ci95_high": float(distribution[metric].quantile(0.975)),
        }
        for metric in distribution.columns
        if metric != "resample"
    }
    return distribution, summary


def exact_mcnemar(baseline_correct: Iterable[bool], candidate_correct: Iterable[bool]) -> dict[str, Any]:
    left = np.asarray(list(baseline_correct), dtype=bool)
    right = np.asarray(list(candidate_correct), dtype=bool)
    if left.shape != right.shape:
        raise ValueError("McNemar inputs must have identical shape")
    baseline_only = int(np.sum(left & ~right))
    candidate_only = int(np.sum(~left & right))
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(0, min(baseline_only, candidate_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(value)))
        adjusted[name] = running
    return adjusted


def annotation_agreement(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    axes = ["visual_required", "temporal_required", "traffic_knowledge_required", "mixed_or_ambiguous"]
    required = {"sample_id", "primary_label", *axes}
    for name, frame in (("annotator_1", left), ("annotator_2", right)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} annotation file is missing columns: {missing}")
    paired = left[list(required)].merge(right[list(required)], on="sample_id", suffixes=("_1", "_2"), validate="one_to_one")
    metrics = {
        axis: float(cohen_kappa_score(paired[f"{axis}_1"], paired[f"{axis}_2"]))
        for axis in axes
    }
    metrics["primary_label"] = float(cohen_kappa_score(paired.primary_label_1, paired.primary_label_2))
    disagreements = paired[
        np.logical_or.reduce(
            [paired[f"{column}_1"].astype(str) != paired[f"{column}_2"].astype(str) for column in [*axes, "primary_label"]]
        )
    ].copy()
    return {"rows": len(paired), "cohen_kappa": metrics, "disagreement_rows": len(disagreements), "disagreements": disagreements}


def write_run_manifest(path: Path, *, config: dict[str, Any], artifacts: Iterable[Path]) -> dict[str, Any]:
    payload = {
        "schema_version": PHASE06A_SCHEMA_VERSION,
        "config": config,
        "config_sha256": sha256_json(config),
        "artifacts": [artifact_record(Path(item)) for item in artifacts],
    }
    save_json(Path(path), payload)
    return payload


def assert_no_public_test_reference(text: str) -> None:
    lowered = str(text).casefold().replace("-", "_")
    forbidden = ["public_test", "submission_phase", "sample_submission"]
    found = [token for token in forbidden if token in lowered]
    if found:
        raise ValueError(f"Phase 06A source contains forbidden public-test/submission references: {found}")
