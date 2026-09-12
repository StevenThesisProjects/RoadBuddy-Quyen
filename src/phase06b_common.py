from __future__ import annotations

"""Shared gates and analysis helpers for RoadBuddy Phase 06B.

Phase 06B starts only after the Phase 06A baseline winner is locked.  The
module deliberately contains no public-test workflow and never converts
validation support timestamps or answer correctness into development inputs.
"""

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from phase06a_common import (
    EXPECTED_VALIDATION_IDS_SHA256,
    EXPECTED_VALIDATION_ROWS,
    compute_classification_metrics,
    fixed_tile_allocation,
    sha256_file,
    sha256_json,
    validate_prediction_artifact,
)
from roadbuddy_common import MODEL_ID, MODEL_REVISION


PHASE06B_SCHEMA_VERSION = 2
LOCKED_BASELINE_WINNER = "L32-F1"
LOCKED_CANDIDATE_COUNT = 32
LOCKED_TOTAL_TILE_BUDGET = 8
MIN_TAXONOMY_KAPPA = 0.70
NOVELTY_TRACKS = {"knowledge_augmented", "traffic_temporal_grounding"}
TAXONOMY_AXES = (
    "visual_required",
    "temporal_required",
    "traffic_knowledge_required",
    "mixed_or_ambiguous",
)
PRIMARY_LABELS = {
    "visual_static",
    "temporal",
    "traffic_knowledge",
    "mixed",
    "ambiguous",
}
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXPERIMENT_ROLES = {
    "L32-F1": ("baseline_control", "midpoint", 1, True),
    **{
        f"{family}-{k}": (role, selector, k, eligible)
        for family, role, selector, eligible in (
            ("U", "uniform_control", "uniform", False),
            ("RND", "negative_control", "seeded_random", False),
            ("QTG", "learned_grounding", "question_guided", True),
            ("TATG", "traffic_aware_grounding", "traffic_aware_question_guided", True),
            ("ORACLE", "diagnostic_only", "human_support", False),
        )
        for k in (1, 3, 8)
    },
}
WINNER_ELIGIBLE_ARMS = {
    name for name, (_, _, _, eligible) in EXPERIMENT_ROLES.items() if eligible
}
PHASE06A_PREDICTION_PATHS = {
    "Z-F1": "outputs/phase06a/zero_shot/Z-F1/full/predictions.csv",
    "Z-F3": "outputs/phase06a/zero_shot/Z-F3/full/predictions.csv",
    "Z-F8": "outputs/phase06a/zero_shot/Z-F8/full/predictions.csv",
    "L16-F1": "outputs/phase06a/lora_r16_evaluation/L16-F1/full/predictions.csv",
    "L16-F3": "outputs/phase06a/lora_r16_evaluation/L16-F3/full/predictions.csv",
    "L16-F8": "outputs/phase06a/lora_r16_evaluation/L16-F8/full/predictions.csv",
    "L8-F1": "outputs/phase06a/rank_ablation/r8/full/predictions.csv",
    "L32-F1": "outputs/phase06a/rank_ablation/r32/full/predictions.csv",
    "L64-F1": "outputs/phase06a/rank_ablation/r64/full/predictions.csv",
}


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_locked_baseline(project_root: Path) -> dict[str, Any]:
    root = Path(project_root)
    manifest_path = root / "outputs/phase06a/final_analysis/baseline_winner_manifest.json"
    status_path = root / "outputs/phase06a/final_analysis/PHASE06A_FINAL_STATUS.json"
    ids_path = root / "data/splits/phase01/validation_sample_ids.json"
    for path in (manifest_path, status_path, ids_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = load_json(manifest_path)
    status = load_json(status_path)
    if manifest.get("status") != "complete" or status.get("status") != "complete":
        raise ValueError("Phase 06A baseline winner is not complete")
    if manifest.get("winner") != LOCKED_BASELINE_WINNER or status.get("winner") != LOCKED_BASELINE_WINNER:
        raise ValueError("Phase 06B requires the locked L32-F1 baseline winner")
    if sha256_file(ids_path) != EXPECTED_VALIDATION_IDS_SHA256:
        raise ValueError("Frozen validation IDs checksum changed")
    winner_record = manifest.get("winner_predictions", {})
    winner_path = Path(str(winner_record.get("path", "")))
    if not winner_path.is_absolute():
        winner_path = root / winner_path
    if not winner_path.is_file() or sha256_file(winner_path) != winner_record.get("sha256"):
        raise ValueError("Locked winner prediction artifact is missing or hash-mismatched")
    if int(manifest.get("validation_rows", -1)) != EXPECTED_VALIDATION_ROWS:
        raise ValueError("Locked winner does not use 298 validation rows")
    if manifest.get("validation_ids_sha256") != EXPECTED_VALIDATION_IDS_SHA256:
        raise ValueError("Baseline manifest validation checksum is not canonical")
    winner = pd.read_csv(winner_path)
    if (
        len(winner) != EXPECTED_VALIDATION_ROWS
        or winner.sample_id.astype(str).nunique() != EXPECTED_VALIDATION_ROWS
        or winner.group_id.astype(str).nunique() != 110
    ):
        raise ValueError("Locked winner prediction membership is not 298 IDs / 110 groups")
    return manifest


def _require_sha256(value: Any, field: str) -> str:
    normalized = str(value).strip().lower()
    if not HASH_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def _require_named_provenance(value: Any, field: str) -> Any:
    if isinstance(value, Mapping):
        if not str(value.get("name", "")).strip() or not str(value.get("completed_at_utc", "")).strip():
            raise ValueError(f"{field} requires name and completed_at_utc")
        return value
    if isinstance(value, list) and value:
        for index, item in enumerate(value):
            _require_named_provenance(item, f"{field}[{index}]")
        return value
    raise ValueError(f"{field} requires explicit human provenance")


def taxonomy_agreement_report(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    """Calculate reliability metrics without adjudicating or fabricating labels."""
    required = {"sample_id", "primary_label", *TAXONOMY_AXES}
    validated = []
    for name, frame in (("annotator_1", left), ("annotator_2", right)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
        copy = frame.copy()
        copy["sample_id"] = copy.sample_id.astype(str)
        if copy.sample_id.duplicated().any():
            raise ValueError(f"{name} contains duplicate sample IDs")
        invalid_primary = set(copy.primary_label.astype(str)) - PRIMARY_LABELS
        if invalid_primary:
            raise ValueError(f"{name} has invalid primary labels: {sorted(invalid_primary)}")
        for axis in TAXONOMY_AXES:
            values = pd.to_numeric(copy[axis], errors="raise")
            if values.isna().any() or not set(values.astype(int)).issubset({0, 1}):
                raise ValueError(f"{name}.{axis} must contain only 0/1")
            copy[axis] = values.astype(int)
        validated.append(copy)
    if sorted(validated[0].sample_id) != sorted(validated[1].sample_id):
        raise ValueError("Annotator memberships differ")
    paired = validated[0][list(required)].merge(
        validated[1][list(required)], on="sample_id", suffixes=("_1", "_2"), validate="one_to_one"
    )
    metrics: dict[str, Any] = {}
    for label in (*TAXONOMY_AXES, "primary_label"):
        one, two = paired[f"{label}_1"], paired[f"{label}_2"]
        kappa = float(cohen_kappa_score(one, two))
        metrics[label] = {
            "cohen_kappa": kappa if math.isfinite(kappa) else None,
            "raw_percent_agreement": float((one == two).mean()),
            "annotator_1_prevalence": one.astype(str).value_counts(normalize=True).sort_index().to_dict(),
            "annotator_2_prevalence": two.astype(str).value_counts(normalize=True).sort_index().to_dict(),
            "passes_kappa": bool(math.isfinite(kappa) and kappa >= MIN_TAXONOMY_KAPPA),
        }
    compared = [*TAXONOMY_AXES, "primary_label"]
    disagreement_mask = np.logical_or.reduce(
        [paired[f"{label}_1"].astype(str) != paired[f"{label}_2"].astype(str) for label in compared]
    )
    failures = [label for label, metric in metrics.items() if not metric["passes_kappa"]]
    return {
        "schema_version": PHASE06B_SCHEMA_VERSION,
        "rows": len(paired), "threshold": MIN_TAXONOMY_KAPPA, "metrics": metrics,
        "agreement_gate_passed": not failures, "failed_labels": failures,
        "disagreement_rows": int(disagreement_mask.sum()),
        "disagreements": paired.loc[disagreement_mask].copy(),
    }


def validate_taxonomy_manifest(
    payload: Mapping[str, Any], *, taxonomy_path: Path | None = None,
    validation_ids_path: Path | None = None, allow_signed_exception: bool = True,
) -> dict[str, Any]:
    required = {
        "schema_version", "status", "rows", "taxonomy_sha256", "validation_ids_sha256",
        "annotator_1_sha256", "annotator_2_sha256", "adjudicated_sha256", "guideline_sha256",
        "agreement", "annotators", "adjudicator", "created_at_utc",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Taxonomy manifest v2 is missing fields: {missing}")
    if int(payload["schema_version"]) != PHASE06B_SCHEMA_VERSION:
        raise ValueError("Taxonomy manifest must use schema_version=2; legacy manifests require explicit migration")
    if int(payload["rows"]) != EXPECTED_VALIDATION_ROWS:
        raise ValueError("Taxonomy manifest must contain exactly 298 rows")
    hash_fields = ("taxonomy_sha256", "validation_ids_sha256", "annotator_1_sha256", "annotator_2_sha256", "adjudicated_sha256", "guideline_sha256")
    for field in hash_fields:
        _require_sha256(payload[field], field)
    if payload["validation_ids_sha256"] != EXPECTED_VALIDATION_IDS_SHA256:
        raise ValueError("Taxonomy manifest validation checksum is not canonical")
    _require_named_provenance(payload["annotators"], "annotators")
    _require_named_provenance(payload["adjudicator"], "adjudicator")
    if not str(payload["created_at_utc"]).strip():
        raise ValueError("created_at_utc is required")
    agreement = payload["agreement"]
    if not isinstance(agreement, Mapping) or set(agreement.get("metrics", {})) != {*TAXONOMY_AXES, "primary_label"}:
        raise ValueError("Taxonomy manifest agreement metrics are incomplete")
    failures = [name for name, metric in agreement["metrics"].items() if metric.get("cohen_kappa") is None or float(metric["cohen_kappa"]) < MIN_TAXONOMY_KAPPA]
    exception = payload.get("signed_exception")
    exception_valid = False
    if failures and allow_signed_exception and isinstance(exception, Mapping):
        exception_valid = all(str(exception.get(key, "")).strip() for key in ("approved_by", "approved_at_utc", "rationale"))
    expected_status = "complete" if not failures or exception_valid else "awaiting_reannotation"
    if payload["status"] != expected_status:
        raise ValueError(f"Taxonomy status must be {expected_status!r} for the recorded agreement")
    if taxonomy_path is not None and sha256_file(Path(taxonomy_path)) != payload["taxonomy_sha256"]:
        raise ValueError("Frozen taxonomy hash mismatch")
    if validation_ids_path is not None and sha256_file(Path(validation_ids_path)) != payload["validation_ids_sha256"]:
        raise ValueError("Frozen validation IDs file hash mismatch")
    return dict(payload)


def build_taxonomy_manifest(
    *, taxonomy_path: Path, validation_ids_path: Path, annotator_1_path: Path,
    annotator_2_path: Path, adjudicated_path: Path, guideline_path: Path,
    agreement: Mapping[str, Any], annotators: Sequence[Mapping[str, Any]],
    adjudicator: Mapping[str, Any], signed_exception: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    clean_agreement = {key: value for key, value in agreement.items() if key != "disagreements"}
    payload: dict[str, Any] = {
        "schema_version": PHASE06B_SCHEMA_VERSION,
        "status": "complete" if not clean_agreement.get("failed_labels") or signed_exception else "awaiting_reannotation",
        "rows": EXPECTED_VALIDATION_ROWS,
        "taxonomy_sha256": sha256_file(taxonomy_path), "validation_ids_sha256": sha256_file(validation_ids_path),
        "annotator_1_sha256": sha256_file(annotator_1_path), "annotator_2_sha256": sha256_file(annotator_2_path),
        "adjudicated_sha256": sha256_file(adjudicated_path), "guideline_sha256": sha256_file(guideline_path),
        "agreement": clean_agreement, "annotators": list(annotators), "adjudicator": dict(adjudicator),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if signed_exception:
        payload["signed_exception"] = dict(signed_exception)
    return validate_taxonomy_manifest(payload, taxonomy_path=taxonomy_path, validation_ids_path=validation_ids_path)


def validate_taxonomy_frame(taxonomy: pd.DataFrame, expected_ids: list[Any]) -> pd.DataFrame:
    required = {"sample_id", "primary_label", *TAXONOMY_AXES}
    missing = sorted(required - set(taxonomy.columns))
    if missing:
        raise ValueError(f"Frozen taxonomy is missing columns: {missing}")
    result = taxonomy.copy()
    result["sample_id"] = result.sample_id.astype(str)
    expected = sorted(str(value) for value in expected_ids)
    if result.sample_id.duplicated().any() or sorted(result.sample_id) != expected:
        raise ValueError("Frozen taxonomy membership differs from frozen validation IDs")
    if result[list(required)].isna().any().any():
        raise ValueError("Frozen taxonomy cannot contain missing labels")
    invalid_primary = set(result.primary_label.astype(str)) - PRIMARY_LABELS
    if invalid_primary:
        raise ValueError(f"Unknown primary taxonomy labels: {sorted(invalid_primary)}")
    for axis in TAXONOMY_AXES:
        values = pd.to_numeric(result[axis], errors="raise").astype(int)
        if not set(values).issubset({0, 1}):
            raise ValueError(f"Taxonomy axis {axis} must contain only 0/1")
        result[axis] = values
    return result


def load_phase06a_predictions(project_root: Path, expected_ids: list[Any]) -> dict[str, pd.DataFrame]:
    root = Path(project_root)
    predictions: dict[str, pd.DataFrame] = {}
    for experiment, relative in PHASE06A_PREDICTION_PATHS.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        validate_prediction_artifact(frame, expected_ids, run_scope="full")
        predictions[experiment] = frame
    return predictions


def taxonomy_slice_metrics(
    predictions: Mapping[str, pd.DataFrame],
    taxonomy: pd.DataFrame,
    *,
    min_rows: int = 30,
    min_groups: int = 15,
) -> pd.DataFrame:
    if min_rows <= 0 or min_groups <= 0:
        raise ValueError("Slice thresholds must be positive")
    slices: list[tuple[str, pd.Series]] = []
    for label in sorted(PRIMARY_LABELS):
        slices.append((f"primary:{label}", taxonomy.primary_label.astype(str).eq(label)))
    for axis in TAXONOMY_AXES:
        slices.append((f"axis:{axis}", taxonomy[axis].astype(int).eq(1)))
    records: list[dict[str, Any]] = []
    taxonomy_keys = taxonomy[["sample_id", "primary_label", *TAXONOMY_AXES]].copy()
    for experiment, raw in predictions.items():
        merged = raw.copy()
        merged["sample_id"] = merged.sample_id.astype(str)
        merged = merged.merge(taxonomy_keys, on="sample_id", validate="one_to_one", suffixes=("", "_taxonomy"))
        for slice_name, taxonomy_mask in slices:
            member_ids = set(taxonomy.loc[taxonomy_mask, "sample_id"].astype(str))
            subset = merged[merged.sample_id.isin(member_ids)].copy()
            rows = len(subset)
            groups = int(subset.group_id.astype(str).nunique()) if rows else 0
            record: dict[str, Any] = {
                "experiment": experiment,
                "slice": slice_name,
                "rows": rows,
                "groups": groups,
                "inference_ready": rows >= min_rows and groups >= min_groups,
            }
            if rows:
                metrics = compute_classification_metrics(subset)
                record.update(
                    accuracy=metrics["accuracy"],
                    macro_f1=metrics["macro_f1"],
                    parse_rate=metrics["parse_rate"],
                    errors=int((~subset.correct.astype(bool)).sum()),
                )
            else:
                record.update(accuracy=None, macro_f1=None, parse_rate=None, errors=0)
            records.append(record)
    return pd.DataFrame(records)


def build_novelty_decision_evidence(
    winner_predictions: pd.DataFrame,
    taxonomy: pd.DataFrame,
    *,
    min_rows: int = 30,
    min_groups: int = 15,
    min_error_share: float = 0.30,
    min_error_share_gap: float = 0.10,
) -> dict[str, Any]:
    if min_rows < 30 or min_groups < 15:
        raise ValueError("Novelty evidence thresholds cannot be weaker than 30 rows / 15 groups")
    if not 0 <= min_error_share <= 1 or not 0 <= min_error_share_gap <= 1:
        raise ValueError("Error-share thresholds must lie in [0, 1]")
    winner = winner_predictions.copy()
    winner["sample_id"] = winner.sample_id.astype(str)
    merged = winner.merge(
        taxonomy[["sample_id", "temporal_required", "traffic_knowledge_required"]],
        on="sample_id", validate="one_to_one",
    )
    errors = ~merged.correct.astype(bool)
    total_errors = int(errors.sum())
    mapping = {"knowledge_augmented": "traffic_knowledge_required", "traffic_temporal_grounding": "temporal_required"}
    tracks: dict[str, Any] = {}
    for track, axis in mapping.items():
        subset = merged[merged[axis].astype(int).eq(1)]
        error_count = int((~subset.correct.astype(bool)).sum())
        error_share = float(error_count / total_errors) if total_errors else 0.0
        row_group_eligible = len(subset) >= min_rows and subset.group_id.astype(str).nunique() >= min_groups
        tracks[track] = {
            "axis": axis, "rows": len(subset), "groups": int(subset.group_id.astype(str).nunique()),
            "errors": error_count, "error_rate": float(error_count / len(subset)) if len(subset) else None,
            "error_share": error_share, "row_group_eligible": row_group_eligible,
            "eligible": row_group_eligible and error_share >= min_error_share,
        }
    ordered = sorted(tracks, key=lambda name: tracks[name]["error_share"], reverse=True)
    gap = tracks[ordered[0]]["error_share"] - tracks[ordered[1]]["error_share"]
    if total_errors == 0 or not any(values["row_group_eligible"] for values in tracks.values()):
        recommendation, reason = "insufficient_evidence", "No target slice has sufficient rows/groups or the baseline has no errors."
    elif tracks[ordered[0]]["eligible"] and gap >= min_error_share_gap:
        recommendation = ordered[0]
        reason = f"Top target meets row/group/error-share gates and gap {gap:.4f} meets {min_error_share_gap:.4f}."
    else:
        recommendation = "manual_review_required"
        reason = "Automatic recommendation gates were not all met; human review remains required."
    return {
        "schema_version": PHASE06B_SCHEMA_VERSION, "winner": LOCKED_BASELINE_WINNER,
        "winner_rows": len(merged), "winner_total_errors": total_errors,
        "thresholds": {"min_rows": min_rows, "min_groups": min_groups, "min_error_share": min_error_share, "min_error_share_gap": min_error_share_gap},
        "tracks": tracks, "error_share_gap": gap, "recommendation": recommendation,
        "recommendation_reason": reason,
        "overlap_note": "Taxonomy axes may overlap; error shares are diagnostic and need not sum to one.",
        "human_decision_required": True,
    }

def validate_track_decision(decision: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    required = {"status", "selected_track", "decision_by", "rationale", "evidence_sha256"}
    missing = sorted(required - set(decision))
    if missing:
        raise ValueError(f"Novelty decision is missing fields: {missing}")
    selected = str(decision["selected_track"])
    if decision["status"] != "locked" or selected not in NOVELTY_TRACKS:
        raise ValueError("Novelty decision must lock exactly one supported track")
    if not str(decision["decision_by"]).strip() or not str(decision["rationale"]).strip():
        raise ValueError("Novelty decision requires a named decision maker and rationale")
    expected_hash = sha256_json(evidence)
    if decision["evidence_sha256"] != expected_hash:
        raise ValueError("Novelty decision does not reference the current evidence hash")
    return dict(decision)


def build_locked_novelty_protocol(
    selected_track: str,
    *,
    baseline_manifest: Mapping[str, Any],
    taxonomy_sha256: str,
    evidence_sha256: str,
) -> dict[str, Any]:
    if selected_track not in NOVELTY_TRACKS:
        raise ValueError(f"Unsupported novelty track: {selected_track}")
    return {
        "schema_version": PHASE06B_SCHEMA_VERSION,
        "status": "locked",
        "selected_track": selected_track,
        "baseline_winner": LOCKED_BASELINE_WINNER,
        "baseline_predictions_sha256": baseline_manifest["winner_predictions"]["sha256"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "validation_rows": EXPECTED_VALIDATION_ROWS,
        "validation_ids_sha256": EXPECTED_VALIDATION_IDS_SHA256,
        "taxonomy_sha256": taxonomy_sha256,
        "decision_evidence_sha256": evidence_sha256,
        "fixed_variables": {
            "adapter": "L32-F1 Phase 06A final adapter",
            "prompt": "Phase 06A canonical MCQ prompt",
            "parser": "Phase 06A A/B/C/D parser",
            "generation": {"max_new_tokens": 16, "do_sample": False, "num_beams": 1},
            "checkpoint_selection": "train-side inner_dev only",
            "bootstrap_resamples": 10_000,
            "mcnemar_role": "secondary",
            "multiple_comparison_correction": "Holm",
        },
        "forbidden": [
            "public-test label access",
            "validation-driven checkpoint selection",
            "validation support-frame selection",
            "simultaneous unregistered model/frame/retrieval changes",
        ],
    }


def validate_knowledge_corpus_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {"corpus_name", "version", "effective_date_cutoff", "documents"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Knowledge corpus manifest is missing fields: {missing}")
    documents = payload["documents"]
    if not isinstance(documents, list) or not documents:
        raise ValueError("Knowledge corpus manifest requires at least one document")
    document_fields = {
        "document_id",
        "title",
        "local_path",
        "source_url",
        "issuing_authority",
        "effective_date",
        "sha256",
        "license_or_access_note",
    }
    ids = []
    for index, document in enumerate(documents):
        missing_document = sorted(document_fields - set(document))
        if missing_document:
            raise ValueError(f"Knowledge document {index} is missing fields: {missing_document}")
        if not all(str(document[field]).strip() for field in document_fields):
            raise ValueError(f"Knowledge document {index} contains blank provenance fields")
        ids.append(str(document["document_id"]))
    if len(ids) != len(set(ids)):
        raise ValueError("Knowledge document IDs must be unique")
    return dict(payload)


def validate_temporal_input_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "visual_encoder", "visual_encoder_revision", "visual_encoder_checkpoint_sha256",
        "question_encoder", "question_encoder_revision", "question_encoder_checkpoint_sha256",
        "preprocessing_sha256", "dtype", "normalization", "candidate_count",
        "support_annotation_split", "support_annotations_path", "support_annotations_sha256",
        "feature_bank_schema_version", "split_membership_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Temporal input manifest is missing fields: {missing}")
    if int(payload["schema_version"]) != PHASE06B_SCHEMA_VERSION or int(payload["feature_bank_schema_version"]) != PHASE06B_SCHEMA_VERSION:
        raise ValueError("Temporal input and feature-bank manifests must use schema version 2")
    if int(payload["candidate_count"]) != LOCKED_CANDIDATE_COUNT:
        raise ValueError("Temporal grounding protocol requires 32 candidates")
    if str(payload["support_annotation_split"]).strip().lower() != "train":
        raise ValueError("Support annotations must be train-only")
    support_path = str(payload["support_annotations_path"]).casefold()
    if any(token in support_path for token in ("validation", "public_test", "public-test")):
        raise ValueError("Support annotation path cannot reference validation/public test")
    for field in ("visual_encoder_checkpoint_sha256", "question_encoder_checkpoint_sha256", "preprocessing_sha256", "support_annotations_sha256", "split_membership_sha256"):
        _require_sha256(payload[field], field)
    for field in ("visual_encoder", "visual_encoder_revision", "question_encoder", "question_encoder_revision", "dtype", "normalization"):
        if not str(payload[field]).strip():
            raise ValueError(f"Temporal input manifest requires nonblank {field}")
    if "traffic_extractor" in payload:
        validate_traffic_extractor_manifest(payload["traffic_extractor"])
    return dict(payload)

def assert_selected_track(protocol: Mapping[str, Any], expected_track: str) -> None:
    if expected_track not in NOVELTY_TRACKS:
        raise ValueError(f"Unknown expected track: {expected_track}")
    if protocol.get("status") != "locked" or protocol.get("selected_track") != expected_track:
        raise ValueError(f"Notebook requires selected track {expected_track!r}")



TEMPORAL_SUPPORT_COLUMNS = {
    "sample_id", "group_id", "video_sha256", "support_start_sec", "support_end_sec",
    "annotation_source", "annotator", "adjudication_status", "split",
}
FEATURE_BANK_SPLITS = {"train_fit", "inner_dev", "train", "validation"}
FULL_TEMPORAL_PREDICTION_COLUMNS = {
    "sample_id", "group_id", "answer", "prediction", "raw_response", "parse_status", "correct",
    "candidate_indices", "candidate_timestamps_sec", "candidate_scores", "selected_indices",
    "selected_timestamps_sec", "selected_k", "tile_allocation", "realized_tile_count",
    "feature_latency_seconds", "selector_latency_seconds", "vlm_latency_seconds",
    "model_sha256", "adapter_sha256", "selector_sha256", "feature_bank_sha256", "config_sha256",
}


def validate_temporal_support_annotations(
    frame: pd.DataFrame, *, expected_train_ids: Sequence[Any] | None = None,
    expected_group_by_id: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    missing = sorted(TEMPORAL_SUPPORT_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Temporal support annotations are missing columns: {missing}")
    result = frame.copy()
    result["sample_id"] = result.sample_id.astype(str)
    result["group_id"] = result.group_id.astype(str)
    if not result.split.astype(str).str.casefold().eq("train").all():
        raise ValueError("Temporal support annotations must contain split=train only")
    for field in ("video_sha256",):
        for value in result[field]:
            _require_sha256(value, field)
    starts = pd.to_numeric(result.support_start_sec, errors="raise")
    ends = pd.to_numeric(result.support_end_sec, errors="raise")
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or (starts < 0).any() or (ends < starts).any():
        raise ValueError("Temporal support intervals must be finite with 0 <= start <= end")
    for field in ("annotation_source", "annotator", "adjudication_status"):
        if result[field].isna().any() or result[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"Temporal support annotations require nonblank {field}")
    if expected_train_ids is not None and not set(result.sample_id).issubset(set(map(str, expected_train_ids))):
        raise ValueError("Temporal support annotations contain non-training sample IDs")
    if expected_group_by_id is not None:
        bad = result[result.apply(lambda row: expected_group_by_id.get(row.sample_id) != row.group_id, axis=1)]
        if len(bad):
            raise ValueError("Temporal support group IDs do not match the canonical training split")
    result["support_start_sec"], result["support_end_sec"] = starts, ends
    return result


def validate_traffic_extractor_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "feature_vocabulary", "model_id", "model_revision", "checkpoint_sha256",
        "confidence_thresholds", "aggregation_policy", "output_dimension",
        "missing_detection_behavior", "license", "provenance",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Traffic extractor manifest is missing fields: {missing}")
    if not isinstance(payload["feature_vocabulary"], list) or not payload["feature_vocabulary"]:
        raise ValueError("Traffic feature vocabulary must be non-empty")
    if len(payload["feature_vocabulary"]) != int(payload["output_dimension"]):
        raise ValueError("Traffic output_dimension must match the feature vocabulary")
    _require_sha256(payload["checkpoint_sha256"], "checkpoint_sha256")
    for field in ("model_id", "model_revision", "aggregation_policy", "missing_detection_behavior", "license", "provenance"):
        if not str(payload[field]).strip():
            raise ValueError(f"Traffic extractor requires nonblank {field}")
    return dict(payload)


def validate_feature_bank_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "split", "rows", "membership_sha256", "candidate_count",
        "source_video_manifest_sha256", "question_manifest_sha256", "visual_encoder_sha256",
        "question_encoder_sha256", "preprocessing_sha256", "dtype", "normalization",
        "frame_feature_dim", "question_feature_dim", "contains_relevance_targets",
        "bank_sha256", "scope",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Feature-bank manifest is missing fields: {missing}")
    if int(payload["schema_version"]) != PHASE06B_SCHEMA_VERSION:
        raise ValueError("Feature-bank schema version mismatch")
    split = str(payload["split"]).casefold()
    if split not in FEATURE_BANK_SPLITS:
        raise ValueError(f"Unknown feature-bank split: {split}")
    if int(payload["candidate_count"]) != LOCKED_CANDIDATE_COUNT:
        raise ValueError("Feature bank must use exactly 32 candidates")
    for field in ("membership_sha256", "source_video_manifest_sha256", "question_manifest_sha256", "visual_encoder_sha256", "question_encoder_sha256", "preprocessing_sha256", "bank_sha256"):
        _require_sha256(payload[field], field)
    if split == "validation" and bool(payload["contains_relevance_targets"]):
        raise ValueError("Validation feature banks cannot contain relevance targets")
    if str(payload["scope"]) not in {"smoke", "full"}:
        raise ValueError("Feature bank scope must be smoke or full")
    if int(payload["rows"]) <= 0 or int(payload["frame_feature_dim"]) <= 0 or int(payload["question_feature_dim"]) <= 0:
        raise ValueError("Feature-bank dimensions and row count must be positive")
    return dict(payload)


def validate_temporal_experiment_registry(payload: Mapping[str, Any], *, protocol: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "status", "selected_track", "parent_protocol_sha256", "validation_ids_sha256", "arms", "winner_candidates"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Experiment registry is missing fields: {missing}")
    if int(payload["schema_version"]) != PHASE06B_SCHEMA_VERSION or payload["status"] != "locked":
        raise ValueError("Experiment registry must be locked schema version 2")
    if payload["selected_track"] != "traffic_temporal_grounding" or protocol.get("selected_track") != "traffic_temporal_grounding":
        raise ValueError("Temporal registry requires the locked traffic_temporal_grounding track")
    if payload["parent_protocol_sha256"] != sha256_json(protocol):
        raise ValueError("Experiment registry parent protocol hash mismatch")
    if payload["validation_ids_sha256"] != EXPECTED_VALIDATION_IDS_SHA256:
        raise ValueError("Experiment registry validation membership mismatch")
    arms = payload["arms"]
    if not isinstance(arms, list):
        raise ValueError("Experiment registry arms must be a list")
    by_name = {str(arm.get("name")): arm for arm in arms}
    if len(by_name) != len(arms) or set(by_name) != set(EXPERIMENT_ROLES):
        raise ValueError("Experiment registry has missing, duplicate, or extra arms")
    required_arm_fields = {
        "name", "experiment_role", "selector_family", "k", "candidate_count", "total_visual_tile_budget",
        "tile_allocation", "config_sha256", "checkpoint_sha256", "feature_bank_sha256",
        "predictions_path", "predictions_sha256", "validation_ids_sha256", "winner_eligible",
    }
    for name, (role, selector, k, eligible) in EXPERIMENT_ROLES.items():
        arm = by_name[name]
        absent = sorted(required_arm_fields - set(arm))
        if absent:
            raise ValueError(f"Registry arm {name} is missing fields: {absent}")
        if (arm["experiment_role"], arm["selector_family"], int(arm["k"]), bool(arm["winner_eligible"])) != (role, selector, k, eligible):
            raise ValueError(f"Registry arm {name} has an invalid role/selector/k/winner eligibility")
        if int(arm["candidate_count"]) != LOCKED_CANDIDATE_COUNT or int(arm["total_visual_tile_budget"]) != LOCKED_TOTAL_TILE_BUDGET:
            raise ValueError(f"Registry arm {name} violates candidate/tile budget")
        if list(arm["tile_allocation"]) != fixed_tile_allocation(k, LOCKED_TOTAL_TILE_BUDGET):
            raise ValueError(f"Registry arm {name} has invalid tile allocation")
        if arm["validation_ids_sha256"] != EXPECTED_VALIDATION_IDS_SHA256:
            raise ValueError(f"Registry arm {name} validation membership mismatch")
        if not str(arm["predictions_path"]).strip():
            raise ValueError(f"Registry arm {name} requires predictions_path")
        for field in ("config_sha256", "checkpoint_sha256", "feature_bank_sha256", "predictions_sha256"):
            _require_sha256(arm[field], f"{name}.{field}")
    candidates = set(map(str, payload["winner_candidates"]))
    if candidates != WINNER_ELIGIBLE_ARMS or any(name.startswith("ORACLE-") for name in candidates):
        raise ValueError("Winner candidates must contain only L32-F1 and all QTG/TATG arms")
    return dict(payload)


def validate_full_temporal_predictions(
    frame: pd.DataFrame, expected_ids: Sequence[Any], *, arm: Mapping[str, Any],
) -> dict[str, Any]:
    missing = sorted(FULL_TEMPORAL_PREDICTION_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Temporal predictions are missing provenance columns: {missing}")
    result = frame.copy()
    result["sample_id"] = result.sample_id.astype(str)
    expected = sorted(map(str, expected_ids))
    if len(result) != EXPECTED_VALIDATION_ROWS or result.sample_id.duplicated().any() or sorted(result.sample_id) != expected:
        raise ValueError("Full temporal predictions must contain exactly the 298 unique frozen IDs")
    invalid_correct = result.parse_status.astype(str).ne("parsed") & result.correct.astype(bool)
    if invalid_correct.any() or result.raw_response.isna().any():
        raise ValueError("Parse failures must retain raw responses and count as incorrect")
    k = int(arm["k"])
    for _, row in frame.iterrows():
        candidates = json.loads(row.candidate_indices) if isinstance(row.candidate_indices, str) else row.candidate_indices
        times = json.loads(row.candidate_timestamps_sec) if isinstance(row.candidate_timestamps_sec, str) else row.candidate_timestamps_sec
        scores = json.loads(row.candidate_scores) if isinstance(row.candidate_scores, str) else row.candidate_scores
        selected = json.loads(row.selected_indices) if isinstance(row.selected_indices, str) else row.selected_indices
        selected_times = json.loads(row.selected_timestamps_sec) if isinstance(row.selected_timestamps_sec, str) else row.selected_timestamps_sec
        allocation = json.loads(row.tile_allocation) if isinstance(row.tile_allocation, str) else row.tile_allocation
        if len(candidates) != LOCKED_CANDIDATE_COUNT or len(times) != LOCKED_CANDIDATE_COUNT or len(scores) != LOCKED_CANDIDATE_COUNT:
            raise ValueError("Every full row requires all 32 candidate indices/timestamps/scores")
        if len(selected) != k or len(selected_times) != k or int(row.selected_k) != k:
            raise ValueError("Selected-frame provenance does not match arm k")
        if list(allocation) != fixed_tile_allocation(k, LOCKED_TOTAL_TILE_BUDGET) or int(row.realized_tile_count) > LOCKED_TOTAL_TILE_BUDGET:
            raise ValueError("Prediction row violates the fixed total tile budget")
        for field in ("config_sha256", "checkpoint_sha256", "feature_bank_sha256"):
            if str(row[field]) != str(arm[field]):
                raise ValueError(f"Prediction {field} does not match the registry")
    return {"rows": len(frame), "unique_ids": frame.sample_id.astype(str).nunique(), "arm": arm["name"], "integrity": "pass"}
