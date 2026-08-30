from __future__ import annotations

"""Shared gates and analysis helpers for RoadBuddy Phase 06B.

Phase 06B starts only after the Phase 06A baseline winner is locked.  The
module deliberately contains no public-test workflow and never converts
validation support timestamps or answer correctness into development inputs.
"""

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from phase06a_common import (
    EXPECTED_VALIDATION_IDS_SHA256,
    EXPECTED_VALIDATION_ROWS,
    compute_classification_metrics,
    sha256_file,
    sha256_json,
    validate_prediction_artifact,
)
from roadbuddy_common import MODEL_ID, MODEL_REVISION


PHASE06B_SCHEMA_VERSION = 1
LOCKED_BASELINE_WINNER = "L32-F1"
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
    return manifest


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
    min_error_share_gap: float = 0.10,
) -> dict[str, Any]:
    if not 0 <= min_error_share_gap <= 1:
        raise ValueError("min_error_share_gap must lie in [0, 1]")
    winner = winner_predictions.copy()
    winner["sample_id"] = winner.sample_id.astype(str)
    merged = winner.merge(
        taxonomy[["sample_id", "temporal_required", "traffic_knowledge_required"]],
        on="sample_id",
        validate="one_to_one",
    )
    errors = ~merged.correct.astype(bool)
    total_errors = int(errors.sum())
    mapping = {
        "knowledge_augmented": "traffic_knowledge_required",
        "traffic_temporal_grounding": "temporal_required",
    }
    tracks: dict[str, Any] = {}
    for track, axis in mapping.items():
        subset = merged[merged[axis].astype(int).eq(1)]
        error_count = int((~subset.correct.astype(bool)).sum())
        tracks[track] = {
            "axis": axis,
            "rows": len(subset),
            "groups": int(subset.group_id.astype(str).nunique()),
            "errors": error_count,
            "error_rate": float(error_count / len(subset)) if len(subset) else None,
            "error_share": float(error_count / total_errors) if total_errors else 0.0,
            "eligible": len(subset) >= min_rows and subset.group_id.astype(str).nunique() >= min_groups,
        }
    eligible = [name for name, values in tracks.items() if values["eligible"]]
    if len(eligible) == 1:
        recommendation = eligible[0]
        reason = "Only one target slice meets the preregistered row/group thresholds."
    elif len(eligible) == 2:
        ordered = sorted(eligible, key=lambda name: tracks[name]["error_share"], reverse=True)
        gap = tracks[ordered[0]]["error_share"] - tracks[ordered[1]]["error_share"]
        if gap >= min_error_share_gap:
            recommendation = ordered[0]
            reason = f"Winner error-share gap {gap:.4f} meets threshold {min_error_share_gap:.4f}."
        else:
            recommendation = "manual_review_required"
            reason = f"Eligible tracks have error-share gap {gap:.4f}, below threshold {min_error_share_gap:.4f}."
    else:
        recommendation = "insufficient_evidence"
        reason = "Neither target slice meets the preregistered row/group thresholds."
    return {
        "schema_version": PHASE06B_SCHEMA_VERSION,
        "winner": LOCKED_BASELINE_WINNER,
        "winner_rows": len(merged),
        "winner_total_errors": total_errors,
        "thresholds": {
            "min_rows": min_rows,
            "min_groups": min_groups,
            "min_error_share_gap": min_error_share_gap,
        },
        "tracks": tracks,
        "recommendation": recommendation,
        "recommendation_reason": reason,
        "overlap_note": "Taxonomy axes may overlap; error shares are diagnostic and need not sum to one.",
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
        "visual_encoder",
        "visual_encoder_revision",
        "question_encoder",
        "question_encoder_revision",
        "candidate_count",
        "support_annotation_split",
        "support_annotations_path",
        "support_annotations_sha256",
        "feature_bank_schema_version",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Temporal input manifest is missing fields: {missing}")
    if int(payload["candidate_count"]) != 32:
        raise ValueError("Temporal grounding protocol requires 32 candidates")
    if str(payload["support_annotation_split"]).strip().lower() != "train":
        raise ValueError("Support annotations must be train-only")
    support_path = str(payload["support_annotations_path"]).casefold()
    if "validation" in support_path or "public_test" in support_path or "public-test" in support_path:
        raise ValueError("Support annotation path cannot reference validation/public test")
    return dict(payload)


def assert_selected_track(protocol: Mapping[str, Any], expected_track: str) -> None:
    if expected_track not in NOVELTY_TRACKS:
        raise ValueError(f"Unknown expected track: {expected_track}")
    if protocol.get("status") != "locked" or protocol.get("selected_track") != expected_track:
        raise ValueError(f"Notebook requires selected track {expected_track!r}")

