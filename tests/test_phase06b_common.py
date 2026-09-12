import copy
import sys
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase06a_common import EXPECTED_VALIDATION_IDS_SHA256, fixed_tile_allocation, sha256_json
from phase06b_common import (
    EXPERIMENT_ROLES,
    PHASE06B_SCHEMA_VERSION,
    WINNER_ELIGIBLE_ARMS,
    build_novelty_decision_evidence,
    taxonomy_agreement_report,
    validate_feature_bank_manifest,
    validate_taxonomy_frame,
    validate_taxonomy_manifest,
    validate_temporal_experiment_registry,
    validate_temporal_input_manifest,
    validate_temporal_support_annotations,
    validate_track_decision,
)

H = "a" * 64


class Phase06BCommonTest(unittest.TestCase):
    def taxonomy(self, n=40):
        return pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "primary_label": (["temporal", "traffic_knowledge", "visual_static", "mixed"] * ((n + 3) // 4))[:n],
            "visual_required": [i % 2 for i in range(n)],
            "temporal_required": [int(i < 30) for i in range(n)],
            "traffic_knowledge_required": [int(i >= 25) for i in range(n)],
            "mixed_or_ambiguous": [int(i % 5 == 0) for i in range(n)],
        })

    def agreement(self, kappa=0.8):
        return {"metrics": {name: {"cohen_kappa": kappa} for name in
            ["visual_required", "temporal_required", "traffic_knowledge_required", "mixed_or_ambiguous", "primary_label"]}}

    def taxonomy_manifest(self, kappa=0.8):
        return {
            "schema_version": 2, "status": "complete", "rows": 298,
            "taxonomy_sha256": H, "validation_ids_sha256": EXPECTED_VALIDATION_IDS_SHA256,
            "annotator_1_sha256": H, "annotator_2_sha256": H,
            "adjudicated_sha256": H, "guideline_sha256": H,
            "agreement": self.agreement(kappa),
            "annotators": [{"name": "a1", "completed_at_utc": "2026-09-12T00:00:00Z"}, {"name": "a2", "completed_at_utc": "2026-09-12T00:00:00Z"}],
            "adjudicator": {"name": "judge", "completed_at_utc": "2026-09-12T00:00:00Z"},
            "created_at_utc": "2026-09-12T00:00:00Z",
        }

    def temporal_manifest(self):
        return {
            "schema_version": 2, "visual_encoder": "ve", "visual_encoder_revision": "rev",
            "visual_encoder_checkpoint_sha256": H, "question_encoder": "qe", "question_encoder_revision": "rev",
            "question_encoder_checkpoint_sha256": H, "preprocessing_sha256": H, "dtype": "float32",
            "normalization": "l2", "candidate_count": 32, "support_annotation_split": "train",
            "support_annotations_path": "data/train/support.csv", "support_annotations_sha256": H,
            "feature_bank_schema_version": 2, "split_membership_sha256": H,
        }

    def protocol(self):
        return {"status": "locked", "selected_track": "traffic_temporal_grounding"}

    def registry(self):
        protocol = self.protocol()
        arms = []
        for name, (role, selector, k, eligible) in EXPERIMENT_ROLES.items():
            arms.append({
                "name": name, "experiment_role": role, "selector_family": selector, "k": k,
                "candidate_count": 32, "total_visual_tile_budget": 8,
                "tile_allocation": fixed_tile_allocation(k, 8), "config_sha256": H,
                "checkpoint_sha256": H, "feature_bank_sha256": H, "predictions_path": f"outputs/{name}.csv", "predictions_sha256": H,
                "validation_ids_sha256": EXPECTED_VALIDATION_IDS_SHA256, "winner_eligible": eligible,
            })
        return protocol, {"schema_version": PHASE06B_SCHEMA_VERSION, "status": "locked",
            "selected_track": "traffic_temporal_grounding", "parent_protocol_sha256": sha256_json(protocol),
            "validation_ids_sha256": EXPECTED_VALIDATION_IDS_SHA256, "arms": arms,
            "winner_candidates": sorted(WINNER_ELIGIBLE_ARMS)}

    def test_taxonomy_rejects_membership_axes_and_primary_labels(self):
        frame = self.taxonomy(4)
        self.assertEqual(len(validate_taxonomy_frame(frame, frame.sample_id)), 4)
        broken = frame.copy(); broken.loc[0, "temporal_required"] = 2
        with self.assertRaises(ValueError): validate_taxonomy_frame(broken, frame.sample_id)
        broken = frame.copy(); broken.loc[0, "primary_label"] = "temporal_required"
        with self.assertRaises(ValueError): validate_taxonomy_frame(broken, frame.sample_id)

    def test_taxonomy_manifest_requires_v2_and_human_provenance(self):
        self.assertEqual(validate_taxonomy_manifest(self.taxonomy_manifest())["schema_version"], 2)
        for key in ("schema_version", "annotators", "adjudicator"):
            broken = self.taxonomy_manifest(); broken.pop(key)
            with self.assertRaises(ValueError): validate_taxonomy_manifest(broken)

    def test_kappa_below_threshold_requires_reannotation_or_signed_exception(self):
        broken = self.taxonomy_manifest(0.69)
        with self.assertRaises(ValueError): validate_taxonomy_manifest(broken)
        broken["status"] = "awaiting_reannotation"
        self.assertEqual(validate_taxonomy_manifest(broken)["status"], "awaiting_reannotation")
        broken["status"] = "complete"; broken["signed_exception"] = {"approved_by": "PI", "approved_at_utc": "2026-09-12", "rationale": "documented imbalance"}
        self.assertEqual(validate_taxonomy_manifest(broken)["status"], "complete")

    def test_agreement_reports_kappa_raw_agreement_and_prevalence(self):
        left = self.taxonomy(40); right = left.copy(); right.loc[:19, "temporal_required"] = 1 - right.loc[:19, "temporal_required"]
        report = taxonomy_agreement_report(left, right)
        metric = report["metrics"]["temporal_required"]
        self.assertIn("raw_percent_agreement", metric); self.assertIn("annotator_1_prevalence", metric)
        self.assertFalse(metric["passes_kappa"])

    def test_novelty_thresholds_never_replace_human_decision(self):
        taxonomy = self.taxonomy(40)
        predictions = pd.DataFrame({"sample_id": taxonomy.sample_id, "group_id": [f"g{i//2}" for i in range(40)], "correct": [False] * 30 + [True] * 10})
        evidence = build_novelty_decision_evidence(predictions, taxonomy)
        self.assertTrue(evidence["human_decision_required"])
        self.assertIn(evidence["recommendation"], {"traffic_temporal_grounding", "manual_review_required", "insufficient_evidence"})
        with self.assertRaises(ValueError): build_novelty_decision_evidence(predictions, taxonomy, min_rows=1)

    def test_human_track_lock_requires_provenance_and_current_evidence(self):
        evidence = {"recommendation": "manual_review_required"}
        decision = {"status": "locked", "selected_track": "traffic_temporal_grounding", "decision_by": "researcher", "rationale": "review", "evidence_sha256": sha256_json(evidence)}
        self.assertEqual(validate_track_decision(decision, evidence)["status"], "locked")
        decision["decision_by"] = ""
        with self.assertRaises(ValueError): validate_track_decision(decision, evidence)

    def test_temporal_manifest_rejects_schema_checksum_and_split_leakage(self):
        self.assertEqual(validate_temporal_input_manifest(self.temporal_manifest())["candidate_count"], 32)
        for key, value in (("schema_version", 1), ("support_annotations_path", "data/validation/support.csv"), ("support_annotations_sha256", "abc")):
            broken = self.temporal_manifest(); broken[key] = value
            with self.assertRaises(ValueError): validate_temporal_input_manifest(broken)

    def test_temporal_support_rejects_nontrain_and_bad_intervals(self):
        frame = pd.DataFrame({"sample_id": ["s1"], "group_id": ["g1"], "video_sha256": [H], "support_start_sec": [1.0], "support_end_sec": [2.0], "annotation_source": ["human"], "annotator": ["a"], "adjudication_status": ["verified"], "split": ["train"]})
        self.assertEqual(len(validate_temporal_support_annotations(frame, expected_train_ids=["s1"])), 1)
        broken = frame.copy(); broken.loc[0, "split"] = "validation"
        with self.assertRaises(ValueError): validate_temporal_support_annotations(broken)
        broken = frame.copy(); broken.loc[0, "support_end_sec"] = 0.5
        with self.assertRaises(ValueError): validate_temporal_support_annotations(broken)

    def test_feature_bank_rejects_validation_targets_and_hash_mismatch(self):
        payload = {"schema_version": 2, "split": "validation", "rows": 298, "membership_sha256": H, "candidate_count": 32, "source_video_manifest_sha256": H, "question_manifest_sha256": H, "visual_encoder_sha256": H, "question_encoder_sha256": H, "preprocessing_sha256": H, "dtype": "float32", "normalization": "l2", "frame_feature_dim": 8, "question_feature_dim": 8, "contains_relevance_targets": False, "bank_sha256": H, "scope": "full"}
        self.assertEqual(validate_feature_bank_manifest(payload)["split"], "validation")
        broken = copy.deepcopy(payload); broken["contains_relevance_targets"] = True
        with self.assertRaises(ValueError): validate_feature_bank_manifest(broken)
        broken = copy.deepcopy(payload); broken["bank_sha256"] = "abc"
        with self.assertRaises(ValueError): validate_feature_bank_manifest(broken)

    def test_registry_requires_exact_arms_roles_and_membership(self):
        protocol, registry = self.registry()
        self.assertEqual(len(validate_temporal_experiment_registry(registry, protocol=protocol)["arms"]), 16)
        broken = copy.deepcopy(registry); broken["arms"].pop()
        with self.assertRaises(ValueError): validate_temporal_experiment_registry(broken, protocol=protocol)
        broken = copy.deepcopy(registry); broken["arms"][1]["winner_eligible"] = True
        with self.assertRaises(ValueError): validate_temporal_experiment_registry(broken, protocol=protocol)
        broken = copy.deepcopy(registry); broken["winner_candidates"].append("ORACLE-1")
        with self.assertRaises(ValueError): validate_temporal_experiment_registry(broken, protocol=protocol)
        broken = copy.deepcopy(registry); broken["arms"][0]["validation_ids_sha256"] = H
        with self.assertRaises(ValueError): validate_temporal_experiment_registry(broken, protocol=protocol)


if __name__ == "__main__":
    unittest.main()
