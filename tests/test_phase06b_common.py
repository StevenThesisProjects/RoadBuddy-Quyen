import unittest
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase06a_common import sha256_json
from phase06b_common import (
    build_novelty_decision_evidence,
    validate_knowledge_corpus_manifest,
    validate_taxonomy_frame,
    validate_temporal_input_manifest,
    validate_track_decision,
)


class Phase06BCommonTest(unittest.TestCase):
    def taxonomy(self):
        return pd.DataFrame(
            {
                "sample_id": ["a", "b", "c", "d"],
                "primary_label": ["temporal", "traffic_knowledge", "visual_static", "mixed"],
                "visual_required": [1, 0, 1, 1],
                "temporal_required": [1, 0, 0, 1],
                "traffic_knowledge_required": [0, 1, 0, 1],
                "mixed_or_ambiguous": [0, 0, 0, 1],
            }
        )

    def test_taxonomy_requires_exact_membership_and_binary_axes(self):
        validated = validate_taxonomy_frame(self.taxonomy(), ["a", "b", "c", "d"])
        self.assertEqual(len(validated), 4)
        broken = self.taxonomy()
        broken.loc[0, "temporal_required"] = 2
        with self.assertRaises(ValueError):
            validate_taxonomy_frame(broken, ["a", "b", "c", "d"])

    def test_decision_evidence_and_human_lock(self):
        predictions = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c", "d"],
                "group_id": ["g1", "g2", "g3", "g4"],
                "correct": [False, True, True, False],
            }
        )
        evidence = build_novelty_decision_evidence(
            predictions,
            self.taxonomy(),
            min_rows=1,
            min_groups=1,
            min_error_share_gap=0.6,
        )
        self.assertEqual(evidence["recommendation"], "manual_review_required")
        decision = {
            "status": "locked",
            "selected_track": "traffic_temporal_grounding",
            "decision_by": "researcher",
            "rationale": "Temporal slice selected after review.",
            "evidence_sha256": sha256_json(evidence),
        }
        self.assertEqual(validate_track_decision(decision, evidence)["status"], "locked")

    def test_knowledge_manifest_requires_provenance(self):
        payload = {
            "corpus_name": "laws",
            "version": "v1",
            "effective_date_cutoff": "2026-08-25",
            "documents": [
                {
                    "document_id": "law-1",
                    "title": "Law",
                    "local_path": "data/knowledge/law-1.txt",
                    "source_url": "https://example.invalid/law",
                    "issuing_authority": "authority",
                    "effective_date": "2025-01-01",
                    "sha256": "abc",
                    "license_or_access_note": "public source",
                }
            ],
        }
        self.assertEqual(validate_knowledge_corpus_manifest(payload)["version"], "v1")

    def test_temporal_manifest_rejects_validation_support(self):
        payload = {
            "visual_encoder": "encoder",
            "visual_encoder_revision": "rev",
            "question_encoder": "encoder",
            "question_encoder_revision": "rev",
            "candidate_count": 32,
            "support_annotation_split": "train",
            "support_annotations_path": "data/train/support.csv",
            "support_annotations_sha256": "abc",
            "feature_bank_schema_version": 1,
        }
        self.assertEqual(validate_temporal_input_manifest(payload)["candidate_count"], 32)
        payload["support_annotations_path"] = "data/validation/support.csv"
        with self.assertRaises(ValueError):
            validate_temporal_input_manifest(payload)


if __name__ == "__main__":
    unittest.main()
