import unittest
import sys
from pathlib import Path

import pandas as pd

try:
    import torch  # noqa: F401
except ModuleNotFoundError:  # Desktop runtime may not include PyTorch.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required by RoadBuddy Phase 06A helpers")
class Phase06ACommonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src_dir = Path(__file__).resolve().parents[1] / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        import phase06a_common as common

        cls.common = common

    def test_fixed_tile_budget(self):
        self.assertEqual(self.common.fixed_tile_allocation(1, 8), [8])
        self.assertEqual(self.common.fixed_tile_allocation(3, 8), [3, 2, 3])
        self.assertEqual(self.common.fixed_tile_allocation(8, 8), [1] * 8)

    def test_inner_split_has_no_group_overlap(self):
        frame = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(12)],
                "group_id": [f"g{i // 2}" for i in range(12)],
                "answer": list("ABCDABCDABCD"),
            }
        )
        fit, dev, report = self.common.group_safe_inner_split(frame, search_trials=16)
        self.assertFalse(set(fit.group_id) & set(dev.group_id))
        self.assertEqual(report["group_overlap"], 0)

    def test_prediction_integrity_rejects_parse_failure_marked_correct(self):
        frame = pd.DataFrame(
            {
                "sample_id": ["s1"],
                "group_id": ["g1"],
                "answer": ["A"],
                "prediction": [None],
                "raw_response": [""],
                "parse_status": ["invalid"],
                "correct": [True],
                "frame_count": [1],
                "frame_indices": ["[1]"],
                "frame_timestamps_sec": ["[0.1]"],
                "num_patches_list": ["[1]"],
                "realized_tile_count": [1],
                "latency_seconds": [0.1],
            }
        )
        with self.assertRaises(ValueError):
            self.common.validate_prediction_artifact(frame, ["s1"], run_scope="smoke")

    def test_exact_mcnemar_and_holm(self):
        result = self.common.exact_mcnemar(
            [True, True, False, False], [True, False, True, False]
        )
        self.assertEqual(result["discordant"], 2)
        adjusted = self.common.holm_adjust({"a": 0.01, "b": 0.04})
        self.assertGreaterEqual(adjusted["a"], 0.01)
        self.assertGreaterEqual(adjusted["b"], adjusted["a"])


if __name__ == "__main__":
    unittest.main()
