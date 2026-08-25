import unittest

try:
    import torch
except ModuleNotFoundError:  # Desktop runtime may not include PyTorch.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required by temporal-grounding tests")
class TemporalGroundingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.traffic_temporal_grounding import (
            TemporalGroundingConfig,
            TrafficAwareTemporalGrounder,
            select_diverse_topk,
            temporal_grounding_loss,
            uniform_candidate_indices,
            weak_targets_from_support_times,
        )

        globals().update(
            {
                "TemporalGroundingConfig": TemporalGroundingConfig,
                "TrafficAwareTemporalGrounder": TrafficAwareTemporalGrounder,
                "select_diverse_topk": select_diverse_topk,
                "temporal_grounding_loss": temporal_grounding_loss,
                "uniform_candidate_indices": uniform_candidate_indices,
                "weak_targets_from_support_times": weak_targets_from_support_times,
            }
        )

    def test_uniform_candidates_are_deterministic_and_unique(self):
        self.assertEqual(uniform_candidate_indices(100, 4), [12, 37, 62, 87])
        self.assertEqual(uniform_candidate_indices(3, 8), [0, 1, 2])

    def test_support_supervision_is_train_only(self):
        candidates = torch.tensor([0.0, 1.0, 2.0])
        target = weak_targets_from_support_times(
            candidates, torch.tensor([1.0]), split_name="train", sigma_seconds=0.25
        )
        self.assertAlmostEqual(float(target.sum()), 1.0, places=6)
        self.assertEqual(int(target.argmax()), 1)
        with self.assertRaises(ValueError):
            weak_targets_from_support_times(
                candidates, torch.tensor([1.0]), split_name="validation"
            )

    def test_grounder_shapes_loss_and_selection(self):
        torch.manual_seed(42)
        config = TemporalGroundingConfig(
            frame_feature_dim=8,
            question_feature_dim=6,
            traffic_feature_dim=4,
            hidden_dim=16,
            candidate_count=5,
            selected_count=2,
        )
        model = TrafficAwareTemporalGrounder(config)
        frames = torch.randn(2, 5, 8)
        questions = torch.randn(2, 6)
        traffic = torch.randn(2, 5, 4)
        normalized_times = torch.linspace(0, 1, 5).repeat(2, 1)
        seconds = torch.arange(5, dtype=torch.float32).repeat(2, 1)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=torch.bool)
        scores = model(frames, questions, normalized_times, traffic, mask)
        self.assertEqual(scores.shape, (2, 5))
        targets = torch.softmax(torch.randn(2, 5), dim=1)
        loss = temporal_grounding_loss(scores, targets, mask)
        self.assertTrue(torch.isfinite(loss))
        selected, selected_scores = model.select(
            frames, questions, normalized_times, seconds, traffic, mask
        )
        self.assertEqual(selected.shape, (2, 2))
        self.assertEqual(selected_scores.shape, (2, 5))
        for row, row_times in zip(selected, seconds):
            chosen_times = row_times[row]
            self.assertTrue(bool(torch.all(chosen_times[:-1] <= chosen_times[1:])))

    def test_diverse_topk_is_temporally_ordered(self):
        scores = torch.tensor([[0.1, 0.9, 0.8, 0.7]])
        timestamps = torch.tensor([[0.0, 1.0, 1.1, 3.0]])
        selected = select_diverse_topk(
            scores, timestamps, k=2, min_temporal_gap=1.0
        )
        self.assertEqual(selected.tolist(), [[1, 3]])


if __name__ == "__main__":
    unittest.main()
