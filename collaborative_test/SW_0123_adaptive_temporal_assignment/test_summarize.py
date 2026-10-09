import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import summarize


def score(values):
    array = np.asarray(values, dtype=np.float64)
    return {"metrics": {"fg_ari": float(array.mean()),
                        "foreground_iou": float(array.mean()),
                        "matched_object_iou": float(array.mean())},
            "per_image": {name: array.copy() for name in summarize.METRICS},
            "secondary_actual_qcc_metrics": {name: float(array.mean())
                                              for name in summarize.METRICS}}


class SummaryGateTests(unittest.TestCase):
    def test_bootstrap_uses_same_image_draw_indices_across_seed_rows(self):
        candidate, reference = {}, {}
        image = np.arange(320, dtype=np.float64) / 320
        for seed in summarize.SEEDS:
            candidate[seed] = score(image + 0.4 + seed * 0.01)
            reference[seed] = score(image * 0.2)
        result = summarize.paired_bootstrap(candidate, reference, draws=37, seed=123)
        per_image = np.stack([candidate[s]["per_image"]["fg_ari"]
                              - reference[s]["per_image"]["fg_ari"]
                              for s in summarize.SEEDS]).mean(axis=0)
        indices = np.random.default_rng(123).integers(0, 320, size=(37, 320))
        expected = np.quantile(per_image[indices].mean(axis=1), [.025, .975])
        np.testing.assert_allclose(result["ci95"], expected, rtol=0, atol=0)
        self.assertEqual(result["draws"], 37)
        self.assertEqual(result["seed"], 123)

    def test_registered_promotion_gate_passes_only_with_all_guards_and_margins(self):
        values = {"source": .40, "legacy_full": .45, "gate_only_control": .46,
                  "adaptive_full": .70}
        scores = {arm: {seed: score(np.full(320, mean)) for seed in summarize.SEEDS}
                  for arm, mean in values.items()}
        slot = {"foreground_iou": .20, "matched_object_iou": .20, "fg_ari": .77}
        guards = {str(seed): {"training_guard_status": True, "activity_guard": True,
                              "assignment_credit_guard": True}
                  for seed in summarize.SEEDS}
        means, bootstrap, gains, gates = summarize.promotion_gates(scores, slot, guards)
        self.assertTrue(all(gates.values()))
        self.assertEqual(set(bootstrap), {"legacy_full", "gate_only_control", "source"})
        for value in gains.values():
            self.assertAlmostEqual(value, .3)
        self.assertGreater(means["adaptive_full"]["fg_ari"], means["source"]["fg_ari"])

        guards["1"]["activity_guard"] = False
        _, _, _, failed = summarize.promotion_gates(scores, slot, guards)
        self.assertFalse(failed["all_adaptive_posttraining_guards_passed"])

    def test_incomplete_task_set_returns_pending_without_metrics(self):
        missing = MagicMock(name="missing_artifact")
        missing.is_file.return_value = False
        with patch.object(summarize.coordinator, "artifact_path", return_value=missing):
            result = summarize.summarize()
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["registered_task_count"], 27)
        self.assertEqual(result["validated_task_count"], 0)
        self.assertEqual(len(result["pending_task_ids"]), 27)
        self.assertNotIn("three_seed_means", result)
        self.assertFalse(result["promotion_gate_passed"])


if __name__ == "__main__":
    unittest.main()
