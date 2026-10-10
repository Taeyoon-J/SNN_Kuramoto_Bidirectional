"""CPU-only queue/report contract tests for SW0131 read-only diagnostics."""
import sys
import unittest

import numpy as np  # load before torch dependencies

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
from collaborative_test.SW_0131_decoder_partition_diagnosis import queue


class DiagnosticQueueTests(unittest.TestCase):
    def test_plan_contains_only_three_read_only_seed_diagnostics(self):
        tasks = queue.task_plan()
        self.assertEqual([row["seed"] for row in tasks], [0, 1, 2])
        self.assertTrue(all(row["stage"] == "read_only_diagnostic" for row in tasks))
        for task in tasks:
            self.assertIn("--seed", queue.command(task))
            self.assertNotIn("train.py", queue.command(task))

    def test_exclusive_gpu_predicate_rejects_owner_and_excess_memory(self):
        self.assertTrue(queue.gpu_is_exclusive_candidate(512, []))
        self.assertFalse(queue.gpu_is_exclusive_candidate(513, []))
        self.assertFalse(queue.gpu_is_exclusive_candidate(0, [1234]))

    def test_report_validation_accepts_negative_diagnostic_delta_but_rejects_bad_evidence(self):
        ids = list(range(100, 164))
        report = {
            "status": "complete", "experiment": "SW0131_decoder_partition_diagnosis",
            "seed": 2, "images": 64, "image_ids": ids,
            "source_core_sha256": "source", "implementation_fingerprint": {"run": "fp"},
            "asset_hashes": {"train_cache_sha256": "cache"},
            "warm_decoder_sha256": "warm", "optimizer_updates": 0,
            "ground_truth_used": False, "source_checkpoint_modified": False,
            "decoder_artifact_modified": False, "time_steps": 64, "settle": 32,
            "batch_size": 16, "decoder_states": ["initial_random", "warmed32"],
            "conditions": ["native", "row_shuffled", "image_mean_content"],
            "per_image": [
                {"image_id": image_id, "K": 2, "native_group_sizes": [128, 128],
                 "mse": {state: {"native": 0.20, "row_shuffled": 0.19,
                                 "image_mean_content": 0.19}
                         for state in ("initial_random", "warmed32")}}
                for image_id in ids],
            "paired64image_deltas": {
                f"{state}_{condition}_minus_native": {
                    "mean": -0.01, "per_image": [-0.01] * 64}
                for state in ("initial_random", "warmed32")
                for condition in ("row_shuffled", "image_mean_content")},
            "batch_gradient_diagnostics": [
                {"condition": "native", "decoder_state": state, "batch_index": batch,
                 "q_gradient_norm": 0.0, "rgb_gradient_norms_by_family": {"encoder": 0.0}}
                for state in ("initial_random", "warmed32") for batch in range(4)],
        }
        expected = dict(expected_ids=ids, source_sha="source", fingerprint={"run": "fp"},
                        assets={"train_cache_sha256": "cache"}, warm_sha="warm")
        self.assertTrue(queue.validate_report(report, 2, **expected))
        inconsistent = dict(report)
        inconsistent["paired64image_deltas"] = dict(report["paired64image_deltas"])
        inconsistent["paired64image_deltas"]["warmed32_row_shuffled_minus_native"] = {
            "mean": -0.01, "per_image": [0.01] * 64}
        self.assertFalse(queue.validate_report(inconsistent, 2, **expected))
        invalid = dict(report)
        invalid["per_image"] = [dict(row) for row in report["per_image"]]
        invalid["per_image"][8] = dict(invalid["per_image"][8], image_id=999)
        self.assertFalse(queue.validate_report(invalid, 2, **expected))
        invalid = dict(report, ground_truth_used=True)
        self.assertFalse(queue.validate_report(invalid, 2, **expected))


if __name__ == "__main__":
    unittest.main()
