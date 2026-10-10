from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0136_native32_transfer import evaluate, transfer_queue as queue


class TransferQueueContracts(unittest.TestCase):
    def test_plan_parallelizes_exactly_two_completed_seed1_arms(self):
        tasks = queue.task_plan()
        self.assertEqual([task["arm"] for task in tasks], ["actual_joint", "actual_frozen"])
        self.assertEqual(queue.MAX_PARALLEL, 2)
        for task in tasks:
            argv = queue.command(task)
            self.assertEqual(argv[1:3], ["-m", "collaborative_test.SW_0136_native32_transfer.evaluate"])
            self.assertIn(task["arm"], argv)
            self.assertIn("--stage", argv)
            self.assertIn("predict", argv)
            self.assertTrue(str(queue.result_dir(task)).endswith(task["arm"]))

    def test_gpu_candidate_requires_no_owner_and_at_most512mib(self):
        self.assertTrue(queue.gpu_is_exclusive_candidate(0, []))
        self.assertTrue(queue.gpu_is_exclusive_candidate(512, []))
        self.assertFalse(queue.gpu_is_exclusive_candidate(513, []))
        self.assertFalse(queue.gpu_is_exclusive_candidate(0, [123]))

    def test_candidate_result_also_requires_the_mapped_source_reference(self):
        prediction = {"protocol": {"primary": "mapped source97 QCC reference only",
                                    "transfer": {"reference_role": "mapped SW0097 native32 QCC only",
                                                 "source_core_sha256": evaluate.sw130.source97.EXPECTED_SOURCE_SHAS[1],
                                                 "converted_state_sha256": "converted"}}}
        with mock.patch.object(evaluate, "validate_prediction", return_value=prediction) as validate:
            self.assertTrue(queue.validate_result({"arm": "actual_joint"}))
            self.assertEqual(validate.call_count, 2)
        bad = {"protocol": {"transfer": {"reference_role": "wrong"}}}
        with mock.patch.object(evaluate, "validate_prediction", return_value=bad):
            self.assertFalse(queue.validate_result({"arm": "actual_joint"}))

    def test_frozen_arm_validity_does_not_depend_on_joint_task_completion(self):
        frozen = {"protocol": {"status": "complete", "arm": "actual_frozen"}}
        with mock.patch.object(evaluate, "validate_prediction", return_value=frozen) as validate:
            self.assertTrue(queue.validate_result({"arm": "actual_frozen"}))
            validate.assert_called_once_with(queue.result_dir({"arm": "actual_frozen"}), arm="actual_frozen")

    def test_score_reducer_preserves_readout_nesting_and_exact_metric_schema(self):
        metrics = {name: {"mean": i / 10, "valid_count": 320, "per_image": []}
                   for i, name in enumerate(evaluate.METRICS)}
        report = {"scores": {"actual_joint": {"primary": metrics, "qcc": metrics},
                             "source97_qcc": {"qcc": metrics}}}
        reduced = queue.score_mean_view(report)
        self.assertEqual(set(reduced["actual_joint"]), {"primary", "qcc"})
        self.assertEqual(reduced["actual_joint"]["primary"]["fg_ari"], 0.0)
        with self.assertRaises(ValueError):
            queue.score_mean_view({"scores": {"arm": {"primary": {"unexpected": {"mean": 1}}}}})


if __name__ == "__main__":
    unittest.main()

