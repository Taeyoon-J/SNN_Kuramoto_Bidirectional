import json
import unittest
from unittest import mock

from collaborative_test.SW_0127_frozen_qcc_attribution import dispatcher


class FrozenQCCDispatcherTests(unittest.TestCase):
    def test_preflight_is_a_dependency_of_evaluation(self):
        tasks = dispatcher.task_plan()
        self.assertEqual([task["stage"] for task in tasks], ["preflight", "evaluate"])
        self.assertEqual(tasks[0]["depends_on"], [])
        self.assertEqual(tasks[1]["depends_on"], [tasks[0]["task_id"]])

    def test_gpu_must_be_idle_and_below_memory_cap(self):
        self.assertTrue(dispatcher.gpu_is_exclusive(512, []))
        self.assertFalse(dispatcher.gpu_is_exclusive(513, []))
        self.assertFalse(dispatcher.gpu_is_exclusive(0, [1234]))

    def test_preflight_artifact_requires_native_and_four_arm_smokes(self):
        payload = {
            "status": "passed", "experiment": "SW0127", "ground_truth_used": False,
            "optimizer_updates": 0,
            "implementation_fingerprint": dispatcher.evaluator._implementation_fingerprint(),
            "native_rollout_parity": {
                "batches": [{"traces_exact": {"theta": True, "component_spikes": True},
                              "qcc_labels_exact": True} for _ in range(2)],
                "four_arm_validation_smokes": [
                    {"count": 2, "actual_spikes_finite": True,
                     "qcc_labels_shape": [2, 16, 16]} for _ in range(4)]}}
        task = {"stage": "preflight", "output": "/not-read-directly/preflight.json"}
        with mock.patch.object(dispatcher.pathlib.Path, "read_text",
                               return_value=json.dumps(payload)):
            self.assertTrue(dispatcher.valid_result(task))
            payload["native_rollout_parity"]["four_arm_validation_smokes"].pop()
            with mock.patch.object(dispatcher.pathlib.Path, "read_text",
                                   return_value=json.dumps(payload)):
                self.assertFalse(dispatcher.valid_result(task))


if __name__ == "__main__":
    unittest.main()
