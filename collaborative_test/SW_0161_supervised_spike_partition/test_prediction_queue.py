"""CPU queue identity and dependency-state contracts for SW161 prediction."""
import os
import unittest
from unittest import mock

from collaborative_test.SW_0161_supervised_spike_partition import prediction_queue as pq
from collaborative_test.SW_0161_supervised_spike_partition import score_queue as sq
from collaborative_test.SW_0161_supervised_spike_partition import train


class PredictionQueueTests(unittest.TestCase):
    def test_child_argv_and_output_are_registered_for_both_arms(self):
        for arm in train.ARMS:
            config = pq.configure_arm(arm)
            argv = pq.command(arm)
            self.assertEqual(argv[argv.index("--arm") + 1], arm)
            self.assertEqual(argv[argv.index("--device") + 1], "cuda:0")
            self.assertEqual(argv[argv.index("--stage") + 1], "predict")
            self.assertEqual(config["output"], pq.evaluation.PREDICTION_ROOT / f"{arm}_seed1")
        self.assertEqual(set(train.ARMS), {"pair_supervised_P", "mask_supervised_P"})

    def test_queue_fingerprints_resolve_real_current_dependencies(self):
        fp = pq.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0161_supervised_spike_partition/predict_evaluate.py", fp)
        self.assertIn("collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py", fp)
        self.assertIn("snn_kuramoto_bidirectional/evaluation.py", pq.evaluation.implementation_fingerprint())
        self.assertEqual(sq.implementation_fingerprint()["collaborative_test/SW_0161_supervised_spike_partition/score_queue.py"],
                         sq.sha256_file(sq.HERE / "score_queue.py"))

    def test_training_wait_requires_live_registered_arm_owner(self):
        arm = train.ARMS[0]
        state = {"experiment": train.EXPERIMENT, "arm": arm,
                 "status": "waiting_for_sw0160_score", "supervisor_pid": 111, "child_pid": None}
        with mock.patch.object(pq, "TRAINING_QUEUE_ROOT", pq.HERE / "_queue_test"):
            folder = pq.TRAINING_QUEUE_ROOT / arm
            folder.mkdir(parents=True, exist_ok=True)
            try:
                (folder / "state.json").write_text(__import__("json").dumps(state), encoding="utf-8")
                with mock.patch.object(pq, "_process_has_args", return_value=True):
                    self.assertTrue(pq._training_queue_is_live(arm))
                with mock.patch.object(pq, "_process_has_args", return_value=False):
                    self.assertFalse(pq._training_queue_is_live(arm))
                state["status"] = "failed"
                (folder / "state.json").write_text(__import__("json").dumps(state), encoding="utf-8")
                with mock.patch.object(pq, "_process_has_args", return_value=True):
                    self.assertFalse(pq._training_queue_is_live(arm))
                # These are the actual waiting/running states emitted by the retry wrapper.
                for status in ("waiting_for_admitted_preflight", "waiting_for_gpu", "running"):
                    state["status"] = status
                    (folder / "state.json").write_text(__import__("json").dumps(state), encoding="utf-8")
                    with mock.patch.object(pq, "_process_has_args", return_value=True):
                        self.assertTrue(pq._training_queue_is_live(arm), status)
            finally:
                import shutil
                shutil.rmtree(pq.TRAINING_QUEUE_ROOT, ignore_errors=True)

    def test_score_queue_waits_for_each_registered_prediction_owner(self):
        arm = train.ARMS[1]
        state = {"experiment": train.EXPERIMENT, "arm": arm,
                 "status": "waiting_for_training_artifact", "supervisor_pid": 222,
                 "child_pid": None}
        with mock.patch.object(sq, "PREDICTION_QUEUE_ROOT", sq.HERE / "_score_queue_test"):
            folder = sq.PREDICTION_QUEUE_ROOT / arm
            folder.mkdir(parents=True, exist_ok=True)
            try:
                (folder / "state.json").write_text(__import__("json").dumps(state), encoding="utf-8")
                with mock.patch.object(sq, "_process_has_args", return_value=True):
                    self.assertTrue(sq._prediction_queue_is_live(arm))
                with mock.patch.object(sq, "_process_has_args", return_value=False):
                    self.assertFalse(sq._prediction_queue_is_live(arm))
            finally:
                import shutil
                shutil.rmtree(sq.HERE / "_score_queue_test", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
