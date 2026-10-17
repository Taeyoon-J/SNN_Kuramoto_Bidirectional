import unittest

from collaborative_test.SW_0187_joint_encoder_unique_scaling import score_queue as q
from collaborative_test.SW_0187_joint_encoder_unique_scaling import prediction_queue as prediction


class _Owner:
    module = prediction.MODULE
    def _proc_record(self, pid):
        if pid != 700: return None
        return {"state": "S", "argv": ["python", "-m", self.module, "--source-seed", "1"]}


class ScoreQueueContracts(unittest.TestCase):
    def test_requires_both_arm_prediction_freezes_for_every_seed(self):
        complete = {seed: {"experiment": "SW0187_prediction_queue", "seed": seed,
            "status": "complete", "ground_truth_used_for_prediction": False,
            "validation_data_read": False, "predictions": {arm: {} for arm in ("control512", "candidate1024")}}
            for seed in q.SEEDS}
        ready, pending = q.predictions_ready(_Owner(), prediction, complete)
        self.assertTrue(ready); self.assertEqual(pending, [])

    def test_waits_for_live_producer_and_rejects_stale_or_incomplete(self):
        rows = {seed: {"experiment": "SW0187_prediction_queue", "seed": seed,
            "status": "complete", "ground_truth_used_for_prediction": False,
            "validation_data_read": False, "predictions": {arm: {} for arm in ("control512", "candidate1024")}}
            for seed in q.SEEDS}
        rows[1] = {"experiment": "SW0187_prediction_queue", "seed": 1,
            "status": "running_worker", "supervisor_pid": 700}
        ready, pending = q.predictions_ready(_Owner(), prediction, rows)
        self.assertFalse(ready); self.assertEqual(pending, [1])
        rows[1] = {"experiment": "SW0187_prediction_queue", "seed": 1,
            "status": "complete", "ground_truth_used_for_prediction": False,
            "validation_data_read": False,
            "predictions": {arm: {} for arm in ("control512", "candidate1024")}}
        rows[1]["predictions"].pop("candidate1024")
        with self.assertRaises(ValueError): q.predictions_ready(_Owner(), prediction, rows)

    def test_pending_state_uses_real_prediction_queue_process_identity(self):
        owner = _Owner()
        state = {"experiment": "SW0187_prediction_queue", "seed": 1,
            "status": "running_worker", "supervisor_pid": 700}
        self.assertFalse(q._prediction_terminal(owner, prediction, 1, state))
        owner.module = "collaborative_test.SW_0182_unique1024_actual_descent.prediction_queue"
        with self.assertRaises(RuntimeError): q._prediction_terminal(owner, prediction, 1, state)

    def test_rejected_training_is_terminal_not_a_wait(self):
        rows = {seed: {"experiment": "SW0187_prediction_queue", "seed": seed,
            "status": "complete", "ground_truth_used_for_prediction": False,
            "validation_data_read": False, "predictions": {arm: {} for arm in ("control512", "candidate1024")}}
            for seed in q.SEEDS}
        rows[2]["status"] = "complete_training_rejected"
        with self.assertRaises(RuntimeError): q.predictions_ready(_Owner(), prediction, rows)

    def test_cli_help(self):
        with self.assertRaises(SystemExit) as result: q.main(["--help"])
        self.assertEqual(result.exception.code, 0)


if __name__ == "__main__": unittest.main()
