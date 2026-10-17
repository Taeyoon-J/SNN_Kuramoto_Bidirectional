"""GT-last readiness and process-identity checks for SW0186 score coordinator."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from collaborative_test.SW_0186_full_encoder_qce import score_queue as q


class ScoreQueueTests(unittest.TestCase):
    def test_score_waits_until_each_seed_is_complete(self):
        states = {seed: {"experiment": "SW0186_prediction_queue", "seed": seed,
            "arm": "product_q_ce", "status": "complete", "ground_truth_used_for_prediction": False,
            "validation_data_read": False} for seed in q.SEEDS}
        states[2].update(status="waiting_for_stable_gpu", supervisor_pid=42)
        owner = SimpleNamespace(_proc_record=lambda _pid: {"state": "S", "argv": [
            "python", "-m", q.PRED_MODULE, "--source-seed", "2"]})
        ready, pending = q.prediction_terminal_set(owner, states)
        self.assertFalse(ready); self.assertEqual(pending, [2])

    def test_pending_seed_requires_live_exact_seed_supervisor(self):
        states = {seed: {"experiment": "SW0186_prediction_queue", "seed": seed,
            "arm": "product_q_ce", "status": "complete", "ground_truth_used_for_prediction": False,
            "validation_data_read": False} for seed in q.SEEDS}
        states[0].update(status="running_worker", supervisor_pid=99)
        owner = SimpleNamespace(_proc_record=lambda _pid: {"state": "S", "argv": [
            "python", "-m", q.PRED_MODULE, "--source-seed", "0"]})
        ready, pending = q.prediction_terminal_set(owner, states)
        self.assertFalse(ready); self.assertEqual(pending, [0])
        bad_owner = SimpleNamespace(_proc_record=lambda _pid: {"state": "S", "argv": [
            "python", "-m", q.PRED_MODULE, "--worker", "--source-seed", "0"]})
        with self.assertRaises(RuntimeError): q.prediction_terminal_set(bad_owner, states)

    def test_gt_prediction_claim_is_rejected_before_scoring(self):
        states = {seed: {"experiment": "SW0186_prediction_queue", "seed": seed,
            "arm": "product_q_ce", "status": "complete", "ground_truth_used_for_prediction": False,
            "validation_data_read": False} for seed in q.SEEDS}
        states[1]["ground_truth_used_for_prediction"] = True
        with self.assertRaises(ValueError): q.prediction_terminal_set(SimpleNamespace(), states)

    def test_score_coordinator_cli_help_is_wired(self):
        with self.assertRaises(SystemExit) as help_exit:
            q.main(["--help"])
        self.assertEqual(help_exit.exception.code, 0)


if __name__ == "__main__": unittest.main()
