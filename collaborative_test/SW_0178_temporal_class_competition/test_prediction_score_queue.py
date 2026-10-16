import json
import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np  # Keep NumPy/OpenMP initialized before production imports.

from collaborative_test.SW_0178_temporal_class_competition import (
    prediction_queue as prediction,
    predict_evaluate as evaluate,
    score_queue as score,
    train,
)


class MemoryStatePath:
    files = {}
    directories = set()

    def __init__(self, key):
        self.key = str(key)
        self.name = self.key.rsplit("/", 1)[-1]

    def __truediv__(self, name):
        return MemoryStatePath(f"{self.key}/{name}")

    def is_file(self):
        return self.key in self.files

    def is_dir(self):
        return self.key in self.directories

    def read_text(self, encoding=None):
        value = self.files[self.key]
        return value if isinstance(value, str) else json.dumps(value)


class SW178PredictionQueueContracts(unittest.TestCase):
    def test_all_literal_training_producer_active_statuses_are_wait_states(self):
        # These literals are emitted by SW0178 training_queue._write_state.
        producer_active = {
            "validating_train_cache", "waiting_for_exclusive_gpu",
            "waiting_for_stable_gpu", "running_worker", "yielding_foreign_owner",
        }
        self.assertEqual(prediction.ACTIVE_TRAIN, producer_active)
        owner = SimpleNamespace(_proc_record=lambda pid: {
            "state": "R", "argv": ["python", "-u", "-m", prediction.TRAIN_MODULE,
                                      "--arm", "mean_rho_ce"]})
        for status in producer_active:
            state = {"status": status, "supervisor_pid": 123}
            self.assertTrue(prediction._training_is_pending(owner, state, "mean_rho_ce"), status)
        self.assertFalse(prediction._training_is_pending(owner,
            {"status": "waiting_for_stable_gpu", "supervisor_pid": 123}, "product_q_ce"))

    def test_supervisor_liveness_requires_expected_module_arm_and_no_worker_flag(self):
        owner = SimpleNamespace(_proc_record=lambda pid: {"state": "S", "argv": [
            "python", "-m", prediction.MODULE, "--arm", "mean_rho_ce"]})
        self.assertTrue(prediction._module_live(owner, 123, prediction.MODULE, "mean_rho_ce"))
        self.assertFalse(prediction._module_live(owner, 123, prediction.MODULE, "product_q_ce"))
        worker_owner = SimpleNamespace(_proc_record=lambda pid: {"state": "S", "argv": [
            "python", "-m", prediction.MODULE, "--worker", "--arm", "mean_rho_ce"]})
        self.assertFalse(prediction._module_live(worker_owner, 123, prediction.MODULE, "mean_rho_ce"))

    def test_rejected_arm_publication_uses_one_arm_argument(self):
        captured = []
        with patch.object(prediction, "_write_state",
                          side_effect=lambda *args, **kwargs: captured.append((args, kwargs))):
            result = prediction._publish_training_rejection(
                "mean_rho_ce", {"arm": "mean_rho_ce", "state_sha256": "a" * 64})
        self.assertEqual(result["status"], "complete_training_rejected")
        args, kwargs = captured[0]
        self.assertEqual(args, ("mean_rho_ce", "complete_training_rejected"))
        self.assertNotIn("arm", kwargs)
        self.assertFalse(kwargs["ground_truth_used_for_prediction"])


class SW178ScoreQueueContracts(unittest.TestCase):
    def test_prediction_supervisor_liveness_rejects_worker_and_wrong_arm(self):
        good, wrong = train.ARMS
        owner = SimpleNamespace(_proc_record=lambda pid: {"state": "S", "argv": [
            "python", "-m", score.PREDICTION_MODULE, "--arm", good]})
        self.assertTrue(score._module_live(owner, 123, score.PREDICTION_MODULE, good))
        self.assertFalse(score._module_live(owner, 123, score.PREDICTION_MODULE, wrong))
        worker_owner = SimpleNamespace(_proc_record=lambda pid: {"state": "S", "argv": [
            "python", "-m", score.PREDICTION_MODULE, "--worker", "--arm", good]})
        self.assertFalse(score._module_live(worker_owner, 123, score.PREDICTION_MODULE, good))

    def test_prediction_dependency_waits_on_literal_supervisor_state(self):
        active_arm, rejected_arm = train.ARMS
        active_state = {"experiment": "SW0178_prediction_queue_v1", "arm": active_arm,
                        "status": "running_prediction", "supervisor_pid": 44,
                        "ground_truth_used_for_prediction": False}
        rejected_state = {"experiment": "SW0178_prediction_queue_v1", "arm": rejected_arm,
                          "status": "complete_training_rejected",
                          "ground_truth_used_for_prediction": False,
                          "training_rejections": []}
        module = prediction.MODULE
        owner = SimpleNamespace(_proc_record=lambda pid: {
            "state": "R", "argv": ["python", "-m", module, "--arm", active_arm]})
        states = {active_arm: active_state, rejected_arm: rejected_state}
        MemoryStatePath.files = {f"state:{arm}": value for arm, value in states.items()}
        with patch.object(score, "_prediction_state_path", side_effect=lambda arm:
                          MemoryStatePath(f"state:{arm}")):
            self.assertIs(score._prediction_dependency(owner), False)

    def test_all_rejected_predictions_are_terminal_without_gt_receipts(self):
        states = {arm: {"experiment": "SW0178_prediction_queue_v1", "arm": arm,
                        "status": "complete_training_rejected",
                        "ground_truth_used_for_prediction": False,
                        "training_rejections": [{"arm": arm}]}
                  for arm in train.ARMS}
        MemoryStatePath.files = {f"state:{arm}": value for arm, value in states.items()}
        with patch.object(score, "_prediction_state_path",
                          side_effect=lambda arm: MemoryStatePath(f"state:{arm}")), \
             patch.object(score, "Path", MemoryStatePath), \
             patch.object(score, "_sha", side_effect=lambda path:
                          hashlib.sha256(str(path.key).encode()).hexdigest()):
            result = score._prediction_dependency(SimpleNamespace())
        self.assertEqual(set(result), {"_rejected"})
        self.assertEqual(set(result["_rejected"]), set(train.ARMS))

    def test_scoring_dependency_retains_successful_arm_when_other_arm_rejected(self):
        good, rejected = "product_q_ce", "mean_rho_ce"
        prediction_path, protocol_path = "out/predictions.npz", "out/protocol.json"
        MemoryStatePath.directories = {"out"}
        prediction_sha = hashlib.sha256(prediction_path.encode()).hexdigest()
        protocol_sha = hashlib.sha256(protocol_path.encode()).hexdigest()
        protocol = {"experiment": "SW0178_temporal_class_competition_fixed320_prediction",
                    "arm": good, "status": "complete",
                    "ground_truth_used_for_prediction": False,
                    "prediction_sha256": prediction_sha}
        MemoryStatePath.files = {
            prediction_path: b"prediction fixture",
            protocol_path: json.dumps(protocol),
            f"state:{good}": {"experiment": "SW0178_prediction_queue_v1", "arm": good,
                "status": "complete", "ground_truth_used_for_prediction": False,
                "prediction_sha256": prediction_sha, "protocol_sha256": protocol_sha,
                "output": "out"},
            f"state:{rejected}": {"experiment": "SW0178_prediction_queue_v1", "arm": rejected,
                "status": "complete_training_rejected", "ground_truth_used_for_prediction": False,
                "training_rejections": [{"arm": rejected}]},
        }
        with patch.object(score, "_prediction_state_path", side_effect=lambda arm:
                          MemoryStatePath(f"state:{arm}")), \
             patch.object(score, "Path", MemoryStatePath), \
             patch.object(score, "_sha", side_effect=lambda path:
                          hashlib.sha256(str(path.key).encode()).hexdigest()):
            result = score._prediction_dependency(SimpleNamespace())
        self.assertEqual(set(result), {good, "_rejected"})
        self.assertEqual(result[good]["state"]["prediction_sha256"], prediction_sha)
        self.assertIn(rejected, result["_rejected"])

    def test_independent_score_gates_accept_one_arm_and_leave_mechanism_unavailable(self):
        def metric(fg, fiou, oiou):
            return {"fg_ari": {"mean": fg}, "foreground_iou": {"mean": fiou},
                    "matched_object_iou": {"mean": oiou}}
        scores = {"SW0164_foreground_induced": metric(.7709, .615, .631)}
        paired = {}
        for endpoint in evaluate.ENDPOINTS:
            scores[f"product_q_ce_{endpoint}"] = metric(.79, .62, .64)
            paired[f"{endpoint}_product_q_ce_minus_SW0164_fg_ari"] = {"ci95": [.01, .03]}
        gates = evaluate._score_gates(scores, paired)
        self.assertTrue(gates["arm_eligibility"]["product_q_ce"]["eligible"])
        self.assertTrue(gates["co_primary_any_arm_eligible"])
        self.assertFalse(gates["mean_rho_mechanism_promoted"])
        self.assertEqual(gates["mean_rho_advantage"]["conditional"]["status"],
                         "unavailable_training_arm_rejected")


if __name__ == "__main__":
    unittest.main()
