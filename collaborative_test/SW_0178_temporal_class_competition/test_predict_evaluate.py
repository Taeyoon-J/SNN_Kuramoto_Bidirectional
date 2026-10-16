import numpy as np  # NumPy/OpenMP precedes Torch through the production import.
import unittest

from collaborative_test.SW_0178_temporal_class_competition import predict_evaluate as evaluate


class SW178EvaluationContracts(unittest.TestCase):
    def test_standalone_foreground_may_differ_from_source_t(self):
        old_count = evaluate.COUNT
        evaluate.COUNT = 2
        try:
            source = np.zeros((2, 32, 32), dtype=bool)
            source[:, :16, :] = True
            qcc_support = np.zeros((2, 32, 32), dtype=np.int32)
            qcc_support[:, 16:, :] = 1
            conditional = np.zeros_like(qcc_support)
            conditional[source] = 1
            standalone = np.zeros_like(qcc_support)
            standalone[qcc_support > 0] = 2
            labels, support = evaluate.validate_prediction_arrays(
                {"conditional": conditional, "standalone": standalone}, qcc_support, source)
            self.assertTrue(np.array_equal(labels["conditional"] > 0, source))
            self.assertTrue(np.array_equal(labels["standalone"] > 0, support > 0))
            self.assertFalse(np.array_equal(labels["standalone"] > 0, source))
        finally:
            evaluate.COUNT = old_count

    def test_foreground_contract_rejects_changed_conditional_and_standalone_masks(self):
        old_count = evaluate.COUNT
        evaluate.COUNT = 1
        try:
            source = np.zeros((1, 32, 32), dtype=bool)
            source[:, :8, :] = True
            support = np.zeros((1, 32, 32), dtype=np.int32)
            support[:, 8:20, :] = 3
            conditional = source.astype(np.int32)
            standalone = (support > 0).astype(np.int32)
            broken_conditional = conditional.copy(); broken_conditional[:, 0, 0] = 0
            with self.assertRaisesRegex(ValueError, "conditional"):
                evaluate.validate_prediction_arrays(
                    {"conditional": broken_conditional, "standalone": standalone}, support, source)
            broken_standalone = standalone.copy(); broken_standalone[:, 8, 0] = 0
            with self.assertRaisesRegex(ValueError, "standalone"):
                evaluate.validate_prediction_arrays(
                    {"conditional": conditional, "standalone": broken_standalone}, support, source)
        finally:
            evaluate.COUNT = old_count

    @staticmethod
    def _metric(fg, fiou, oiou):
        return {"fg_ari": {"mean": fg}, "foreground_iou": {"mean": fiou},
                "matched_object_iou": {"mean": oiou}}

    def test_each_arm_has_independent_six_gate_eligibility(self):
        scores = {"SW0164_foreground_induced": self._metric(.7709, .615, .631)}
        paired = {}
        for endpoint in evaluate.ENDPOINTS:
            scores[f"product_q_ce_{endpoint}"] = self._metric(.79, .62, .64)
            scores[f"mean_rho_ce_{endpoint}"] = self._metric(.78, .60, .64)
            paired[f"{endpoint}_product_q_ce_minus_SW0164_fg_ari"] = {"ci95": [.01, .03]}
            paired[f"{endpoint}_mean_rho_ce_minus_SW0164_fg_ari"] = {"ci95": [.001, .02]}
            paired[f"{endpoint}_mean_rho_ce_minus_product_q_ce_fg_ari"] = {"ci95": [-.02, .0]}
        gates = evaluate._score_gates(scores, paired)
        self.assertTrue(gates["arm_eligibility"]["product_q_ce"]["eligible"])
        self.assertFalse(gates["arm_eligibility"]["mean_rho_ce"]["eligible"])
        self.assertTrue(gates["co_primary_any_arm_eligible"])
        self.assertFalse(gates["mean_rho_mechanism_promoted"])

    def test_mean_rho_promotion_requires_both_endpoints_to_beat_q_control(self):
        scores = {"SW0164_foreground_induced": self._metric(.77, .61, .63)}
        paired = {}
        for endpoint in evaluate.ENDPOINTS:
            scores[f"product_q_ce_{endpoint}"] = self._metric(.78, .62, .64)
            scores[f"mean_rho_ce_{endpoint}"] = self._metric(.80, .63, .65)
            paired[f"{endpoint}_product_q_ce_minus_SW0164_fg_ari"] = {"ci95": [.001, .02]}
            paired[f"{endpoint}_mean_rho_ce_minus_SW0164_fg_ari"] = {"ci95": [.01, .04]}
            paired[f"{endpoint}_mean_rho_ce_minus_product_q_ce_fg_ari"] = {"ci95": [.001, .03]}
        gates = evaluate._score_gates(scores, paired)
        self.assertTrue(gates["arm_eligibility"]["mean_rho_ce"]["eligible"])
        self.assertTrue(gates["mean_rho_mechanism_promoted"])
        paired["standalone_mean_rho_ce_minus_product_q_ce_fg_ari"]["ci95"][0] = -0.001
        self.assertFalse(evaluate._score_gates(scores, paired)["mean_rho_mechanism_promoted"])


if __name__ == "__main__":
    unittest.main()
