"""Small synthetic schema/aggregation tests for SW0054 and SW0055 summaries."""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "SW_0054_causal_mechanism_ablation"))
sys.path.insert(0, str(HERE / "SW_0055_unique_data_scale"))
import summarize as sw55  # noqa: E402

del sys.modules["summarize"]
sys.path.insert(0, str(HERE / "SW_0054_causal_mechanism_ablation"))
import summarize as sw54  # noqa: E402


def make_sw54_report():
    metrics = {"fg_ari": 0.4, "foreground_iou": 0.3, "matched_object_iou": 0.2}
    auc = {name: {"macro_distance_auc": 0.6} for name in sw54.AUC_SIGNALS}
    conditions = {}
    for name in sw54.PILOT_CONDITIONS:
        conditions[name] = {
            "fixed_readouts": {r: {"metrics": dict(metrics)} for r in ("spike_cc", "membrane_spatial")},
            "distance_controlled_macro_auc": auc,
            "activity_scale": {"spike_event_rate_mean": 0.2, "membrane_temporal_variance_mean": 0.1, "membrane_abs_mean": 0.3},
        }
        if name.startswith(("gate_", "carrier_")):
            conditions[name]["gate_invariants"] = [{"graph_bitwise_equal": True, "theta_allclose": True}]
    conditions["K0"]["kuramoto_K_during_rollout"] = 0.0
    conditions["gate_perm_s0"]["fixed_readouts"]["spike_cc"]["metrics"]["fg_ari"] = 0.5
    return {
        "experiment": "SW0054 causal mechanism ablation",
        "checkpoint": "/x/" + sw54.CHECKPOINT_SUFFIX,
        "checkpoint_sha256": "a" * 64,
        "count": 32, "ids": [1320, 1351],
        "core": {"steps": 256, "settle": 64},
        "interventions": {"condition_set": "pilot"},
        "target": {"ids": [1320, 1351], "ground_truth_used_for_prediction": False},
        "conditions": conditions,
    }


class SummarizerTests(unittest.TestCase):
    def test_sw54_delta_and_protocol_validation(self):
        result = sw54.summarize(make_sw54_report())
        self.assertAlmostEqual(result["intervention_deltas"]["gate_perm_s0"]["fixed_readout_metric_delta_vs_normal"]["spike_cc"]["fg_ari"], 0.1)
        report = make_sw54_report()
        report["conditions"]["gate_mean"]["gate_invariants"][0]["theta_allclose"] = False
        with self.assertRaises(ValueError):
            sw54.validate_report(report)

    def test_sw55_fixed_threshold_policy_and_baselines(self):
        self.assertEqual(sw55.WINDOWS, {"short": (256, 64), "long": (1024, 512)})
        self.assertEqual(sw55.BASELINES[0.50]["fg_ari"], 0.562902)
        self.assertEqual(sw55.BASELINES[0.35]["matched_object_iou"], 0.349261)
        self.assertEqual(sw55.THRESHOLDS, (0.50, 0.35))


if __name__ == "__main__":
    unittest.main()
