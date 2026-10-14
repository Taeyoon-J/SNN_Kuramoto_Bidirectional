import unittest

import numpy as np
import torch

from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143
from collaborative_test.SW_0158_membrane_spike_credit import preflight
from collaborative_test.SW_0155_supervised_spike_relations import train as sw155
from collaborative_test.SW_0158_membrane_spike_credit import spike_credit


class PreflightContractTests(unittest.TestCase):
    def _receipt(self):
        upstream = {"source_training_ids": list(range(20)), "source_sha256": "s" * 64}
        return upstream, {
            "experiment": preflight.EXPERIMENT, "status": "complete", "seed": 1,
            "implementation_fingerprint": {"code": "sha"}, "upstream": upstream,
            "optimizer_updates": 2, "source_positions": [0, 1, 2, 3],
            "image_ids": [0, 1, 2, 3], "ground_truth_used": True,
            "ground_truth_scope": "the four registered source TRAIN masks only",
            "validation_or_test_ground_truth_used": False,
            "rgb_batch_sha256": "a" * 64,
            "source_encoder_patcher_and_graph_frozen_state_sha256_before": "b" * 64,
            "source_encoder_patcher_and_graph_frozen_state_sha256_after": "b" * 64,
            "eta_zero_source_T_parity": {
                "six_trace_exact": True, "production_Q_exact": True,
                "production_QCC_exact": True,
                "trace_parity": {key: True for key in sw143.TRACE_FIELDS}},
            "matched_temporal_credit": {
                "same_state_same_rng_forward_exact": True,
                "live_gradient_finite": True, "detached_gradient_finite": True,
                "additional_membrane_credit_detected": True,
                "live_detached_membrane_gradient_delta_l2": 0.2,
                "repeated_live_gradient_residual_l2": 1e-8},
            "post_update1_recurrence_reset": {
                "finite": True, "same_spike_used_by_dendrite_and_reset": True,
                "recurrent_assignment_exact": True, "native_reset_equation_exact": True},
            "post_update1_amplitude_diagnostics": {
                "per_node_amplitude_factor_temporal_variance_mean": .1,
                "active_native_event_gate_count": 20,
                "eligible_node_count": 20,
                "fraction_eligible_nodes_with_nonzero_temporal_variance": .5,
                "aggregate_amplitude_factor_mean": 1.01, "aggregate_amplitude_factor_std": .01},
            "post_update1_zero_eta_Q_mean_abs_delta": .001,
            "updates": [{"update": step,
                "family_gradient_norms": {"native_core": 1., "integration_tensors": .2, "eta": .1},
                "family_parameter_change_l2": {"native_core": .001,
                    "integration_tensors": .001, "eta": .001}} for step in (1, 2)],
            "training_admission": True,
        }

    def test_credit_gate_uses_repeated_live_numeric_residual(self):
        live = torch.tensor([1.0, -2.0])
        repeated = live + torch.tensor([1e-8, 0.0])
        detached = torch.tensor([.9, -1.9])
        summary = preflight.temporal_credit_summary(
            live, repeated, detached, {"trace": True, "Q": True})
        self.assertTrue(summary["additional_membrane_credit_detected"])
        no_delta = preflight.temporal_credit_summary(live, live, live,
            {"trace": True})
        self.assertFalse(no_delta["additional_membrane_credit_detected"])
        with self.assertRaises(ValueError):
            preflight.temporal_credit_summary(live, repeated, detached, {"trace": False})

    def test_cpu_source_paths_and_calibration_are_resolvable(self):
        fingerprint = preflight.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0158_membrane_spike_credit/spike_credit.py", fingerprint)
        self.assertEqual(spike_credit.registered_margin_rms(), spike_credit.SOURCE_T_MARGIN_RMS)

    def test_actual_train_loader_sorts_then_restores_requested_id_order(self):
        ids = [4, 1, 3, 2]
        images = np.zeros((5, 128, 128, 3), dtype=np.uint8)
        masks = np.zeros((5, 128, 128), dtype=np.uint8)
        for index in range(5):
            images[index, 0, 0, 0] = index
            masks[index, 0, 0] = index
        cache = images[np.asarray(ids)].copy()
        class Handle:
            def __getitem__(self, key):
                return {"image": images, "mask": masks}[key]
        aligned_rgb, aligned_masks = sw155.load_aligned_train_rows(
            "unused", ids, cache, 0, 4, hdf5_handle=Handle(), expected_rows=5)
        self.assertTrue(np.array_equal(aligned_rgb.numpy(), cache))
        self.assertEqual(aligned_masks[:, 0, 0].tolist(), ids)

    def test_recurrence_helper_checks_actual_reset_and_recurrent_spike(self):
        from collaborative_test.SW_0158_membrane_spike_credit.test_spike_credit import (
            MembraneSpikeCreditTests)
        torch.manual_seed(15812)
        wrapped = MembraneSpikeCreditTests.tiny_wrapped()
        spike_credit.install_spike_credit(wrapped, spike_credit.SOURCE_T_MARGIN_RMS,
                                          eta_initial=torch.tensor([.2, -.1, .3, .1]))
        trace = spike_credit.late_rollout_credit(wrapped, torch.randn(1, 8, 4),
            total_steps=16, live_tail_steps=0, capture_state=True)
        result = preflight.validate_recurrence_reset(trace)
        self.assertTrue(result["native_reset_equation_exact"])
        broken = dict(trace)
        broken["credit_state"] = dict(trace["credit_state"])
        broken["credit_state"]["dendrite_previous_spike"] = trace["credit_state"][
            "dendrite_previous_spike"].clone()
        broken["credit_state"]["dendrite_previous_spike"][0, 0, 1] += 1
        with self.assertRaises(AssertionError):
            preflight.validate_recurrence_reset(broken)

    def test_amplitude_diagnostic_measures_within_node_time_variance(self):
        events = torch.ones(1, 4, 2, 4)
        gates = torch.ones_like(events)
        spikes = torch.ones_like(events)
        spikes[:, :, 1] = 1.2  # spatial difference alone is not temporal variance
        trace = {"component_events": events, "component_gates": gates,
                 "component_spikes": spikes}
        summary = preflight.amplitude_diagnostics(trace, settle=0)
        self.assertEqual(summary["per_node_amplitude_factor_temporal_variance_mean"], 0.0)
        spikes[:, :, 0, 2] = 1.3
        changed = preflight.amplitude_diagnostics(trace, settle=0)
        self.assertGreater(changed["per_node_amplitude_factor_temporal_variance_mean"], 0.0)

    def test_receipt_validator_binds_source_parity_and_train_scope(self):
        upstream, receipt = self._receipt()
        fp = {"code": "sha"}
        self.assertTrue(preflight.validate_report_record(
            receipt, expected_fingerprint=fp, upstream=upstream))
        for key, value in (("image_ids", [0, 1, 2, 4]),
                           ("ground_truth_used", False),
                           ("implementation_fingerprint", {"code": "other"})):
            bad = dict(receipt); bad[key] = value
            self.assertFalse(preflight.validate_report_record(
                bad, expected_fingerprint=fp, upstream=upstream), key)
        bad = dict(receipt)
        bad["eta_zero_source_T_parity"] = dict(receipt["eta_zero_source_T_parity"])
        bad["eta_zero_source_T_parity"]["production_QCC_exact"] = False
        self.assertFalse(preflight.validate_report_record(
            bad, expected_fingerprint=fp, upstream=upstream))


if __name__ == "__main__":
    unittest.main()
