"""CPU integration contracts for the source-bound resource probe."""
import numpy as np  # Import NumPy before Torch on Windows.
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from collaborative_test.SW_0173_gradient_fidelity import resource_probe as probe


class ResourceProbeTests(unittest.TestCase):
    def test_probe_fingerprint_binds_real_sources_and_design(self):
        fingerprint = probe.fingerprint()
        self.assertEqual(fingerprint["steps"], 1024)
        self.assertEqual(fingerprint["tail_live_steps"], 64)
        self.assertEqual(fingerprint["full_live_steps"], 1024)
        self.assertFalse(fingerprint["validation_data_read"])
        self.assertTrue(all(len(value) == 64 for value in fingerprint["files"].values()))
        self.assertIn("collaborative_test/SW_0171_contextual_instance_graph/train_prefix.py",
                      fingerprint["files"])

    def test_cpu_prepare_reuses_only_admitted_train_rows_and_preserves_visibility(self):
        if torch.cuda.is_initialized():
            self.skipTest("a prior local test initialized CUDA")
        ids = list(range(40000, 40512))
        arrays = {
            "rgb": np.broadcast_to(np.zeros((1, 1, 1, 3), dtype=np.uint8),
                                   (512, 128, 128, 3)),
            "source_support": np.broadcast_to(np.zeros((1, 1, 1), dtype=np.int32),
                                              (512, 32, 32)),
            "feasible_targets": np.broadcast_to(np.zeros((1, 1, 1), dtype=np.int32),
                                                 (512, 32, 32)),
        }
        admitted = {"prepared": {"arrays": arrays, "cache_manifest_sha256": "a" * 64,
                                 "arrays_sha256": "b" * 64},
                    "proof": {"ordered_train_ids": ids}}
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "probe-test"}), \
             patch.object(probe, "_import", return_value=SimpleNamespace(
                 _load_admitted_inputs=lambda: admitted)):
            result = probe.prepare_cpu()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "probe-test")
        self.assertEqual(result["position"], 0)
        self.assertEqual(result["image_id"], ids[0])
        self.assertEqual(result["source_proof"], admitted["proof"])
        self.assertFalse(result["cuda_initialized"])
        self.assertFalse(result["validation_data_read"])

    def test_real_native_path_validates_short_cpu_geometry_and_tail_full_parity(self):
        """Use actual native rollout/coupling; shorten only this tiny CPU fixture."""
        from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
        from collaborative_test.SW_0162_connected_signed_partition.preflight import capture_source_coupling
        from collaborative_test.SW_0171_contextual_instance_graph import train_prefix
        from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
        from snn_kuramoto_bidirectional.s2net_cls import S2NetCore

        n = 1024
        torch.manual_seed(17311)
        hp = S2NetHyperparameters(
            num_feature_maps=8, num_regions=n, sc=torch.eye(n), osc_dim=4,
            gamma_drive_mode="static", gamma_phase_mode="standardize_tanh",
            theta_init="gamma", graph_mode="learned", graph_top_k=2,
            graph_spatial_decay=.35, geodesic_steps=0, geodesic_radius=1.5,
            geodesic_contrast=2.0, geodesic_temperature=.5, geodesic_cap=16.0,
            kuramoto_backend="factorized", k=256.0, freq_gain=2.0,
            num_time_steps=4, spike_per_component=True, gate_mode="raw",
            membrane_vth=.06, spike_pulse_gain=0.0, dendritic_projection="shared",
            spike_spatial_grid_size=32,
        ).validate()
        wrapped = PhaseStateIntegration(S2NetCore(hp, device="cpu"), "phase").eval()
        gamma = torch.randn((1, 8, n), dtype=torch.float32)

        class Found:
            def __init__(self):
                self.wrapped = wrapped

            def encode(self, _rgb):
                return gamma

        found = Found()
        rgb = torch.zeros((1, 128, 128, 3), dtype=torch.uint8)
        old_steps, old_settle = train_prefix.preflight.STEPS, train_prefix.preflight.SETTLE
        try:
            train_prefix.preflight.STEPS = 4
            train_prefix.preflight.SETTLE = 0
            before_rng = train_prefix.preflight._state_rng(torch.device("cpu"))
            with torch.enable_grad():
                tail = train_prefix._native_forward(found, rgb, live_tail_steps=2)
            probe.validate_native_trace(tail[1], batch=1, steps=4)
            tail_trace = probe._trace_copy(tail[1])
            tail_q = tail[4].detach().clone()
            train_prefix.preflight._restore_rng(torch.device("cpu"), before_rng)
            with torch.enable_grad():
                full = train_prefix._native_forward(found, rgb, live_tail_steps=4)
            probe.validate_native_trace(full[1], batch=1, steps=4)
            self.assertTrue(all(torch.equal(tail_trace[name], full[1][name].detach())
                                for name in probe.TRACE_FIELDS))
            self.assertTrue(torch.equal(tail_q, full[4].detach()))
            self.assertEqual(tuple(full[4].shape), (1, 1024, 1024))
        finally:
            train_prefix.preflight.STEPS = old_steps
            train_prefix.preflight.SETTLE = old_settle

    def test_native_trace_geometry_rejects_each_short_horizon_before_q(self):
        trace = {
            "theta": torch.zeros((1, 64, 1024, 4)),
            "membrane": torch.zeros((1, 1024, 64)),
            "spikes": torch.zeros((1, 1024, 64)),
            "component_membrane": torch.zeros((1, 4, 1024, 64)),
            "component_spikes": torch.zeros((1, 4, 1024, 64)),
            "component_gates": torch.zeros((1, 4, 1024, 64)),
        }
        with self.assertRaisesRegex(ValueError, "theta.*expected"):
            probe.validate_native_trace(trace, batch=1)


if __name__ == "__main__":
    unittest.main()
