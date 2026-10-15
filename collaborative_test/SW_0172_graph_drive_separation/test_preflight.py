"""SW0172 CPU-side source preparation and report helper contracts."""
import numpy as np  # Import NumPy before Torch to avoid the Windows OpenMP import conflict.
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from collaborative_test.SW_0172_graph_drive_separation import preflight


class PreflightTests(unittest.TestCase):
    def test_fingerprint_dependencies_exist_and_bind_actual_native_helpers(self):
        fp = preflight.implementation_fingerprint()
        files = fp["files"]
        self.assertIn("collaborative_test/SW_0172_graph_drive_separation/dual_gamma_rollout.py", files)
        self.assertIn("collaborative_test/SW_0134_native_spike_binding/rollout.py", files)
        self.assertIn("collaborative_test/SW_0162_connected_signed_partition/preflight.py", files)
        self.assertIn("collaborative_test/SW_0164_foreground_preserving_partition/foreground_signed.py", files)
        self.assertTrue(all(len(value) == 64 for value in files.values()))

    def test_atomic_report_writer_import_resolves_to_real_producer(self):
        writer = preflight._module(
            "collaborative_test.SW_0171_contextual_instance_graph.preflight")
        self.assertTrue(callable(writer._write_report))

    def test_cpu_prepare_selects_first_eight_admitted_rows_without_cuda(self):
        ids = list(range(9000, 9512))
        rgb = np.zeros((512, 128, 128, 3), dtype=np.uint8)
        for i in range(512):
            rgb[i, 0, 0, 0] = i % 256
        support = np.zeros((512, 32, 32), dtype=np.int32)
        support[:, 0, 0] = np.arange(512, dtype=np.int32)
        prepared = {
            "arrays": {"rgb": rgb, "source_support": support},
            "source_proof": {
                "ordered_train_ids": ids,
                "source_cache_manifest_sha256": "a" * 64,
                "context_cache_manifest_sha256": "b" * 64,
            },
            "source_checkpoint_sha256": "c" * 64,
            "source_manifest_sha256": "d" * 64,
            "producer_bindings": {"source_gain": 8.0},
            "archived_reference_proof": {"fixed320_ids": list(range(1320, 1640))},
            "cache_manifest_sha256": "e" * 64,
            "arrays_sha256": "f" * 64,
        }
        admitted = {
            "prepared": prepared, "proof": prepared["source_proof"],
            "lambda": preflight.REGISTERED_LAMBDA,
            "admission_report_sha256": "1" * 64,
            "queue_fingerprint_sha256": "2" * 64,
            "preflight_fingerprint": {"ready": True},
        }
        with patch.object(preflight, "_module", return_value=SimpleNamespace(
                _load_admitted_inputs=lambda: admitted)):
            result = preflight.prepare_inputs_cpu()
        self.assertEqual(result["selected_positions"], list(range(8)))
        self.assertEqual(result["selected_ids"], ids[:8])
        self.assertEqual(result["rgb8"].shape, (8, 128, 128, 3))
        self.assertEqual(result["source_support8"].shape, (8, 32, 32))
        self.assertEqual(result["rgb8"][:, 0, 0, 0].tolist(), list(range(8)))
        self.assertEqual(result["source_support8"][:, 0, 0].tolist(), list(range(8)))
        self.assertFalse(result["cuda_initialized"])
        self.assertFalse(result["validation_data_read"])

    def test_cpu_prepare_rejects_changed_fixed_lambda(self):
        ids = list(range(512))
        prepared = {
            "arrays": {"rgb": np.zeros((512, 128, 128, 3), dtype=np.uint8),
                       "source_support": np.zeros((512, 32, 32), dtype=np.int32)},
            "source_proof": {"ordered_train_ids": ids,
                "source_cache_manifest_sha256": "a" * 64,
                "context_cache_manifest_sha256": "b" * 64},
            "source_checkpoint_sha256": "c" * 64, "source_manifest_sha256": "d" * 64,
            "producer_bindings": {"source_gain": 8.0},
            "archived_reference_proof": {}, "cache_manifest_sha256": "e" * 64,
            "arrays_sha256": "f" * 64,
        }
        admitted = {"prepared": prepared, "proof": prepared["source_proof"],
                    "lambda": preflight.REGISTERED_LAMBDA + 1e-9}
        with patch.object(preflight, "_module", return_value=SimpleNamespace(
                _load_admitted_inputs=lambda: admitted)):
            with self.assertRaisesRegex(ValueError, "completed SW0171 TRAIN-only lambda"):
                preflight.prepare_inputs_cpu()

    def test_trace_summary_binds_all_six_native_fields(self):
        trace = {name: torch.full((2, 3), index, dtype=torch.float32)
                 for index, name in enumerate(preflight.TRACE_FIELDS)}
        summary = preflight._cpu_trace_summary(trace)
        self.assertEqual(set(summary), set(preflight.TRACE_FIELDS))
        self.assertEqual(len(set(summary.values())), len(preflight.TRACE_FIELDS))
        changed = dict(trace)
        changed["theta"] = trace["theta"] + 1
        self.assertNotEqual(summary, preflight._cpu_trace_summary(changed))

    def _trace(self, steps):
        return {
            "theta": torch.zeros((2, steps, 1024, 4)),
            "membrane": torch.zeros((2, 1024, steps)),
            "spikes": torch.zeros((2, 1024, steps)),
            "component_membrane": torch.zeros((2, 4, 1024, steps)),
            "component_spikes": torch.zeros((2, 4, 1024, steps)),
            "component_gates": torch.zeros((2, 4, 1024, steps)),
            "live_tail_steps": 64,
            "prefix_steps": steps - 64,
        }

    def test_dual_runner_overrides_native_64_step_default_and_rejects_short_trace(self):
        calls = []

        def native_like_runner(_wrapped, gamma_graph, gamma_drive, *, total_steps=64,
                               live_tail_steps=64):
            calls.append((total_steps, live_tail_steps))
            trace = self._trace(total_steps)
            return {"trace": trace}

        gamma = torch.zeros((2, 8, 1024))
        preflight._dual_run(object(), gamma, gamma, native_like_runner)
        self.assertEqual(calls, [(1024, 64)])
        with self.assertRaisesRegex(ValueError, "wrong 1024-step geometry"):
            preflight._validate_trace_geometry(self._trace(64), 2)

    def test_all_native_trace_fields_require_registered_temporal_axis(self):
        trace = self._trace(1024)
        preflight._validate_trace_geometry(trace, 2)
        trace["theta"] = torch.zeros((2, 1024, 1024, 4))
        # theta's registered layout is [B,T,N,fold]; time is axis 1.
        preflight._validate_trace_geometry(trace, 2)
        trace["component_gates"] = torch.zeros((2, 4, 1024, 64))
        with self.assertRaisesRegex(ValueError, "component_gates.*1024-step"):
            preflight._validate_trace_geometry(trace, 2)
        trace = self._trace(1024)
        trace["theta"] = torch.zeros((2, 1024, 1023, 4))
        with self.assertRaisesRegex(ValueError, "theta.*1024-step"):
            preflight._validate_trace_geometry(trace, 2)
        trace = self._trace(1024)
        trace["membrane"] = torch.zeros((2, 1023, 1024))
        with self.assertRaisesRegex(ValueError, "membrane.*1024-step"):
            preflight._validate_trace_geometry(trace, 2)
        trace = self._trace(1024)
        trace["component_spikes"] = torch.zeros((2, 4, 1023, 1024))
        with self.assertRaisesRegex(ValueError, "component_spikes.*1024-step"):
            preflight._validate_trace_geometry(trace, 2)

    def test_real_native_1024_step_trace_passes_geometry_before_settled_q(self):
        """Run the real core at the registered length; its configured default is only 64."""
        from collaborative_test.SW_0172_graph_drive_separation import dual_gamma_rollout
        from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
        from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
        from snn_kuramoto_bidirectional.s2net_cls import S2NetCore

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(17204)
            n = 1024
            hp = S2NetHyperparameters(
                num_feature_maps=8, num_regions=n, sc=torch.eye(n), osc_dim=4,
                gamma_drive_mode="static", gamma_phase_mode="standardize_tanh",
                theta_init="gamma", graph_mode="learned", graph_top_k=2,
                graph_spatial_decay=.35, geodesic_steps=0, geodesic_radius=1.5,
                geodesic_contrast=2.0, geodesic_temperature=.5, geodesic_cap=16.0,
                kuramoto_backend="factorized", k=256.0, freq_gain=2.0,
                num_time_steps=64, spike_per_component=True, gate_mode="raw",
                membrane_vth=.06, spike_pulse_gain=0.0, dendritic_projection="shared",
                spike_spatial_grid_size=32,
            ).validate()
            wrapped = PhaseStateIntegration(S2NetCore(hp, device="cpu"), "phase").eval()
            for parameter in wrapped.parameters():
                parameter.requires_grad_(False)
            gamma = torch.randn((1, 8, n), dtype=torch.float32)
            with torch.no_grad():
                actual = preflight._dual_run(
                    wrapped, gamma, gamma, dual_gamma_rollout.run_dual_gamma_rollout)
                trace = actual["trace"]
                preflight._validate_trace_geometry(trace, 1)
                q = preflight._q(trace)
            self.assertEqual(tuple(trace["theta"].shape), (1, 1024, 1024, 4))
            self.assertEqual(tuple(trace["component_spikes"].shape), (1, 4, 1024, 1024))
            self.assertEqual(tuple(q.shape), (1, 1024, 1024))
            self.assertTrue(torch.isfinite(q).all())
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
