from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for import_root in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from collaborative_test.SW_0151_source_preserving_event_history.preflight import (
    accumulate_threshold_rms,
    assert_exact_trace_parity,
    event_summaries,
    finish_threshold_rms,
    implementation_fingerprint,
    merge_history_moments,
    module_state_digest,
    new_rms_accumulator,
    validate_registered_train_order,
    validate_train_rgb_schema,
)


class ActualSourcePreflightHelpersTests(unittest.TestCase):
    def test_float64_rms_accumulates_exact_per_component_threshold_residual(self):
        threshold = torch.tensor([.1, .2, .3, .4], dtype=torch.float32)
        membrane = torch.zeros(2, 4, 3, 6, dtype=torch.float32)
        for component in range(4):
            membrane[:, component] = threshold[component] + (component + 1) * .1
        state = new_rms_accumulator()
        state = accumulate_threshold_rms(state, membrane, threshold, settle=2)
        observed = finish_threshold_rms(state)
        expected = [0.1, 0.2, 0.3, 0.4]
        for got, want in zip(observed, expected):
            self.assertAlmostEqual(got, want, places=7)

    def test_scale_rejects_zero_nonfinite_wrong_shape_and_no_settled_steps(self):
        for scale_accumulator in (
            (torch.zeros(4, dtype=torch.float64), 0),
            (torch.tensor([1., 1., 1., float("nan")], dtype=torch.float64), 4),
            (torch.ones(3, dtype=torch.float64), 4),
        ):
            with self.assertRaises(ValueError):
                finish_threshold_rms(scale_accumulator)
        with self.assertRaises(ValueError):
            finish_threshold_rms((torch.zeros(4, dtype=torch.float64), 4))

    def test_exact_trace_and_production_q_parity_is_fail_closed(self):
        native = {key: torch.randn(2, 4, 3, 8) for key in (
            "component_membrane", "component_spikes", "component_gates")}
        native.update({key: torch.randn(2, 3, 8) for key in ("membrane", "spikes")})
        native["theta"] = torch.randn(2, 8, 3, 4)
        adaptive = {key: value.clone() for key, value in native.items()}
        q = torch.randn(2, 3, 3)
        assert_exact_trace_parity(native, adaptive, q, q.clone())
        adaptive["component_spikes"][0, 0, 0, 0] += 1
        with self.assertRaisesRegex(AssertionError, "component_spikes"):
            assert_exact_trace_parity(native, adaptive, q, q)
        adaptive["component_spikes"] = native["component_spikes"].clone()
        with self.assertRaisesRegex(AssertionError, "Q mismatch"):
            assert_exact_trace_parity(native, adaptive, q, q + 1e-7)

    def test_event_summaries_report_finite_settled_statistics(self):
        events = torch.zeros(2, 4, 3, 1024)
        events[..., 512::2] = 1
        history = torch.arange(1024, dtype=torch.float32).view(1, 1, 1, 1024).expand_as(events)
        stats = event_summaries({"component_events": events, "component_event_history": history})
        self.assertEqual(stats["settled_event_on_fraction_by_component"], [0.5] * 4)
        self.assertEqual(stats["settled_event_exact_zero_fraction_by_component"], [0.5] * 4)
        self.assertEqual(stats["settled_event_exact_one_fraction_by_component"], [0.5] * 4)
        self.assertTrue(all(value > 0 for value in stats["settled_history_variance_by_component"]))
        with self.assertRaises(ValueError):
            event_summaries({"component_events": events[..., :32],
                             "component_event_history": history[..., :32]})

    def test_implementation_fingerprint_binds_actual_source_and_preflight_code(self):
        fingerprint = implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0135_native32_spike_binding/foundation.py", fingerprint)
        self.assertIn("snn_kuramoto_bidirectional/graph_generator.py", fingerprint)
        self.assertIn("snn_kuramoto_bidirectional/kuramoto_layer.py", fingerprint)
        self.assertIn("collaborative_test/SW_0145_native32_boundary_credit_probe/run.py", fingerprint)
        self.assertIn("collaborative_test/SW_0151_source_preserving_event_history/adaptive_rollout.py", fingerprint)
        self.assertIn("collaborative_test/SW_0151_source_preserving_event_history/event_history.py", fingerprint)
        self.assertTrue(all(len(value) == 64 for value in fingerprint.values()))

    def test_source_contract_order_and_train_cache_schema_are_strict(self):
        ids = list(range(4096))
        rows = list(range(1000, 5096))
        validate_registered_train_order(ids, rows, expected_ids=ids, expected_rows=rows)
        with self.assertRaisesRegex(ValueError, "image IDs"):
            validate_registered_train_order(ids, rows, expected_ids=ids[::-1], expected_rows=rows)
        with self.assertRaisesRegex(ValueError, "pool rows"):
            validate_registered_train_order(ids, rows, expected_ids=ids, expected_rows=rows[::-1])
        validate_train_rgb_schema(type("CacheHeader", (), {
            "shape": (70000, 128, 128, 3), "dtype": np.dtype(np.uint8)})())
        for shape, dtype in (((4096, 128, 128, 3), np.dtype(np.uint8)),
                             ((70000, 128, 128, 3), np.dtype(np.float32))):
            with self.assertRaises(ValueError):
                validate_train_rgb_schema(type("CacheHeader", (), {"shape": shape, "dtype": dtype})())

    def test_parameter_and_persistent_or_nonpersistent_buffers_are_bound(self):
        class ModuleWithBuffers(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scalar = torch.nn.Parameter(torch.tensor(1.))
                self.weight = torch.nn.Parameter(torch.tensor([1., 2.]))
                self.register_buffer("scalar_buffer", torch.tensor(5.))
                self.register_buffer("persistent", torch.tensor([3.]))
                self.register_buffer("temporary", torch.tensor([4.]), persistent=False)

        module = ModuleWithBuffers()
        before = module_state_digest({"model": module})
        with torch.no_grad():
            module.temporary.add_(1)
        self.assertNotEqual(before, module_state_digest({"model": module}))

    def test_history_variance_uses_stable_pooled_moments_for_constant_and_mixed_batches(self):
        mean = np.zeros(4, dtype=np.float64)
        m2 = np.zeros(4, dtype=np.float64)
        count = 0
        constant = torch.full((1, 4, 1, 32), 1.0 - 2 ** -24, dtype=torch.float64)
        mean, m2, count = merge_history_moments(mean, m2, count, constant)
        mean, m2, count = merge_history_moments(mean, m2, count, constant.clone())
        self.assertEqual(count, 64)
        self.assertEqual(m2.tolist(), [0.0] * 4)
        self.assertEqual((m2 / count).tolist(), [0.0] * 4)

        mean, m2, count = np.zeros(4), np.zeros(4), 0
        mean, m2, count = merge_history_moments(
            mean, m2, count, torch.ones(1, 4, 1, 1, dtype=torch.float64))
        mean, m2, count = merge_history_moments(
            mean, m2, count, torch.full((1, 4, 1, 1), 3.0, dtype=torch.float64))
        self.assertEqual(mean.tolist(), [2.0] * 4)
        self.assertEqual((m2 / count).tolist(), [1.0] * 4)


if __name__ == "__main__":
    unittest.main()
