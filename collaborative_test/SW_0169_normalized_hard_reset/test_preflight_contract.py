from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0169_normalized_hard_reset import preflight
from collaborative_test.SW_0169_normalized_hard_reset import hard_reset
from collaborative_test.SW_0169_normalized_hard_reset.test_hard_reset import _TinyWrapped


class PreflightContractTests(unittest.TestCase):
    def test_event_transitions_run_along_time_not_nodes(self):
        # Two nodes each flip at every adjacent time pair: 2 * 3 = 6.
        raw = torch.tensor([[[[0., 1., 0., 1.], [1., 0., 1., 0.]]]])
        gate = torch.ones_like(raw)
        trace = {"raw_events": raw, "component_gates": gate,
                 "component_spikes": raw * gate}
        with mock.patch.object(preflight, "SETTLE", 0):
            rows = preflight._event_diagnostics(trace)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["active_to_active_event_transitions"], 6)
        self.assertEqual(rows[0]["active_samples"], 8)

    def test_event_diagnostics_reports_fraction_of_nodes_with_temporal_flip(self):
        # Exactly one of ten eligible nodes flips. Aggregate transition count
        # alone is insufficient; the registered admission unit is the node.
        raw = torch.zeros((1, 1, 10, 4))
        raw[0, 0, 0] = torch.tensor([0., 1., 0., 1.])
        gate = torch.ones_like(raw)
        trace = {"raw_events": raw, "component_gates": gate,
                 "component_spikes": raw * gate}
        with mock.patch.object(preflight, "SETTLE", 0):
            row = preflight._event_diagnostics(trace)[0]
        self.assertEqual(row["eligible_node_units"], 10)
        self.assertEqual(row["node_units_with_active_event_flip"], 1)
        self.assertAlmostEqual(row["active_event_flip_node_fraction"], 0.1)
        self.assertEqual(preflight._event_gate_failures([row]), [])

    def test_event_gate_rejects_one_flip_below_ten_percent_node_units(self):
        row = {"image_index": 0, "component": 1, "active_samples": 100,
               "raw_event_on_fraction_when_active": 0.5,
               "eligible_node_units": 11, "total_node_units": 11,
               "node_units_with_active_event_flip": 1,
               "active_event_flip_node_fraction": 1 / 11}
        self.assertEqual(preflight._event_gate_failures([row])[0]["reason"],
                         "component_image_node_flip_fraction_below_10pct")

    def test_both_history_arms_use_the_identical_maximin_objective(self):
        calls = []

        def constrained(q, target, foreground):
            calls.append((q, target, foreground.copy()))
            return q.sum()

        fake = types.SimpleNamespace(
            constrained_maximin_loss=constrained,
            matched_local_edge_loss=lambda *_: self.fail("local objective is not registered"))
        q = torch.eye(4, requires_grad=True)
        target = np.asarray([[1, 1], [2, 2]], dtype=np.int32)
        support = np.ones((2, 2), dtype=np.int32)
        with mock.patch.object(preflight, "_import_keep_cuda", return_value=fake):
            live = preflight._objective("history_live", q, target, support)
            detached = preflight._objective("history_detached", q, target, support)
        self.assertTrue(torch.equal(live, detached))
        self.assertEqual(len(calls), 2)
        self.assertTrue(np.array_equal(calls[0][1], calls[1][1]))
        self.assertTrue(np.array_equal(calls[0][2], calls[1][2]))

    def test_source_support_is_recomputed_cpu_b4_from_cached_spikes(self):
        calls = []
        q_batches = [np.zeros((4, 1024, 1024), dtype=np.float32),
                     np.ones((4, 1024, 1024), dtype=np.float32)]

        def recompute(components):
            calls.append(np.asarray(components).copy())
            return q_batches[len(calls) - 1]

        audit = types.SimpleNamespace(_recompute_cpu_q=recompute)
        qcc = types.SimpleNamespace(qcc_labels_from_affinity=lambda q: torch.ones(
            (q.shape[0], 32, 32), dtype=torch.int32))
        structured = types.SimpleNamespace(
            feasible_target=lambda labels, target: np.where(labels > 0, target, 0))

        def importer(name):
            if name.endswith("audit_train_cache"):
                return audit
            if name.endswith("qcc_readout"):
                return qcc
            if name.endswith("structured_energy"):
                return structured
            raise AssertionError(name)

        fingerprint = {"fixed": "source"}
        calibration_input = {
            "implementation_fingerprint": fingerprint,
            "source_cache_state_sha256": "a" * 64,
            "selected_ids": list(range(8)),
            "cache_component_spikes": [np.full((4, 1024, 512), i, np.float32)
                                       for i in range(8)],
            # Deliberately unusable: production support must not be taken from it.
            "cache_contexts": [(None, None)] * 8,
        }
        report = {"source_six_trace_parity": True, "source_q_parity": True,
                  "source_a_parity": True, "source_cache_replay_exact": True,
                  "implementation_fingerprint": fingerprint,
                  "source_model_state_sha256_after": "a" * 64}
        masks = np.ones((8, 32, 32), dtype=np.int32)
        with mock.patch.object(preflight, "_import_keep_cuda", side_effect=importer):
            result = preflight.prepare_train_targets(calibration_input, report, masks)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].shape, (4, 4, 1024, 512))
        self.assertEqual(calls[1].shape, (4, 4, 1024, 512))
        self.assertEqual(len(result["supports"]), 8)

    def test_actual_rollout_capture_and_direct_partial_replay_contract(self):
        def run(detached):
            torch.manual_seed(16903)
            wrapped = _TinyWrapped()
            hard_reset.initialize_slow_membrane_(wrapped)
            found = types.SimpleNamespace(wrapped=wrapped)
            gamma = torch.randn(1, 2, 3, requires_grad=True)
            with mock.patch.object(preflight, "STEPS", 6), mock.patch.object(
                    preflight, "TAIL", 2):
                trace, observed = preflight._rollout_capture_final_carries(
                    found, gamma, torch.ones(4), detach_history=detached)
                self.assertEqual(set(observed), {
                    "previous_membrane", "previous_event", "step_kwargs", "actual_emitted"})
                trace["component_spikes"].sum().backward()
                direct = preflight._direct_carry_partial_gradient(
                    observed, detach_history=detached)
            return observed["actual_emitted"].detach().clone(), direct

        live_emitted, live = run(False)
        detached_emitted, detached = run(True)
        self.assertTrue(torch.equal(live_emitted, detached_emitted))
        self.assertGreater(live["previous_membrane"], 0.0)
        self.assertGreater(live["previous_event"], 0.0)
        self.assertEqual(detached["previous_membrane"], 0.0)
        self.assertEqual(detached["previous_event"], 0.0)
        self.assertIn("local final-step partial", live["scope"])


if __name__ == "__main__":
    unittest.main()
