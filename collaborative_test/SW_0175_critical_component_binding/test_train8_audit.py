"""Focused contracts for the non-gating SW0175 fixed TRAIN8 audit."""
from __future__ import annotations

import numpy as np  # Must precede torch on Windows.
import json
import torch
import unittest

from collaborative_test.SW_0175_critical_component_binding import train8_audit as audit
from collaborative_test.SW_0175_critical_component_binding import train_v2 as train


class Train8AuditTests(unittest.TestCase):
    def test_cached_multiclass_support_and_bool_mask_have_same_semantics(self):
        cached = np.asarray([[13, 13, 0], [4, 0, 4]], dtype=np.int32)
        mask = cached > 0
        self.assertTrue(np.array_equal(train._support_mask(cached), mask))
        self.assertTrue(np.array_equal(train._support_mask(mask), mask))
        self.assertTrue(np.array_equal(audit._support_mask(cached), mask))
        self.assertTrue(np.array_equal(audit._support_mask(mask), mask))

    def test_adjusted_rand_is_label_permutation_invariant(self):
        source = np.asarray([[0, 0, 2], [0, 4, 4]], dtype=np.int32)
        relabeled = np.asarray([[19, 19, 7], [19, 2, 2]], dtype=np.int32)
        changed = np.asarray([[0, 0, 2], [4, 4, 4]], dtype=np.int32)
        self.assertAlmostEqual(audit._adjusted_rand(source, relabeled), 1.0, places=12)
        self.assertLess(audit._adjusted_rand(source, changed), 1.0)

    def _route_case(self, zero_kind):
        support = np.asarray([[5, 5], [0, 5]], dtype=np.int32)
        target = np.asarray([[8, 8], [0, 8]], dtype=np.int32)
        rho = torch.ones((4, 4, 4), dtype=torch.float32)
        rho[:, 1, 3] = rho[:, 3, 1] = 0.9
        rho[0, 0, 1] = rho[0, 1, 0] = -0.2 if zero_kind == "negative" else 0.0
        if zero_kind == "two_zero_components":
            rho[1, 0, 1] = rho[1, 1, 0] = 0.0
        q = rho.clamp_min(0).prod(dim=0)
        return q, rho, target, support

    def test_clamp_blocked_pair_mass_uses_exact_component_partials(self):
        q, rho, target, support = self._route_case("one_zero_component")
        stats = audit._routes_and_clamp(q, rho, target, support)
        self.assertEqual(stats["positive_route_edges"], 2)
        self.assertEqual(stats["positive_route_pair_mass"], 3)
        self.assertEqual(stats["positive_clamp_blocked_pair_mass"], 0)
        self.assertEqual(stats["positive_route_zero_q_pair_mass"], 2)
        self.assertEqual([x["count"] for x in stats["component_rho_quantiles_on_positive_routes"]],
                         [2, 2, 2, 2])

        for case in ("negative", "two_zero_components"):
            q, rho, target, support = self._route_case(case)
            stats = audit._routes_and_clamp(q, rho, target, support)
            self.assertEqual(stats["positive_clamp_blocked_pair_mass"], 2)
            self.assertEqual(stats["positive_route_zero_q_pair_mass"], 2)

    def test_event_occupancy_distinguishes_continuous_amplitude_from_events(self):
        trace = {"component_gates": torch.full((1, 4, 3, 5), .25),
                 "component_spikes": torch.full((1, 4, 3, 5), .25)}
        result = audit._event_stats(trace)
        self.assertEqual(result["component_gate_mean_amplitude"], [.25] * 4)
        self.assertEqual(result["component_spike_mean_amplitude"], [.25] * 4)
        self.assertEqual(result["component_gate_active_fraction"], [1.0] * 4)
        self.assertEqual(result["component_spike_event_fraction"], [1.0] * 4)
        self.assertEqual(result["component_events_given_active_gate"], [1.0] * 4)
        with self.assertRaises(ValueError):
            audit._event_stats({"component_gates": torch.ones((1, 3, 3, 5)),
                                "component_spikes": torch.ones((1, 3, 3, 5))})

    def test_report_seals_partition_arrays_as_json_safe_hashes(self):
        rows = {"native_Q": [{"endpoints": {
            endpoint: {"labels": np.arange(1024, dtype=np.int32).reshape(32, 32),
                       "scores": {"fg_ari": .25}}
            for endpoint in audit.ENDPOINTS}}]}
        audit._seal_partition_labels(rows)
        for endpoint in audit.ENDPOINTS:
            item = rows["native_Q"][0]["endpoints"][endpoint]
            self.assertNotIn("labels", item)
            self.assertEqual(len(item["partition_sha256"]), 64)
        json.dumps(rows, allow_nan=False)

    def test_fixed_positions_and_audit_are_explicitly_non_gating(self):
        self.assertEqual(audit.POSITIONS, tuple(range(256, 264)))
        fingerprint = audit.implementation_fingerprint()
        self.assertEqual(fingerprint["optimizer_updates"], 0)
        self.assertFalse(fingerprint["selection_or_gate_use"])
        self.assertIn("train8_audit.py", " ".join(fingerprint["files"].keys()))


if __name__ == "__main__":
    unittest.main()
