import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from collaborative_test.SW_0170_causal_synaptic_readout import preflight
from collaborative_test.SW_0170_causal_synaptic_readout import causal_synapse
from collaborative_test.SW_0168_end_to_end_connectivity import connectivity_objective
from collaborative_test.SW_0167_structured_relation_energy import audit_train_cache


class PreflightContractTests(unittest.TestCase):
    def test_cpu_prepare_exports_ordered_source_proof_without_gt(self):
        payload = {
            "selected_ids": list(range(10, 18)), "positions": list(range(8)),
            "source_checkpoint_sha256": "a" * 64, "source_manifest_sha256": "b" * 64,
            "source_cache_manifest_sha256": "c" * 64,
            "source_cache_bindings_sha256": "d" * 64,
            "source_cache_state_sha256": "e" * 64,
            "context_cache_manifest_sha256": "f" * 64,
            "context_cache_bindings_sha256": "1" * 64,
            "context_cache_state_sha256": "2" * 64,
            "rgb_sha256": ["3" * 64] * 8, "train_rgb_manifest_sha256": "4" * 64,
            "train_hdf5_path": "train.h5", "train_hdf5_size_bytes": 123,
            "train_hdf5_mtime_ns": 456, "implementation_fingerprint": {"source": "fixed"},
            "ground_truth_read": False, "validation_data_read": False,
        }
        with mock.patch.object(preflight.calibration, "prepare_calibration_cpu", return_value=payload), \
                mock.patch.object(torch.cuda, "is_initialized", return_value=False):
            result = preflight.prepare_inputs_cpu()
        self.assertEqual(result["source_proof"]["selected_ids"], list(range(10, 18)))
        self.assertEqual(result["source_proof"]["source_cache_state_sha256"], "e" * 64)
        self.assertFalse(result["source_proof"]["ground_truth_read"])
        self.assertFalse(result["source_proof"]["validation_data_read"])

    def test_runtime_fingerprint_resolves_registered_dependency_paths(self):
        fingerprint = preflight.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0170_causal_synaptic_readout/causal_synapse.py",
                      fingerprint["files"])
        self.assertIn("collaborative_test/SW_0169_normalized_hard_reset/preflight_v3.py",
                      fingerprint["files"])
        self.assertTrue(all(len(value) == 64 for value in fingerprint["files"].values()))

    def test_filtered_qcc_splits_native_train8_into_registered_b4_calls(self):
        # Exercise the public eight-row helper and its actual CPU-QCC consumer
        # with production geometry, while replacing only the expensive affinity
        # multiply with a deterministic valid Q fixture.
        components = np.zeros((8, 4, 1024, 512), dtype=np.float32)
        calls = []

        def q_b4(batch):
            calls.append(tuple(batch.shape))
            self.assertEqual(batch.dtype, np.float32)
            return np.zeros((4, 1024, 1024), dtype=np.float32)

        with mock.patch.object(audit_train_cache, "_recompute_cpu_q", side_effect=q_b4):
            q, labels = preflight._cpu_filtered_qcc(components)
        self.assertEqual(calls, [(4, 4, 1024, 512), (4, 4, 1024, 512)])
        self.assertEqual(q.shape, (8, 1024, 1024))
        self.assertEqual(labels.shape, (8, 32, 32))

    def test_weighted_forest_route_means_use_pair_mass(self):
        q = torch.tensor([[1., .8, .2, .1], [.8, 1., .4, .3],
                          [.2, .4, 1., .9], [.1, .3, .9, 1.]])
        target = np.array([[1, 1, 2, 2]], dtype=np.int32)
        support = np.ones_like(target, dtype=bool)
        plan = connectivity_objective.forest_routes(q, target, support)
        result = preflight._weighted_route_means(q, plan)
        self.assertGreater(result["positive_pair_mass"], 0)
        self.assertGreater(result["negative_pair_mass"], 0)
        self.assertAlmostEqual(result["positive"], 0.85)
        self.assertAlmostEqual(result["negative"], 0.4)
        self.assertAlmostEqual(result["separation"], 0.45)

    def test_direct_live_causal_state_carry_partial_uses_actual_recurrence_refs(self):
        spikes = torch.ones((1, 4, 2, 6), dtype=torch.float32, requires_grad=True)
        z, refs = causal_synapse.causal_synapse(spikes, live_start=2, capture_step=4)
        z[..., 4].sum().backward()
        direct = preflight._direct_synapse_carry_gradient(refs)
        self.assertEqual(direct["step"], 4)
        self.assertGreater(direct["previous_state_gradient_norm"], 0.0)
        self.assertIn("S_t held fixed", direct["scope"])

    def test_fixed_eligibility_and_stage1_gates_are_explicit(self):
        rows = [{"index": i, "finite": True, "nonconstant_offdiagonal": True,
                 "filtered_activity_variance": .1} for i in range(8)]
        same = [{"separation": .2, "positive_pair_mass": 2, "negative_pair_mass": 3}
                for _ in range(8)]
        better = [{"separation": .3, "positive_pair_mass": 2, "negative_pair_mass": 3}
                  for _ in range(8)]
        self.assertEqual(preflight._gate_failures(rows, .8, .7, same, better, [0, 1, 2, 3]), [])
        self.assertIn("filtered_maximin_loss_not_lower",
                      preflight._gate_failures(rows, .8, .8, same, better, [0, 1, 2, 3]))
        self.assertIn("filtered_qcc_foreground_below_four_of_eight",
                      preflight._gate_failures(rows, .8, .7, same, better, [0, 1, 2]))
        unmatched = [dict(row) for row in better]
        unmatched[-1]["separation"] = None
        self.assertIn("no_fixed_eligible_forest_rows",
                      preflight._gate_failures(rows, .8, .7, same, unmatched, [0, 1, 2, 3]))

    def test_invalid_affinity_routes_fail_as_operational_schema_error(self):
        q = torch.eye(4)
        target = np.array([[1, 1, 2, 2]], dtype=np.int32)
        support = np.ones_like(target, dtype=bool)
        plan = connectivity_objective.forest_routes(q, target, support)
        bad = dict(plan)
        bad["positive_pair_mass"] = 0
        route = preflight._weighted_route_means(q, bad)
        self.assertIsNone(route["separation"])


if __name__ == "__main__":
    unittest.main()
