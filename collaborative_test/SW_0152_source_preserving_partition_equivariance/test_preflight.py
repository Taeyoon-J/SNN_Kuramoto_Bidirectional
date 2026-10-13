import sys
import unittest
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0152_source_preserving_partition_equivariance import preflight, training
from collaborative_test.SW_0152_source_preserving_partition_equivariance.deep_encoder import SourcePreservingDeepEncoder


class FakeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_generator = nn.Linear(2, 2)
        self.fixed = nn.Parameter(torch.ones(1))


class FakeWrapped(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = FakeCore()
        self.a_d = nn.Parameter(torch.ones(4))
        self.a_m = nn.Parameter(torch.ones(4))
        self.b = nn.Parameter(torch.zeros(4))

    def project_integrations_(self):
        with torch.no_grad():
            self.a_d.clamp_(min=0.0)
            self.a_m.clamp_(min=0.0)


class FakeFoundation:
    def __init__(self):
        self.encoder = SourcePreservingDeepEncoder(nn.Conv2d(3, 8, 3))
        self.wrapped = FakeWrapped()


class PreflightTests(unittest.TestCase):
    def test_real_implementation_fingerprint_resolves_all_dependencies(self):
        fingerprint = preflight.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0152_source_preserving_partition_equivariance/preflight.py", fingerprint)
        self.assertIn("collaborative_test/SW_0134_native_spike_binding/rollout.py", fingerprint)
        self.assertTrue(all(len(value) == 64 for value in fingerprint.values()))

    def test_within_color_permutation_helper_measures_real_patch_rows(self):
        rgb = torch.full((1, 128, 128, 3), 100, dtype=torch.uint8)
        labels = ((torch.arange(1024) // 16) % 2).reshape(1, 1024)
        p = torch.nn.functional.one_hot(labels, 2).float() * 0.98 + 0.01
        p = p / p.sum(-1, keepdim=True)
        changed, record = training.color_bin_row_permutation(rgb, p)
        self.assertEqual(record["eligible_patch_count"], 1024)
        self.assertGreater(record["changed_hard_assignment_count"], 0)
        self.assertGreater(float(preflight.alignment_js(p, changed)), 0.1)
        self.assertTrue(torch.equal(rgb, torch.full_like(rgb, 100)))

    def test_trainability_selects_registered_families_and_optimizer_rates(self):
        foundation = FakeFoundation()
        groups = training.configure_trainability(foundation, "equivariance")
        self.assertEqual(set(groups), {"spatial_residual", "graph", "native_core", "integration_scalars"})
        self.assertTrue(all(p.requires_grad for params in groups.values() for p in params))
        self.assertTrue(all(not p.requires_grad for p in foundation.encoder.base_encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in groups["native_core"] + groups["integration_scalars"]))
        self.assertTrue(foundation.wrapped.core.fixed.requires_grad)
        optimizer = training.make_optimizer(groups)
        self.assertEqual([g["name"] for g in optimizer.param_groups],
                         ["spatial_residual", "graph", "native_core", "integration_scalars"])
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [3e-6, 3e-5, 3e-5, 1e-3])

    def test_logical_update_accumulates_four_microbatches_before_single_step(self):
        weight = nn.Parameter(torch.tensor(0.25))
        optimizer = torch.optim.Adam([{"params": [weight], "name": "spatial_residual", "lr": 0.01}])
        seen = []
        def compute(micro):
            seen.append(micro)
            return {"total": (weight - float(micro + 1)).square()}
        before = weight.detach().clone()
        wrapped = FakeWrapped()
        row = training.logical_update(optimizer=optimizer,
                                     family_map={"spatial_residual": [weight]},
                                     compute_micro=compute, wrapped=wrapped)
        self.assertEqual(seen, [0, 1, 2, 3])
        self.assertGreater(row["family_gradient_norms"]["spatial_residual"], 0)
        self.assertNotEqual(float(weight.detach()), float(before))

    def test_preflight_refuses_cpu_before_touching_outputs(self):
        output = HERE / "should_not_be_created_by_cpu_test.json"
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(RuntimeError, "requires assigned CUDA"):
            preflight.run_preflight(device="cpu", output=output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
