import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import sys

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint as durable
from collaborative_test.SW_0152_source_preserving_partition_equivariance import train, training, preflight_queue


class Wrapped(nn.Module):
    def __init__(self):
        super().__init__()
        self.values = nn.ParameterList([nn.Parameter(torch.tensor(v)) for v in (.2, .3, .4, .5)])

    def project_integrations_(self):
        return None


def fixture():
    wrapped, encoder = Wrapped(), nn.Linear(1, 1, bias=False)
    families = {name: [wrapped.values[i]] for i, name in enumerate(
        ("spatial_residual", "graph", "native_core", "integration_scalars"))}
    optimizer = torch.optim.Adam([
        {"name": name, "params": params, "lr": .01} for name, params in families.items()])
    return wrapped, encoder, families, optimizer


def update(wrapped, families, optimizer, step):
    def micro(index):
        total = sum((p - (step + index + 1) * .1).square() for ps in families.values() for p in ps)
        return {"total": total}
    return training.logical_update(optimizer=optimizer, family_map=families,
                                  compute_micro=micro, wrapped=wrapped)


class TrainContractTests(unittest.TestCase):
    def test_color_permutation_gate_requires_positive_finite_delta(self):
        report = {"within_color_permutation_diagnostic": {
            "baseline_alignment_js_mean": .1, "within_color_permuted_alignment_js_mean": .2}}
        self.assertAlmostEqual(train.require_color_permutation_gate(report), .1)
        report["within_color_permutation_diagnostic"]["within_color_permuted_alignment_js_mean"] = .1
        with self.assertRaisesRegex(ValueError, "admission"):
            train.require_color_permutation_gate(report)

    def test_importing_cpu_queue_validator_restores_gpu_mapping(self):
        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        existed = "CUDA_VISIBLE_DEVICES" in os.environ
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
            with mock.patch.object(preflight_queue, "validate_report", return_value=True):
                self.assertTrue(train.preflight_queue_valid({}, {"ids": []}, {}, {}))
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "2")
        finally:
            if existed: os.environ["CUDA_VISIBLE_DEVICES"] = original
            else: os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def test_checkpoint_resume_matches_uninterrupted_optimizer_boundary(self):
        torch.manual_seed(17)
        a = fixture(); wa, ea, fa, oa = a
        update(wa, fa, oa, 0)
        with tempfile.TemporaryDirectory() as temp:
            bindings, runtime = {"lineage": "fixed"}, durable.runtime_signature("cpu")
            durable.save_boundary(temp, completed_updates=1, wrapped=wa, encoder=ea,
                optimizer=oa, history=[{"update": 1}], bindings=bindings,
                runtime=runtime, device="cpu")
            update(wa, fa, oa, 1)
            expected = [p.detach().clone() for p in wa.parameters()]
            expected_steps = [int(state["step"].item()) for state in oa.state.values()]

            wb, eb, fb, ob = fixture()
            loaded = durable.load_latest(temp, wrapped=wb, encoder=eb, optimizer=ob,
                expected_bindings=bindings, expected_runtime=runtime, device="cpu", max_updates=128)
            self.assertEqual(loaded["completed_updates"], 1)
            update(wb, fb, ob, 1)
            for left, right in zip(expected, wb.parameters()):
                self.assertTrue(torch.equal(left, right))
            self.assertEqual(expected_steps, [int(state["step"].item()) for state in ob.state.values()])


if __name__ == "__main__":
    unittest.main()
