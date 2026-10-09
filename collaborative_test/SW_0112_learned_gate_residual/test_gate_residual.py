import unittest
import contextlib
import io
import json
from unittest import mock

import numpy as np
import torch
from torch import nn

from gate_residual import SharedGateResidual, actual_gate_binding
from collaborative_test.SW_0112_learned_gate_residual import run as runner
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components


def sinusoidal_gating(theta_hist, t, phase_delay_steps, gate_mode="sigmoid"):
    theta = theta_hist[t]
    delayed = theta_hist[max(0, t - phase_delay_steps)]
    gate = 0.5 * (1.0 + torch.sin(delayed.mean(dim=-1)))
    if gate_mode == "raw":
        return torch.sin(theta) * gate.unsqueeze(-1), gate
    return torch.sin(theta) * torch.sigmoid(gate).unsqueeze(-1), torch.sigmoid(gate)


class FakeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_residual = SharedGateResidual()


class GateResidualTests(unittest.TestCase):
    def _reference_core(self, state, gate):
        core = runner.common.make_core("cpu", steps=4)
        if gate:
            runner.attach_gate_residual(core)
        core.load_state_dict(state, strict=True)
        # This is the terminal hook used by short/registered training and
        # isolates the dynamics from unused internal object grouping.
        core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
        core.eval()
        if gate:
            original_forward = core.forward
            def bound_forward(*args, **kwargs):
                with actual_gate_binding(core):
                    return original_forward(*args, **kwargs)
            core.forward = bound_forward
        return core

    @staticmethod
    def _fixed_labels(spikes, components):
        groups = spike_synchrony_components(
            spikes, synchrony_threshold=0.5, min_group_size=2,
            settle=2, components=components, background="largest_component")
        return spatial_components_to_patch_labels(groups, 16)

    def test_eval_terminal_group_skip_preserves_control_and_candidate_cc(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(min(old_threads, 4))
        try:
            torch.manual_seed(190112)
            cases = []
            for gate in (False, True):
                source = runner.common.make_core("cpu", steps=4)
                if gate:
                    runner.attach_gate_residual(source)
                    with torch.no_grad():
                        source.gate_residual.w.copy_(torch.linspace(-0.2, 0.2, 8))
                        source.gate_residual.b.fill_(0.1)
                state = {key: value.detach().clone() for key, value in source.state_dict().items()}
                reference = self._reference_core(state, gate)
                with mock.patch.object(runner.torch, "load", return_value=state):
                    evaluated = runner._eval_core("cpu", "synthetic-checkpoint.pt", 4, gate)
                gamma = torch.randn(1, 8, 256)
                with torch.no_grad():
                    ref_out = reference(gamma, return_core_out=True, return_theta=True)
                    eval_out = evaluated(gamma, return_core_out=True, return_theta=True)
                for a, b in zip(ref_out[1:], eval_out[1:]):
                    self.assertTrue(torch.equal(a, b))
                self.assertTrue(torch.equal(reference.last_component_spikes,
                                            evaluated.last_component_spikes))
                self.assertTrue(torch.equal(reference.last_component_out,
                                            evaluated.last_component_out))
                ref_labels = self._fixed_labels(ref_out[1], reference.last_component_spikes)
                eval_labels = self._fixed_labels(eval_out[1], evaluated.last_component_spikes)
                self.assertTrue(torch.equal(ref_labels, eval_labels))
                cases.append(True)
            self.assertEqual(cases, [True, True])
        finally:
            torch.set_num_threads(old_threads)

    def test_cli_success_json_flush_is_a_print_argument(self):
        from collaborative_test.SW_0112_learned_gate_residual import run as runner
        old_argv = runner.sys.argv
        output = io.StringIO()
        try:
            runner.sys.argv = ["run.py", "preflight", "--seed", "0", "--arm", "control",
                               "--device", "cpu", "--output", "unused.json"]
            with mock.patch.object(runner, "preflight", return_value={"status": "passed"}), \
                 mock.patch.object(runner, "write"), contextlib.redirect_stdout(output):
                runner.main()
        finally:
            runner.sys.argv = old_argv
        record = json.loads(output.getvalue())
        self.assertEqual(record, {"status": "complete", "stage": "preflight", "seed": 0, "arm": "control"})

    def test_zero_initialization_is_bitwise_identity(self):
        gate = SharedGateResidual()
        delayed = torch.randn(3, 7, 4)
        base = torch.rand(3, 7)
        self.assertTrue(torch.equal(gate(delayed, base), base))

    def test_residual_is_bounded_and_has_finite_parameter_gradients(self):
        gate = SharedGateResidual()
        with torch.no_grad():
            gate.w.copy_(torch.linspace(-1.0, 1.0, 8))
            gate.b.fill_(0.3)
        delayed = torch.randn(2, 5, 4)
        base = torch.rand(2, 5)
        corrected = gate(delayed, base)
        self.assertTrue(torch.all(corrected >= base.square() - 1e-7))
        self.assertTrue(torch.all(corrected <= 2 * base - base.square() + 1e-7))
        grad = torch.autograd.grad(corrected.sum(), (gate.w, gate.b))
        self.assertTrue(all(torch.isfinite(g).all() for g in grad))

    def test_actual_binding_uses_delayed_history_and_restores_on_error(self):
        core = FakeCore()
        original = sinusoidal_gating
        history = [torch.full((2, 3, 4), value) for value in (0.1, 0.5, 1.0)]
        with actual_gate_binding(core):
            drive, corrected = sinusoidal_gating(history, 2, 2, gate_mode="raw")
        self.assertIs(sinusoidal_gating, original)
        base = 0.5 * (1 + torch.sin(history[0].mean(-1)))
        self.assertTrue(torch.equal(corrected, base))
        self.assertTrue(torch.equal(drive, torch.sin(history[2]) * base.unsqueeze(-1)))
        with self.assertRaises(RuntimeError):
            with actual_gate_binding(core):
                raise RuntimeError("sentinel")
        self.assertIs(sinusoidal_gating, original)

    def test_rejects_wrong_fold_or_component_contract(self):
        gate = SharedGateResidual()
        with self.assertRaises(ValueError):
            gate(torch.randn(2, 3, 5), torch.rand(2, 3))


if __name__ == "__main__":
    unittest.main()
