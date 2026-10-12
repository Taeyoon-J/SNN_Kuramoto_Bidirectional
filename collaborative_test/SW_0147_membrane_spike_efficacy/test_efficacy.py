from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional")]

import torch

from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0147_membrane_spike_efficacy.efficacy import (
    MembraneSpikeEfficacy, efficacy_rollout,
)
from collaborative_test.SW_0134_native_spike_binding.test_rollout import _core


class MembraneEfficacyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14701)

    def _wrapped(self):
        return PhaseStateIntegration(_core(64), "phase")

    def test_zero_rho_reproduces_native_rollout_exactly(self):
        wrapped = self._wrapped()
        gamma = torch.randn(1, 8, 64)
        expected = late_rollout(wrapped, gamma, total_steps=64, live_tail_steps=32)
        module = MembraneSpikeEfficacy(torch.zeros(4), torch.ones(4))
        actual = efficacy_rollout(wrapped, gamma, module, total_steps=64,
                                  live_tail_steps=32)
        for key in ("component_membrane", "component_spikes", "component_gates",
                    "membrane", "spikes", "theta"):
            self.assertTrue(torch.equal(actual[key], expected[key]), key)
        self.assertTrue(torch.equal(actual["component_native_spikes"],
                                    actual["component_spikes"]))

    def test_efficacy_is_bounded_and_uses_native_event_times_gate(self):
        module = MembraneSpikeEfficacy(torch.zeros(4), torch.ones(4))
        with torch.no_grad():
            module.rho.copy_(torch.tensor([-8., -1., 1., 8.]))
        native = torch.tensor([[0., 1., 0., .5], [1., 0., .5, 0.],
                               [0., .5, 1., 0.], [.5, 0., 0., 1.]])
        margin = torch.tensor([[1., -1., 0., 2.], [-1., 0., 1., 0.],
                               [0., 2., -1., 1.], [2., 1., 0., -1.]])
        ids = torch.tensor([0, 1, 2, 3])
        factor, _ = module.efficacy(margin, ids)
        self.assertEqual(factor.shape, native.shape)
        emitted = native + factor * native
        ratio = torch.where(native != 0, emitted / native.clamp_min(1e-9), 1.0)
        self.assertTrue(torch.all((ratio >= .5) & (ratio <= 1.5)))
        self.assertTrue(torch.equal(emitted[native == 0], torch.zeros_like(emitted[native == 0])))
        with self.assertRaisesRegex(ValueError, "component ID"):
            module.efficacy(torch.ones(1, 4), torch.tensor([0, 1, 2, 3]))

    def test_margin_kernel_has_zero_init_rho_credit_and_live_only_margin_credit(self):
        module = MembraneSpikeEfficacy(torch.zeros(4), torch.ones(4))
        ids = torch.tensor([0, 1, 2, 3])
        native = torch.ones(4, 5)
        margin = torch.tensor([[.2, .4, .6, .8, 1.0]] * 4, requires_grad=True)

        # At zero initialization the added efficacy is exactly zero. Its
        # first-step signal can train rho, while the margin derivative is zero.
        zero_factor, _ = module.efficacy(margin, ids, credit="live")
        zero_added = zero_factor * native
        margin0, rho0 = torch.autograd.grad(
            zero_added.sum(), (margin, module.rho), allow_unused=True)
        self.assertTrue(torch.equal(zero_added, torch.zeros_like(zero_added)))
        self.assertIsNotNone(margin0)
        self.assertTrue(torch.equal(margin0, torch.zeros_like(margin0)))
        self.assertTrue(torch.isfinite(rho0).all())
        self.assertGreater(float(rho0.detach().norm()), 0.)

        # Once rho has moved, live and detached arms retain identical forward
        # efficacy, but only live phi carries the added margin gradient.
        with torch.no_grad():
            module.rho.fill_(.25)
        live_factor, _ = module.efficacy(margin, ids, credit="live")
        detached_factor, _ = module.efficacy(margin, ids, credit="detached")
        live_added = live_factor * native
        detached_added = detached_factor * native
        self.assertTrue(torch.equal(live_added, detached_added))
        live_margin = torch.autograd.grad(live_added.sum(), margin, retain_graph=True)[0]
        detached_margin = torch.autograd.grad(
            detached_added.sum(), margin, allow_unused=True)[0]
        self.assertTrue(torch.isfinite(live_margin).all())
        self.assertGreater(float(live_margin.detach().norm()), 0.)
        self.assertIsNone(detached_margin)

    def test_live_second_step_membrane_credit_and_detached_forward_identity(self):
        gamma = torch.randn(1, 8, 64, requires_grad=True)
        live_wrap = self._wrapped()
        detached_wrap = PhaseStateIntegration(_core(64), "phase")
        detached_wrap.core.load_state_dict(copy.deepcopy(live_wrap.core.state_dict()))
        detached_wrap.load_state_dict(copy.deepcopy(live_wrap.state_dict()))
        live = MembraneSpikeEfficacy(torch.zeros(4), torch.ones(4))
        detached = MembraneSpikeEfficacy(torch.zeros(4), torch.ones(4))
        live.load_state_dict(copy.deepcopy(detached.state_dict()))
        live_out = efficacy_rollout(live_wrap, gamma, live, total_steps=64,
                                    live_tail_steps=32, credit="live")
        detach_out = efficacy_rollout(detached_wrap, gamma, detached, total_steps=64,
                                      live_tail_steps=32, credit="detached")
        self.assertTrue(torch.equal(live_out["component_spikes"],
                                    detach_out["component_spikes"]))
        # At rho=0, first-step efficacy is zero. A first update moves rho;
        # the second pass then carries live phi credit into membrane/core.
        optimizer = torch.optim.SGD([live.rho], lr=.1)
        first = live_out["component_spikes"][..., 32:].sum()
        grad_rho = torch.autograd.grad(first, live.rho, retain_graph=True)[0]
        self.assertTrue(torch.isfinite(grad_rho).all())
        self.assertGreater(float(grad_rho.detach().norm()), 0.)
        optimizer.zero_grad(set_to_none=True)
        first.backward()
        optimizer.step()
        self.assertGreater(float(live.rho.detach().abs().max()), 0.)

        # Re-run with the changed rho; only the live branch has the added
        # efficacy derivative into membrane. Native event-surrogate credit
        # remains present in both arms.
        with torch.no_grad():
            detached.rho.copy_(live.rho)
        live2 = efficacy_rollout(live_wrap, gamma.detach(), live, total_steps=64,
                                 live_tail_steps=32, credit="live")
        det2 = efficacy_rollout(detached_wrap, gamma.detach(), detached, total_steps=64,
                                live_tail_steps=32, credit="detached")
        self.assertTrue(torch.equal(live2["component_spikes"], det2["component_spikes"]))
        loss_live = live2["component_spikes"][..., 32:].sum()
        loss_det = det2["component_spikes"][..., 32:].sum()
        gl = torch.autograd.grad(loss_live, live_wrap.core.membrane_layer.tau_m,
                                 allow_unused=True)[0]
        gd = torch.autograd.grad(loss_det, detached_wrap.core.membrane_layer.tau_m,
                                 allow_unused=True)[0]
        # Both receive native surrogate gradients; the detached arm is not
        # asserted to have zero total membrane gradient.
        self.assertIsNotNone(gl)
        self.assertIsNotNone(gd)
        self.assertTrue(torch.isfinite(gl).all() and torch.isfinite(gd).all())
        self.assertGreater(float((gl - gd).detach().norm()), 1e-8,
                           "live efficacy must add membrane credit beyond detached efficacy")

    def test_same_adjusted_spike_is_recurrent_reset_and_readout_value(self):
        wrapped = self._wrapped()
        gamma = torch.randn(1, 8, 64)
        module = MembraneSpikeEfficacy(torch.zeros(4), torch.ones(4))
        with torch.no_grad():
            module.rho.fill_(.5)
        trace = efficacy_rollout(wrapped, gamma, module, total_steps=64,
                                 live_tail_steps=64)
        emitted = trace["component_spikes"]
        previous = trace["component_previous_spikes"]
        self.assertTrue(torch.equal(previous[..., 1:], emitted[..., :-1]))
        self.assertTrue(torch.equal(wrapped.core.last_component_spikes, emitted))
        native = trace["component_native_spikes"]
        self.assertTrue(torch.all((emitted >= .5 * native - 1e-7)
                                  & (emitted <= 1.5 * native + 1e-7)))

    def test_rejects_invalid_calibration_and_credit_arm(self):
        with self.assertRaises(ValueError):
            MembraneSpikeEfficacy(torch.zeros(4), torch.tensor([1., 1., 0., 1.]))
        wrapped = self._wrapped()
        module = MembraneSpikeEfficacy(torch.zeros(4), torch.ones(4))
        with self.assertRaisesRegex(ValueError, "credit"):
            efficacy_rollout(wrapped, torch.randn(1, 8, 64), module,
                             total_steps=64, live_tail_steps=0, credit="other")


if __name__ == "__main__":
    unittest.main()
