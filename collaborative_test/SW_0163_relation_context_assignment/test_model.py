"""Meaningful CPU contracts for matched SW0163 Q/A-context heads."""
from __future__ import annotations

import unittest

import torch

from collaborative_test.SW_0163_relation_context_assignment.model import (
    ARMS, RelationContextAssignment,
)
from collaborative_test.SW_0161_supervised_spike_partition.objective import mask_hungarian_loss


torch.set_num_threads(1)


def inputs(seed=163):
    gen = torch.Generator().manual_seed(seed)
    spikes = torch.rand((1, 4, 1024, 512), generator=gen, dtype=torch.float32)
    raw_q = torch.rand((1, 1024, 1024), generator=gen, dtype=torch.float32)
    raw_a = torch.rand((1, 1024, 1024), generator=gen, dtype=torch.float32)
    q = (raw_q + raw_q.transpose(-1, -2)) * .5
    a = (raw_a + raw_a.transpose(-1, -2)) * .5
    q.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    a.diagonal(dim1=-2, dim2=-1).fill_(.25)
    return spikes, q, a


class RelationContextTests(unittest.TestCase):
    def test_matched_arm_initialization_and_inactive_channels(self):
        spikes, q, a = inputs()
        models = {arm: RelationContextAssignment(arm) for arm in ARMS}
        states = [model.state_dict() for model in models.values()]
        for other in states[1:]:
            self.assertEqual(states[0].keys(), other.keys())
            for key in states[0]:
                torch.testing.assert_close(states[0][key], other[key], rtol=0, atol=0)
        q2, a2 = a, q
        outputs = {}
        captured = {}
        hooks = []
        for arm, model in models.items():
            def capture(module, args, arm=arm):
                captured[arm] = args[0].detach().clone()
            hooks.append(model.relation_mlp.register_forward_pre_hook(capture))
            outputs[arm] = model(spikes, q, a)[0].detach()
        for hook in hooks:
            hook.remove()

        # Zero final message projection makes all four matched heads exactly the
        # original raw binder before the first optimizer update.
        for other in outputs.values():
            torch.testing.assert_close(outputs["raw_only"], other, rtol=0, atol=0)
        raw_binder = models["raw_only"].binder(spikes)[0].detach()
        torch.testing.assert_close(outputs["raw_only"], raw_binder, rtol=0, atol=0)

        # Concatenated input is [h,Qmsg,Amsg]; inactive relation channels are exact zero.
        self.assertEqual(float(captured["raw_only"][..., 64:128].abs().max()), 0.0)
        self.assertEqual(float(captured["raw_only"][..., 128:192].abs().max()), 0.0)
        self.assertGreater(float(captured["Q_only"][..., 64:128].abs().max()), 0.0)
        self.assertEqual(float(captured["Q_only"][..., 128:192].abs().max()), 0.0)
        self.assertEqual(float(captured["A_only"][..., 64:128].abs().max()), 0.0)
        self.assertGreater(float(captured["A_only"][..., 128:192].abs().max()), 0.0)
        self.assertGreater(float(captured["Q_A"][..., 64:128].abs().max()), 0.0)
        self.assertGreater(float(captured["Q_A"][..., 128:192].abs().max()), 0.0)

        # With exact same initial parameters, each arm ignores the topology it excludes.
        torch.testing.assert_close(models["raw_only"](spikes, q, a)[0],
            models["raw_only"](spikes, q2, a2)[0], rtol=0, atol=0)
        torch.testing.assert_close(models["Q_only"](spikes, q, a)[0],
            models["Q_only"](spikes, q, a2)[0], rtol=0, atol=0)
        torch.testing.assert_close(models["A_only"](spikes, q, a)[0],
            models["A_only"](spikes, q2, a)[0], rtol=0, atol=0)
        # Once message weights learn, the combined arm responds to both channels.
        labels = torch.zeros((1, 1024), dtype=torch.int64)
        labels[:, 256:512] = 1
        labels[:, 512:768] = 2
        labels[:, 768:] = 3
        optimizer = torch.optim.Adam(models["Q_A"].parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = mask_hungarian_loss(models["Q_A"](spikes, q, a)[0], labels)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(models["Q_A"](spikes, q, a)[0].detach(),
                                     models["Q_A"](spikes, q2, a2)[0].detach()))

    def test_two_disposable_updates_have_live_relation_and_binder_credit(self):
        spikes, q, a = inputs(164)
        model = RelationContextAssignment("Q_A")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        labels = torch.zeros((1, 1024), dtype=torch.int64)
        labels[:, 256:512] = 1
        labels[:, 512:768] = 2
        labels[:, 768:] = 3
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        grads = {}
        losses = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            probabilities, slots, h = model(spikes, q, a)
            self.assertEqual(tuple(probabilities.shape), (1, 1024, 11))
            torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones((1, 1024)), atol=2e-6, rtol=0)
            loss, _ = mask_hungarian_loss(probabilities, labels)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    grads[name] = float(parameter.grad.norm())
            if step == 0:
                self.assertEqual(grads.get("relation_mlp.0.weight", 0.0), 0.0)
                self.assertGreater(grads["relation_mlp.2.weight"], 0.0)
            else:
                self.assertGreater(grads["relation_mlp.0.weight"], 0.0)
                self.assertGreater(grads["relation_mlp.2.weight"], 0.0)
            self.assertGreater(grads["binder.input_projection.weight"], 0.0)
            self.assertGreater(grads["binder.gru.weight_hh"], 0.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertEqual(len(losses), 2)
        for name, parameter in model.named_parameters():
            if name in ("relation_mlp.0.weight", "binder.input_projection.weight", "binder.gru.weight_hh"):
                self.assertGreater(float((parameter.detach() - before[name]).norm()), 0.0, name)

    def test_topology_diagonal_is_ignored_and_invalid_relations_reject(self):
        spikes, q, a = inputs(165)
        model = RelationContextAssignment("Q_A").eval()
        q_without_diag, a_without_diag = q.clone(), a.clone()
        q_without_diag.diagonal(dim1=-2, dim2=-1).zero_()
        a_without_diag.diagonal(dim1=-2, dim2=-1).zero_()
        with torch.no_grad():
            base = model(spikes, q_without_diag, a_without_diag)[0]
            diag_q = q_without_diag + torch.eye(1024).unsqueeze(0)
            diag_a = a_without_diag + torch.eye(1024).unsqueeze(0) * 9.0
            changed = model(spikes, diag_q, diag_a)[0]
        torch.testing.assert_close(base, changed, rtol=0, atol=0)
        bad = q.clone(); bad[0, 0, 1] += .5
        with self.assertRaisesRegex(ValueError, "symmetric"):
            model(spikes, bad, a)
        bad = q.clone(); bad[0, 0, 1] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            model(spikes, bad, a)
        bad = q.clone(); bad[0, 0, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            model(spikes, bad, a)
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            model(spikes, q * 2, a)
        with self.assertRaisesRegex(ValueError, "arm"):
            RelationContextAssignment("unknown")

    def test_patch_node_permutation_covariance(self):
        spikes, q, a = inputs(166)
        model = RelationContextAssignment("Q_A").eval()
        generator = torch.Generator().manual_seed(167)
        permutation = torch.randperm(1024, generator=generator)
        with torch.no_grad():
            original = model(spikes, q, a)[0]
            permuted = model(spikes[:, :, permutation, :],
                q[:, permutation, :][:, :, permutation],
                a[:, permutation, :][:, :, permutation])[0]
        torch.testing.assert_close(permuted, original[:, permutation, :], rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
