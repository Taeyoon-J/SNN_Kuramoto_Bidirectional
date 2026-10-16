import numpy as np  # Keep NumPy/OpenMP initialization before torch on Windows.
import torch
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0178_temporal_class_competition.class_competition import leave_one_out_class_loss
from collaborative_test.SW_0178_temporal_class_competition.train import (
    ARMS, MASK_SUPERVISED, UNSUPERVISED_LEADERBOARD_ELIGIBLE,
    _backward_microbatch_losses, _credit_failures, _expected_batches, _train_update,
    _validate_cursor, _vectorized_class_loss,
)
from collaborative_test.SW_0178_temporal_class_competition import train as train_module
from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint


class SW178TrainingContracts(unittest.TestCase):
    def test_vectorized_objective_value_and_gradient_match_reference(self):
        generator = torch.Generator().manual_seed(17821)
        raw = torch.randn((3, 11, 11), generator=generator, dtype=torch.float64)
        affinity = ((raw + raw.transpose(1, 2)) * .5).requires_grad_(True)
        # Image 0 includes BG plus a singleton negative class and noncontiguous
        # IDs; image 1 is one-class/zero-loss; image 2 has two valid classes
        # with unequal query counts and another singleton class.
        labels = torch.tensor([
            [0, 0, 0, 4, 4, 9, 9, 9, 9, 22, 22],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [0, 0, 7, 7, 7, 7, 7, 31, 31, 44, 55],
        ], dtype=torch.int64)
        fast_losses = torch.stack([_vectorized_class_loss(affinity[i], labels[i]) for i in range(3)])
        reference = leave_one_out_class_loss(affinity, labels)
        fast = fast_losses.mean()
        self.assertTrue(torch.allclose(fast, reference, atol=1e-11, rtol=1e-11))
        grad_fast, = torch.autograd.grad(fast, affinity, retain_graph=True)
        grad_ref, = torch.autograd.grad(reference, affinity)
        self.assertTrue(torch.allclose(grad_fast, grad_ref, atol=1e-10, rtol=1e-10))
        self.assertTrue(bool(torch.isfinite(grad_fast).all()))

    def test_registered_order_is_two_consecutive_numpy_permutations(self):
        rng = np.random.RandomState(171001)
        expected = []
        for _ in range(2):
            order = rng.permutation(512).astype(np.int64)
            expected.extend([order[i:i + 16].tolist() for i in range(0, 512, 16)])
        actual = _expected_batches()
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 64)
        self.assertTrue(all(len(batch) == 16 for batch in actual))

    def test_first_two_commit_credit_contract_requires_emitted_spike_gradient(self):
        row = {"update": 1, "loss_to_emitted_component_spikes_gradient_l2": 1e-5,
            "gradient_norm_before_clip": .3,
            "family_gradient_norms": {name: 1e-6 for name in
                ("context_encoder", "graph", "native_core", "integration")},
            "family_parameter_delta_l2": {name: 1e-6 for name in
                ("context_encoder", "graph", "native_core", "integration")}}
        self.assertEqual(_credit_failures(row), [])
        row["loss_to_emitted_component_spikes_gradient_l2"] = 0.0
        self.assertTrue(any("emitted-S" in item for item in _credit_failures(row)))

    def test_training_is_explicitly_mask_supervised_not_unsupervised_eligible(self):
        self.assertIs(MASK_SUPERVISED, True)
        self.assertIs(UNSUPERVISED_LEADERBOARD_ELIGIBLE, False)

    def test_shared_microbatch_forward_graph_backpropagates_once_as_sum(self):
        x = torch.tensor([1.25, -0.5], dtype=torch.float64, requires_grad=True)
        shared = (x.square()).sum()
        losses = [shared * factor for factor in (1., 2., 3., 4.)]
        _backward_microbatch_losses(losses)
        expected = (2.0 * x * (1. + 2. + 3. + 4.)) / 16.0
        self.assertTrue(torch.allclose(x.grad, expected, atol=1e-12, rtol=0))

    def test_train_update_backprops_shared_native_forward_once(self):
        # Exercise the production update function with one B2 shared graph. If
        # it backpropagates once per image, the second backward raises because
        # the shared native forward graph has already been released.
        old_batch, old_micro = train_module.BATCH, train_module.MICRO
        train_module.BATCH = train_module.MICRO = 2
        try:
            params = [torch.nn.Parameter(torch.tensor(.2)) for _ in range(4)]
            groups = []
            group_names = ("context_encoder", "graph", "native_core", "integration")
            for name, parameter in zip(group_names, params):
                groups.append({"name": name, "params": [parameter],
                    "parameter_names": [name + ".weight"], "lr": .01})
            optimizer = torch.optim.SGD(groups, lr=.01)
            wrapped = SimpleNamespace(project_integrations_=lambda: None)
            found = SimpleNamespace(wrapped=wrapped)

            # A block structure makes the loss depend on the labels, while a
            # scalar shared across the two images forces both image losses to
            # traverse the exact same forward autograd graph.
            target = np.indices((32, 32))[1].astype(np.int32) % 2
            labels = torch.from_numpy(target.reshape(-1))
            same = (labels[:, None] == labels[None, :]).float()
            different = 1.0 - same
            base = same * 0.8 + different * 0.1
            rgb = np.zeros((2, 128, 128, 3), dtype=np.uint8)
            modal = np.stack([target, target]).astype(np.int32)
            inputs = {"arrays": {"rgb": rgb, "modal_gt": modal},
                      "ordered_train_ids": [1001, 1002]}

            def native_forward(_found, rgb_batch, *, live_tail_steps):
                self.assertEqual(tuple(rgb_batch.shape), (2, 128, 128, 3))
                self.assertEqual(live_tail_steps, 64)
                shared = sum(params)
                q_one = base.to(shared.device) + shared * (same.to(shared.device) -
                                                             different.to(shared.device))
                q = q_one.unsqueeze(0).expand(2, -1, -1)
                rho = q.unsqueeze(1).expand(-1, 4, -1, -1)
                return (torch.zeros((2, 8), device=shared.device),
                        {"component_spikes": None},
                        torch.zeros((2, 1024, 1024), device=shared.device),
                        torch.zeros((2, 8, 1024), device=shared.device), q, rho)

            with mock.patch.object(train_module, "_backward_microbatch_losses",
                                   wraps=_backward_microbatch_losses):
                with mock.patch("collaborative_test.SW_0175_critical_component_binding.train_v2._native_forward",
                                side_effect=native_forward):
                    row = _train_update(ARMS[0], found, optimizer, np.asarray([0, 1]),
                                        inputs, torch.device("cpu"), 3)
            self.assertEqual(row["microbatches"][0]["image_indices"], [0, 1])
            self.assertEqual(set(row["family_gradient_norms"]), set(group_names))
            self.assertTrue(all(torch.isfinite(p.detach()).all() for p in params))
            self.assertTrue(all(p.grad is not None for p in params))
        finally:
            train_module.BATCH, train_module.MICRO = old_batch, old_micro

    def test_boundary_resume_restores_wrapped_encoder_and_optimizer(self):
        temp_parent = Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
        temp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sw178_boundary_", dir=temp_parent) as td:
            root = Path(td).resolve()
            self.assertEqual(root.parent, temp_parent.resolve())
            wrapped, encoder = torch.nn.Linear(3, 2), torch.nn.Linear(2, 2)
            optimizer = torch.optim.Adam([
                {"name": "native_core", "params": list(wrapped.parameters()), "parameter_names": ["w", "b"]},
                {"name": "context_encoder", "params": list(encoder.parameters()), "parameter_names": ["w", "b"]},
            ], lr=.01)
            bindings = {"experiment": "SW0178", "arm": ARMS[0], "initial": "a" * 64}
            runtime = checkpoint.runtime_signature("cpu")
            loss = (wrapped(torch.ones((2, 3))) + encoder(torch.ones((2, 2)))).square().mean()
            loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            expected = [{k: v.detach().clone() for k, v in m.state_dict().items()}
                        for m in (wrapped, encoder)]
            history = [{"update": 1, "image_indices": _expected_batches()[0]}]
            checkpoint.save_boundary(root, completed_updates=1, wrapped=wrapped,
                encoder=encoder, optimizer=optimizer, history=history, bindings=bindings,
                runtime=runtime, device="cpu")
            with torch.no_grad():
                for module in (wrapped, encoder):
                    for parameter in module.parameters(): parameter.add_(5)
            loaded = checkpoint.load_latest(root, wrapped=wrapped, encoder=encoder,
                optimizer=optimizer, expected_bindings=bindings, expected_runtime=runtime,
                device="cpu", max_updates=64)
            self.assertEqual(loaded["completed_updates"], 1)
            self.assertEqual(loaded["history"], history)
            for module, saved in zip((wrapped, encoder), expected):
                for name, value in module.state_dict().items():
                    self.assertTrue(torch.equal(value, saved[name]))
            # The actual optimizer state is restored, not reconstructed empty.
            self.assertTrue(any(optimizer.state.values()))
            _validate_cursor(1, history)


if __name__ == "__main__":
    unittest.main()
