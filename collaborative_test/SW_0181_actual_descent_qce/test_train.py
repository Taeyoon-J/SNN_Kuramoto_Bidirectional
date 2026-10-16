"""CPU contracts for SW0181 proposal decisions and durable cursor semantics."""
from __future__ import annotations

import unittest
import os
from pathlib import Path

import numpy as np
import torch

from collaborative_test.SW_0181_actual_descent_qce import train, state_transaction as tx
from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint


class ProposalDecisionTests(unittest.TestCase):
    def setUp(self):
        random_state = 18101
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        self.model = torch.nn.Linear(2, 1)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.05)
        self.modules = {"model": self.model}
        self.opt.zero_grad(set_to_none=True)

    def _proposal(self):
        x = torch.tensor([[1.0, -0.5], [0.25, 0.75]])
        loss = self.model(x).square().mean()
        loss.backward()
        self.opt.step()

    def test_rejected_proposal_restores_weights_adam_rng_and_clears_grad(self):
        snapshot = tx.capture_transaction(self.modules, self.opt)
        start_sha = tx.transaction_digest(snapshot)
        start_rng = snapshot["rng"]["torch_cpu"].clone()
        self._proposal()
        self.assertNotEqual(tx.transaction_digest(tx.capture_transaction(self.modules, self.opt)), start_sha)
        torch.rand(3)
        after_gradient_rng = tx.capture_rng_state()
        accepted, threshold = train._commit_decision(
            snapshot, self.modules, self.opt, before=1.0, after=1.0,
            rng_after_gradient=after_gradient_rng)
        self.assertFalse(accepted)
        self.assertEqual(threshold, 1e-6)
        restored = tx.capture_transaction(self.modules, self.opt)
        self.assertEqual(tx.transaction_digest(restored), start_sha)
        self.assertTrue(torch.equal(torch.get_rng_state(), start_rng))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_accepted_proposal_keeps_adam_and_restores_post_gradient_rng(self):
        snapshot = tx.capture_transaction(self.modules, self.opt)
        before = [parameter.detach().clone() for parameter in self.model.parameters()]
        self._proposal()
        after_gradient_rng = tx.capture_rng_state()
        expected_rng = after_gradient_rng["torch_cpu"].clone()
        accepted, threshold = train._commit_decision(
            snapshot, self.modules, self.opt, before=1.0, after=0.9,
            rng_after_gradient=after_gradient_rng)
        self.assertTrue(accepted)
        self.assertEqual(threshold, 1e-6)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, self.model.parameters())))
        self.assertEqual(len(self.opt.state), len(list(self.model.parameters())))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_registered_guard_rejects_insufficient_or_nonfinite_descent(self):
        accepted, threshold = train._acceptance(0.5, 0.5 - 0.49e-6)
        self.assertFalse(accepted)
        self.assertEqual(threshold, 0.5e-6)
        accepted, _ = train._acceptance(0.5, 0.5 - 1.01e-6)
        self.assertTrue(accepted)
        accepted, _ = train._acceptance(0.5, float("nan"))
        self.assertFalse(accepted)

    def test_cursor_tracks_proposals_separately_from_accepted_adam_steps(self):
        batches = train._expected_batches()
        rows = [
            {"proposal": 1, "image_indices": batches[0], "accepted": True,
             "before_loss": 1.0, "after_loss": 0.8, "acceptance_threshold": 1e-6,
             "accepted_update_count": 1},
            {"proposal": 2, "image_indices": batches[1], "accepted": False,
             "before_loss": 1.0, "after_loss": 1.0, "acceptance_threshold": 1e-6,
             "accepted_update_count": 1},
        ]
        train._validate_cursor(2, 1, rows)
        with self.assertRaises(ValueError):
            train._validate_cursor(2, 2, rows)
        rows[1]["image_indices"] = batches[2]
        with self.assertRaises(ValueError):
            train._validate_cursor(2, 1, rows)

    def test_real_boundary_roundtrip_restores_model_adam_and_proposal_cursor(self):
        # Exercise the production atomic checkpoint container with SW0181's
        # distinct proposal-cursor / accepted-Adam-step metadata.
        temp_root = train.ROOT / "collaborative_testing" / "temporary"
        temp_root.mkdir(parents=True, exist_ok=True)
        work = temp_root / f"sw181-boundary-{os.getpid()}-{id(self)}"
        work.mkdir(exist_ok=False)
        try:
            wrapped = torch.nn.Linear(2, 1)
            encoder = torch.nn.Linear(2, 2)
            optimizer = torch.optim.Adam(
                list(wrapped.parameters()) + list(encoder.parameters()), lr=0.01)
            modules = {"wrapped": wrapped, "encoder": encoder}
            for scale in (1.0, 0.5):
                optimizer.zero_grad(set_to_none=True)
                loss = (wrapped(torch.ones(1, 2)) ** 2).mean() + scale * (
                    encoder(torch.ones(1, 2)) ** 2).mean()
                loss.backward()
                optimizer.step()

            batches = train._expected_batches()
            history = []
            for index, accepted in enumerate((True, False)):
                history.append({
                    "proposal": index + 1,
                    "proposal_seed": 171000 + 2 + index,
                    "image_indices": batches[index],
                    "accepted": accepted,
                    "before_loss": 1.0,
                    "after_loss": 0.5 if accepted else 1.0,
                    "acceptance_threshold": 1e-6,
                    "accepted_update_count": 1,
                })
            train._validate_cursor(2, 1, history, seed=2)
            runtime = {**checkpoint.runtime_signature("cpu"),
                "sw0181_proposal_cursor_semantics": True,
                "proposal_cursor": 2, "accepted_updates": 1}
            bindings = {"experiment": train.EXPERIMENT, "seed": 2,
                "training_fingerprint": {"files": {"train.py": "ab" * 32}}}
            checkpoint.save_boundary(work, completed_updates=2, wrapped=wrapped,
                encoder=encoder, optimizer=optimizer, history=history, bindings=bindings,
                runtime=runtime, device="cpu")
            saved_modules = {name: {key: value.detach().clone()
                for key, value in module.state_dict().items()}
                for name, module in modules.items()}
            saved_transaction_sha = train._state_digest(modules, optimizer)

            with torch.no_grad():
                for parameter in list(wrapped.parameters()) + list(encoder.parameters()):
                    parameter.add_(3.0)
            optimizer.zero_grad(set_to_none=True)
            (wrapped(torch.zeros(1, 2)).sum() + encoder(torch.zeros(1, 2)).sum()).backward()
            optimizer.step()

            loaded = checkpoint.load_latest(work, wrapped=wrapped, encoder=encoder,
                optimizer=optimizer, expected_bindings=bindings, expected_runtime=runtime,
                device="cpu", max_updates=train.PROPOSALS)
            self.assertEqual(loaded["completed_updates"], 2)
            self.assertEqual(loaded["history"], history)
            self.assertEqual(checkpoint.inspect_committed_metadata(work)["runtime"][
                "accepted_updates"], 1)
            for name, module in modules.items():
                for key, value in module.state_dict().items():
                    self.assertTrue(torch.equal(value, saved_modules[name][key]))
            self.assertEqual(train._state_digest(modules, optimizer), saved_transaction_sha)
            self.assertEqual(len(optimizer.state), 4)
            for state in optimizer.state.values():
                self.assertEqual(int(state["step"].item()), 2)
        finally:
            for child in work.iterdir():
                child.unlink()
            work.rmdir()
            if temp_root.is_dir() and not any(temp_root.iterdir()):
                temp_root.rmdir()


if __name__ == "__main__":
    unittest.main()
