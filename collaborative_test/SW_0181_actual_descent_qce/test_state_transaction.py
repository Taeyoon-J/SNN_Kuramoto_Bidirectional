import random
import unittest

import numpy as np
import torch

from collaborative_test.SW_0181_actual_descent_qce import state_transaction as tx


class BufferedScalar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.7], dtype=torch.float64))
        self.register_buffer("calls", torch.tensor(0, dtype=torch.int64))

    def forward(self, x):
        self.calls.add_(1)
        return self.weight * x + self.calls.to(dtype=x.dtype)


def _optimizer_digest(optimizer):
    return tx.transaction_digest({"modules": {}, "optimizer": optimizer.state_dict()})


class StateTransactionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1801)
        np.random.seed(1802)
        random.seed(1803)

    def test_rejected_proposal_restores_parameters_buffers_adam_and_all_cpu_rng(self):
        model = BufferedScalar()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        # Establish real Adam moments/step state before the proposal boundary.
        optimizer.zero_grad(set_to_none=True)
        model(torch.tensor([0.2], dtype=torch.float64)).sum().backward()
        optimizer.step()
        before = tx.capture_transaction({"model": model}, optimizer)
        before_digest = tx.transaction_digest(before)
        before_step = int(optimizer.state[model.weight]["step"].item())

        expected_torch = torch.rand(3)
        expected_python = random.random()
        expected_numpy = np.random.random(3)
        tx.restore_transaction(before, {"model": model}, optimizer)
        # A rejected full proposal mutates the parameter, registered buffer,
        # Adam moments, and every CPU random stream.
        optimizer.zero_grad(set_to_none=True)
        objective = model(torch.rand(1, dtype=torch.float64)).sum()
        objective.backward()
        optimizer.step()
        random.random(); np.random.random(); torch.rand(2)
        self.assertNotEqual(tx.transaction_digest(tx.capture_transaction({"model": model}, optimizer)),
                            before_digest)
        tx.restore_transaction(before, {"model": model}, optimizer)

        self.assertEqual(tx.transaction_digest(tx.capture_transaction({"model": model}, optimizer)),
                         before_digest)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(int(optimizer.state[model.weight]["step"].item()), before_step)
        self.assertTrue(torch.equal(torch.rand(3), expected_torch))
        self.assertEqual(random.random(), expected_python)
        self.assertTrue(np.array_equal(np.random.random(3), expected_numpy))

    def test_accepted_proposal_is_kept_and_snapshot_does_not_alias_live_state(self):
        model = BufferedScalar()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        optimizer.zero_grad(set_to_none=True)
        model(torch.tensor([1.0], dtype=torch.float64)).sum().backward()
        optimizer.step()
        snapshot = tx.capture_transaction({"model": model}, optimizer)
        digest_before = tx.transaction_digest(snapshot)
        saved_weight = snapshot["modules"]["model"]["weight"].clone()
        saved_buffer = snapshot["modules"]["model"]["calls"].clone()

        optimizer.zero_grad(set_to_none=True)
        model(torch.tensor([-0.5], dtype=torch.float64)).square().sum().backward()
        optimizer.step()
        self.assertTrue(torch.equal(snapshot["modules"]["model"]["weight"], saved_weight))
        self.assertTrue(torch.equal(snapshot["modules"]["model"]["calls"], saved_buffer))
        self.assertNotEqual(tx.transaction_digest(tx.capture_transaction({"model": model}, optimizer)),
                            digest_before)
        self.assertEqual(int(optimizer.state[model.weight]["step"].item()), 2)

    def test_restore_rejects_changed_module_schema(self):
        model = BufferedScalar()
        optimizer = torch.optim.Adam(model.parameters())
        snapshot = tx.capture_transaction({"model": model}, optimizer)
        other = torch.nn.Linear(1, 1)
        with self.assertRaises(ValueError):
            tx.restore_transaction(snapshot, {"model": other}, optimizer)

    def test_cpu_rng_capture_does_not_initialize_cuda(self):
        initialized_before = torch.cuda.is_initialized()
        snapshot = tx.capture_rng_state()
        self.assertEqual(snapshot["torch_cuda_initialized"], initialized_before)
        self.assertEqual(torch.cuda.is_initialized(), initialized_before)


if __name__ == "__main__":
    unittest.main()
