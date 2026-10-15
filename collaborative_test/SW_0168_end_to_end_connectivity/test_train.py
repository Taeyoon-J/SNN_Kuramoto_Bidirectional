"""CPU-side schedule and optimizer-artifact checks for SW0168."""
from __future__ import annotations

import unittest

import torch

from collaborative_test.SW_0168_end_to_end_connectivity import train


class TrainingContractTests(unittest.TestCase):
    def test_fixed_exposures_are_two_deterministic_full_train_permutations(self):
        first = train._epoch_orders()
        self.assertEqual(first, train._epoch_orders())
        self.assertEqual(len(first), 2)
        self.assertTrue(all(len(order) == 512 and sorted(order) == list(range(512))
                            for order in first))
        self.assertEqual((train.BATCH, train.MICRO, train.UPDATES), (16, 4, 64))
        self.assertEqual(train.EPOCHS * train.COUNT, train.UPDATES * train.BATCH)

    def test_optimizer_receipt_names_updated_and_structurally_unused_parameters(self):
        live = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        unused = torch.nn.Parameter(torch.tensor([2.0]))
        integration = torch.nn.Parameter(torch.tensor([0.25]))
        optimizer = torch.optim.Adam([
            {"params": [live, unused], "parameter_names": ["core.live", "core.unused"],
             "lr": 1e-3, "name": "native_core"},
            {"params": [integration], "parameter_names": ["integration.b"],
             "lr": 1e-3, "name": "integration"},
        ])
        for _ in range(train.UPDATES):
            optimizer.zero_grad(set_to_none=True)
            (live.square().sum() + integration.square().sum()).backward()
            optimizer.step()
        receipt = train._optimizer_step_summary(optimizer)
        self.assertEqual(receipt["native_core"]["updated_parameter_names"], ["core.live"])
        self.assertEqual(receipt["native_core"]["unused_parameter_names"], ["core.unused"])
        self.assertEqual(receipt["native_core"]["steps"], [64])
        self.assertEqual(receipt["integration"]["max_step"], 64)

    def test_summary_accepts_late_or_unused_params_but_requires_critical_credit(self):
        critical = lambda: {"parameter_count": 2, "updated_parameter_count": 2,
            "unused_parameter_count": 0, "steps": [63, 64], "max_step": 64}
        receipt = {name: critical() for name in ("encoder", "graph", "native_core")}
        receipt["integration"] = {"parameter_count": 3, "updated_parameter_count": 1,
            "unused_parameter_count": 2, "steps": [17], "max_step": 17}
        train._validate_optimizer_summary(receipt)
        receipt["integration"] = {"parameter_count": 3, "updated_parameter_count": 0,
            "unused_parameter_count": 3, "steps": [], "max_step": 0}
        train._validate_optimizer_summary(receipt)
        receipt["graph"] = {"parameter_count": 2, "updated_parameter_count": 1,
            "unused_parameter_count": 1, "steps": [63], "max_step": 63}
        with self.assertRaises(ValueError):
            train._validate_optimizer_summary(receipt)

    def test_artifact_admission_rejects_unregistered_or_missing_arm_artifact(self):
        with self.assertRaises(ValueError):
            train.validate_training_artifact("not_an_arm")
        with self.assertRaises((FileNotFoundError, ValueError)):
            train.validate_training_artifact("constrained_maximin", train.HERE / "missing_fixture")


if __name__ == "__main__":
    unittest.main()
