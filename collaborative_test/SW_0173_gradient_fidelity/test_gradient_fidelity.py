"""CPU contracts for fixed-direction gradient-fidelity accounting."""
import numpy as np  # Keep NumPy ahead of Torch on Windows.
import unittest

import torch
from torch import nn

from collaborative_test.SW_0173_gradient_fidelity import gradient_fidelity as gf


class GradientFidelityTests(unittest.TestCase):
    def test_symmetric_difference_matches_autograd_dot_and_restores_exact_state(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([0.4, -0.7], dtype=torch.float64))
                self.register_buffer("calls", torch.tensor(0, dtype=torch.int64))

        model = Model()
        modules = {"model": model}
        group = gf.parameter_union(modules)[0]
        direction = torch.tensor([0.25, -0.3], dtype=torch.float64)
        directions = {group.name: direction}
        initial_rng = torch.random.get_rng_state().clone()
        initial_parameter = model.weight.detach().clone()

        def closure():
            with torch.no_grad():
                model.calls.add_(1)
            # A random additive constant checks common-random-number evaluation.
            noise = torch.rand((), dtype=torch.float64)
            return (model.weight.pow(3)).sum() + noise + model.calls.to(torch.float64) * 2

        result = gf.directional_gradient_check(
            named_modules=modules, directions=directions,
            epsilons=(1e-3, 5e-3), closure=closure)
        expected = float((3 * initial_parameter.square() * direction).sum())
        self.assertAlmostEqual(result["autograd_dot_direction"], expected, places=12)
        self.assertEqual(len(result["evaluations"]), 2)
        self.assertEqual([r["epsilon"] for r in result["evaluations"]], [1e-3, 5e-3])
        for row in result["evaluations"]:
            self.assertAlmostEqual(row["symmetric_difference"], expected, delta=3e-5)
            self.assertLess(row["absolute_error"], 3e-5)
        self.assertEqual(result["state_before_sha256"], result["state_after_sha256"])
        self.assertTrue(result["state_restored"])
        self.assertTrue(torch.equal(model.weight, initial_parameter))
        self.assertEqual(int(model.calls), 0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), initial_rng))
        self.assertEqual(result["epsilon_selection"],
                         "none; all supplied values reported in input order")

    def test_baseline_buffer_mutation_is_reset_before_each_finite_difference(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([0.5], dtype=torch.float64))
                self.register_buffer("offset", torch.tensor(0.0, dtype=torch.float64))

        model = Model()
        group = gf.parameter_union({"m": model})[0]

        def closure():
            with torch.no_grad():
                model.offset.add_(3.0)
            return model.weight.square().sum() + model.offset

        result = gf.directional_gradient_check(
            named_modules={"m": model},
            directions={group.name: torch.ones_like(model.weight)},
            epsilons=(1e-3,), closure=closure)
        row = result["evaluations"][0]
        self.assertAlmostEqual(row["symmetric_difference"], 1.0, places=10)
        self.assertAlmostEqual(result["autograd_dot_direction"], 1.0, places=10)
        self.assertTrue(result["state_restored"])

    def test_projected_boundary_uses_one_sided_directional_check(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # At the physical lower boundary, the supplied direction points inward.
                self.theta = nn.Parameter(torch.tensor([0.0], dtype=torch.float64))

        model = Model()
        group = gf.parameter_union({"m": model})[0]
        direction = torch.tensor([0.1], dtype=torch.float64)

        def constrained_objective():
            if bool((model.theta < 0).any()):
                raise AssertionError("one-sided probe evaluated an infeasible negative state")
            return 2.0 * model.theta.sum() + model.theta.square().sum()

        result = gf.directional_forward_check(
            named_modules={"m": model}, directions={group.name: direction},
            scales=(1 / 16, 1.0), closure=constrained_objective)
        self.assertEqual(result["difference_scheme"], "one-sided feasible forward difference")
        self.assertEqual([row["scale"] for row in result["evaluations"]], [1 / 16, 1.0])
        self.assertAlmostEqual(result["autograd_dot_direction"], 0.2, places=12)
        for row in result["evaluations"]:
            self.assertAlmostEqual(row["actual_change"],
                                   0.2 * row["scale"] + 0.01 * row["scale"] ** 2,
                                   places=12)
        self.assertTrue(result["state_restored"])
        self.assertEqual(float(model.theta.detach()), 0.0)

    def test_parameter_union_counts_tied_parameter_once_and_lists_both_aliases(self):
        class Tied(nn.Module):
            def __init__(self):
                super().__init__()
                parameter = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
                self.left = parameter
                self.right = parameter

        model = Tied()
        groups = gf.parameter_union({"net": model})
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].name, "net.left")
        self.assertEqual(groups[0].aliases, ("net.left", "net.right"))
        direction = torch.ones_like(model.left)
        result = gf.directional_gradient_check(
            named_modules={"net": model}, directions={"net.left": direction},
            epsilons=(1e-4,), closure=lambda: (model.left + model.right).sum())
        # The tied value occurs twice in the loss, while the union contains one slot.
        self.assertEqual(result["autograd_dot_direction"], 4.0)
        self.assertAlmostEqual(result["evaluations"][0]["symmetric_difference"], 4.0, places=10)

    def test_exception_restores_parameter_buffer_and_rng(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
                self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))

        model = Model()
        before_sha = gf.state_sha256({"m": model})
        before_rng = torch.random.get_rng_state().clone()
        calls = 0

        def closure():
            nonlocal calls
            calls += 1
            with torch.no_grad():
                model.counter.add_(1)
            if calls == 2:
                raise RuntimeError("injected finite-difference failure")
            return model.weight.square().sum()

        group = gf.parameter_union({"m": model})[0]
        with self.assertRaisesRegex(RuntimeError, "injected"):
            gf.directional_gradient_check(
                named_modules={"m": model},
                directions={group.name: torch.ones_like(model.weight)},
                epsilons=(1e-3,), closure=closure)
        self.assertEqual(gf.state_sha256({"m": model}), before_sha)
        self.assertEqual(int(model.counter), 0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before_rng))

    def test_invalid_direction_or_epsilon_is_rejected_without_mutation(self):
        model = nn.Linear(2, 1, bias=False, dtype=torch.float64)
        modules = {"m": model}
        group = gf.parameter_union(modules)[0]
        before = gf.state_sha256(modules)
        good = {group.name: torch.ones_like(group.parameter)}
        with self.assertRaisesRegex(ValueError, "positive finite"):
            gf.directional_gradient_check(named_modules=modules, directions=good,
                epsilons=(0.0,), closure=lambda: model.weight.square().sum())
        with self.assertRaisesRegex(ValueError, "exactly match"):
            gf.directional_gradient_check(named_modules=modules, directions={},
                epsilons=(1e-3,), closure=lambda: model.weight.square().sum())
        self.assertEqual(gf.state_sha256(modules), before)


if __name__ == "__main__":
    unittest.main()
