from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0173_gradient_fidelity import audit
from collaborative_test.SW_0173_gradient_fidelity import gradient_fidelity as gf
from collaborative_test.SW_0168_end_to_end_connectivity import connectivity_objective as objective


class AuditContractTests(unittest.TestCase):
    def test_fingerprint_binds_queue_and_frozen_production_dependencies(self):
        value = audit.fingerprint()
        self.assertEqual(value["experiment"], "SW0173_native_Q_gradient_horizon_fidelity_audit")
        self.assertIn("collaborative_test/SW_0173_gradient_fidelity/audit_queue.py", value["files"])
        self.assertIn("collaborative_test/SW_0173_gradient_fidelity/test_audit.py", value["files"])
        self.assertIn("collaborative_test/SW_0136_native32_transfer/evaluate.py", value["files"])
        self.assertEqual(value["optimizer_updates"], 0)

    def test_fixed_route_loss_matches_native_objective_and_gradient(self):
        target = torch.tensor([[1, 1, 2], [1, 2, 2], [0, 0, 0]])
        support = target > 0
        base = torch.linspace(.08, .92, 81, dtype=torch.float64).reshape(9, 9)
        q1 = ((base + base.T) / 2).requires_grad_(True)
        plan = objective.forest_routes(q1.detach(), target, support)
        native = objective.constrained_maximin_loss(q1, target, support)
        fixed = audit.fixed_route_loss(q1, plan)
        grad_native, = torch.autograd.grad(native, q1, retain_graph=True)
        grad_fixed, = torch.autograd.grad(fixed, q1)
        self.assertTrue(torch.equal(native, fixed))
        self.assertTrue(torch.equal(grad_native, grad_fixed))

    def test_production_rho_product_and_four_edge_clamp_jacobian(self):
        generator = torch.Generator().manual_seed(1703)
        spikes = (torch.rand((1, 4, 3, 516), generator=generator) > .5).float().requires_grad_()
        trace = {"component_spikes": spikes}
        rho = audit._component_rho(trace)
        q = rho.clamp_min(0).prod(1)[0]
        edges = [(0, 1, 3), (0, 2, 2), (1, 2, 4), (2, 0, 1)]
        q_loss = sum((q[i, j] - .31).square() for i, j, _ in edges)
        result = audit.clamp_blocking_diagnostics(q, trace, q_loss,
            {"positive_edges": edges})
        self.assertEqual(result["positive_route_edges"], 4)
        self.assertGreaterEqual(result["all_component_partials_zero_edges"], 0)
        self.assertAlmostEqual(result["positive_pair_mass"], 10)
        self.assertTrue(0 <= result["blocked_edge_fraction"] <= 1)
        self.assertTrue(0 <= result["blocked_positive_pair_mass_fraction"] <= 1)
        self.assertTrue(0 <= result["blocked_positive_loss_gradient_mass_fraction"] <= 1)

    def test_event_change_uses_common_active_samples_and_distinct_thresholds(self):
        base = {
            "component_gates": torch.ones(1, 4, 2, 3),
            "component_spikes": torch.zeros(1, 4, 2, 3),
            "component_membrane": torch.full((1, 4, 2, 3), .5),
        }
        trial = {name: value.clone() for name, value in base.items()}
        trial["component_gates"][..., 0] = 0
        trial["component_spikes"][..., 1] = 1
        trial["component_membrane"][..., 2] = .9
        result = audit._event_change(base, trial, torch.full((1, 4, 1, 1), .4),
                                     torch.full((1, 4, 1, 1), .8))
        self.assertEqual(result["common_active_samples"], 16)
        self.assertEqual(result["event_flips_on_common_active"], 8)
        self.assertEqual(result["threshold_crossings_on_common_active"], 8)

    def test_single_image_native_metrics_use_registered_metric_names(self):
        from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
        labels = np.arange(1024, dtype=np.int32).reshape(32, 32) % 5
        metrics = audit._single_train_metrics(contract, labels, labels.copy())
        self.assertEqual(set(metrics), set(contract.METRICS))
        self.assertTrue(all(value == 1.0 for value in metrics.values()))

    def test_virtual_adam_delta_uses_qualified_names_and_restores_state(self):
        class Wrapped(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.tensor([.4], dtype=torch.float64))
                self.a_d = torch.nn.Parameter(torch.tensor([.2], dtype=torch.float64))

            def project_integrations_(self):
                with torch.no_grad():
                    self.a_d.clamp_(0, 1)

        wrapped = Wrapped()
        encoder = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        found = types.SimpleNamespace(wrapped=wrapped, encoder=encoder)

        class FakeTrain:
            @staticmethod
            def _make_optimizer(_found):
                group = {"params": [wrapped.w, wrapped.a_d],
                         "parameter_names": ["graph.weight", "a_d"],
                         "name": "native_core", "lr": 1e-2}
                return torch.optim.Adam([group], lr=1e-2), {}

        gradients = {"native_core::graph.weight": torch.tensor([1.], dtype=torch.float64),
                     "native_core::a_d": torch.tensor([-1.], dtype=torch.float64)}
        before = gf.state_sha256({"wrapped": wrapped, "encoder": encoder})
        rng = gf._rng_snapshot()
        with mock.patch.object(audit.probe, "_import", return_value=FakeTrain):
            delta, reported_before = audit._virtual_delta(found, gradients)
        self.assertEqual(before, reported_before)
        self.assertEqual(before, gf.state_sha256({"wrapped": wrapped, "encoder": encoder}))
        self.assertIn("native_core::graph.weight", delta)
        self.assertIn("native_core::a_d", delta)
        self.assertGreater(float(delta["native_core::graph.weight"].abs().item()), 0)
        self.assertGreater(float(delta["native_core::a_d"].abs().item()), 0)
        self.assertTrue(torch.equal(gf._rng_snapshot()["cpu"], rng["cpu"]))

    def test_preregistered_decision_checks_use_both_scales_and_populations(self):
        parity = [{"repeat_full": {"trace_fields_exact": True, "q_exact": True,
                                    "gamma_exact": True, "actual_A_exact": True,
                                    "projection_exact": True},
                   "tail_full": {"all_trace_fields_exact": True, "q_exact": True,
                                 "gamma_exact": True, "actual_A_exact": True,
                                 "projection_exact": True}}]
        trials = {h: {p: [] for p in ("cohort", "transfer")} for h in ("tail64", "full1024")}
        for h in trials:
            for p in trials[h]:
                for scale in audit.SCALES:
                    change = -.1 if h == "full1024" else -.01
                    trials[h][p].append({"scale": scale,
                        "predicted_directional_change": -.2,
                        "actual_native_change": change,
                        "actual_fixed_route_change": change / 2})
        result = audit._decision_checks(parity, trials)
        self.assertTrue(result["same_rng_repeat_jitter_zero"])
        self.assertTrue(result["tail_full_exact_source_parity"])
        self.assertTrue(result["all_full1024_direction_checks_pass"])
        self.assertTrue(result["full_transfer_more_negative_than_tail_at_both_scales"])
        trials["full1024"]["transfer"][1]["actual_native_change"] = .01
        self.assertFalse(audit._decision_checks(parity, trials)["all_full1024_direction_checks_pass"])

    def test_cross_horizon_gradient_delta_matrix_uses_parameter_union(self):
        parameter = torch.nn.Parameter(torch.tensor([2., -1.], dtype=torch.float64))
        optimizer = types.SimpleNamespace(param_groups=[{
            "parameter_names": ["w"], "params": [parameter], "name": "native_core"}])
        gradients = {
            "tail64": {"by_id": {id(parameter): torch.tensor([1., 3.], dtype=torch.float64)}},
            "full1024": {"by_id": {id(parameter): torch.tensor([-2., 4.], dtype=torch.float64)}},
        }
        deltas = {
            "tail64": {"native_core::w": torch.tensor([.5, -.25], dtype=torch.float64)},
            "full1024": {"native_core::w": torch.tensor([.125, .75], dtype=torch.float64)},
        }
        matrix = audit._cross_directional_dot_matrix(optimizer, gradients, deltas)
        self.assertEqual(matrix["tail64"]["tail64"], -.25)
        self.assertEqual(matrix["tail64"]["full1024"], 2.375)
        self.assertEqual(matrix["full1024"]["tail64"], -2.0)
        self.assertEqual(matrix["full1024"]["full1024"], 2.75)

    def test_trial_matrix_executes_both_horizons_populations_and_scales(self):
        class Graph(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.log_coupling_gain = torch.nn.Parameter(torch.tensor(0.0))

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.graph_generator = Graph()

        class Wrapped(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.tensor(.2, dtype=torch.float64))
                self.a_d = torch.nn.Parameter(torch.full((4,), .2, dtype=torch.float64))
                self.a_m = torch.nn.Parameter(torch.full((4,), .2, dtype=torch.float64))
                self.b = torch.nn.Parameter(torch.zeros((1, 4, 1, 1), dtype=torch.float64))
                self.core = Core()

            def project_integrations_(self):
                with torch.no_grad():
                    self.a_d.clamp_(0, 1)
                    self.a_m.clamp_(0, 1)

        wrapped = Wrapped()
        found = types.SimpleNamespace(wrapped=wrapped, encoder=torch.nn.Linear(1, 1))
        group = {"name": "native_core", "parameter_names": ["w"], "params": [wrapped.w], "lr": .01}

        class FakeTrain:
            @staticmethod
            def _make_optimizer(_found):
                return torch.optim.Adam([group], lr=.01), {}

        arrays = {
            "rgb": np.zeros((4, 2, 2, 3), dtype=np.uint8),
            "feasible_targets": np.asarray([
                [[1, 1], [2, 2]], [[1, 2], [1, 2]],
                [[1, 1], [2, 2]], [[1, 2], [1, 2]]], dtype=np.int32),
            "source_support": np.ones((4, 2, 2), dtype=np.int32),
            "image_ids": np.arange(4, dtype=np.int64),
        }
        targets, supports = arrays["feasible_targets"], arrays["source_support"]
        base = torch.tensor([[.8, .6, .3, .2], [.6, .8, .4, .3],
                             [.3, .4, .8, .7], [.2, .3, .7, .8]], dtype=torch.float64)
        slope = torch.tensor([[0., -.1, .02, .01], [-.1, 0., .03, .02],
                              [.02, .03, 0., -.08], [.01, .02, -.08, 0.]], dtype=torch.float64)

        def forward(_train, _found, _arrays, position, _device, _horizon):
            q = (base + wrapped.w * slope).unsqueeze(0)
            trace = {
                "component_gates": torch.ones((1, 4, 2, 4)),
                "component_spikes": torch.zeros((1, 4, 2, 4)),
                "component_membrane": torch.full((1, 4, 2, 4), .5),
            }
            return wrapped.w.reshape(1, 1), trace, torch.eye(4)[None], torch.zeros((1, 4, 8)), q

        plans, source_events = {}, {}
        source_baseline = {"cohort": [], "transfer": []}
        for position in range(4):
            q0 = base + wrapped.w.detach() * slope
            loss = objective.constrained_maximin_loss(q0, targets[position], supports[position])
            fixed = audit.fixed_route_loss(q0, objective.forest_routes(
                q0, targets[position], supports[position] > 0))
            plans[position] = objective.forest_routes(q0, targets[position], supports[position] > 0)
            source_baseline["cohort" if position < 2 else "transfer"].append({
                "native_loss": float(loss), "fixed_route_loss": float(fixed)})
            source_events[position] = {"trace": forward(None, None, None, position, None, None)[1],
                "threshold": torch.full((1, 4, 1, 1), .4)}

        rng = {position: gf._rng_snapshot() for position in range(4)}
        gradients = {}
        deltas = {}
        for horizon, value in (("tail64", 1.), ("full1024", 2.)):
            gradients[horizon] = {"by_name": {"native_core::w": torch.tensor(.3)},
                "by_id": {id(wrapped.w): torch.tensor(.3)}}
            deltas[horizon] = {"native_core::w": torch.tensor(-.01 * value, dtype=torch.float64)}
        with mock.patch.object(audit, "_forward", side_effect=forward), \
             mock.patch.object(audit, "_readouts", return_value=({}, {}, {}, np.zeros((2, 2), dtype=np.int32))), \
             mock.patch.object(audit, "clamp_blocking_diagnostics", return_value={"blocked_edge_fraction": 0.0}):
            result = audit._evaluate_trial_matrix(
                train=FakeTrain, found=found, arrays=arrays, device=torch.device("cpu"),
                deltas=deltas, gradients=gradients, image_rng=rng, source_plans=plans,
                source_events=source_events, gain=1.0, objective=objective, solver=None,
                contract=None, sw136=None, source_baseline=source_baseline,
                horizons={"tail64": 64, "full1024": 1024}, scales=(1/16, 1.0),
                populations={"cohort": (0, 1), "transfer": (2, 3)})
        self.assertEqual(set(result), {"tail64", "full1024"})
        self.assertEqual(sum(len(rows) for horizon in result.values()
                             for rows in horizon.values()), 8)
        self.assertTrue(all(len(result[h][p]) == 2 for h in result for p in result[h]))
        self.assertTrue(all(len(row["images"]) == 2 for h in result for p in result[h]
                            for row in result[h][p]))
        self.assertTrue(all("predicted_directional_change" in row
                            for h in result for p in result[h] for row in result[h][p]))


if __name__ == "__main__":
    unittest.main()
