from __future__ import annotations

import importlib
import sys
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import torch

from collaborative_test.SW_0172_graph_drive_separation import train_prefix as train


class PrefixTrainerTests(unittest.TestCase):
    def test_registered_matched_prefix_is_two_pass_b16(self):
        batches = train._expected_batches()
        self.assertEqual(len(batches), train.UPDATES)
        self.assertTrue(all(len(row) == 16 for row in batches))
        self.assertEqual(set(i for batch in batches[:4] for i in batch), set(range(64)))
        self.assertEqual(set(i for batch in batches[4:] for i in batch), set(range(64)))
        self.assertEqual(train._expected_batches(), batches)

    def test_both_routing_arms_have_identical_q_plus_instance_objective(self):
        qloss = torch.tensor(0.4, requires_grad=True)
        iloss = torch.tensor(0.7, requires_grad=True)
        expected = qloss + train.preflight.REGISTERED_LAMBDA * iloss
        for arm in train.ARMS:
            got = train._compose_arm_loss(arm, qloss, iloss, train.preflight.REGISTERED_LAMBDA)
            self.assertTrue(torch.equal(got, expected))
        with self.assertRaises(ValueError):
            train._compose_arm_loss("shared_gamma", qloss, None, 0.3)
        expected.backward()
        self.assertGreater(float(qloss.grad), 0)
        self.assertGreater(float(iloss.grad), 0)

    def test_native_forward_routes_graph_context_and_drive_by_arm(self):
        dual = importlib.import_module("collaborative_test.SW_0172_graph_drive_separation.dual_gamma_rollout")
        base = torch.full((1, 8, 1024), 1.0)
        context = torch.full((1, 8, 1024), 2.0)
        trace = {name: torch.ones((1, 2, 4)) for name in train.preflight.TRACE_FIELDS}
        trace["spikes"] = torch.ones((1, 1024, 2))
        trace["component_spikes"] = torch.ones((1, 4, 1024, 2))
        z = torch.nn.functional.normalize(torch.ones((1, 1024, 8)), dim=-1)
        adjacency = torch.eye(1024).unsqueeze(0)
        routed = []

        def rollout(wrapped, gamma_graph, gamma_drive, **kwargs):
            routed.append((gamma_graph, gamma_drive, kwargs))
            return {"trace": trace, "adjacency": adjacency, "z": z}

        fake_pair = {"gamma_base": base, "gamma_context": context}
        classifier = importlib.import_module("snn_kuramoto_bidirectional.spike_classifier")
        with mock.patch.object(dual, "encode_gamma_pair", return_value=fake_pair), \
             mock.patch.object(dual, "run_dual_gamma_rollout", side_effect=rollout), \
             mock.patch.object(classifier, "spike_synchrony_affinity", return_value=torch.eye(1024).unsqueeze(0)):
            train._native_forward(SimpleNamespace(wrapped=object()), torch.zeros((1, 128, 128, 3), dtype=torch.uint8),
                                  arm="shared_gamma", live_tail_steps=64)
            train._native_forward(SimpleNamespace(wrapped=object()), torch.zeros((1, 128, 128, 3), dtype=torch.uint8),
                                  arm="separated_gamma", live_tail_steps=64)
        self.assertIs(routed[0][0], context)
        self.assertIs(routed[0][1], context)
        self.assertIs(routed[1][0], context)
        self.assertIs(routed[1][1], base)
        self.assertEqual(routed[0][2]["total_steps"], 1024)
        self.assertEqual(routed[1][2]["live_tail_steps"], 64)

    def test_micro4_accumulation_matches_logical_b16_mean_gradient(self):
        x = torch.tensor(.37, requires_grad=True)
        data = torch.linspace(-1.0, 1.0, 16)
        target = torch.linspace(.2, -.3, 16)
        expected = torch.autograd.grad((x * data - target).square().mean(), x)[0]
        y = torch.tensor(.37, requires_grad=True)
        for start in range(0, 16, 4):
            train._backward_microbatch_loss((y * data[start:start+4] - target[start:start+4]).square(), 16)
        self.assertTrue(torch.allclose(y.grad, expected, rtol=0, atol=1e-7))

    def test_first_two_update_gate_requires_q_route_credit_not_auxiliary_only_credit(self):
        context_rows = [
            {"name": "context.8.weight", "gradient_l2": 1.0, "parameter_delta_l2": 1.0},
            {"name": "context.0.weight", "gradient_l2": 1.0, "parameter_delta_l2": 1.0},
        ]
        row = {"update": 1, "q_to_component_spikes_gradient_l2": 0.2,
            "q_to_context_gamma_gradient_l2": 0.3,
            "family_gradient_norms": {"graph": 0.1, "native_core": 0.1},
            "family_parameter_delta_l2": {"graph": 0.1, "native_core": 0.1},
            "named_parameter_credit": {"context_encoder": context_rows}}
        self.assertEqual(train._credit_rejection(row), [])
        broken = dict(row, q_to_context_gamma_gradient_l2=0.0)
        failures = train._credit_rejection(broken)
        self.assertTrue(any("Q-to-context-gamma" in reason for reason in failures))

    def test_trainer_fingerprint_binds_dual_routing_leaf_and_registered_cache_dependencies(self):
        fp = train._training_fingerprint()
        self.assertEqual(fp["experiment"], train.EXPERIMENT)
        self.assertIn("collaborative_test/SW_0172_graph_drive_separation/dual_gamma_rollout.py", fp["files"])
        self.assertIn("collaborative_test/SW_0172_graph_drive_separation/admission_queue.py", fp["files"])
        self.assertIn("collaborative_test/SW_0171_contextual_instance_graph/instance_objective.py", fp["files"])
        self.assertTrue(all(len(v) == 64 for v in fp["files"].values()))


if __name__ == "__main__":
    unittest.main()
