from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import unittest
import uuid
from types import SimpleNamespace

import numpy
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from collaborative_test.SW_0171_contextual_instance_graph import train_prefix as train


class PrefixTrainerTests(unittest.TestCase):
    def test_registered_prefix_exposes_first64_twice_in_eight_b16_updates(self):
        batches = train._expected_batches()
        self.assertEqual(len(batches), 8)
        self.assertTrue(all(len(batch) == 16 for batch in batches))
        pass1 = [i for batch in batches[:4] for i in batch]
        pass2 = [i for batch in batches[4:] for i in batch]
        self.assertEqual(set(pass1), set(range(64)))
        self.assertEqual(set(pass2), set(range(64)))
        self.assertEqual(pass1, [i for i in train._orders()[0]])
        self.assertEqual(pass2, [i for i in train._orders()[1]])
        self.assertEqual(train._expected_batches(), batches)

    def test_resume_cursor_rejects_skipped_or_reordered_updates(self):
        batches = train._expected_batches()
        history = [{"update": i + 1, "image_indices": batches[i]} for i in range(4)]
        train._validate_cursor(4, history)
        bad = [dict(row) for row in history]
        bad[2]["image_indices"] = list(reversed(bad[2]["image_indices"]))
        with self.assertRaises(ValueError):
            train._validate_cursor(4, bad)
        with self.assertRaises(ValueError):
            train._validate_cursor(3, history)

    def test_q_only_and_instance_arm_use_registered_loss_composition(self):
        objective = importlib.import_module(
            "collaborative_test.SW_0168_end_to_end_connectivity.connectivity_objective")
        q = torch.tensor([[.9, .8, .2, .1], [.8, .9, .2, .1],
                          [.2, .2, .9, .7], [.1, .1, .7, .9]], requires_grad=True)
        support = numpy.ones((2, 2), dtype=numpy.bool_)
        target = numpy.asarray([[1, 1], [2, 2]], dtype=numpy.int32)
        q_loss, instance_loss = train._loss_for_arm(
            "Q_plus_instance", q, torch.nn.functional.normalize(torch.eye(4), dim=-1),
            target, support, numpy.asarray([[0, 0], [1, 1]], dtype=numpy.int32), objective)
        self.assertIsNotNone(instance_loss)
        self.assertTrue(torch.isfinite(q_loss) and torch.isfinite(instance_loss))
        combined = train._compose_arm_loss("Q_plus_instance", q_loss, instance_loss, .75)
        self.assertTrue(torch.allclose(combined, q_loss + .75 * instance_loss))
        q_only, no_instance = train._loss_for_arm(
            "Q_only", q, torch.nn.functional.normalize(torch.eye(4), dim=-1),
            target, support, numpy.asarray([[0, 0], [1, 1]], dtype=numpy.int32), objective)
        self.assertIsNone(no_instance)
        self.assertTrue(torch.allclose(train._compose_arm_loss("Q_only", q_only, None, .75), q_only))
        combined.backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_micro4_accumulation_matches_logical_b16_mean_gradient(self):
        x = torch.tensor(.37, requires_grad=True)
        data = torch.linspace(-1.0, 1.0, 16)
        target = torch.linspace(.2, -.3, 16)
        full_loss = (x * data - target).square().mean()
        expected = torch.autograd.grad(full_loss, x)[0]

        y = torch.tensor(.37, requires_grad=True)
        for start in range(0, 16, 4):
            row = (y * data[start:start+4] - target[start:start+4]).square()
            train._backward_microbatch_loss(row, 16)
        self.assertTrue(torch.allclose(y.grad, expected, rtol=0, atol=1e-7))

    def test_named_families_enable_frozen_native_graph_and_core_but_keep_source_frozen(self):
        class Graph(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.geodesic_contrast = torch.nn.Parameter(torch.tensor(1.0))
                self.weight = torch.nn.Parameter(torch.tensor(2.0))

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.graph_generator = Graph()
                self.kuramoto_weight = torch.nn.Parameter(torch.tensor(3.0))

        class Wrapped(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.core = Core()
                self.a_d = torch.nn.Parameter(torch.tensor(1.0))
                self.a_m = torch.nn.Parameter(torch.tensor(1.0))
                self.b = torch.nn.Parameter(torch.tensor(1.0))

        base = torch.nn.Linear(2, 2)
        context = torch.nn.Linear(2, 2)
        encoder = torch.nn.Module()
        encoder.base_encoder = base
        encoder.context = context
        encoder.add_module("base_encoder", base)
        encoder.add_module("context", context)
        wrapped = Wrapped()
        for parameter in list(encoder.parameters()) + list(wrapped.parameters()):
            parameter.requires_grad_(False)
        families = train._named_families(SimpleNamespace(encoder=encoder, wrapped=wrapped))
        self.assertEqual(set(families), {"context_encoder", "graph", "native_core", "integration"})
        self.assertTrue(all(p.requires_grad for _, p in families["context_encoder"]))
        self.assertTrue(all(p.requires_grad for _, p in families["graph"]))
        self.assertTrue(all(p.requires_grad for _, p in families["native_core"]))
        self.assertTrue(all(p.requires_grad for _, p in families["integration"]))
        self.assertFalse(base.weight.requires_grad)
        self.assertFalse(wrapped.core.graph_generator.geodesic_contrast.requires_grad)

    def test_first_two_update_credit_gate_requires_q_spike_graph_and_context_evolution(self):
        def row(update, hidden=False):
            context = [
                {"name": "context.8.weight", "gradient_l2": 1.0, "parameter_delta_l2": 1.0},
                {"name": "context.0.weight", "gradient_l2": float(hidden), "parameter_delta_l2": float(hidden)},
            ]
            return {"update": update, "q_to_component_spikes_gradient_l2": 1.0,
                "family_gradient_norms": {"graph": 1.0, "native_core": 1.0},
                "family_parameter_delta_l2": {"graph": 1.0, "native_core": 1.0},
                "named_parameter_credit": {"context_encoder": context}}
        self.assertEqual(train._credit_rejection(row(1)), [])
        self.assertEqual(train._credit_rejection(row(2, hidden=True)), [])
        failures = train._credit_rejection(row(2, hidden=False))
        self.assertTrue(any("hidden layers" in item for item in failures))
        dead_q = row(1); dead_q["q_to_component_spikes_gradient_l2"] = 0.0
        self.assertTrue(any("Q-to-component-spikes" in item for item in train._credit_rejection(dead_q)))

    def test_admission_source_digest_uses_actual_source_cache_field(self):
        proof = {"source_cache_state_sha256": "a" * 64,
                 "ordered_train_ids": list(range(512))}
        self.assertEqual(train._validate_source_proof_digest(proof, "a" * 64), "a" * 64)
        with self.assertRaises(ValueError):
            train._validate_source_proof_digest({"source_state_sha256": "a" * 64}, "a" * 64)
        self.assertEqual(len(proof["ordered_train_ids"]), 512)

    def test_scientific_rejection_is_terminal_for_exact_boundary_on_resume(self):
        fixture_root = ROOT / "collaborative_testing" / "temporary"
        fixture_root.mkdir(parents=True, exist_ok=True)
        own_root = fixture_root / f"sw171_reject_{os.getpid()}_{uuid.uuid4().hex}"
        own_root.mkdir()
        try:
            bindings = {"source_proof": {"source_cache_state_sha256": "a" * 64},
                        "training_fingerprint": {"files": {"x": "b" * 64}}}
            history = [{"update": 1, "mean_loss": 1.0}]
            boundary = own_root / "boundary.ptx"
            boundary.write_bytes(b"committed boundary")
            rejection = {"experiment": train.EXPERIMENT, "status": "scientific_rejection",
                "arm": "Q_only", "completed_updates": 1,
                "history_sha256": train._history_sha(history),
                "bindings_sha256": train._binding_sha(bindings),
                "boundary_sha256": train._sha_file(boundary),
                "source_proof": bindings["source_proof"],
                "training_fingerprint": bindings["training_fingerprint"],
                "validation_data_read": False, "rejection_reasons": ["dead credit"]}
            (own_root / "credit_rejection.json").write_text(json.dumps(rejection), encoding="utf-8")
            accepted = train._read_credit_rejection(own_root, arm="Q_only", bindings=bindings,
                                                    completed=1, history=history)
            self.assertEqual(accepted, rejection)
            with self.assertRaises(ValueError):
                train._read_credit_rejection(own_root, arm="Q_only", bindings=bindings,
                                             completed=0, history=[])
            # Simulate a crash after committing a failed update boundary but
            # before creating its immutable rejection sidecar.
            (own_root / "credit_rejection.json").unlink()
            boundary.write_bytes(b"failed update boundary")
            committed_history = [
                {"update": 1, "q_to_component_spikes_gradient_l2": 1.0,
                 "family_gradient_norms": {"graph": 1.0, "native_core": 1.0},
                 "family_parameter_delta_l2": {"graph": 1.0, "native_core": 1.0},
                 "named_parameter_credit": {"context_encoder": [
                    {"name": "context.8.weight", "gradient_l2": 1.0, "parameter_delta_l2": 1.0}]}},
                {"update": 2, "q_to_component_spikes_gradient_l2": 1.0,
                 "family_gradient_norms": {"graph": 1.0, "native_core": 1.0},
                 "family_parameter_delta_l2": {"graph": 1.0, "native_core": 1.0},
                 "named_parameter_credit": {"context_encoder": [
                    {"name": "context.0.weight", "gradient_l2": 0.0, "parameter_delta_l2": 0.0}]}}
            ]
            recovered = train._recover_terminal_rejection(own_root, arm="Q_only",
                bindings=bindings, completed=2, history=committed_history)
            self.assertEqual(recovered["completed_updates"], 2)
            self.assertTrue((own_root / "credit_rejection.json").is_file())
            # A subsequent invocation reads the terminal receipt; it cannot
            # continue the optimizer beyond the already committed boundary.
            self.assertEqual(train._recover_terminal_rejection(own_root,
                arm="Q_only", bindings=bindings, completed=2, history=committed_history), recovered)
        finally:
            if os.path.commonpath([str(fixture_root.resolve()), str(own_root.resolve())]) != str(fixture_root.resolve()):
                raise RuntimeError("refusing to remove fixture outside central temporary root")
            shutil.rmtree(own_root)

    def test_boundary_checkpoint_resume_matches_uninterrupted_optimizer(self):
        durable = importlib.import_module(
            "collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint")
        fixture_root = ROOT / "collaborative_testing" / "temporary"
        fixture_root.mkdir(parents=True, exist_ok=True)
        own_root = fixture_root / f"sw171_boundary_{os.getpid()}_{uuid.uuid4().hex}"
        own_root.mkdir()
        try:
            torch.manual_seed(811)
            initial_wrapped = torch.nn.Linear(3, 2)
            initial_encoder = torch.nn.Linear(2, 2)
            init_w = {k: v.detach().clone() for k, v in initial_wrapped.state_dict().items()}
            init_e = {k: v.detach().clone() for k, v in initial_encoder.state_dict().items()}
            wrapped_full, encoder_full = torch.nn.Linear(3, 2), torch.nn.Linear(2, 2)
            wrapped_full.load_state_dict(init_w); encoder_full.load_state_dict(init_e)
            wrapped_part, encoder_part = torch.nn.Linear(3, 2), torch.nn.Linear(2, 2)
            wrapped_part.load_state_dict(init_w); encoder_part.load_state_dict(init_e)
            params_full = list(wrapped_full.parameters()) + list(encoder_full.parameters())
            params_part = list(wrapped_part.parameters()) + list(encoder_part.parameters())
            opt_full = torch.optim.Adam(params_full, lr=3e-4)
            opt_part = torch.optim.Adam(params_part, lr=3e-4)
            bindings = {"arm": "Q_only", "cursor": 8, "ids_sha256": "c" * 64}
            runtime = {"device_type": "cpu", "torch_version": torch.__version__}
            rows_full = []
            xs = [torch.tensor([[.1, .2, .3], [.4, -.2, .8]]) for _ in range(3)]
            ys = [torch.tensor([[.3, -.1], [.2, .7]]) for _ in range(3)]

            def step(wrapped, encoder, optimizer, x, y):
                optimizer.zero_grad(set_to_none=True)
                loss = (encoder(wrapped(x)) - y).square().mean()
                loss.backward(); optimizer.step()
                return {"update": len(rows_full) + 1, "loss": float(loss.detach())}

            for x, y in zip(xs, ys):
                rows_full.append(step(wrapped_full, encoder_full, opt_full, x, y))
            first = step(wrapped_part, encoder_part, opt_part, xs[0], ys[0])
            history = [first]
            durable.save_boundary(own_root, completed_updates=1,
                wrapped=wrapped_part, encoder=encoder_part, optimizer=opt_part,
                history=history, bindings=bindings, runtime=runtime, device="cpu")

            resumed_w, resumed_e = torch.nn.Linear(3, 2), torch.nn.Linear(2, 2)
            resumed_w.load_state_dict(init_w); resumed_e.load_state_dict(init_e)
            resumed_opt = torch.optim.Adam(list(resumed_w.parameters()) + list(resumed_e.parameters()), lr=3e-4)
            loaded = durable.load_latest(own_root, wrapped=resumed_w, encoder=resumed_e,
                optimizer=resumed_opt, expected_bindings=bindings, expected_runtime=runtime,
                device="cpu", max_updates=3)
            self.assertEqual(loaded["completed_updates"], 1)
            history = loaded["history"]
            for x, y in zip(xs[1:], ys[1:]):
                loss = (resumed_e(resumed_w(x)) - y).square().mean()
                resumed_opt.zero_grad(set_to_none=True); loss.backward(); resumed_opt.step()
                history.append({"update": len(history) + 1, "loss": float(loss.detach())})
                durable.save_boundary(own_root, completed_updates=len(history),
                    wrapped=resumed_w, encoder=resumed_e, optimizer=resumed_opt,
                    history=history, bindings=bindings, runtime=runtime, device="cpu")
            for a, b in zip(wrapped_full.parameters(), resumed_w.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for a, b in zip(encoder_full.parameters(), resumed_e.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            state_a, state_b = opt_full.state_dict()["state"], resumed_opt.state_dict()["state"]
            self.assertEqual(state_a.keys(), state_b.keys())
            for key in state_a:
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(state_a[key][field], state_b[key][field], rtol=0, atol=0)
        finally:
            resolved_base = fixture_root.resolve()
            resolved_target = own_root.resolve()
            if os.path.commonpath([str(resolved_base), str(resolved_target)]) != str(resolved_base):
                raise RuntimeError("refusing to remove fixture outside central temporary root")
            shutil.rmtree(own_root)


if __name__ == "__main__":
    unittest.main()
