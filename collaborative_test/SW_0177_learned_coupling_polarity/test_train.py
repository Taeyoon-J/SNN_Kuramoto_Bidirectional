"""CPU contract checks for the SW0177 matched paired trainer."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
from unittest import mock

import numpy as np  # Must initialize before torch.
import torch
from torch import nn

from collaborative_test.SW_0177_learned_coupling_polarity import train
from collaborative_test.SW_0177_learned_coupling_polarity.polarity_operator import seeded_edge_multiplier


class _Graph(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2)
        self.geodesic_contrast = nn.Parameter(torch.ones(()))
        self.log_coupling_gain = nn.Parameter(torch.zeros(()))


class _Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_generator = _Graph()
        self.native = nn.Linear(2, 2)


class _Wrapped(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = _Core()
        self.a_d = nn.Parameter(torch.ones(()))
        self.a_m = nn.Parameter(torch.ones(()))
        self.b = nn.Parameter(torch.ones(()))

    def project_integrations_(self):
        with torch.no_grad():
            self.a_d.clamp_(0, 1); self.a_m.clamp_(0, 1)


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_encoder = nn.Linear(2, 2)
        self.context = nn.Linear(2, 2)


def _fake_found():
    return SimpleNamespace(wrapped=_Wrapped(), encoder=_Encoder())


class TrainerTests(unittest.TestCase):
    def test_registered_two_pass_order_is_64_complete_b16_updates(self):
        orders = [np.random.RandomState(171001).permutation(512).tolist()]
        rng = np.random.RandomState(171001)
        expected = [rng.permutation(512).astype(int).tolist() for _ in range(2)]
        batches = train._expected_batches()
        self.assertEqual(len(batches), 64)
        self.assertTrue(all(len(row) == 16 for row in batches))
        self.assertEqual(batches[:32], [expected[0][i:i + 16] for i in range(0, 512, 16)])
        self.assertEqual(batches[32:], [expected[1][i:i + 16] for i in range(0, 512, 16)])
        self.assertEqual(orders[0], expected[0])

    def test_head_initialization_matches_arms_and_restores_cpu_rng(self):
        torch.manual_seed(93)
        before = torch.random.get_rng_state().clone()
        signed = seeded_edge_multiplier(seed=train.SEED, mode="signed")
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))
        nonnegative = seeded_edge_multiplier(seed=train.SEED, mode="nonnegative")
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))
        self.assertEqual(set(signed.state_dict()), set(nonnegative.state_dict()))
        for name in signed.state_dict():
            self.assertTrue(torch.equal(signed.state_dict()[name], nonnegative.state_dict()[name]))
        z = torch.zeros((1, 4, 16))
        support = torch.tensor([[[0., 1., 0., 0.], [1., 0., 1., 0.],
                                 [0., 1., 0., 0.], [0., 0., 0., 0.]]])
        self.assertTrue(torch.equal(signed(z, support), torch.ones_like(support)))
        self.assertTrue(torch.equal(nonnegative(z, support), torch.ones_like(support)))

    def test_optimizer_has_five_disjoint_named_families_including_head(self):
        found = _fake_found()
        head = seeded_edge_multiplier(seed=train.SEED, mode="signed")
        optimizer, families = train._make_optimizer(found, head)
        self.assertEqual([g["name"] for g in optimizer.param_groups],
            ["context_encoder", "graph", "native_core", "integration", "polarity_head"])
        ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(families), {"context_encoder", "graph", "native_core", "integration", "polarity_head"})
        self.assertFalse(any(p.requires_grad for p in found.encoder.base_encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in head.parameters()))
        self.assertEqual(optimizer.param_groups[-1]["lr"], 1e-3)
        contract = train._optimizer_contract(optimizer)
        self.assertEqual([row["name"] for row in contract],
            ["context_encoder", "graph", "native_core", "integration", "polarity_head"])
        self.assertTrue(all(row["parameter_names"] for row in contract))
        self.assertTrue(all(row["betas"] == [0.9, 0.999] and row["eps"] > 0 for row in contract))
        self.assertEqual(contract[-1]["lr"], 1e-3)
        json.dumps({"groups": contract, "defaults": optimizer.defaults}, allow_nan=False)
        other_head = seeded_edge_multiplier(seed=train.SEED, mode="nonnegative")
        other_optimizer, _ = train._make_optimizer(_fake_found(), other_head)
        self.assertEqual(contract, train._optimizer_contract(other_optimizer))

    def test_training_factory_checks_cpu_then_rebuilds_exact_gpu_source(self):
        calls = []
        cpu_base, gpu_base = object(), object()
        cpu_found, gpu_found = SimpleNamespace(), SimpleNamespace()
        admitted = {"prepared": object(), "v2_receipt": {"report": {
            "fresh_initial_model_state_sha256": "state"}}}
        module_api = SimpleNamespace(_module_sha=lambda module: "base" if module in (cpu_base, gpu_base) else "bad")
        source_api = SimpleNamespace(_source_sha=lambda found, base: "source")
        def cpu_build(prepared, device):
            calls.append(("cpu", device))
            self.assertIs(prepared, admitted["prepared"])
            return cpu_found, cpu_base, "source", "state"
        def gpu_build(prepared, device):
            calls.append(("gpu", str(device)))
            self.assertIs(prepared, admitted["prepared"])
            return gpu_found, gpu_base, "source", "state"
        with mock.patch.object(train.torch.cuda, "is_initialized", return_value=False), \
             mock.patch.object(train.torch.cuda, "set_device", side_effect=lambda device: calls.append(("set", str(device)))), \
             mock.patch.object(train.preflight, "build_source", side_effect=cpu_build), \
             mock.patch.object(train.preflight, "_build_source", side_effect=gpu_build), \
             mock.patch.object(train.preflight, "_module", return_value=module_api), \
             mock.patch.object(train, "_legacy", return_value=source_api):
            found, base, source_sha, state_sha = train._build_training_source(admitted, "cuda:0")
        self.assertIs(found, gpu_found)
        self.assertIs(base, gpu_base)
        self.assertEqual((source_sha, state_sha), ("source", "state"))
        self.assertEqual(calls, [("cpu", "cpu"), ("set", "cuda:0"), ("gpu", "cuda:0")])

    def test_boundary_zero_roundtrip_restores_registered_head_and_empty_adam(self):
        from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint
        found = _fake_found()
        head = seeded_edge_multiplier(seed=train.SEED, mode="signed")
        optimizer, _ = train._make_optimizer(found, head)
        bindings = {"experiment": train.EXPERIMENT, "arm": "signed",
                    "head_sha256": train._legacy()._state_sha(nn.Identity(), head)}
        runtime = checkpoint.runtime_signature("cpu")
        central = train.ROOT / "collaborative_testing" / "temporary"
        central.mkdir(parents=True, exist_ok=True)
        root = (central / f"sw177_boundary_{os.getpid()}_{uuid.uuid4().hex}").resolve()
        root.mkdir()
        try:
            original_head = {k: v.detach().clone() for k, v in head.state_dict().items()}
            original_context = {k: v.detach().clone() for k, v in found.encoder.state_dict().items()}
            checkpoint.save_boundary(root, completed_updates=0, wrapped=found.wrapped,
                encoder=found.encoder, optimizer=optimizer, history=[], bindings=bindings,
                runtime=runtime, device="cpu")
            optimizer_payload = torch.load(root / "boundary.ptx", map_location="cpu", weights_only=False)
            raw_payload = torch.load(__import__("io").BytesIO(optimizer_payload["payload_bytes"]),
                                     map_location="cpu", weights_only=False)
            train._legacy()._validate_optimizer_state(raw_payload["optimizer_state"],
                train._legacy()._optimizer_summary(optimizer))
            with torch.no_grad():
                for parameter in head.parameters(): parameter.add_(0.25)
                found.encoder.context.weight.add_(0.5)
            restored = checkpoint.load_latest(root, wrapped=found.wrapped, encoder=found.encoder,
                optimizer=optimizer, expected_bindings=bindings, expected_runtime=runtime,
                device="cpu", max_updates=train.UPDATES)
            self.assertEqual(restored["completed_updates"], 0)
            self.assertEqual(optimizer.state, {})
            for key, value in original_head.items():
                self.assertTrue(torch.equal(head.state_dict()[key], value))
            for key, value in original_context.items():
                self.assertTrue(torch.equal(found.encoder.state_dict()[key], value))
        finally:
            if root.resolve().parent != central.resolve():
                raise AssertionError("refusing to clean checkpoint fixture outside central temporary")
            for path in root.iterdir(): path.unlink()
            root.rmdir()

    def test_checkpoint_validator_rehashes_actual_head_tensor_bytes(self):
        found = _fake_found()
        head = seeded_edge_multiplier(seed=train.SEED, mode="signed")
        found.encoder.add_module("polarity_head", head)
        base = train._legacy()
        payload = {"wrapped": {k: v.detach().clone() for k, v in found.wrapped.state_dict().items()},
                   "encoder": {k: v.detach().clone() for k, v in found.encoder.state_dict().items()}}
        declared = base._state_sha_from_state(payload)
        payload["model_state_sha256"] = declared
        train._validate_checkpoint_model_state(payload, declared)
        key = next(key for key in payload["encoder"] if "polarity_head" in key and key.endswith("weight"))
        payload["encoder"][key].view(-1)[0] += 1
        with self.assertRaisesRegex(ValueError, "tensor bytes"):
            train._validate_checkpoint_model_state(payload, declared)

    def test_v1_failure_cannot_substitute_for_missing_v2_success(self):
        central = train.ROOT / "collaborative_testing" / "temporary"
        central.mkdir(parents=True, exist_ok=True)
        root = central / f"sw177_admission_{os.getpid()}_{uuid.uuid4().hex}"
        root.mkdir()
        try:
            state_path, report_path = root / "state.json", root / "preflight.json"
            report_bytes = b'{"status":"scientific_rejection"}\n'
            report_path.write_bytes(report_bytes)
            state_path.write_text(json.dumps({"status": "failed",
                "report_file_sha256": __import__("hashlib").sha256(report_bytes).hexdigest(),
                "queue_fingerprint_sha256": "a" * 64}), encoding="utf-8")
            fake_queue = SimpleNamespace(STATE=state_path, REPORT=report_path,
                                         fingerprint_sha256=lambda: "a" * 64)
            with mock.patch.object(train, "_module", return_value=fake_queue):
                with self.assertRaisesRegex(ValueError, "source_preflight_v2"):
                    train._preflight_receipt()
        finally:
            if root.resolve().parent != central.resolve():
                raise AssertionError("refusing to clean trainer fixture outside central temporary")
            for path in root.iterdir(): path.unlink()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
