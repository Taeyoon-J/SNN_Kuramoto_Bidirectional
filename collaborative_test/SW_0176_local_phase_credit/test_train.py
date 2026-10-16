"""CPU contracts for SW0176's matched fresh-source trainer."""
from __future__ import annotations

import tempfile
import os
import json
import io
from pathlib import Path
import unittest
from unittest import mock

import numpy as np  # Keep NumPy/OpenMP initialization before torch.
import torch

from collaborative_test.SW_0176_local_phase_credit import train
from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint


ROOT = Path(__file__).resolve().parents[2]
TEMP_ROOT = Path(os.environ.get("TEMP", ROOT / "collaborative_testing" / "temporary"))


def _toy_modules_and_optimizer(lr=3e-4):
    wrapped = torch.nn.Linear(2, 1)
    encoder = torch.nn.Linear(1, 1)
    groups = [
        {"params": [wrapped.weight, wrapped.bias], "parameter_names": ["weight", "bias"],
         "name": "context_encoder", "lr": lr},
    ]
    return wrapped, encoder, torch.optim.Adam(groups)


class TrainerContractTests(unittest.TestCase):
    def test_cpu_factory_uses_fresh_builder_and_uninitialized_adam(self):
        wrapped, encoder, optimizer = _toy_modules_and_optimizer()
        base_encoder = torch.nn.Linear(2, 2)
        found = type("Found", (), {})()
        found.wrapped, found.encoder = wrapped, encoder
        source_sha, base_sha = "1" * 64, "2" * 64
        initial_sha = train.base._state_sha(wrapped, encoder)
        families = {"context_encoder": [("weight", wrapped.weight), ("bias", wrapped.bias)]}
        admitted = {"prepared": {}, "source_proof": {"source_cache_state_sha256": source_sha},
            "preflight_report_sha256": "3" * 64,
            "preflight_report": {"status": "complete_source_preflight"},
            "control_receipt": {"manifest_sha256": "4" * 64}}
        bindings = {"sentinel": "bindings"}
        with mock.patch.object(train.torch.cuda, "is_initialized", return_value=False), \
             mock.patch.object(train.preflight_v4, "_build_source", return_value=(
                found, base_encoder, source_sha, base_sha, initial_sha, families)) as builder, \
             mock.patch.object(train.base, "_make_optimizer", return_value=(optimizer, families)), \
             mock.patch.object(train, "_control_bindings") as control_bindings:
            receipt = train.prepare_fresh_training_state_cpu(admitted)
        builder.assert_called_once_with({}, "cpu")
        control_bindings.assert_called_once_with(admitted, initial_sha, source_sha, base_sha,
                                                  families, optimizer)
        self.assertEqual(receipt["status"], "complete_fresh_cpu_factory")
        self.assertEqual(receipt["initial_model_state_sha256"], initial_sha)
        self.assertEqual(receipt["optimizer_state_entries"], 0)
        self.assertFalse(receipt["cuda_initialized"])
        self.assertEqual(admitted["fresh_cpu_factory_receipt"], receipt)
        self.assertEqual(receipt["receipt_sha256"], train._json_sha(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}))

    def test_schedule_is_exact_registered_64_update_numpy_schedule(self):
        batches = train._expected_batches()
        self.assertEqual(len(batches), 64)
        self.assertTrue(all(len(batch) == 16 for batch in batches))
        self.assertEqual(batches, train.base._expected_batches())
        history = [{"update": i + 1, "image_indices": list(batch)}
                   for i, batch in enumerate(batches[:2])]
        train._validate_cursor(2, history)
        corrupted = [dict(history[0]), dict(history[1])]
        corrupted[1]["image_indices"] = list(reversed(corrupted[1]["image_indices"]))
        with self.assertRaisesRegex(ValueError, "exposure order"):
            train._validate_cursor(2, corrupted)

    def test_control_projection_binds_fresh_optimizer_group_order_and_defaults(self):
        wrapped, encoder, optimizer = _toy_modules_and_optimizer()
        contract = train._optimizer_group_contract(optimizer)
        initial = "a" * 64
        source = "b" * 64
        base_sha = "c" * 64
        control = {"bindings": {"initial_model_state_sha256": initial,
            "source_state_sha256": source, "base_encoder_sha256": base_sha}}
        admitted = {"prepared": {"control_manifest": control},
                    "control_receipt": {"optimizer_group_contract": contract}}
        families = {"context_encoder": [("weight", wrapped.weight), ("bias", wrapped.bias)]}
        train._control_bindings(admitted, initial, source, base_sha, families, optimizer)

        _, _, changed_lr = _toy_modules_and_optimizer(lr=1e-3)
        with self.assertRaisesRegex(ValueError, "group names/order/defaults"):
            train._control_bindings(admitted, initial, source, base_sha,
                                    families, changed_lr)
        with self.assertRaisesRegex(ValueError, "initialization"):
            train._control_bindings(admitted, "d" * 64, source, base_sha,
                                    families, optimizer)

    def test_saved_torch_optimizer_contract_matches_fresh_json_contract(self):
        wrapped, encoder, optimizer = _toy_modules_and_optimizer()
        fresh = train._optimizer_group_contract(optimizer)
        buffer = io.BytesIO()
        torch.save(optimizer.state_dict(), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, map_location="cpu", weights_only=False)
        keys = ("name", "parameter_names", "lr", "betas", "eps", "weight_decay",
                "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused")
        saved_contract = [{key: train._json_safe_optimizer_value(group.get(key))
                           for key in keys} for group in saved["param_groups"]]
        self.assertEqual(saved_contract, fresh)
        # The JSON round trip is the representation persisted in the control receipt.
        self.assertEqual(json.loads(json.dumps(saved_contract)), fresh)

    def test_real_boundary_roundtrip_restores_model_optimizer_and_cursor(self):
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        wrapped, encoder, optimizer = _toy_modules_and_optimizer()
        value = wrapped(torch.ones(1, 2)).square().sum() + encoder(torch.ones(1, 1)).square().sum()
        value.backward(); optimizer.step()
        expected_wrapped = {key: tensor.detach().clone() for key, tensor in wrapped.state_dict().items()}
        expected_encoder = {key: tensor.detach().clone() for key, tensor in encoder.state_dict().items()}
        expected_opt = optimizer.state_dict()
        batch = train._expected_batches()[0]
        history = [{"update": 1, "image_indices": list(batch), "loss": 0.25}]
        bindings = {"experiment": train.EXPERIMENT, "arm": train.ARM,
                    "source_proof_sha256": "e" * 64}
        runtime = checkpoint.runtime_signature("cpu")
        with tempfile.TemporaryDirectory(prefix="sw176_boundary_", dir=TEMP_ROOT) as temp:
            # The runner-provided temporary root is already uniquely owned;
            # checkpoint directly there to avoid nested temp trees on Windows.
            work = Path(temp)
            checkpoint.save_boundary(work, completed_updates=1, wrapped=wrapped,
                encoder=encoder, optimizer=optimizer, history=history,
                bindings=bindings, runtime=runtime, device="cpu")
            self.assertEqual(Path(work).resolve(), Path(temp).resolve())
            with torch.no_grad():
                wrapped.weight.add_(17); encoder.weight.sub_(9)
            for state in optimizer.state.values():
                for key, item in state.items():
                    if torch.is_tensor(item) and item.is_floating_point():
                        item.add_(1)
            restored = checkpoint.load_latest(work, wrapped=wrapped, encoder=encoder,
                optimizer=optimizer, expected_bindings=bindings, expected_runtime=runtime,
                device="cpu", max_updates=train.UPDATES)
            self.assertEqual(restored["completed_updates"], 1)
            self.assertEqual(restored["history"], history)
            self.assertEqual(train._validate_cursor(1, restored["history"]), None)
            for key, tensor in expected_wrapped.items():
                self.assertTrue(torch.equal(wrapped.state_dict()[key], tensor))
            for key, tensor in expected_encoder.items():
                self.assertTrue(torch.equal(encoder.state_dict()[key], tensor))
            self.assertEqual(checkpoint.optimizer_schema(optimizer),
                             checkpoint.optimizer_schema(_optimizer_from_state(expected_opt,
                                 wrapped, encoder)))


def _optimizer_from_state(state, wrapped, encoder):
    groups = [{"params": [wrapped.weight, wrapped.bias],
               "parameter_names": ["weight", "bias"], "name": "context_encoder", "lr": 3e-4}]
    result = torch.optim.Adam(groups)
    result.load_state_dict(state)
    return result


if __name__ == "__main__":
    unittest.main()
