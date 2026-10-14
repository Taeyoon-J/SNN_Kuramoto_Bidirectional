"""CPU tests for SW0163 schedule, mask update, and durable resume primitives."""
from __future__ import annotations

import shutil
import json
import unittest
from pathlib import Path
import uuid
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint as durable
from collaborative_test.SW_0163_relation_context_assignment import model, train

torch.set_num_threads(1)


class TrainingTests(unittest.TestCase):
    def _mini_batch(self, seed=17):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        spikes = (torch.rand((1, 4, 1024, 512), generator=gen) > 0.82).float()
        target = torch.zeros((1, 1024), dtype=torch.int64)
        target[:, 500:] = 1
        q = torch.zeros((1, 1024, 1024), dtype=torch.float32)
        adjacency = q.clone()
        return [(spikes, target, q, adjacency)]

    def _step(self, binder, optimizer, batch, ix):
        return train._logical_update(binder=binder, optimizer=optimizer,
            microbatches=batch, update_index=ix, pass_index=1,
            cache_indices=[0], image_ids=[70000])

    def test_registered_schedule_is_fixed_and_each_pass_is_a_permutation(self):
        order = train.make_training_order()
        self.assertEqual(order.shape, (16, 512))
        expected = np.arange(512)
        for row in order:
            np.testing.assert_array_equal(np.sort(row), expected)
        np.testing.assert_array_equal(order, train.make_training_order())
        with self.assertRaises(ValueError):
            train._load_logical_batch(None, None, list(range(15)), "cpu")

    def test_all_arms_share_identical_initialization_and_exact_parameter_shapes(self):
        models = [model.RelationContextAssignment(arm) for arm in train.ARMS]
        states = [m.state_dict() for m in models]
        self.assertTrue(all(states[0].keys() == state.keys() for state in states[1:]))
        for state in states[1:]:
            for name in states[0]:
                torch.testing.assert_close(states[0][name], state[name], rtol=0, atol=0)
        self.assertEqual(len({train._model_state_sha(state) for state in states}), 1)

    def test_production_mask_loss_update_changes_model_with_finite_gradient(self):
        binder = model.RelationContextAssignment("Q_A")
        optimizer = torch.optim.Adam([{"name": "assignment",
            "params": list(binder.parameters()), "lr": train.LR}])
        before = {k: v.detach().clone() for k, v in binder.state_dict().items()}
        record = self._step(binder, optimizer, self._mini_batch(), 1)
        self.assertGreater(record["mean_loss"], 0)
        self.assertGreater(record["gradient_norm"], 0)
        self.assertLessEqual(record["row_sum_error"], 2e-5)
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in binder.state_dict().items()))

    def test_durable_boundary_resume_matches_uninterrupted_next_update(self):
        batch = self._mini_batch(seed=31)
        uninterrupted = model.RelationContextAssignment("Q_only")
        uninterrupted_opt = torch.optim.Adam([{"name": "assignment",
            "params": list(uninterrupted.parameters()), "lr": train.LR}])
        self._step(uninterrupted, uninterrupted_opt, batch, 1)

        td = Path(train.HERE) / f".train_resume_fixture_{uuid.uuid4().hex}"
        td.mkdir()
        try:
            boundary = td / "resume"
            sentinel = torch.nn.Identity()
            bindings = {"experiment": train.EXPERIMENT, "arm": "Q_only",
                        "boundary_test": "same-update-resume"}
            runtime = durable.runtime_signature("cpu")
            durable.save_boundary(boundary, completed_updates=1, wrapped=uninterrupted,
                encoder=sentinel, optimizer=uninterrupted_opt, history=[{"update": 1}],
                bindings=bindings, runtime=runtime, device="cpu")
            self._step(uninterrupted, uninterrupted_opt, batch, 2)
            expected = {k: v.detach().clone() for k, v in uninterrupted.state_dict().items()}

            resumed = model.RelationContextAssignment("Q_only")
            resumed_opt = torch.optim.Adam([{"name": "assignment",
                "params": list(resumed.parameters()), "lr": train.LR}])
            loaded = durable.load_latest(boundary, wrapped=resumed, encoder=torch.nn.Identity(),
                optimizer=resumed_opt, expected_bindings=bindings, expected_runtime=runtime,
                device="cpu", max_updates=train.UPDATES)
            self.assertEqual(loaded["completed_updates"], 1)
            self._step(resumed, resumed_opt, batch, 2)
            for name, value in resumed.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            state_a, state_b = uninterrupted_opt.state_dict()["state"], resumed_opt.state_dict()["state"]
            self.assertEqual(state_a.keys(), state_b.keys())
            for key in state_a:
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    if torch.is_tensor(state_a[key][field]):
                        torch.testing.assert_close(state_a[key][field], state_b[key][field], rtol=0, atol=0)
                    else:
                        self.assertEqual(state_a[key][field], state_b[key][field])
        finally:
            shutil.rmtree(td)

    def test_logical_loader_uses_verified_source_and_context_row_contracts(self):
        class SourceRows:
            image_ids = list(range(512))
            def load_row(self, ix):
                return (np.zeros((4, 1024, 512), dtype=np.float32) + ix,
                        np.full((1024,), ix % 2, dtype=np.int64))
        class ContextRows:
            def load(self, ix):
                q = np.zeros((1024, 1024), dtype=np.float32)
                a = np.zeros_like(q)
                return q + ix, a + ix
        micros, ids = train._load_logical_batch(SourceRows(), ContextRows(), list(range(16)), "cpu")
        self.assertEqual(ids, list(range(16)))
        self.assertEqual(len(micros), 4)
        for index, (spikes, targets, q, a) in enumerate(micros):
            self.assertEqual(tuple(spikes.shape), (4, 4, 1024, 512))
            self.assertEqual(tuple(targets.shape), (4, 1024))
            self.assertEqual(tuple(q.shape), (4, 1024, 1024))
            self.assertEqual(tuple(a.shape), (4, 1024, 1024))
            self.assertEqual(float(spikes[0, 0, 0, 0]), index * 4)
            self.assertEqual(int(targets[0, 0]), (index * 4) % 2)

    def test_optimizer_endpoint_guard_rejects_mixed_or_wrong_moment_shapes(self):
        binder = model.RelationContextAssignment("raw_only")
        optimizer = torch.optim.Adam([{"name": "assignment",
            "params": list(binder.parameters()), "lr": train.LR}])
        batch = self._mini_batch(seed=44)
        self._step(binder, optimizer, batch, 1)
        state = optimizer.state_dict()
        state["param_groups"][0]["params"] = list(state["param_groups"][0]["params"])
        with self.assertRaisesRegex(ValueError, "endpoint"):
            train._optimizer_summary(state, list(binder.parameters()), endpoint=2)

    def test_public_cpu_admission_binds_actual_producer_contract_before_gpu(self):
        root = Path(train.HERE) / f".train_prepare_fixture_{uuid.uuid4().hex}"
        cache_root, context_root = root / "source", root / "context"
        cache_root.mkdir(parents=True); context_root.mkdir()
        try:
            source_sha, context_sha = "a" * 64, "b" * 64
            (cache_root / "manifest.json").write_text(json.dumps({"status": "complete", "count": 512,
                "bindings_sha256": "c" * 64, "bindings": {"image_ids": list(range(1000, 1512)),
                "source_checkpoint_sha256": "d" * 64, "source_manifest_sha256": "e" * 64,
                "source_model_state_sha256": "f" * 64}}), encoding="utf-8")
            source_sha = train.sha256_file(cache_root / "manifest.json")
            (context_root / "manifest.json").write_text(json.dumps({"status": "complete", "count": 512,
                "bindings_sha256": "1" * 64}), encoding="utf-8")
            context_sha = train.sha256_file(context_root / "manifest.json")
            report_path = root / "receipt.json"
            report_path.write_text('{"training_admission": true}', encoding="utf-8")
            ids = list(range(1000, 1512))
            source_manifest = json.loads((cache_root / "manifest.json").read_text())
            context_manifest = json.loads((context_root / "manifest.json").read_text())
            upstream = {"experiment": train.EXPERIMENT, "ordered_image_ids": ids,
                "source_trace_manifest_sha256": source_sha,
                "context_cache_manifest_sha256": context_sha}
            receipt = {"training_admission": True}
            with mock.patch.object(train.preflight, "validate_upstream", return_value=upstream), \
                 mock.patch.object(train.preflight, "validate_report_record", return_value=receipt) as validate_receipt, \
                 mock.patch.object(train.preflight, "implementation_fingerprint",
                                   return_value={"preflight.py": "f" * 64}), \
                 mock.patch.object(train, "implementation_fingerprint",
                                   return_value={"train.py": "1" * 64}):
                result = train.prepare_training_inputs_cpu(cache_root=cache_root,
                    context_root=context_root, preflight_path=report_path)
            self.assertEqual(result["image_ids"], ids)
            validate_receipt.assert_called_once_with(receipt, expected=upstream)
            self.assertEqual(result["preflight_report"], receipt)
            report_path.write_text('{"training_admission": false}', encoding="utf-8")
            with mock.patch.object(train.preflight, "validate_upstream", return_value=upstream), \
                 mock.patch.object(train.preflight, "validate_report_record",
                                   return_value={"training_admission": False}), \
                 mock.patch.object(train.preflight, "implementation_fingerprint",
                                   return_value={"preflight.py": "f" * 64}), \
                 mock.patch.object(train, "implementation_fingerprint",
                                   return_value={"train.py": "1" * 64}):
                with self.assertRaisesRegex(ValueError, "rejected"):
                    train.prepare_training_inputs_cpu(cache_root=cache_root,
                        context_root=context_root, preflight_path=report_path)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
