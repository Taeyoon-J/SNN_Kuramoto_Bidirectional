from __future__ import annotations

import copy
import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np  # Import NumPy before Torch on the local Windows runtime.
import torch
from torch import nn

from collaborative_test.SW_0135_native32_spike_binding.binder import NativeSpikeSlotBinder
from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint as durable
from collaborative_test.SW_0161_supervised_spike_partition import cache, train


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)


def _tiny_step(model, optimizer):
    x = torch.rand(4, 3)
    target = torch.randn(4, 2)
    loss = (model.linear(x) - target).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


class SW161TrainingTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(train.HERE) / ("train_test_" + uuid.uuid4().hex)

    def tearDown(self):
        if self.root.exists():
            resolved, parent = self.root.resolve(), Path(train.HERE).resolve()
            if resolved.parent != parent:
                raise AssertionError("refusing to remove a path outside this test-owned directory")
            shutil.rmtree(resolved)

    def test_fixed_sixteen_pass_order_is_complete_and_reproducible(self):
        a = train.make_training_order()
        b = train.make_training_order()
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(a.shape, (16, 512))
        expected = np.arange(512)
        for pass_order in a:
            self.assertTrue(np.array_equal(np.sort(pass_order), expected))
        self.assertEqual(train.UPDATES, 512)
        self.assertEqual(train.BATCH * train.UPDATES, train.PASSES * 512)

    def test_actual_binder_microbatch_gradient_accumulation_updates_head(self):
        torch.manual_seed(161003)
        binder = NativeSpikeSlotBinder(seed=train.HEAD_SEED)
        reference = copy.deepcopy(binder)
        optimizer = train.make_optimizer(binder)
        # Nonconstant actual-shaped cached traces and four distinct instances.
        source = torch.randn(1, 4, 1024, 512, dtype=torch.float32) * 0.1
        traces = [source.expand(4, -1, -1, -1).contiguous() for _ in range(4)]
        labels_one = (torch.arange(1024, dtype=torch.int64) // 256).clamp_max(3)
        targets = [labels_one.expand(4, -1).contiguous() for _ in range(4)]
        ref_losses = []
        for x, y in zip(traces, targets):
            p, _, _ = reference(x)
            loss, _ = train.objective.balanced_relation_loss(p, y)
            ref_losses.append(loss)
        torch.stack(ref_losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), train.CLIP)
        before = {name: value.detach().clone() for name, value in binder.state_dict().items()}
        row = train.logical_update(binder=binder, optimizer=optimizer,
            arm="pair_supervised_P", trace_microbatches=traces,
            target_microbatches=targets, update_index=1, pass_index=1,
            cache_indices=list(range(16)), image_ids=list(range(1000, 1016)))
        self.assertTrue(np.isfinite(row["mean_loss"]))
        self.assertTrue(np.isfinite(row["binder_gradient_norm"]) and row["binder_gradient_norm"] > 0)
        for actual, expected in zip(binder.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=2e-6)
        changed = [not torch.equal(before[name], value) for name, value in binder.state_dict().items()]
        self.assertTrue(any(changed))

        candidate = NativeSpikeSlotBinder(seed=train.HEAD_SEED)
        candidate_opt = train.make_optimizer(candidate)
        candidate_before = {name: value.detach().clone() for name, value in candidate.state_dict().items()}
        candidate_row = train.logical_update(binder=candidate, optimizer=candidate_opt,
            arm="mask_supervised_P", trace_microbatches=traces,
            target_microbatches=targets, update_index=1, pass_index=1,
            cache_indices=list(range(16)), image_ids=list(range(1000, 1016)))
        self.assertTrue(np.isfinite(candidate_row["mean_loss"]))
        self.assertGreater(candidate_row["binder_gradient_norm"], 0)
        self.assertTrue(any(not torch.equal(candidate_before[name], value)
                            for name, value in candidate.state_dict().items()))

    def test_optimizer_step_validator_rejects_bad_range_and_moment_shape(self):
        binder = NativeSpikeSlotBinder(seed=train.HEAD_SEED)
        optimizer = train.make_optimizer(binder)
        for parameter in binder.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        state = optimizer.state_dict()
        self.assertEqual(train.optimizer_step_summary(state, max_step=1), {"binder": 1})
        param_id = next(iter(state["state"]))
        bad_step = {**state, "state": {**state["state"], param_id: dict(state["state"][param_id])}}
        bad_step["state"][param_id]["step"] = torch.tensor(2.0)
        with self.assertRaisesRegex(ValueError, "outside registered range"):
            train.optimizer_step_summary(bad_step, max_step=1)
        bad_shape = {**state, "state": {**state["state"], param_id: dict(state["state"][param_id])}}
        bad_shape["state"][param_id]["exp_avg"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "moments"):
            train.optimizer_step_summary(bad_shape, max_step=1)
        mixed_steps = {**state, "state": {key: dict(value) for key, value in state["state"].items()}}
        for value in mixed_steps["state"].values():
            value["step"] = torch.tensor(2.0)
        mixed_param = next(iter(mixed_steps["state"]))
        mixed_steps["state"][mixed_param]["step"] = torch.tensor(1.0)
        with self.assertRaisesRegex(ValueError, "did not reach the registered endpoint"):
            train.optimizer_step_summary(mixed_steps, max_step=2)

    def test_durable_boundary_resume_matches_uninterrupted_cpu_updates(self):
        torch.manual_seed(161004)
        reference = TinyHead()
        ref_opt = torch.optim.Adam([{"name": "tiny", "params": list(reference.parameters()), "lr": 1e-3}])
        ref_history = []
        ref_history.append({"loss": _tiny_step(reference, ref_opt)})
        ref_history.append({"loss": _tiny_step(reference, ref_opt)})
        expected_rng = torch.get_rng_state().clone()

        torch.manual_seed(161004)
        partial = TinyHead()
        partial_opt = torch.optim.Adam([{"name": "tiny", "params": list(partial.parameters()), "lr": 1e-3}])
        partial_history = [{"loss": _tiny_step(partial, partial_opt)}]
        boundary = self.root / "boundary"
        runtime = durable.runtime_signature("cpu")
        durable.save_boundary(boundary, completed_updates=1, wrapped=partial,
            encoder=nn.Identity(), optimizer=partial_opt, history=partial_history,
            bindings={"experiment": train.EXPERIMENT, "fixture": True},
            runtime=runtime, device="cpu")

        resumed = TinyHead()
        resumed_opt = torch.optim.Adam([{"name": "tiny", "params": list(resumed.parameters()), "lr": 1e-3}])
        restored = durable.load_latest(boundary, wrapped=resumed, encoder=nn.Identity(),
            optimizer=resumed_opt, expected_bindings={"experiment": train.EXPERIMENT, "fixture": True},
            expected_runtime=runtime, device="cpu", max_updates=2)
        history = restored["history"]
        history.append({"loss": _tiny_step(resumed, resumed_opt)})
        for left, right in zip(reference.state_dict().values(), resumed.state_dict().values()):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(ref_history, history)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        ref_state, resumed_state = ref_opt.state_dict()["state"], resumed_opt.state_dict()["state"]
        self.assertEqual(set(ref_state), set(resumed_state))
        for parameter_id in ref_state:
            self.assertEqual(ref_state[parameter_id]["step"], resumed_state[parameter_id]["step"])
            torch.testing.assert_close(ref_state[parameter_id]["exp_avg"], resumed_state[parameter_id]["exp_avg"])
            torch.testing.assert_close(ref_state[parameter_id]["exp_avg_sq"], resumed_state[parameter_id]["exp_avg_sq"])

    def test_training_rejects_short_fixture_cache_as_source(self):
        ids, rgb_hashes = [1, 2], ["a" * 64, "b" * 64]
        bindings = {"fixture": True}
        store = cache._AtomicTraceCache(self.root, bindings, ids, rgb_hashes)
        rows = [np.zeros((4, 1024, 512), dtype=np.float32) for _ in ids]
        targets = [np.zeros((1024,), dtype=np.int64) for _ in ids]
        for i in range(2):
            store.write_row(i, ids[i], rgb_hashes[i], rows[i], targets[i])
        store.finalize(bindings=bindings, source_model_state_before="c" * 64,
                       source_model_state_after="c" * 64)
        with self.assertRaisesRegex(ValueError, "manifest is invalid"):
            train.validate_cache(self.root, expected_bindings=bindings)

    def test_manifest_producer_roundtrips_through_endpoint_consumer(self):
        output = self.root / "endpoint"
        output.mkdir(parents=True)
        cache_root = self.root / "cache"
        cache_root.mkdir()
        (cache_root / "manifest.json").write_text("{}", encoding="utf-8")
        ids = list(range(10000, 10512))
        cache_manifest = {"count": 512, "bindings_sha256": "a" * 64,
                          "bindings": {"image_ids": ids}}
        cache_manifest_sha = train.sha256_file(cache_root / "manifest.json")
        binder = NativeSpikeSlotBinder(seed=train.HEAD_SEED)
        initial = train.source_training.state_sha(binder)
        optimizer = train.make_optimizer(binder)
        for parameter in binder.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer_state = optimizer.state_dict()
        for value in optimizer_state["state"].values():
            value["step"] = torch.tensor(float(train.UPDATES))
        final_state = {name: value.detach().cpu().clone()
                       for name, value in binder.state_dict().items()}
        final_sha = train.source_training.state_sha(final_state)
        checkpoint_path, optimizer_path, history_path = (
            output / "checkpoint.pt", output / "optimizer.pt", output / "history.json")
        with checkpoint_path.open("xb") as stream:
            torch.save({"binder": final_state, "binder_state_sha256": final_sha,
                        "completed_updates": train.UPDATES}, stream)
        with optimizer_path.open("xb") as stream:
            torch.save(optimizer_state, stream)
        order = train.make_training_order()
        history = []
        for update in range(train.UPDATES):
            pass_ix, in_pass = divmod(update, 32)
            indices = order[pass_ix, in_pass * train.BATCH:(in_pass + 1) * train.BATCH].tolist()
            history.append({"update": update + 1, "cache_indices": indices,
                "image_ids": [ids[index] for index in indices],
                "mean_loss": 0.5, "binder_gradient_norm": 0.25})
        history_path.write_text(json.dumps(history, allow_nan=False), encoding="utf-8")
        schedule_sha = __import__("hashlib").sha256(order.tobytes()).hexdigest()
        preflight_fp = {"preflight.py": "b" * 64}
        bindings = {"experiment": train.EXPERIMENT, "arm": train.ARMS[0],
            "seed": train.SEED, "head_seed": train.HEAD_SEED,
            "shuffle_seed": train.SHUFFLE_SEED, "passes": train.PASSES,
            "updates": train.UPDATES, "logical_batch": train.BATCH,
            "microbatch": train.MICRO, "learning_rate": train.LR,
            "clip_norm": train.CLIP, "ordered_train_ids_sha256": train.sw155.ids_sha(ids),
            "cache_manifest_sha256": cache_manifest_sha,
            "cache_bindings_sha256": cache_manifest["bindings_sha256"],
            "preflight_report_sha256": "c" * 64,
            "preflight_implementation_fingerprint": preflight_fp,
            "schedule_sha256": schedule_sha, "training_order": order.tolist(),
            "implementation_fingerprint": train.implementation_fingerprint(),
            "initial_binder_state_sha256": initial}
        manifest = train.build_training_manifest(arm=train.ARMS[0],
            output_files={"checkpoint": checkpoint_path, "optimizer": optimizer_path,
                          "history": history_path},
            bindings=bindings, cache_manifest=cache_manifest, order=order,
            schedule_sha=schedule_sha,
            step_summary={"binder": train.UPDATES},
            initial_binder_state_sha256=initial, binder_state_sha256=final_sha)
        (output / "manifest.json").write_text(json.dumps(manifest, allow_nan=False),
                                                encoding="utf-8")
        preflight_binding = {"report_sha256": "c" * 64,
                             "preflight_implementation_fingerprint": preflight_fp}
        with (mock.patch.object(train, "validate_cache", return_value=cache_manifest),
              mock.patch.object(train, "validate_preflight_admission", return_value=preflight_binding)):
            accepted = train.validate_training_artifact(train.ARMS[0], directory=output,
                                                         cache_root=cache_root)
        self.assertEqual(accepted["completed_updates"], 512)
        tampered = dict(manifest, preflight_report_sha256="d" * 64)
        (output / "manifest.json").write_text(json.dumps(tampered), encoding="utf-8")
        with (mock.patch.object(train, "validate_cache", return_value=cache_manifest),
              mock.patch.object(train, "validate_preflight_admission", return_value=preflight_binding)):
            with self.assertRaisesRegex(ValueError, "endpoint binding"):
                train.validate_training_artifact(train.ARMS[0], directory=output,
                                                   cache_root=cache_root)


if __name__ == "__main__":
    unittest.main()
