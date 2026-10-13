from __future__ import annotations

import os
import copy
from pathlib import Path
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # preload before torch in local Windows CPU environments
_NUMPY_PRELOADED = np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint as durable
from collaborative_test.SW_0150_spike_spatial_mixture import preflight, train


class TinyWrapped(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph = nn.Linear(2, 2)
        self.native = nn.Linear(2, 2)
        self.spatial_decoder = nn.Linear(2, 1)
        self.a_d = nn.Parameter(torch.tensor(0.4))
        self.a_m = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.1))

    def project_integrations_(self):
        self.a_d.clamp_(0, 1)
        self.a_m.clamp_(0, 1)


class TinyFoundation:
    def __init__(self):
        self.wrapped = TinyWrapped()
        self.encoder = nn.Linear(1, 2)


def _families(foundation):
    wrapped = foundation.wrapped
    return {
        "encoder": list(foundation.encoder.parameters()),
        "graph": list(wrapped.graph.parameters()),
        "native_core": list(wrapped.native.parameters()),
        "integration_scalars": [wrapped.a_d, wrapped.a_m, wrapped.b],
        "decoder": list(wrapped.spatial_decoder.parameters()),
    }


def _params(wrapped, encoder):
    return {"wrapped": {key: value.detach().cpu().clone()
                        for key, value in wrapped.state_dict().items()},
            "encoder": {key: value.detach().cpu().clone()
                        for key, value in encoder.state_dict().items()}}


def _assert_nested_equal(test, left, right):
    if torch.is_tensor(left):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            _assert_nested_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class DurableTrainingTests(unittest.TestCase):
    def test_implementation_fingerprint_resolves_the_real_transitive_source_files(self):
        fingerprint = train.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0134_native_spike_binding/rollout.py", fingerprint)
        self.assertIn("snn_kuramoto_bidirectional/sinusoidal_gating.py", fingerprint)
        self.assertIn("collaborative_test/SW_0150_spike_spatial_mixture/preflight_queue.py", fingerprint)
        self.assertNotIn("collaborative_test/SW_0135_native32_spike_binding/rollout.py", fingerprint)

    def test_warm_artifact_path_is_rejected_if_it_points_inside_repository(self):
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            train.validate_training_inputs(warm_path=HERE / "should_not_read.pt")

    def _optimizer_fixture(self, arm):
        foundation = TinyFoundation()
        family_map = _families(foundation)
        optimizer, _ = train._group_optimizer(family_map, arm)
        state = optimizer.state_dict()
        for saved_group, live_group in zip(state["param_groups"], optimizer.param_groups):
            for parameter_id, parameter in zip(saved_group["params"], live_group["params"]):
                parameter_state = state["state"].setdefault(parameter_id, {})
                group_name = saved_group["name"]
                step = (train.WARM_UPDATES + train.UPDATES
                        if group_name == "decoder" else 1)
                parameter_state.update({
                    "step": torch.tensor(float(step)),
                    "exp_avg": torch.zeros_like(parameter),
                    "exp_avg_sq": torch.zeros_like(parameter),
                })
        schema = durable.optimizer_schema(optimizer)
        summary = {name: {"count": 0, "minimum": None, "maximum": None}
                   for name in ("encoder", "graph", "native_core", "integration_scalars", "decoder")}
        for group in state["param_groups"]:
            name = group["name"]
            steps = [int(state["state"][parameter_id]["step"].item())
                     for parameter_id in group["params"]]
            summary[name] = {"count": len(steps), "minimum": min(steps),
                             "maximum": max(steps)}
        rates = {group["name"]: group["lr"] for group in state["param_groups"]}
        return state, schema, rates, summary

    def test_serialized_optimizer_groups_and_step_ranges_are_validated(self):
        for arm in train.ARMS:
            with self.subTest(arm=arm):
                state, schema, rates, summary = self._optimizer_fixture(arm)
                actual = train._validate_optimizer_checkpoint(
                    arm, state, schema, rates, summary)
                self.assertEqual(actual, summary)
                if arm == "encoder_graph":
                    self.assertEqual(actual["native_core"]["count"], 0)
                    self.assertEqual(actual["integration_scalars"]["count"], 0)
                else:
                    native_group = next(g for g in state["param_groups"]
                                        if g["name"] == "native_core")
                    native_id = native_group["params"][0]
                    state["state"][native_id]["step"] = torch.tensor(257.)
                    with self.assertRaisesRegex(ValueError, "step is outside"):
                        train._validate_optimizer_checkpoint(arm, state, schema, rates, summary)

    def test_serialized_optimizer_rejects_bad_decoder_step_moments_and_summary(self):
        state, schema, rates, summary = self._optimizer_fixture("encoder_graph")
        decoder_group = next(g for g in state["param_groups"] if g["name"] == "decoder")
        decoder_id = decoder_group["params"][0]
        bad_step = copy.deepcopy(state)
        bad_step["state"][decoder_id]["step"] = torch.tensor(287.)
        with self.assertRaisesRegex(ValueError, "step is outside"):
            train._validate_optimizer_checkpoint("encoder_graph", bad_step, schema, rates, summary)
        bad_moment = copy.deepcopy(state)
        bad_moment["state"][decoder_id]["exp_avg"] = torch.zeros(99)
        with self.assertRaisesRegex(ValueError, "schema/values differ"):
            train._validate_optimizer_checkpoint("encoder_graph", bad_moment, schema, rates, summary)
        with self.assertRaisesRegex(ValueError, "disagree"):
            train._validate_optimizer_checkpoint(
                "encoder_graph", state, schema, rates,
                dict(summary, decoder={"count": 1, "minimum": 288, "maximum": 288}))

    def _make_started(self, arm):
        # Deterministic source-shaped student and a true 32-step warm decoder Adam.
        torch.manual_seed(15031)
        foundation = TinyFoundation()
        families = _families(foundation)
        if arm == "encoder_graph":
            foundation.wrapped.requires_grad_(False)
            foundation.wrapped.graph.requires_grad_(True)
            foundation.wrapped.spatial_decoder.requires_grad_(True)
            foundation.encoder.requires_grad_(True)
        else:
            foundation.wrapped.requires_grad_(True)
            foundation.encoder.requires_grad_(True)
        warm_params = list(foundation.wrapped.spatial_decoder.parameters())
        warm = torch.optim.Adam(warm_params, lr=1e-3)
        for index in range(32):
            warm.zero_grad(set_to_none=True)
            x = torch.full((4, 1), (index + 1) / 32)
            with torch.no_grad():
                hidden = foundation.wrapped.graph(foundation.encoder(x))
            foundation.wrapped.spatial_decoder(hidden).square().mean().backward()
            warm.step()
        warm_rng = durable.capture_rng("cpu")
        pilot, _ = train._group_optimizer(families, arm)
        copied = preflight.transplant_decoder_adam(
            warm, pilot, warm_params, warm_params)
        self.assertEqual(copied, len(warm_params))
        for parameter in warm_params:
            self.assertEqual(int(pilot.state[parameter]["step"]), 32)
        durable.restore_rng(warm_rng, "cpu")
        return foundation, families, pilot, warm_rng

    def _run_updates(self, arm, root, *, start=0, stop=4, restore=None):
        foundation, families, optimizer, warm_rng = self._make_started(arm)
        selected = train.trainable_parameter_families(families, arm)
        ids = list(range(500, 500 + 16 * 4))
        runtime = durable.runtime_signature("cpu")
        bindings = {"experiment": train.NAME, "arm": arm, "warm_sha256": "a" * 64,
                    "source_sha256": "b" * 64, "ordered_ids_sha256": train.ids_sha(ids)}
        history = []
        if not root.exists():
            root.mkdir()
        if restore is None:
            durable.save_boundary(root, completed_updates=0,
                                  wrapped=foundation.wrapped, encoder=foundation.encoder,
                                  optimizer=optimizer, history=history, bindings=bindings,
                                  runtime=runtime, device="cpu")
        else:
            state = durable.load_latest(root, wrapped=foundation.wrapped,
                                        encoder=foundation.encoder, optimizer=optimizer,
                                        expected_bindings=bindings,
                                        expected_runtime=runtime, device="cpu", max_updates=4)
            start, history = state["completed_updates"], state["history"]
            self.assertEqual(start, restore)

        for update in range(start, stop):
            update_ids = ids[update * 16:(update + 1) * 16]

            def compute_micro(_micro):
                x = torch.rand((4, 1))
                encoded = foundation.encoder(x)
                hidden = foundation.wrapped.graph(encoded)
                if arm == "joint":
                    hidden = foundation.wrapped.native(hidden) * foundation.wrapped.a_d + foundation.wrapped.b
                prediction = foundation.wrapped.spatial_decoder(hidden)
                loss = (prediction - (x * 0.3 + 0.2)).square().mean()
                return loss, {"fixture": "same microbatch objective"}

            def commit(row):
                history.append(row)
                durable.save_boundary(root, completed_updates=row["update"],
                                      wrapped=foundation.wrapped, encoder=foundation.encoder,
                                      optimizer=optimizer, history=history, bindings=bindings,
                                      runtime=runtime, device="cpu")

            train.logical_update(update, optimizer=optimizer, family_map=families,
                                 selected=selected, ordered_ids=update_ids,
                                 compute_micro=compute_micro, wrapped=foundation.wrapped,
                                 arm=arm, commit=commit)
        return foundation, optimizer, history, _params(foundation.wrapped, foundation.encoder)

    def test_interrupted_resume_matches_uninterrupted_with_warm_adam_for_both_arms(self):
        # Kept under the repo to avoid Windows TEMP ACL behavior; cleanup is exact and scoped.
        from uuid import uuid4
        base = ROOT / f".sw150_train_test_{uuid4().hex}"
        base.mkdir()
        try:
            for arm in train.ARMS:
                with self.subTest(arm=arm):
                    full_dir = base / f"{arm}_full"
                    full, full_opt, full_history, full_state = self._run_updates(
                        arm, full_dir, stop=4)
                    partial_dir = base / f"{arm}_resumed"
                    self._run_updates(arm, partial_dir, stop=2)
                    resumed, resumed_opt, resumed_history, resumed_state = self._run_updates(
                        arm, partial_dir, start=0, stop=4, restore=2)
                    _assert_nested_equal(self, full_state, resumed_state)
                    _assert_nested_equal(self, full_opt.state_dict(), resumed_opt.state_dict())
                    self.assertEqual(full_history, resumed_history)
                    self.assertEqual(len(resumed_history), 4)
                    self.assertEqual(int(resumed_opt.state[
                        list(resumed.wrapped.spatial_decoder.parameters())[0]]["step"]), 36)
                    self.assertEqual(max(int(state["step"]) for state in resumed_opt.state.values()), 36)
                    if arm == "encoder_graph":
                        self.assertFalse(any(p.requires_grad for p in resumed.wrapped.native.parameters()))
                        self.assertFalse(resumed.wrapped.a_d.requires_grad)
        finally:
            root = base.resolve()
            if not root.is_relative_to(ROOT.resolve()):
                raise RuntimeError("refusing to clean test data outside repository")
            for child in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            root.rmdir()

    def test_checkpoint_rejects_changed_warm_lineage_before_restoring(self):
        from uuid import uuid4
        directory = ROOT / f".sw150_wrong_lineage_{uuid4().hex}"
        directory.mkdir()
        try:
            foundation, families, optimizer, _ = self._make_started("encoder_graph")
            bindings = {"warm_sha256": "a" * 64, "source_sha256": "b" * 64}
            runtime = durable.runtime_signature("cpu")
            durable.save_boundary(directory, completed_updates=0,
                                  wrapped=foundation.wrapped, encoder=foundation.encoder,
                                  optimizer=optimizer, history=[], bindings=bindings,
                                  runtime=runtime, device="cpu")
            changed = dict(bindings, warm_sha256="c" * 64)
            fresh, _, fresh_optimizer, _ = self._make_started("encoder_graph")
            with self.assertRaisesRegex(ValueError, "bindings mismatch"):
                durable.load_latest(directory, wrapped=fresh.wrapped,
                                    encoder=fresh.encoder, optimizer=fresh_optimizer,
                                    expected_bindings=changed, expected_runtime=runtime,
                                    device="cpu", max_updates=4)
        finally:
            root = directory.resolve()
            if root.is_relative_to(ROOT.resolve()):
                for child in root.iterdir():
                    child.unlink()
                root.rmdir()


if __name__ == "__main__":
    unittest.main()
