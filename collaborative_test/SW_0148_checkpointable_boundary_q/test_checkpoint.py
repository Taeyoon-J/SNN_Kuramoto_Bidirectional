from __future__ import annotations

import random
import unittest
from pathlib import Path
import uuid
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn

from collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint import (
    is_only_uncommitted_temp, load_latest, runtime_signature, save_boundary,
)
from collaborative_test.SW_0148_checkpointable_boundary_q import train as sw148


def _models():
    wrapped = nn.Sequential(nn.Linear(2, 5), nn.Tanh(), nn.Linear(5, 1))
    encoder = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
    optimizer = torch.optim.Adam([
        {"params": list(encoder.parameters()), "lr": 3e-6, "name": "encoder"},
        {"params": list(wrapped.parameters()), "lr": 3e-5, "name": "native_core"},
    ])
    return wrapped, encoder, optimizer


def _update(wrapped, encoder, optimizer, index):
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for _ in range(4):
        x = torch.randn(3, 3)
        jitter = (random.random() + float(np.random.rand())) * 0.01
        target = torch.randn(3, 2) + jitter
        prediction = wrapped(encoder(x))
        loss = (prediction - target[:, :1]).square().mean()
        (loss / 4).backward()
        total += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(wrapped.parameters()), 1.)
    optimizer.step()
    return {"update": index, "loss_sum": total}


def _run(wrapped, encoder, optimizer, start, stop, history, checkpoint_dir=None,
         bindings=None, runtime=None):
    for step in range(start + 1, stop + 1):
        history.append(_update(wrapped, encoder, optimizer, step))
        if checkpoint_dir is not None:
            save_boundary(checkpoint_dir, completed_updates=step, wrapped=wrapped,
                          encoder=encoder, optimizer=optimizer, history=history,
                          bindings=bindings, runtime=runtime, device="cpu")


def _equal_nested(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal_nested(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_equal_nested(a, b) for a, b in zip(left, right))
    return left == right


class BoundaryCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).resolve().parent / f".test-run-{uuid.uuid4().hex}"
        self.tmp.mkdir(parents=True, exist_ok=False)

    def tearDown(self):
        # Remove only the unique fixture tree created by this test class.
        if self.tmp.parent.resolve() != Path(__file__).resolve().parent:
            raise AssertionError("refusing to remove a fixture outside this test folder")
        for child in sorted(self.tmp.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        self.tmp.rmdir()

    def test_interrupted_and_resumed_training_matches_uninterrupted(self):
        bindings = {"source_sha256": "a" * 64, "ordered_ids_sha256": "b" * 64,
                    "arm": "joint", "updates": 6, "microbatch_count": 4}
        runtime = runtime_signature("cpu")
        random.seed(481)
        np.random.seed(482)
        torch.manual_seed(483)
        ref_w, ref_e, ref_o = _models()
        ref_history = []
        _run(ref_w, ref_e, ref_o, 0, 6, ref_history)
        ref_rng = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state())

        random.seed(481)
        np.random.seed(482)
        torch.manual_seed(483)
        first_w, first_e, first_o = _models()
        resumed_history = []
        root = self.tmp / "checkpoints"
        save_boundary(root, completed_updates=0, wrapped=first_w, encoder=first_e,
                      optimizer=first_o, history=[], bindings=bindings,
                      runtime=runtime, device="cpu")
        _run(first_w, first_e, first_o, 0, 3, resumed_history, root, bindings, runtime)

        # New construction may consume random values; loading restores the
        # committed RNG states after all model/optimizer objects are created.
        random.seed(998)
        np.random.seed(999)
        torch.manual_seed(1000)
        resumed_w, resumed_e, resumed_o = _models()
        state = load_latest(root, wrapped=resumed_w, encoder=resumed_e,
                            optimizer=resumed_o, expected_bindings=bindings,
                            expected_runtime=runtime, device="cpu", max_updates=6)
        self.assertEqual(state["completed_updates"], 3)
        resumed_history = state["history"]
        _run(resumed_w, resumed_e, resumed_o, 3, 6, resumed_history,
             root, bindings, runtime)

        for actual, expected in zip(resumed_w.state_dict().values(), ref_w.state_dict().values()):
            self.assertTrue(torch.equal(actual, expected))
        for actual, expected in zip(resumed_e.state_dict().values(), ref_e.state_dict().values()):
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(_equal_nested(resumed_o.state_dict(), ref_o.state_dict()))
        self.assertEqual(resumed_history, ref_history)
        self.assertTrue(torch.equal(torch.get_rng_state(), ref_rng[0]))
        self.assertEqual(random.getstate(), ref_rng[1])
        nstate = np.random.get_state()
        self.assertTrue(np.array_equal(nstate[1], ref_rng[2][1]))
        self.assertEqual(nstate[2:], ref_rng[2][2:])

    def test_rejects_changed_source_or_runtime_before_loading(self):
        wrapped, encoder, optimizer = _models()
        root = self.tmp / "checkpoints"
        bindings = {"source": "source-A"}
        runtime = runtime_signature("cpu")
        save_boundary(root, completed_updates=0, wrapped=wrapped, encoder=encoder,
                      optimizer=optimizer, history=[], bindings=bindings,
                      runtime=runtime, device="cpu")
        before = {k: v.clone() for k, v in wrapped.state_dict().items()}
        with self.assertRaisesRegex(ValueError, "bindings"):
            load_latest(root, wrapped=wrapped, encoder=encoder, optimizer=optimizer,
                        expected_bindings={"source": "source-B"},
                        expected_runtime=runtime, device="cpu", max_updates=6)
        with self.assertRaisesRegex(ValueError, "runtime"):
            load_latest(root, wrapped=wrapped, encoder=encoder, optimizer=optimizer,
                        expected_bindings=bindings,
                        expected_runtime={**runtime, "torch_version": "different"},
                        device="cpu", max_updates=6)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in wrapped.state_dict().items()))

    def test_rejects_truncated_or_corrupt_container(self):
        wrapped, encoder, optimizer = _models()
        root = self.tmp / "checkpoints"
        save_boundary(root, completed_updates=0, wrapped=wrapped, encoder=encoder,
                      optimizer=optimizer, history=[], bindings={"x": 1},
                      runtime=runtime_signature("cpu"), device="cpu")
        checkpoint = root / "boundary.ptx"
        checkpoint.write_bytes(checkpoint.read_bytes()[:64])
        with self.assertRaises(Exception):
            load_latest(root, wrapped=wrapped, encoder=encoder, optimizer=optimizer,
                        expected_bindings={"x": 1}, expected_runtime=runtime_signature("cpu"),
                        device="cpu", max_updates=6)

    def test_actual_logical_update_commits_only_after_projection_and_keeps_order(self):
        class Wrapped(nn.Module):
            def __init__(self):
                super().__init__()
                self.a_d = nn.Parameter(torch.full((4,), 1.2))
                self.a_m = nn.Parameter(torch.full((4,), 0.2))

        wrapped = Wrapped()
        encoder = nn.Linear(1, 1, bias=False)
        f = SimpleNamespace(wrapped=wrapped, encoder=encoder)
        groups = {"encoder": list(encoder.parameters()),
                  "integration_scalars": [wrapped.a_d, wrapped.a_m]}
        params = [p for values in groups.values() for p in values]
        optimizer = torch.optim.Adam([
            {"params": groups["encoder"], "lr": 3e-6, "name": "encoder"},
            {"params": groups["integration_scalars"], "lr": 1e-3,
             "name": "integration_scalars"},
        ])
        cache = np.zeros((40, 128, 128, 3), dtype=np.uint8)
        rows = np.array([21, 2, 18, 4, 17, 6, 15, 8, 13, 10, 11, 12, 9, 14, 7, 16])
        for row in range(len(cache)):
            cache[row, 0, 0, 0] = row
        image_ids = list(range(500, 516))
        seen = []

        def fake_loss(_f, _teacher, rgb, _scale, _lambda):
            seen.append(rgb[:, 0, 0, 0].detach().cpu().tolist())
            signal = rgb[:, 0, 0, 0].float().mean() / 255.0 + 1.0
            val = signal * (encoder.weight.sum() + wrapped.a_d.sum() + wrapped.a_m.sum())
            loss = val.square()
            return loss, loss.detach(), (loss.detach() * .5), {"fixture": True}

        committed = []

        def commit_after_projection(row):
            self.assertTrue(torch.isfinite(wrapped.a_d).all())
            self.assertTrue(torch.isfinite(wrapped.a_m).all())
            self.assertTrue(bool((wrapped.a_d >= 0).all() and (wrapped.a_d <= 1).all()))
            self.assertTrue(bool((wrapped.a_m >= 0).all() and (wrapped.a_m <= 1).all()))
            self.assertTrue(all(int(state["step"].item()) == 1
                                for state in optimizer.state.values()))
            committed.append(row)

        with mock.patch.object(sw148.sw146, "_loss_micro", side_effect=fake_loss):
            row = sw148._one_logical_update(
                "joint", 0, f=f, teacher=object(), opt=optimizer, params=params,
                groups=groups, cache=cache, rows=rows, ordered_ids=image_ids,
                scale=1.0, lambda_a=.2, commit=commit_after_projection)
        expected_groups = [rows[i:i + 4].tolist() for i in range(0, 16, 4)]
        self.assertEqual(seen, expected_groups)
        self.assertEqual(row["image_ids"], image_ids)
        self.assertEqual(row["update"], 1)
        self.assertEqual(len(row["microbatches"]), 4)
        self.assertEqual(committed, [row])

    def test_k256_boundary_finalizes_without_replaying_an_update(self):
        calls = []
        sw148._execute_remaining_updates(256, 256, calls.append)
        self.assertEqual(calls, [])
        with self.assertRaises(ValueError):
            sw148._execute_remaining_updates(257, 256, calls.append)

    def test_only_empty_or_our_uncommitted_atomic_temps_can_fresh_start(self):
        root = self.tmp / "work"
        root.mkdir()
        self.assertTrue(is_only_uncommitted_temp(root))
        (root / ".boundary.ptx.abc.tmp").write_bytes(b"partial temp")
        self.assertTrue(is_only_uncommitted_temp(root))
        (root / "unrecognized.bin").write_bytes(b"keep")
        self.assertFalse(is_only_uncommitted_temp(root))


if __name__ == "__main__":
    unittest.main()
