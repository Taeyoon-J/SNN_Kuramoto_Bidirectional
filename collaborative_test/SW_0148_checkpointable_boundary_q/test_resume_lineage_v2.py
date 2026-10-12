from __future__ import annotations

import copy
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from torch import nn

from collaborative_test.SW_0148_checkpointable_boundary_q import recovery_v2, train
from collaborative_test.SW_0148_checkpointable_boundary_q.resume_lineage_v2 import (
    normalized_resume_bindings,
)


class ResumeLineageNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.current = {
            "arm": "joint",
            "seed": 1,
            "source_checkpoint_sha256": "a" * 64,
            "initial_source_Q_parity": None,
            "updates": 256,
        }
        self.saved = {
            "arm": "joint",
            "seed": 1,
            "source_checkpoint_sha256": "a" * 64,
            "initial_source_Q_parity": {"exact": True, "max_abs": 0.0},
            "updates": 256,
        }

    def test_verified_parity_normalizes_without_mutating_inputs(self):
        current_before, saved_before = copy.deepcopy(self.current), copy.deepcopy(self.saved)
        result = normalized_resume_bindings(self.current, self.saved)
        self.assertEqual(result["initial_source_Q_parity"], {"exact": True, "max_abs": 0.0})
        self.assertEqual(self.current, current_before)
        self.assertEqual(self.saved, saved_before)

    def test_missing_or_nonexact_saved_parity_is_rejected(self):
        for parity in (None, {}, {"exact": False}, {"exact": 1}):
            with self.subTest(parity=parity):
                saved = copy.deepcopy(self.saved)
                saved["initial_source_Q_parity"] = parity
                with self.assertRaisesRegex(ValueError, "exact successful"):
                    normalized_resume_bindings(self.current, saved)

    def test_only_none_placeholder_is_ignored(self):
        current = copy.deepcopy(self.current)
        current["initial_source_Q_parity"] = {"exact": True}
        with self.assertRaisesRegex(ValueError, "unset parity placeholder"):
            normalized_resume_bindings(current, self.saved)

    def test_any_other_lineage_difference_is_rejected(self):
        for key, value in (("arm", "encoder_graph"), ("seed", 2),
                           ("source_checkpoint_sha256", "b" * 64),
                           ("updates", 255), ("unexpected", "value")):
            with self.subTest(key=key):
                saved = copy.deepcopy(self.saved)
                saved[key] = value
                with self.assertRaisesRegex(ValueError, "lineage differs"):
                    normalized_resume_bindings(self.current, saved)

    def test_actual_train_arm_resume_branch_accepts_only_saved_parity_difference(self):
        """Exercise frozen train_arm's resume compare/load path with costly IO mocked."""
        root = Path.cwd() / f".lineage-resume-fixture-{id(self)}"
        work = root / "work"
        output = root / "training"
        boundary_dir = work / "joint_seed1" / "checkpoints"
        boundary = boundary_dir / "boundary.ptx"

        wrapped, encoder = nn.Linear(2, 2), nn.Linear(2, 2)
        student = SimpleNamespace(
            wrapped=wrapped, encoder=encoder,
            provenance={"source_training_ids_sha256": "i" * 64,
                        "rgb_asset_validation": {"train_cache_sha256": "r" * 64}},
        )
        teacher = SimpleNamespace(wrapped=nn.Linear(2, 2), encoder=nn.Linear(2, 2))
        src = (None, Path(train.sw146.__file__), None, None, None, "s" * 64)
        params = list(wrapped.parameters()) + list(encoder.parameters())
        groups = ["native", "encoder", "integration"]
        ctx = {"admission_sha256": "a" * 64, "lambda_A": .25, "scale": 1.0}
        preflights = {arm: {"post_parameter_sha256": arm[0] * 64}
                      for arm in train.sw146.ARMS}
        ids_rows = ([100], [0], None)

        patchers = [
            mock.patch.object(train.sw146, "_admission_context", return_value=ctx),
            mock.patch.object(train.sw146, "_require_both_preflights", return_value=preflights),
            mock.patch.object(train.sw146, "preflight_path",
                              return_value=Path(train.sw146.__file__).with_name("protocol.json")),
            mock.patch.object(train.sw146, "_source_foundation",
                              side_effect=[(student, src), (teacher, src)]),
            mock.patch.object(train.sw146, "_groups", return_value=groups),
            mock.patch.object(train.sw146, "_set_trainable"),
            mock.patch.object(train.sw146, "_optimizer", return_value=(object(), params)),
            mock.patch.object(train.sw146, "ordered_train", return_value=ids_rows),
            mock.patch.object(np, "load", return_value=object()),
            mock.patch.object(train.sw146, "_param_sha", return_value="p" * 64),
            mock.patch.object(train.sw146, "_frozen_native_sha", return_value="f" * 64),
            mock.patch.object(train.sw146, "fingerprint", return_value={"fixture": True}),
            mock.patch.object(train, "runtime_signature", return_value={
                "device_type": "cuda", "device_index": 0, "cuda_visible_devices": "2"}),
        ]
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            # Construct the exact saved lineage emitted by the frozen builder.
            saved_bindings = train._train_source_bindings(
                "joint", ctx, src, ids_rows[0], student, None, preflights)
            saved_bindings["initial_source_Q_parity"] = {"exact": True, "max_abs": 0.0}
            committed_bindings = copy.deepcopy(saved_bindings)
            runtime_record = {"device_type": "cuda", "device_index": 0,
                              "cuda_visible_devices": "2"}
            metadata = {"bindings": copy.deepcopy(saved_bindings),
                        "completed_updates": train.sw146.UPDATES,
                        "runtime": runtime_record}
            stack.enter_context(mock.patch.object(train, "inspect_committed_metadata",
                                                  return_value=metadata))
            stack.enter_context(mock.patch.object(recovery_v2, "inspect_committed_metadata",
                                                  return_value=metadata))
            loaded = {}
            def load_latest(_checkpoint_dir, *, expected_bindings, expected_runtime, **_kwargs):
                if expected_bindings != committed_bindings:
                    raise ValueError("checkpoint lineage mismatch")
                if expected_runtime != runtime_record:
                    raise ValueError("checkpoint runtime mismatch")
                loaded["bindings"] = copy.deepcopy(expected_bindings)
                loaded["runtime"] = copy.deepcopy(expected_runtime)
                return {"completed_updates": train.sw146.UPDATES,
                        "history": [{"update": i + 1} for i in range(train.sw146.UPDATES)]}
            stack.enter_context(mock.patch.object(train, "load_latest", side_effect=load_latest))
            stack.enter_context(mock.patch.object(
                train, "_finalize", return_value={"status": "training_complete",
                                                   "completed_updates": train.sw146.UPDATES}))
            original_exists = Path.exists
            def fixture_exists(path):
                path = Path(path)
                if path in (boundary_dir, boundary):
                    return True
                if path == output / "joint_seed1":
                    return False
                return original_exists(path)
            receipts = []
            stack.enter_context(mock.patch.object(Path, "exists", fixture_exists))
            stack.enter_context(mock.patch.object(Path, "is_file",
                                                  lambda path: Path(path) == boundary))
            actual_sha = recovery_v2.sha256_file
            def fixture_sha(path):
                if Path(path) == boundary:
                    return "b" * 64
                return actual_sha(path)
            stack.enter_context(mock.patch.object(
                recovery_v2, "sha256_file", side_effect=fixture_sha))
            stack.enter_context(mock.patch.object(
                recovery_v2, "_atomic_create_json", side_effect=lambda _path, payload: receipts.append(payload)))
            result = recovery_v2.train_arm("joint", device="cuda:0", receipt_root=root / "receipts",
                                           work_root=work, output_root=output)

        self.assertEqual(result["status"], "training_complete")
        self.assertEqual(loaded["bindings"], saved_bindings)
        self.assertEqual(loaded["runtime"]["cuda_visible_devices"], "2")
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]["completed_updates_at_start"], train.sw146.UPDATES)
        self.assertEqual(receipts[0]["checkpoint_boundary_sha256"], "b" * 64)
        self.assertEqual(receipts[0]["frozen_sw0148_file_sha256"],
                         recovery_v2.EXPECTED_FROZEN_HASHES)

    def test_process_local_adapter_restores_original_callable_after_error(self):
        class FakeTrainer:
            def _train_source_bindings(self):
                return {"lineage": "same", "initial_source_Q_parity": None}
        fake = FakeTrainer()
        original = fake._train_source_bindings
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with recovery_v2.install_parity_normalizer(fake):
                self.assertEqual(fake._train_source_bindings(), {"lineage": "same"})
                raise RuntimeError("stop")
        self.assertIs(fake._train_source_bindings.__func__, original.__func__)


if __name__ == "__main__":
    unittest.main()
