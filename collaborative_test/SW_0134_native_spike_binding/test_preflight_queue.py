"""CPU contracts for SW0134 preflight evidence and task ordering."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"),
                str(ROOT / "collaborative_test")]

import numpy as np
import torch

from collaborative_test.SW_0134_native_spike_binding import preflight_queue as queue
from collaborative_test.SW_0134_native_spike_binding import run


class PreflightQueueTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.gettempdir()) / f"sw134_pf_{uuid.uuid4().hex[:10]}"
        self.root.mkdir()
        self.archive = self.root / "results_archive"
        self.archive.mkdir()
        self.old_archive = queue.ARCHIVE
        queue.ARCHIVE = self.archive
        self.ids = list(range(4096))
        self.pool = np.arange(4096, dtype=np.int64)
        self.source_sha = "a" * 64
        self.assets = {"train_rgb": "b" * 64, "val_rgb": "c" * 64}
        self.fingerprint = {"run.py": "d" * 64}
        self.manifest = self.root / "source_manifest.json"
        self.manifest.write_text('{"steps":256}', encoding="utf-8")
        self.manifest_sha = run.sha(self.manifest)
        self.pool_sha = hashlib.sha256(np.asarray(self.pool, dtype="<i8").tobytes()).hexdigest()

    def tearDown(self):
        queue.ARCHIVE = self.old_archive
        for path in self.root.rglob("*"):
            if path.is_file():
                path.unlink()
        for path in sorted(self.root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir():
                path.rmdir()
        self.root.rmdir()

    def _warm(self, arm):
        torch.manual_seed(13400 + (arm == "gate_joint"))
        binder = run.NativeSpikeSlotBinder(seed=134)
        decoder = run.RelativeSlotRGBDecoder(seed=106)
        params = list(binder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=run.HEAD_LR)
        for _ in range(32):
            optimizer.zero_grad(set_to_none=True)
            sum((i + 1) * p.square().sum() for i, p in enumerate(params)).backward()
            optimizer.step()
        path = self.archive / f"warm_{arm}_seed0.pt"
        payload = {
            "experiment": "SW0134_native_spike_binding", "seed": 0, "arm": arm,
            "updates": 32, "batch_size": 16, "training_ids": self.ids[:512],
            "training_ids_sha256": hashlib.sha256(
                np.asarray(self.ids[:512], dtype="<i8").tobytes()).hexdigest(),
            "all_training_ids_sha256": hashlib.sha256(
                np.asarray(self.ids, dtype="<i8").tobytes()).hexdigest(),
            "source_core_sha256": self.source_sha,
            "source_manifest_sha256": self.manifest_sha,
            "asset_hashes": self.assets, "implementation_fingerprint": self.fingerprint,
            "binder_state_dict": binder.state_dict(), "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(payload, path)
        return path

    def _report(self):
        warm = {}
        for arm in ("actual_joint", "gate_joint"):
            path = self._warm(arm)
            warm[arm] = {"path": str(path.resolve()), "sha256": run.sha(path),
                         "updates": 32, "loss_first_last": [0.5, 0.25]}
        parity = [{"arm": arm, "steps": steps, "settle": settle,
                   "full_component_trace_gate_q_h_oldloss_exact": True,
                   "production_adapter_gate_trace_exact": True}
                  for arm in ("phase", "constant")
                  for steps, settle in ((64, 32), (1024, 512))]
        norms = {key: 1.0 for key in queue.RGB_SOURCE_FAMILIES}
        updates = [
            {"arm": "actual_joint", "old_loss": 1.0, "rgb_loss": .3,
             "head_gradient_norm": 1.0, "joint_gradient_norm": 2.0,
             "head_changed_parameter_count": 4, "joint_changed_parameter_count": 8,
             "rgb_gradient_norms_by_source_family": norms,
             "source_frozen": False, "source_parameters_unchanged": False,
             "source_training_mode": "train", "throwaway_only": True},
            {"arm": "gate_joint", "old_loss": 1.0, "rgb_loss": .3,
             "head_gradient_norm": 1.0, "joint_gradient_norm": 2.0,
             "head_changed_parameter_count": 4, "joint_changed_parameter_count": 8,
             "rgb_gradient_norms_by_source_family": {
                 **norms, "dendrite": 0.0, "membrane": 0.0,
                 "a_d": 0.0, "a_m": 0.0, "b": 0.0},
             "source_frozen": False, "source_parameters_unchanged": False,
             "source_training_mode": "train", "throwaway_only": True},
            {"arm": "actual_frozen", "old_loss": 1.0, "rgb_loss": .3,
             "head_gradient_norm": 1.0, "joint_gradient_norm": 0.0,
             "head_changed_parameter_count": 4, "joint_changed_parameter_count": 0,
             "rgb_gradient_norms_by_source_family": {},
             "source_frozen": True, "source_parameters_unchanged": True,
             "source_training_mode": "eval", "throwaway_only": True},
        ]
        return {
            "status": "passed", "experiment": "SW0134_native_spike_binding", "seed": 0,
            "shuffle_seed": 117, "batch_size": 16, "train_time_steps": 1024,
            "settle_steps": 512, "live_tail_steps": 64,
            "source_core_sha256": self.source_sha,
            "source_manifest_sha256": self.manifest_sha, "source_updates": 256,
            "training_ids": self.ids,
            "training_ids_sha256": hashlib.sha256(
                np.asarray(self.ids, dtype="<i8").tobytes()).hexdigest(),
            "pool_indices_sha256": self.pool_sha, "asset_hashes": self.assets,
            "ground_truth_used": False, "source_checkpoint_modified": False,
            "joint_training_updates": 0, "warmup_optimizer_updates_per_head": 32,
            "throwaway_optimizer_updates_per_arm": 1,
            "implementation_fingerprint": self.fingerprint,
            "live_gamma_cache_max_abs_diff_first_batch": 1e-6,
            "native_zero_init_late_parity": parity,
            "warm_artifacts": warm, "warm_actual_arms_shared": True,
            "lambda_source_seed": 0, "lambda": .5,
            "lambda_seed0_record_sha256": None,
            "lambda_calibration": [
                {"batch_index": i, "old_joint_norm": float(2 + i),
                 "rgb_joint_norm": float(1 + i / 2), "lambda_ratio": .5}
                for i in range(4)],
            "disposable_updates": updates,
        }

    def test_queue_orders_seed0_before_two_parallel_dependents_and_refuses_sharing(self):
        tasks = queue.task_plan()
        self.assertEqual([row["seed"] for row in tasks], [0, 1, 2])
        self.assertEqual(tasks[0]["depends_on"], [])
        self.assertEqual(tasks[1]["depends_on"], [tasks[0]["task_id"]])
        self.assertEqual(tasks[2]["depends_on"], [tasks[0]["task_id"]])
        self.assertTrue(queue.gpu_is_exclusive_candidate(512, []))
        self.assertFalse(queue.gpu_is_exclusive_candidate(513, []))
        self.assertFalse(queue.gpu_is_exclusive_candidate(1, [123]))
        self.assertIn("--stage", queue.command(tasks[0]))

    def test_valid_result_binds_parity_warm_adam_lambda_and_rgb_credit(self):
        report = self._report()
        path = self.archive / "preflight_seed0.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        original_fp = run.implementation_fingerprint
        try:
            with patch.object(run, "source_contract", return_value=(
                    self.root / "source.pt", self.manifest, {"steps": 256},
                    self.pool, self.ids, self.source_sha)), \
                 patch.object(run, "implementation_fingerprint", return_value=self.fingerprint), \
                 patch.object(run.sw130, "validate_rgb_assets", return_value=self.assets):
                task = {"stage": "preflight", "seed": 0}
                self.assertTrue(queue.valid_result(task, self.assets))
                report["native_zero_init_late_parity"][0]["production_adapter_gate_trace_exact"] = False
                path.write_text(json.dumps(report), encoding="utf-8")
                self.assertFalse(queue.valid_result(task, self.assets))
                report = self._report()
                report["disposable_updates"][1]["rgb_gradient_norms_by_source_family"]["membrane"] = 1e-3
                path.write_text(json.dumps(report), encoding="utf-8")
                self.assertFalse(queue.valid_result(task, self.assets))
        finally:
            run.implementation_fingerprint = original_fp


if __name__ == "__main__":
    unittest.main()
