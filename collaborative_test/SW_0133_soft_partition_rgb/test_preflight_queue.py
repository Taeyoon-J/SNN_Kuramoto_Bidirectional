"""Focused SW0133 preflight and queue contracts (CPU only)."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"),
                str(ROOT / "collaborative_test")]
import numpy as np  # preserve registered NumPy-before-Torch import order
import torch

from collaborative_test.SW_0133_soft_partition_rgb import preflight_queue as queue
from collaborative_test.SW_0133_soft_partition_rgb import run


def _warm_file(source_sha, ids, assets):
    directory = Path(tempfile.gettempdir()) / f"sw133_{uuid.uuid4().hex}"
    directory.mkdir()
    path = directory / "warm.pt"
    parameters = [torch.nn.Parameter(torch.zeros(())) for _ in range(6)]
    optimizer = torch.optim.Adam(parameters)
    for _ in range(32):
        optimizer.zero_grad(set_to_none=True)
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    payload = {
        "decoder_state_dict": {"weight": torch.ones(1)},
        "optimizer_state_dict": optimizer.state_dict(),
        "source_core_sha256": source_sha,
        "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "asset_hashes": assets,
        "warmup_updates": 32,
    }
    torch.save(payload, path)
    return directory, path


def _report(seed, warm_path, fingerprint, source_sha, source_manifest_sha,
            training_ids, pool_sha, assets, seed0_sha=None, seed0_report=None):
    ids_sha = hashlib.sha256(np.asarray(training_ids, dtype="<i8").tobytes()).hexdigest()
    ratios = [0.5] * 4
    calibration = [{"batch_index": i, "old_joint_norm": 10.0 * (i + 1),
                    "rgb_joint_norm": 5.0 * (i + 1), "lambda_ratio": ratios[i]}
                   for i in range(4)]
    updates = []
    for arm in queue.ARMS:
        live = arm != "phase_detached"
        updates.append({
            "arm": arm, "optimizer_updates": 1, "throwaway_only": True,
            "joint_preclip_norm": 2.0, "decoder_preclip_norm": 3.0,
            "old_loss": 1.25, "rgb_loss": 0.4,
            "rgb_to_q_gradient_norm": 1.0 if live else 0.0,
            "rgb_gradient_norms_by_family": ({name: 1.0 for name in queue.LIVE_FAMILIES}
                                               if live else {}),
            "joint_and_decoder_changed": True,
            "changed_core_families": {name: True for name in queue.NATIVE_CORE_FAMILIES},
            "changed_encoder_parameter_count": 1 if live else 0,
            "changed_decoder_parameter_count": 1,
            "changed_integration_parameters": ["b"] if live else [],
            "integration_parameter_gradient_abs_by_component": {
                "core.a_d": [1.0 if live else 0.0] * 4,
                "core.a_m": [1.0 if live else 0.0] * 4,
                "core.b": [1.0 if live else 0.0] * 4},
            "rgb_gradient_unused_integration": {
                "core.a_d": not live, "core.a_m": not live, "core.b": not live},
        })
    report = {
        "status": "passed", "experiment": "SW0133_soft_partition_rgb", "seed": seed,
        "shuffle_seed": 117 + seed, "batch_size": 16, "train_time_steps": 64,
        "settle_steps": 32, "source_core_sha256": source_sha,
        "source_manifest_sha256": source_manifest_sha, "source_steps": 256,
        "training_ids": training_ids, "training_ids_sha256": ids_sha,
        "pool_indices_sha256": pool_sha, "asset_hashes": assets,
        "ground_truth_used": False, "source_core_checkpoint_modified": False,
        "source_encoder_modified": False, "optimizer_updates": 0,
        "implementation_fingerprint": fingerprint,
        "live_source_gamma_max_abs_diff_first_batch": 1e-6,
        "native_zero_initial_parity": {
            "source_core_sha256": source_sha,
            "checks": [{"arm": arm, "time_steps": t, "settle": s,
                        "theta_component_membrane_spikes_q_h_primary_oldloss_exact": True}
                       for arm in ("phase", "constant")
                       for t, s in ((64, 32), (1024, 512))]},
        "paired_zero_initial_arm_check": {
            "initial_decoder_equal": True, "initial_gamma_q_hard_traces_equal": True,
            "initial_soft_p_rgb_equal": True, "warm_decoder_optimizer_shared_across_arms": True},
        "decoder_warmup_artifact": str(warm_path.resolve()),
        "decoder_warmup_artifact_sha256": hashlib.sha256(warm_path.read_bytes()).hexdigest(),
        "decoder_warmup_updates": 32, "decoder_warmup_loss_first_last": [1.0, 0.5],
        "row_scramble_count_per_arm": 64,
        "row_scramble_by_arm": {"phase_live": {
            "per_image_excess": [0.0] * 64, "mean_excess": 0.0, "positive_count": 0}},
        "lambda_source_seed": 0, "lambda": 0.5,
        "lambda_seed0_calibration": calibration if seed == 0 else "reused frozen seed0 preflight",
        "lambda_batch_ratios": ratios if seed == 0 else None,
        "seed0_lambda_record_sha256": seed0_sha,
        "disposable_updates": updates,
    }
    return report


class QueueContractTests(unittest.TestCase):
    def test_fixed_three_seed_plan_and_exclusive_gpu_rule(self):
        tasks = queue.task_plan()
        self.assertEqual([row["seed"] for row in tasks], [0, 1, 2])
        self.assertEqual(tasks[0]["depends_on"], [])
        self.assertEqual(tasks[1]["depends_on"], [tasks[0]["task_id"]])
        self.assertEqual(tasks[2]["depends_on"], [tasks[0]["task_id"]])
        self.assertTrue(queue.gpu_is_exclusive_candidate(512, []))
        self.assertFalse(queue.gpu_is_exclusive_candidate(513, []))
        self.assertFalse(queue.gpu_is_exclusive_candidate(1, [123]))

    def test_validator_checks_calibration_arithmetic_and_live_vs_detached_credit(self):
        training_ids = list(range(4096))
        assets = {"train": "a" * 64, "encoder": "b" * 64}
        source_sha, manifest_sha, pool_sha = "d" * 64, "e" * 64, "f" * 64
        directory, warm = _warm_file(source_sha, training_ids, assets)
        try:
            fingerprint = {"run.py": "c" * 64}
            seed0 = _report(0, warm, fingerprint, source_sha, manifest_sha,
                            training_ids, pool_sha, assets)
            kwargs = dict(fingerprint=fingerprint, source_sha=source_sha,
                          source_manifest_sha=manifest_sha, training_ids=training_ids,
                          pool_sha=pool_sha, asset_hashes=assets, warmup_path=warm,
                          warmup_sha_fn=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
            self.assertTrue(queue.validate_preflight_record(seed0, 0, **kwargs))
            bad_ratio = json.loads(json.dumps(seed0))
            bad_ratio["lambda_seed0_calibration"][0]["lambda_ratio"] = 0.9
            self.assertFalse(queue.validate_preflight_record(bad_ratio, 0, **kwargs))
            bad_detached = json.loads(json.dumps(seed0))
            detached = next(r for r in bad_detached["disposable_updates"]
                            if r["arm"] == "phase_detached")
            detached["rgb_to_q_gradient_norm"] = 0.1
            self.assertFalse(queue.validate_preflight_record(bad_detached, 0, **kwargs))
            bad_live = json.loads(json.dumps(seed0))
            live = next(r for r in bad_live["disposable_updates"] if r["arm"] == "phase_live")
            live["rgb_to_q_gradient_norm"] = 0.0
            self.assertFalse(queue.validate_preflight_record(bad_live, 0, **kwargs))
        finally:
            warm.unlink(missing_ok=True)
            directory.rmdir()

    def test_later_seed_must_bind_passed_seed0_report_and_lambda(self):
        ids = list(reversed(range(4096))); assets = {"train": "a" * 64}
        fp = {"runner": "b" * 64}; source = "c" * 64
        manifest = "d" * 64; pool = "e" * 64
        directory, warm = _warm_file("1" * 64, ids, assets)
        try:
            seed0_dir, seed0_warm = _warm_file(source, list(range(4096)), assets)
            self.addCleanup(lambda: (seed0_warm.unlink(missing_ok=True), seed0_dir.rmdir()))
            seed0 = _report(0, seed0_warm, fp, source, manifest, list(range(4096)), pool, assets)
            seed0_sha = "f" * 64
            seed1 = _report(1, warm, fp, "1" * 64, "2" * 64,
                            ids, "3" * 64, assets,
                            seed0_sha=seed0_sha, seed0_report=seed0)
            kwargs = dict(fingerprint=fp, source_sha="1" * 64,
                          source_manifest_sha="2" * 64, training_ids=ids,
                          pool_sha="3" * 64, asset_hashes=assets, warmup_path=warm,
                          seed0_report=seed0, seed0_report_sha=seed0_sha,
                          warmup_sha_fn=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
            self.assertTrue(queue.validate_preflight_record(seed1, 1, **kwargs))
            mismatched = json.loads(json.dumps(seed1))
            mismatched["lambda"] = 0.5001
            self.assertFalse(queue.validate_preflight_record(mismatched, 1, **kwargs))
        finally:
            warm.unlink(missing_ok=True)
            directory.rmdir()

    def test_valid_result_reads_runtime_source_manifest_and_warm_artifact(self):
        directory = Path(tempfile.gettempdir()) / f"sw133_{uuid.uuid4().hex}"
        directory.mkdir()
        archive = directory / "archive"
        archive.mkdir()
        warm = archive / "preflight_decoder_seed0.pt"
        warm_dir, warm_source = _warm_file("d" * 64, list(range(4096)), {"train": "a" * 64})
        warm_source.replace(warm)
        warm_dir.rmdir()
        manifest = directory / "source_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        pool = np.arange(4096, dtype=np.int64)
        ids = list(range(4096))
        assets = {"train": "a" * 64}
        fingerprint = {"run": "f" * 64}
        report = _report(0, warm, fingerprint, "d" * 64,
                         hashlib.sha256(b"{}").hexdigest(), ids,
                         hashlib.sha256(np.asarray(pool, dtype="<i8").tobytes()).hexdigest(),
                         assets)
        report_path = archive / "preflight_seed0.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        original_archive = queue.ARCHIVE
        try:
            queue.ARCHIVE = archive
            with mock.patch.object(run, "source_contract", return_value=(
                    directory / "source.pt", manifest, {"steps": 256}, pool, ids, "d" * 64)), \
                 mock.patch.object(run.sw130, "validate_rgb_assets", return_value=assets), \
                 mock.patch.object(run, "implementation_fingerprint", return_value=fingerprint):
                self.assertTrue(queue.valid_result({"stage": "preflight", "seed": 0}, assets))
            report["live_source_gamma_max_abs_diff_first_batch"] = float("nan")
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with mock.patch.object(run, "source_contract", return_value=(
                    directory / "source.pt", manifest, {"steps": 256}, pool, ids, "d" * 64)), \
                 mock.patch.object(run.sw130, "validate_rgb_assets", return_value=assets), \
                 mock.patch.object(run, "implementation_fingerprint", return_value=fingerprint):
                self.assertFalse(queue.valid_result({"stage": "preflight", "seed": 0}, assets))
        finally:
            queue.ARCHIVE = original_archive
            report_path.unlink(missing_ok=True)
            warm.unlink(missing_ok=True)
            manifest.unlink(missing_ok=True)
            archive.rmdir()
            directory.rmdir()


if __name__ == "__main__":
    unittest.main()
