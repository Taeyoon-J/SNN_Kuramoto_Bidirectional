"""Preflight artifact, seed-0 lambda and queue-order contract checks."""
import hashlib
import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0132_partition_relative_rgb import preflight_queue as queue
from collaborative_test.SW_0132_partition_relative_rgb import run


class PreflightQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = run.HERE / ("test-artifacts-" + uuid.uuid4().hex[:10])
        self.tmp.mkdir()

    def tearDown(self):
        archive = self.tmp / "archive"
        if archive.exists():
            for filename in ("preflight_seed0.json", "preflight_decoder_seed0.pt"):
                path = archive / filename
                if path.exists():
                    path.unlink()
            archive.rmdir()
        for path in list(self.tmp.iterdir()):
            if path.is_file():
                path.unlink()
        self.tmp.rmdir()

    def test_plan_seeds_dependencies_and_gpu_exclusivity(self):
        tasks = queue.task_plan()
        self.assertEqual([row["seed"] for row in tasks], [0, 1, 2])
        self.assertEqual(tasks[0]["depends_on"], [])
        self.assertEqual(tasks[1]["depends_on"], [tasks[0]["task_id"]])
        self.assertEqual(tasks[2]["depends_on"], [tasks[0]["task_id"]])
        self.assertTrue(queue.gpu_is_exclusive_candidate(512, []))
        self.assertFalse(queue.gpu_is_exclusive_candidate(513, []))
        self.assertFalse(queue.gpu_is_exclusive_candidate(0, [123]))

    def test_warm_optimizer_requires_exact_32_step_state(self):
        valid = {"optimizer_state_dict": {"state": {
            i: {"step": torch.tensor(32.)} for i in range(6)}}}
        run._validate_warm_optimizer(valid)
        invalid = {"optimizer_state_dict": {"state": {
            i: {"step": torch.tensor(31. if i == 0 else 32.)} for i in range(6)}}}
        with self.assertRaisesRegex(ValueError, "step-32"):
            run._validate_warm_optimizer(invalid)

    def test_seed0_lambda_requires_matching_source_assets_and_four_batch_calibration(self):
        ids = list(range(4096))
        source_checkpoint = self.tmp / "source.pt"
        manifest = self.tmp / "source.json"
        manifest.write_text("{}", encoding="utf-8")
        warm = self.tmp / "warm.pt"
        warm.write_bytes(b"registered warm artifact")
        assets = {"cache": "abc"}
        ratios = [0.5, 1.0, 1.5, 2.0]
        calibration = [{"batch_index": i, "lambda_ratio": value,
                        "old_joint_norm": 4 * value, "rgb_joint_norm": 1.0}
                       for i, value in enumerate(ratios)]
        source_sha = run.source97.EXPECTED_SOURCE_SHAS[0]
        record = {"status": "passed", "seed": 0, "lambda_source_seed": 0,
                  "implementation_fingerprint": {"run": "fp"},
                  "source_core_sha256": source_sha, "source_manifest_sha256": run.sha(manifest),
                  "training_ids": ids,
                  "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                  "asset_hashes": assets, "lambda_batch_ratios": ratios,
                  "lambda_seed0_calibration": calibration, "lambda": 1.25,
                  "decoder_warmup_artifact": str(warm),
                  "decoder_warmup_artifact_sha256": run.sha(warm)}
        record_path = self.tmp / "preflight_seed0.json"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        source_tuple = (source_checkpoint, manifest, {}, np.arange(4096), ids, source_sha)
        with mock.patch.object(run, "source_contract", return_value=source_tuple), \
                mock.patch.object(run, "implementation_fingerprint", return_value={"run": "fp"}):
            value, record_sha = run._seed0_lambda_record(self.tmp / "preflight_seed1.json", 1,
                                                         ids, assets)
            self.assertEqual(value, 1.25)
            self.assertEqual(record_sha, run.sha(record_path))
            record["implementation_fingerprint"] = {"run": "stale"}
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                run._seed0_lambda_record(self.tmp / "preflight_seed1.json", 1, ids, assets)

    def test_valid_result_exercises_artifact_validator_and_seed0_schema(self):
        archive = self.tmp / "archive"
        archive.mkdir()
        old_archive = queue.ARCHIVE
        queue.ARCHIVE = archive
        try:
            ids = list(range(4096))
            pool = np.arange(4096, dtype=np.int64)
            manifest = self.tmp / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            source = self.tmp / "source.pt"
            source.write_bytes(b"source")
            sha_source = "source-core-sha"
            assets = {"train": "cache-sha"}
            fp = {"run": "current-fingerprint"}
            ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
            pool_sha = hashlib.sha256(np.asarray(pool, dtype="<i8").tobytes()).hexdigest()
            warm_path = archive / "preflight_decoder_seed0.pt"
            warm = {"warmup_updates": 32, "source_core_sha256": sha_source,
                    "training_ids_sha256": ids_sha, "asset_hashes": assets,
                    "decoder_state_dict": {},
                    "optimizer_state_dict": {"state": {
                        i: {"step": torch.tensor(32.)} for i in range(6)}}}
            torch.save(warm, warm_path)
            ratios = [1., 1., 1., 1.]
            calibration = [{"batch_index": i, "old_joint_norm": 4.,
                            "rgb_joint_norm": 1., "lambda_ratio": 1.}
                           for i in range(4)]
            report = {
                "status": "passed", "experiment": "SW0132_partition_relative_rgb",
                "seed": 0, "implementation_fingerprint": fp,
                "source_core_sha256": sha_source, "source_manifest_sha256": run.sha(manifest),
                "source_steps": 256, "training_ids": ids, "training_ids_sha256": ids_sha,
                "pool_indices_sha256": pool_sha, "shuffle_seed": 117, "batch_size": 16,
                "train_time_steps": 64, "settle_steps": 32, "asset_hashes": assets,
                "ground_truth_used": False, "optimizer_updates": 0,
                "source_core_checkpoint_modified": False, "source_encoder_modified": False,
                "live_source_gamma_max_abs_diff_first_batch": 1e-6,
                "row_scramble_count_per_arm": 64,
                "row_scramble_by_arm": {arm: {"per_image_excess": [.01] * 64,
                    "mean_excess": .01, "positive_count": 64} for arm in ("phase", "constant")},
                "paired_zero_initial_arm_check": {"initial_decoder_equal": True,
                    "initial_gamma_q_hard_traces_equal": True,
                    "warm_decoder_optimizer_shared_across_arms": True},
                "decoder_warmup_artifact": str(warm_path.resolve()),
                "decoder_warmup_artifact_sha256": run.sha(warm_path),
                "native_zero_initial_parity": {"source_core_sha256": sha_source,
                    "checks": [{"arm": arm, "time_steps": t, "settle": s,
                        "theta_component_membrane_spikes_q_h_primary_oldloss_exact": True}
                        for arm in ("phase", "constant") for t, s in ((64, 32), (1024, 512))]},
                "disposable_updates": [], "lambda": 1., "lambda_source_seed": 0,
                "seed0_lambda_record_sha256": None, "lambda_batch_ratios": ratios,
                "lambda_seed0_calibration": calibration,
            }
            for arm in ("phase", "constant"):
                report["disposable_updates"].append({
                    "arm": arm, "optimizer_updates": 1, "throwaway_only": True,
                    "rgb_gradient_norms_by_family": {family: 1. for family in queue.FAMILIES},
                    "joint_and_decoder_changed": True, "changed_encoder_parameter_count": 1,
                    "changed_decoder_parameter_count": 1,
                    "changed_core_families": {name: True for name in
                        ("graph", "oscillator_drive", "kuramoto", "dendrite", "membrane")},
                    "changed_integration_parameters": ["b"],
                    "integration_parameter_gradient_abs_by_component": {
                        name: [1.] * 4 for name in ("core.a_d", "core.a_m", "core.b")}})
            result_path = archive / "preflight_seed0.json"
            result_path.write_text(json.dumps(report), encoding="utf-8")
            fake_contract = (source, manifest, {"steps": 256}, pool, ids, sha_source)
            with mock.patch.object(queue.sw132, "source_contract", return_value=fake_contract), \
                    mock.patch.object(queue.sw132, "implementation_fingerprint", return_value=fp), \
                    mock.patch.object(queue.sw132.sw130, "validate_rgb_assets", return_value=assets):
                self.assertTrue(queue.valid_result({"stage": "preflight", "seed": 0}, assets))
                report["disposable_updates"][0]["changed_core_families"] = {}
                result_path.write_text(json.dumps(report), encoding="utf-8")
                self.assertFalse(queue.valid_result({"stage": "preflight", "seed": 0}, assets))
        finally:
            queue.ARCHIVE = old_archive


if __name__ == "__main__":
    unittest.main()
