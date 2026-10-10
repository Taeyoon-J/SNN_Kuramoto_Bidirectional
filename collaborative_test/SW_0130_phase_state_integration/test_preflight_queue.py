"""Preflight-only queue ordering and evidence-validation regressions."""
import copy
import hashlib
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np  # import before torch in the local Windows environment

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0130_phase_state_integration import preflight_queue as queue


def passing_report(seed, warmup_path, fingerprint, source_sha, source_manifest_sha,
                   ids, pool_sha, assets, lambda_value=12.5, seed0_sha=None):
    checks = [
        {"arm": arm, "time_steps": steps, "settle": settle,
         "theta_component_membrane_spikes_q_h_primary_oldloss_exact": True}
        for arm in queue.REQUIRED_ARMS for steps, settle in ((64, 32), (1024, 512))
    ]
    disposable = []
    norms = {name: 1.0 for name in queue.REQUIRED_FAMILIES}
    for arm in queue.REQUIRED_ARMS:
        disposable.append({
            "arm": arm, "optimizer_updates": 1, "throwaway_only": True,
            "rgb_gradient_norms_by_family": dict(norms),
            "joint_gradient_norm": 1.0, "decoder_gradient_norm": 1.0,
            "changed_native_core_parameter_count": 4,
            "changed_encoder_parameter_count": 2,
            "changed_decoder_parameter_count": 3,
            "changed_integration_parameters": ["b"],
        })
    return {
        "status": "passed", "experiment": "SW0130_phase_state_integration",
        "seed": seed, "arm": "phase", "implementation_fingerprint": fingerprint,
        "source_core_sha256": source_sha, "source_manifest_sha256": source_manifest_sha,
        "source_updates": 256, "matched_training_ids": ids,
        "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "pool_indices_sha256": pool_sha, "shuffle_seed": 117 + seed,
        "batch_size": 16, "updates": 256, "train_time_steps": 64,
        "settle_steps": 32, "asset_hashes": assets,
        "ground_truth_used": False, "original_core_checkpoint_modified": False,
        "registered_gamma_cache_max_abs_diff_first_batch": 1e-6,
        "row_scramble_count": 64, "row_scramble_positive_count": 35,
        "row_scramble_mean_excess": 1e-5,
        "decoder_warmup_artifact": str(warmup_path.resolve()),
        "decoder_warmup_artifact_sha256": "warm-sha",
        "native_zero_initial_parity": {"source_core_sha256": source_sha, "checks": checks},
        "disposable_update_finite": True, "disposable_updates": disposable,
        "lambda_source_seed": 0, "lambda": lambda_value,
        "seed0_lambda_record_sha256": seed0_sha,
        "lambda_seed0_calibration": ([
            {"batch": index, "old_joint_gradient_norm": 50.0,
             "rgb_joint_gradient_norm": 1.0, "ratio": 12.5}
            for index in range(4)] if seed == 0 else
            "reused immutable seed0 preflight calibration"),
    }


class PreflightQueueTests(unittest.TestCase):
    def test_fixed_three_task_plan_orders_seed0_dependency_and_has_no_training(self):
        tasks = queue.task_plan()
        self.assertEqual([row["seed"] for row in tasks], [0, 1, 2])
        self.assertEqual(tasks[0]["depends_on"], [])
        self.assertEqual(tasks[1]["depends_on"], [tasks[0]["task_id"]])
        self.assertEqual(tasks[2]["depends_on"], [tasks[0]["task_id"]])
        self.assertTrue(all(row["stage"] == "preflight" for row in tasks))
        self.assertEqual(len(queue.command(tasks[0])), 10)

    def test_gpu_requires_no_compute_owner_and_at_most_registered_memory(self):
        self.assertTrue(queue.gpu_is_exclusive_candidate(512, []))
        self.assertFalse(queue.gpu_is_exclusive_candidate(513, []))
        self.assertFalse(queue.gpu_is_exclusive_candidate(16, [1234]))

    def test_full_preflight_contract_accepts_and_rejects_family_or_history_gaps(self):
        ids = list(range(4096))
        fp, source_sha, manifest_sha = {"run.py": "fingerprint"}, "source", "manifest"
        pool_sha, assets = "pool", {"cache": "cache-sha"}
        temp = queue.HERE / f"queue_test_{uuid.uuid4().hex[:8]}"
        temp.mkdir()
        warmup = temp / "warmup.pt"
        try:
            warmup.write_bytes(b"warmup")
            report = passing_report(0, warmup, fp, source_sha, manifest_sha,
                                    ids, pool_sha, assets)
            kwargs = dict(fingerprint=fp, source_sha=source_sha,
                          source_manifest_sha=manifest_sha, training_ids=ids,
                          pool_sha=pool_sha, asset_hashes=assets, warmup_path=warmup,
                          warmup_sha_fn=lambda path: "warm-sha")
            self.assertTrue(queue.validate_preflight_record(report, 0, **kwargs))
            bad_family = copy.deepcopy(report)
            bad_family["disposable_updates"][0]["rgb_gradient_norms_by_family"]["b"] = 0.0
            self.assertFalse(queue.validate_preflight_record(bad_family, 0, **kwargs))
            bad_parity = copy.deepcopy(report)
            bad_parity["native_zero_initial_parity"]["checks"].pop()
            self.assertFalse(queue.validate_preflight_record(bad_parity, 0, **kwargs))
        finally:
            warmup.unlink(missing_ok=True)
            temp.rmdir()

    def test_later_seeds_require_exact_seed0_record_hash_and_fixed_lambda(self):
        ids = list(range(4096))
        fp, source_sha, manifest_sha = {"run.py": "fingerprint"}, "source", "manifest"
        pool_sha, assets = "pool", {"cache": "cache-sha"}
        temp = queue.HERE / f"queue_test_{uuid.uuid4().hex[:8]}"
        temp.mkdir()
        warmup = temp / "warmup.pt"
        try:
            warmup.write_bytes(b"warmup")
            seed0 = passing_report(0, warmup, fp, source_sha, manifest_sha,
                                   ids, pool_sha, assets)
            seed1 = passing_report(1, warmup, fp, source_sha, manifest_sha,
                                   ids, pool_sha, assets, seed0_sha="seed0-record-sha")
            kwargs = dict(fingerprint=fp, source_sha=source_sha,
                          source_manifest_sha=manifest_sha, training_ids=ids,
                          pool_sha=pool_sha, asset_hashes=assets, warmup_path=warmup,
                          seed0_report=seed0, seed0_report_sha="seed0-record-sha",
                          warmup_sha_fn=lambda path: "warm-sha")
            self.assertTrue(queue.validate_preflight_record(seed1, 1, **kwargs))
            changed = dict(kwargs, seed0_report_sha="changed-record-sha")
            self.assertFalse(queue.validate_preflight_record(seed1, 1, **changed))
            wrong_lambda = copy.deepcopy(seed1)
            wrong_lambda["lambda"] += 1.0
            self.assertFalse(queue.validate_preflight_record(wrong_lambda, 1, **kwargs))
        finally:
            warmup.unlink(missing_ok=True)
            temp.rmdir()


if __name__ == "__main__":
    unittest.main()
