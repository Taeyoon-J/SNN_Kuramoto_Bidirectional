import numpy as np  # NumPy must precede torch on this Windows runtime.
import contextlib
import io
import json
import hashlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from collaborative_test.SW_0115_analytic_partition_rgb import coordinator, run


class FakeRGBCache:
    def __getitem__(self, rows):
        return np.full((len(rows), 128, 128, 3), 255, dtype=np.uint8)


class RegisteredRunnerTests(unittest.TestCase):
    def test_seed0_task_plan_has_two_matched_arms_and_ordered_dependencies(self):
        tasks = coordinator.task_plan()
        self.assertEqual(len(tasks), 6)
        by_id = {task["task_id"]: task for task in tasks}
        for arm in run.ARMS:
            pre = f"sw0115_preflight_s0_{arm}"
            train = f"sw0115_train_s0_{arm}"
            evaluate = f"sw0115_eval_s0_{arm}"
            self.assertEqual(by_id[train]["depends_on"], [pre])
            self.assertEqual(by_id[evaluate]["depends_on"], [train])
            self.assertEqual(by_id[train]["seed"], 0)
        self.assertTrue(all(task["seed"] == 0 for task in tasks))

    def test_cli_emits_parseable_success_json_after_preflight(self):
        old_argv = run.sys.argv
        output = io.StringIO()
        try:
            run.sys.argv = ["run.py", "preflight", "--seed", "0", "--arm", "control",
                            "--device", "cpu", "--output", "unused.json"]
            with mock.patch.object(run, "preflight", return_value={"status": "passed"}), \
                    contextlib.redirect_stdout(output):
                run.main()
        finally:
            run.sys.argv = old_argv
        self.assertEqual(json.loads(output.getvalue()), {
            "status": "passed", "command": "preflight", "seed": 0, "arm": "control"
        })

    def test_pool_row_mapping_matches_registered_train_rgb_order(self):
        ids = np.array([0, 999, 1640, 3139, 70639], dtype=np.int64)
        rows = run.base.gamma_rows(ids)
        self.assertEqual(rows.tolist(), [0, 999, 1000, 2499, 69999])

    def test_source_manifest_requires_registered_budget_seed_and_ids(self):
        ids = np.concatenate((np.arange(1000), np.arange(1640, 4736))).astype(int).tolist()
        record = {"status": "complete", "unique_images_seen": 4096, "steps": 256,
                  "batch": 16, "seed": 117, "train_steps": 64, "train_settle": 32,
                  "training_ids": ids, "ground_truth_used_for_training": False}
        run.validate_source_manifest(0, record, np.asarray(ids, dtype=np.int64))
        bad = dict(record, train_settle=31)
        with self.assertRaises(AssertionError):
            run.validate_source_manifest(0, bad, np.asarray(ids, dtype=np.int64))

    def test_rgb_cache_manifest_binds_pool_mapping_and_source_stat(self):
        ids = np.concatenate((np.arange(1000, dtype="<i8"),
                              np.arange(1640, 70640, dtype="<i8")))
        source = run.ROOT / "fake_cache_source.hdf5"
        meta = {"status": "complete", "kind": "train", "cache_shape": [70000, 128, 128, 3],
                "cache_dtype": "uint8", "ids_mapping_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                "source_path": str(source), "source_size_bytes": 111,
                "source_mtime_ns": 222, "source_image_shape": [100000, 128, 128, 3],
                "source_image_dtype": "uint8", "blocks": [{"count": 70000}]}
        run.validate_rgb_cache_metadata(meta, source, 111, 222)
        with self.assertRaises(AssertionError):
            run.validate_rgb_cache_metadata(meta, source, 111, 223)

    def test_rgb_block_ledger_checks_order_coverage_source_chunk_and_bytes(self):
        cache = np.arange(6, dtype=np.uint8)
        expected_ids = np.arange(6, dtype=np.int64)
        first, second = cache[:3], cache[3:]
        blocks = [
            {"pool_start": 0, "source_id_start": 0, "count": 3,
             "source_chunk": [0, 3], "sha256": hashlib.sha256(memoryview(first).cast("B")).hexdigest()},
            {"pool_start": 3, "source_id_start": 3, "count": 3,
             "source_chunk": [3, 6], "sha256": hashlib.sha256(memoryview(second).cast("B")).hexdigest()},
        ]
        self.assertEqual(run.audit_cache_block_ledger(cache, blocks, expected_ids),
                         hashlib.sha256(memoryview(cache).cast("B")).hexdigest())
        corrupt = [dict(row) for row in blocks]
        corrupt[1]["sha256"] = "0" * 64
        with self.assertRaises(AssertionError):
            run.audit_cache_block_ledger(cache, corrupt, expected_ids)
        gap = [dict(row) for row in blocks]
        gap[1]["pool_start"] = 4
        with self.assertRaises(AssertionError):
            run.audit_cache_block_ledger(cache, gap, expected_ids)

    def test_rgb_block_ledger_rejects_wrong_source_mapping(self):
        cache = np.arange(3, dtype=np.uint8)
        digest = hashlib.sha256(memoryview(cache).cast("B")).hexdigest()
        blocks = [{"pool_start": 0, "source_id_start": 10, "count": 3,
                   "source_chunk": [10, 13], "sha256": digest}]
        with self.assertRaises(AssertionError):
            run.audit_cache_block_ledger(cache, blocks, np.arange(3, dtype=np.int64))

    def test_npy_storage_size_includes_real_header_offset(self):
        array = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        buffer = io.BytesIO()
        np.save(buffer, array)
        physical_size = buffer.tell()
        buffer.seek(0)
        version = np.lib.format.read_magic(buffer)
        self.assertEqual(version, (1, 0))
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(buffer)
        offset = buffer.tell()
        mmap_contract = SimpleNamespace(shape=shape, dtype=dtype, offset=offset, nbytes=array.nbytes)
        fake_path = SimpleNamespace(stat=lambda: SimpleNamespace(st_size=physical_size))
        run.validate_npy_storage(fake_path, mmap_contract, array.shape, np.uint8)
        self.assertGreater(offset, 0)
        bad_path = SimpleNamespace(stat=lambda: SimpleNamespace(st_size=physical_size - 1))
        with self.assertRaises(AssertionError):
            run.validate_npy_storage(bad_path, mmap_contract, array.shape, np.uint8)

    def test_native_uint8_conversion_occurs_once_before_patch_means(self):
        means = run.read_rgb(FakeRGBCache(), np.arange(16), "cpu")
        self.assertEqual(tuple(means.shape), (16, 256, 3))
        self.assertTrue(torch.equal(means, torch.ones_like(means)))

    def test_evaluation_validator_rejects_nonfinite_metric_entries(self):
        good = {"valid_count": {}, "per_image": {}, "mean": {}}
        for key in ("fg_ari", "foreground_iou", "matched_object_iou"):
            good["valid_count"][key] = 320
            good["per_image"][key] = [0.5] * 320
        good["metrics"] = {key: 0.5 for key in good["per_image"]}
        self.assertTrue(coordinator._finite_metrics(good))
        good["per_image"]["fg_ari"][17] = float("nan")
        self.assertFalse(coordinator._finite_metrics(good))

    def test_training_command_targets_manifest_parent_directory(self):
        task = next(t for t in coordinator.task_plan() if t["stage"] == "train")
        args = coordinator.command(task, "cuda:0")
        self.assertEqual(args[args.index("--output") + 1], str(coordinator.artifact_path(task).parent))
        self.assertTrue(coordinator.artifact_path(task).name == "manifest.json")

    def test_training_validator_binds_existing_preflight_sha(self):
        task = next(t for t in coordinator.task_plan() if t["stage"] == "train")
        with mock.patch.object(coordinator.Path, "is_file", return_value=True), \
                mock.patch.object(coordinator, "sha", return_value="registered-preflight-sha"):
            self.assertTrue(coordinator.training_preflight_matches(
                task, {"preflight_sha256": "registered-preflight-sha"}))
            self.assertFalse(coordinator.training_preflight_matches(
                task, {"preflight_sha256": "different-sha"}))
        with mock.patch.object(coordinator.Path, "is_file", return_value=False):
            self.assertFalse(coordinator.training_preflight_matches(
                task, {"preflight_sha256": "registered-preflight-sha"}))

    def test_scramble_aggregation_uses_per_image_mean_not_extra_batch_division(self):
        q = torch.zeros(2, 256, 256)
        h = [torch.ones(256, 1), torch.ones(256, 1)]
        rgb = torch.stack((torch.linspace(0, 1, 256)[:, None].expand(-1, 3),
                           torch.zeros(256, 3)))
        _, real_per, _, _ = run.batch_reconstruction_loss(q, h, rgb)
        scrambled_h = [torch.eye(256), torch.eye(256)]
        _, scrambled_per, _, _ = run.batch_reconstruction_loss(q, scrambled_h, rgb)
        expected = float((scrambled_per - real_per).mean())
        self.assertAlmostEqual(run.mean_scramble_excess(real_per.tolist(), scrambled_per.tolist()), expected)
        self.assertAlmostEqual(expected,
                               float(scrambled_per.mean() - real_per.mean()))


if __name__ == "__main__":
    unittest.main()
