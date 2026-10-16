from __future__ import annotations

import unittest
import uuid
from pathlib import Path

import numpy as np

from collaborative_test.SW_0182_unique1024_actual_descent import data, schedule
from collaborative_test.SW_0179_qce_seed_replication import train as base


class DataExtensionTests(unittest.TestCase):
    def test_extension_preserves_prefix_bytes_and_appends_exact_modal_rows(self):
        prefix = {
            "rgb": np.zeros((512, 128, 128, 3), dtype=np.uint8),
            "modal_gt": np.zeros((512, 32, 32), dtype=np.int32),
        }
        prefix["rgb"][511, 0, 0, 0] = 199
        prefix["modal_gt"][511, 0, 0] = 6
        old_rgb = prefix["rgb"].tobytes()
        old_modal = prefix["modal_gt"].tobytes()
        added_rgb = np.full((512, 128, 128, 3), 17, dtype=np.uint8)
        added_modal = np.full((512, 32, 32), 4, dtype=np.int32)
        ids = np.arange(10000, 11024, dtype=np.int64)
        extended = data.extend_arrays(prefix, full_ids=ids,
                                      next_rgb=added_rgb, next_modal=added_modal)
        self.assertEqual(extended["rgb"][:512].tobytes(), old_rgb)
        self.assertEqual(extended["modal_gt"][:512].tobytes(), old_modal)
        self.assertTrue(np.all(extended["rgb"][512:] == 17))
        self.assertTrue(np.all(extended["modal_gt"][512:] == 4))
        data._validate_arrays(extended, ids)

    def test_rejects_reserved_ids_and_wrong_new_row_shape(self):
        prefix = {"rgb": np.zeros((512,128,128,3), np.uint8),
                  "modal_gt": np.zeros((512,32,32), np.int32)}
        ids = np.arange(10000, 11024, dtype=np.int64)
        ids[700] = 1320
        with self.assertRaisesRegex(ValueError, "validation or reserve"):
            data.extend_arrays(prefix, full_ids=ids,
                               next_rgb=np.zeros((512,128,128,3), np.uint8),
                               next_modal=np.zeros((512,32,32), np.int32))
        with self.assertRaisesRegex(ValueError, "RGB rows"):
            data.extend_arrays(prefix, full_ids=np.arange(10000,11024),
                               next_rgb=np.zeros((511,128,128,3), np.uint8),
                               next_modal=np.zeros((512,32,32), np.int32))

    def test_registered_schedule_keeps_first32_and_maps_secondpass(self):
        old = base._expected_batches()
        batches = schedule.expected_batches_1024()
        self.assertEqual(batches[:32], old[:32])
        self.assertEqual(batches[32:], [[index + 512 for index in row] for row in old[32:]])
        all_indices = [index for batch in batches for index in batch]
        self.assertEqual(len(all_indices), 1024)
        self.assertEqual(set(all_indices), set(range(1024)))

    def test_ids_require_full_unique_non_eval_source_order(self):
        ids = list(range(20000, 21024))
        self.assertEqual(schedule.validate_ordered_ids(ids), ids)
        with self.assertRaisesRegex(ValueError, "unique"):
            schedule.validate_ordered_ids(ids[:-1] + [ids[0]])
        with self.assertRaisesRegex(ValueError, "validation/reserve"):
            schedule.validate_ordered_ids(ids[:800] + [1320] + ids[801:])

    def test_completed_cache_is_reused_and_corruption_is_rejected(self):
        prefix = {"rgb": np.zeros((512,128,128,3), np.uint8),
                  "modal_gt": np.zeros((512,32,32), np.int32)}
        prefix["rgb"][0, 0, 0, 0] = 23
        prefix["modal_gt"][0, 0, 0] = 5
        ids = np.arange(30000, 31024, dtype=np.int64)
        arrays = data.extend_arrays(
            prefix, full_ids=ids,
            next_rgb=np.full((512,128,128,3), 1, np.uint8),
            next_modal=np.full((512,32,32), 2, np.int32))
        binding = {"prefix_512_manifest_sha256": "a" * 64,
                   "prefix_512_arrays_sha256": "b" * 64,
                   "source_seed": 1, "source_checkpoint_sha256": "c" * 64,
                   "source_manifest_sha256": "d" * 64,
                   "train_rgb_manifest_sha256": "e" * 64,
                   "source_pool_rows_sha256": "f" * 64,
                   "canonical_hdf5_path": "canonical-train.hdf5",
                   "canonical_hdf5_size_bytes": 100,
                   "canonical_hdf5_mtime_ns": 200,
                   "new_rows_start": 512, "new_rows_count": 512,
                   "new_image_ids": ids[512:].tolist(),
                   "modal_builder": "SW0155 objective.modal_train_targets32 on canonical HDF5 TRAIN masks",
                   "ground_truth_scope": "TRAIN modal32 only; no VAL or reserve data",
                   "gamma_or_spike_cache_created": False,
                   "feasible_projected_targets_created": False,
                   "optimizer_updates": 0, "validation_data_read": False,
                   "cuda_initialized": False}
        temp_root = data.ROOT / "collaborative_testing" / "temporary"
        temp_root.mkdir(parents=True, exist_ok=True)
        case_root = temp_root / ("sw0182-cache-" + uuid.uuid4().hex)
        case_root.mkdir(parents=False, exist_ok=False)
        if case_root.resolve().parent != temp_root.resolve():
            raise AssertionError("temporary cache fixture escaped central temporary root")
        try:
            output = case_root / "cache"
            expected_manifest = {"experiment": "SW0182_canonical_source_TRAIN1024_inputs",
                                 "status": "complete", "count": 1024,
                                 "image_ids": ids.tolist(), **binding}
            data._write_cache(output, arrays, expected_manifest)
            reused = data._read_cache(output, ids=ids, prefix=prefix,
                                      expected_binding=binding)
            self.assertTrue(np.array_equal(reused["arrays"]["rgb"], arrays["rgb"]))
            with (output / "arrays.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "arrays hash"):
                data._read_cache(output, ids=ids, prefix=prefix,
                                 expected_binding=binding)
        finally:
            if case_root.resolve().parent == temp_root.resolve():
                for child in case_root.iterdir():
                    if child.is_dir():
                        for leaf in child.iterdir():
                            leaf.unlink()
                        child.rmdir()
                    else:
                        child.unlink()
                case_root.rmdir()


if __name__ == "__main__":
    unittest.main()
