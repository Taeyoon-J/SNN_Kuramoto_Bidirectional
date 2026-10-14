from __future__ import annotations

import numpy as np
import unittest
import json
import shutil
import uuid
from pathlib import Path
from unittest import mock

import h5py

from collaborative_test.SW_0159_one_shot_event_regime import predict_evaluate as module


class SW0159PredictionTests(unittest.TestCase):
    def test_atomic_per_image_row_roundtrip_and_tamper_rejection(self):
        root = module.ROOT / f".test159row{uuid.uuid4().hex[:8]}"
        root.mkdir()
        try:
            labels = (np.arange(32 * 32, dtype=np.int64).reshape(32, 32) % 5)
            module._write_row(root, 0, 1320, "a" * 64, "b" * 64, labels)
            restored = module._load_row(root, 0, 1320, "a" * 64, "b" * 64)
            self.assertTrue(np.array_equal(restored, labels))
            with self.assertRaisesRegex(ValueError, "corrupt or misbound"):
                module._load_row(root, 0, 1320, "c" * 64, "b" * 64)
            row_npz = module._row_dir(root, 0) / "row.npz"
            with row_npz.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                module._load_row(root, 0, 1320, "a" * 64, "b" * 64)
        finally:
            shutil.rmtree(root)

    def test_score_producer_validates_predictions_before_gt_and_emits_actual_flags(self):
        root = module.ROOT / f".test159score{uuid.uuid4().hex[:8]}"
        root.mkdir()
        try:
            dataset = root / "train.h5"
            target_mask = np.zeros((128, 128, 1), dtype=np.uint8)
            target_mask[16:112, 16:112, 0] = 1
            with h5py.File(dataset, "w") as h5:
                ds = h5.create_dataset("mask", shape=(1640, 128, 128, 1), dtype=np.uint8,
                                       compression="gzip", chunks=(1, 128, 128, 1))
                ds[1320:1640] = np.repeat(target_mask[None], 320, axis=0)
            records = {}
            calls = []
            ids = module.IMAGE_IDS.copy()
            labels = np.zeros((module.COUNT, module.GRID, module.GRID), dtype=np.int64)
            labels[:, 4:28, 4:28] = 1
            stat = dataset.stat()
            for name in ("control", "candidate", "source97", "SW0143_T"):
                base = root / name
                base.mkdir()
                archive = base / "predictions.npz"
                protocol = base / "protocol.json"
                with archive.open("wb") as stream:
                    np.savez_compressed(stream, image_ids=ids, qcc_labels=labels)
                protocol.write_text(json.dumps({"name": name}), encoding="utf-8")
                row = {"ids": ids, "labels": labels.copy(),
                    "npz_path": archive, "protocol_path": protocol,
                    "prediction_sha256": module.sha256_file(archive),
                    "protocol_sha256": module.sha256_file(protocol)}
                if name in module.ARMS:
                    row["protocol"] = {"prediction_bindings": {
                        "dataset_path": str(dataset.resolve()), "dataset_size_bytes": stat.st_size,
                        "dataset_mtime_ns": stat.st_mtime_ns}}
                else:
                    row["archive_path"] = archive
                records[name] = row

            def validate(arm, **kwargs):
                calls.append(f"prediction:{arm}")
                return records[arm]

            def refs():
                calls.append("references")
                return records["source97"], records["SW0143_T"]

            real_h5_file = h5py.File
            def recording_h5(*args, **kwargs):
                calls.append("ground_truth_open")
                return real_h5_file(*args, **kwargs)

            out = root / "score.json"
            with mock.patch.object(module, "validate_prediction", side_effect=validate), \
                 mock.patch.object(module, "_source_and_t", side_effect=refs), \
                 mock.patch.object(module.h5py, "File", side_effect=recording_h5):
                report = module.score_all(output=out, dataset=dataset)
                self.assertEqual(calls[:4], ["prediction:control", "prediction:candidate",
                                             "references", "ground_truth_open"])
                self.assertTrue(report["ground_truth_opened_after_all_predictions_validated"])
                self.assertIs(report["ground_truth_used_for_prediction"], False)
                self.assertIs(report["ground_truth_used_for_scoring"], True)
                self.assertEqual(set(report["means"]), {"control", "candidate", "source97", "SW0143_T"})
                self.assertEqual(set(report["means"]["candidate"]), set(module.contract.METRICS))
                self.assertTrue(out.is_file())
                self.assertEqual(module.validate_score(out, dataset=dataset), report)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
