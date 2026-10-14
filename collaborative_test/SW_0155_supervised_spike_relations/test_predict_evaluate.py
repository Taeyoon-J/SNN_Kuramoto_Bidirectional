from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from collaborative_test.SW_0155_supervised_spike_relations import predict_evaluate as module


class SupervisedPredictionEvaluationTests(unittest.TestCase):
    def test_registered_fixed320_prediction_contract_and_no_gt(self):
        self.assertEqual(module.IMAGE_IDS.tolist(), list(range(1320, 1640)))
        self.assertEqual(module.ARMS, ("encoder_graph", "joint"))
        for arm in module.ARMS:
            protocol = {"arm": arm, "seed": 1, "count": 320, "grid_size": [32, 32],
                        "steps": 1024, "settle": 512, "live_tail": 0,
                        "ground_truth_used_for_prediction": False}
            self.assertFalse(protocol["ground_truth_used_for_prediction"])
            self.assertEqual(protocol["seed"], module.SEED)
        with self.assertRaises(ValueError):
            module.predict("source_p")

    def test_prediction_validator_rejects_tampered_protocol_before_loading_labels(self):
        with mock.patch.object(Path, "read_bytes", return_value=b'{"status":"wrong"}'):
            with self.assertRaises((ValueError, KeyError, OSError)):
                module.validate_prediction("encoder_graph", prediction_root=Path("missing"))

    def test_bootstrap_uses_common_fixed320_image_indices(self):
        a = np.arange(320, dtype=np.float64)
        b = a - 0.25
        result = module._paired_bootstrap(a, b)
        self.assertAlmostEqual(result["mean"], 0.25)
        self.assertAlmostEqual(result["ci95"][0], 0.25)
        self.assertAlmostEqual(result["ci95"][1], 0.25)
        with self.assertRaises(ValueError):
            module._paired_bootstrap(a[:319], b[:319])

    def test_gt_is_opened_after_all_predictions_and_references_validate(self):
        events = []
        ids = module.IMAGE_IDS.copy()
        labels = np.zeros((320, 32, 32), dtype=np.int64)
        dataset = module.HERE / "results_archive" / f"synthetic_dataset_{os.getpid()}.h5"
        stat = type("Stat", (), {"st_size": 1234, "st_mtime_ns": 5678})()

        def prediction(arm, **_kwargs):
            events.append(f"pred:{arm}")
            return {"ids": ids, "labels": labels, "npz_path": Path(f"{arm}.npz"),
                    "protocol_path": Path(f"{arm}.json"), "prediction_sha256": "a" * 64,
                    "protocol_sha256": "b" * 64, "protocol": {
                        "training_manifest_sha256": "c" * 64,
                        "training_checkpoint_sha256": "d" * 64,
                        "training_hdf5_path": str(dataset.resolve()),
                        "training_hdf5_size": stat.st_size,
                        "training_hdf5_mtime_ns": stat.st_mtime_ns}}

        source = {"ids": ids, "labels": labels, "archive_path": Path("source.npz"),
                  "protocol_path": Path("source.json"), "prediction_sha256": "e" * 64,
                  "protocol_sha256": "f" * 64}
        incumbent = {"ids": ids, "labels": labels, "npz_path": Path("T.npz"),
                     "protocol_path": Path("T.json"), "prediction_sha256": "1" * 64,
                     "protocol_sha256": "2" * 64}

        class MaskDataset:
            @property
            def shape(self):
                return (100000, 128, 128, 1)

            def __getitem__(self, key):
                if isinstance(key, slice):
                    events.append("gt-read")
                    return np.zeros((320, 128, 128, 1), dtype=np.uint8)
                raise KeyError(key)

        class H5:
            def __contains__(self, name):
                return name == "mask"

            def __getitem__(self, name):
                if name == "mask":
                    return MaskDataset()
                raise KeyError(name)

        class Context:
            def __enter__(self):
                events.append("h5-open")
                return H5()

            def __exit__(self, *_args):
                return False

        output = module.HERE / "results_archive" / f"eval_gt_order_test_{os.getpid()}.json"
        if output.exists():
            self.fail("preserve pre-existing test output")
        known_hashes = {"encoder_graph.npz": "a" * 64, "encoder_graph.json": "b" * 64,
            "joint.npz": "a" * 64, "joint.json": "b" * 64,
            "source.npz": "e" * 64, "source.json": "f" * 64,
            "T.npz": "1" * 64, "T.json": "2" * 64}

        def fake_hash(path):
            return known_hashes.get(Path(path).name, hashlib_stub(str(path)))

        original_stat = Path.stat
        def fake_stat(path, *args, **kwargs):
            if Path(path) == dataset:
                return stat
            return original_stat(path, *args, **kwargs)

        def metrics(_labels, _gt):
            events.append("metrics")
            return {name: {"mean": 0.1, "per_image": [0.1] * 320}
                    for name in ("fg_ari", "foreground_iou", "matched_object_iou")}

        try:
            with mock.patch.object(module, "validate_prediction", side_effect=prediction), \
                 mock.patch.object(module, "_validate_source_prediction", side_effect=lambda: events.append("source") or source), \
                 mock.patch.object(module, "_validate_t_prediction", side_effect=lambda: events.append("T") or incumbent), \
                 mock.patch.object(module, "sha256_file", side_effect=fake_hash), \
                 mock.patch.object(module, "_source_lineage", return_value=({}, {}, source)), \
                 mock.patch.object(module.Path, "is_file", return_value=True), \
                 mock.patch.object(module.Path, "stat", fake_stat), \
                 mock.patch.object(module.h5py, "File", return_value=Context()), \
                 mock.patch.object(module.contract, "modal_native32", return_value=labels), \
                 mock.patch.object(module.contract, "production_metrics32", side_effect=metrics):
                record = module.score_all(output=output, dataset=dataset)
            first_gt = events.index("gt-read")
            self.assertTrue(all(events.index(f"pred:{arm}") < first_gt for arm in module.ARMS))
            self.assertLess(events.index("source"), first_gt)
            self.assertLess(events.index("T"), first_gt)
            self.assertTrue(record["ground_truth_opened_after_all_predictions_validated"])
            self.assertFalse(record["ground_truth_used_for_prediction"])
            self.assertTrue(record["ground_truth_used_for_scoring"])
            self.assertIn("not a supervised Slot comparison", record["scientific_limits"])
        finally:
            if output.exists():
                output.unlink()


def hashlib_stub(text):
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
