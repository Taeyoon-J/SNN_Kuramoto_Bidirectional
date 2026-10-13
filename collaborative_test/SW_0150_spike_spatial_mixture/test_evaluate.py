from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock
from uuid import uuid4

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import h5py

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0150_spike_spatial_mixture import evaluate


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _slot_protocol():
    return {
        "seed": 1, "image_ids": [1320, 1639], "count": 320,
        "inference_seed": 0, "batch_size": 1, "iterations": 3,
        "preprocessing": "full HDF5 RGB image, float32 /127.5 - 1; no crop",
        "resolution": [128, 128], "num_slots": 11,
        "checkpoint_sha256": contract.SLOT_CHECKPOINT_SHA256[1],
        "model_sha256": contract.SLOT_MODEL_SHA256,
        "ground_truth_used_for_prediction": False,
        "background_rule": "most hard-assigned one-pixel perimeter pixels; smallest slot ID wins ties",
        "label_rule": "slot ID + 1; selected background slot set to 0",
        "training_protocol": {
            "seed": 1, "unique_training_images": 70000, "epochs": 10,
            "batch_size": 32, "steps_per_epoch": 2188, "total_steps": 21880,
            "iterations": 3, "learning_rate": 0.0004,
            "selected_images_sha256": contract.SLOT_SELECTED_TRAINING_IDS_SHA256,
            "num_slots": 11, "ground_truth_used_for_training": False,
            "training_id_segments_inclusive": [[0, 999], [1640, 70639]],
        },
    }


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.base = ROOT / f".sw150_eval_test_{uuid4().hex}"
        self.base.mkdir()
        self.prediction_root = self.base / "predictions"
        self.prediction_root.mkdir()
        self.training_root = self.base / "training"
        self.training_root.mkdir()
        self.source_manifest = self.base / "source_manifest.json"
        _write_json(self.source_manifest, {"source": "fixture"})
        self.ids = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
        self.source_ids = list(range(1000, 5096))
        self.rows = np.arange(4096, dtype=np.int64)
        self.source_sha = "c" * 64
        self.assets = {"encoder_source_sha256": "d" * 64,
                       "feature_preprocessing_sha256": "e" * 64,
                       "train_cache_sha256": "f" * 64}
        self.source = {"checkpoint": self.base / "source.pt",
                       "manifest_path": self.source_manifest,
                       "manifest": {}, "rows": self.rows,
                       "ids": self.source_ids, "core_sha256": self.source_sha}
        self.fingerprint = {"fixture": "a" * 64}
        self.training_records = {}
        for arm in evaluate.ARMS:
            checkpoint = self.base / f"{arm}.pt"
            checkpoint.write_bytes(b"checkpoint-" + arm.encode())
            manifest = {"checkpoint_sha256": _sha(checkpoint),
                        "preflight_sha256": "1" * 64,
                        "preflight_warm_sha256": "2" * 64}
            self.training_records[arm] = {
                "manifest": manifest, "manifest_sha256": hashlib.sha256(
                    (arm + "-manifest").encode()).hexdigest(), "checkpoint": checkpoint,
            }
        # Fixed two-foreground-object target, already at the registered 32-grid.
        self.grid = np.zeros((32, 32), dtype=np.int64)
        self.grid[4:, :16] = 1
        self.grid[4:, 16:] = 2
        self.labels = np.broadcast_to(self.grid, (contract.COUNT, 32, 32)).copy()
        self.source_qcc_dir = self.base / "source_qcc"
        self.source_qcc_dir.mkdir()
        self.source_qcc_npz = self.source_qcc_dir / "predictions.npz"
        with self.source_qcc_npz.open("xb") as stream:
            np.savez_compressed(stream, image_ids=self.ids, qcc_labels=self.labels)
        self.source_qcc_protocol = self.source_qcc_dir / "protocol.json"
        _write_json(self.source_qcc_protocol, {"status": "fixture-source-qcc"})
        self.source_qcc_row = {
            "seed": 1, "ids": self.ids, "labels": self.labels,
            "protocol": {}, "archive_path": self.source_qcc_npz,
            "protocol_path": self.source_qcc_protocol,
            "prediction_sha256": _sha(self.source_qcc_npz),
            "protocol_sha256": _sha(self.source_qcc_protocol),
        }
        self._write_prediction("source_p", role="source_p", primary=self.labels,
                               qcc=self.labels, training=None)
        for arm in evaluate.ARMS:
            self._write_prediction(arm, role="trained", primary=self.labels,
                                   qcc=self.labels, training=self.training_records[arm])
        self.slot_root = self.base / "slot"
        slot_dir = self.slot_root / "seed1_epoch10"
        slot_dir.mkdir(parents=True)
        pixels = np.repeat(np.repeat(self.grid, 4, axis=0), 4, axis=1)
        pixels = np.broadcast_to(pixels, (contract.COUNT, 128, 128)).copy().astype(np.int64)
        self.slot_npz = slot_dir / "predictions.npz"
        with self.slot_npz.open("xb") as stream:
            np.savez_compressed(stream, labels=pixels, image_ids=self.ids)
        _write_json(slot_dir / "protocol.json", _slot_protocol())
        self.baseline = self.base / "slot_baseline.json"
        self.baseline_value = {
            "status": "complete", "comparison": "slot32 baseline only",
            "evaluation_contract": {"image_ids": [1320, 1639, 320],
                                    "native_image_size": [128, 128],
                                    "patch_grid": [32, 32], "patch_pixels": 4,
                                    "metrics": list(contract.METRICS)},
            "scores": {"slot32": {metric: {"three_seed_mean": 0.2}
                                   for metric in contract.METRICS}},
        }
        _write_json(self.baseline, self.baseline_value)
        self.dataset = self.base / "targets.h5"
        with h5py.File(self.dataset, "w") as h5:
            mask = h5.create_dataset("mask", shape=(1640, 128, 128), dtype="u1",
                                     chunks=(1, 128, 128), compression="gzip")
            pixels = np.repeat(np.repeat(self.grid.astype(np.uint8), 4, axis=0), 4, axis=1)
            mask[1320:1640] = np.broadcast_to(pixels, (320, 128, 128))
        self._patchers = [
            mock.patch.object(evaluate, "_source_record", return_value=self.source),
            mock.patch.object(evaluate.sw130, "validate_rgb_assets", return_value=self.assets),
            mock.patch.object(evaluate, "implementation_fingerprint", return_value=self.fingerprint),
            mock.patch.object(evaluate.train, "validate_training", side_effect=self._validate_training),
            mock.patch.object(evaluate.sw137, "validate_prediction", side_effect=self._validate_source_qcc),
            mock.patch.object(evaluate, "SLOT_ROOT", self.slot_root),
            mock.patch.object(evaluate, "SLOT_BASELINE", self.baseline),
            mock.patch.object(evaluate, "SLOT_BASELINE_SHA256", _sha(self.baseline)),
        ]
        for patcher in self._patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(getattr(self, "_patchers", [])):
            patcher.stop()
        root = self.base.resolve()
        if not root.is_relative_to(ROOT.resolve()):
            raise RuntimeError("refusing to remove evaluation fixture outside repository")
        for child in sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        root.rmdir()

    def _write_prediction(self, name, *, role, primary, qcc, training):
        directory = self.prediction_root / ("source_p_seed1" if name == "source_p"
                                            else f"{name}_seed1")
        directory.mkdir()
        npz = directory / "predictions.npz"
        with npz.open("xb") as stream:
            np.savez_compressed(stream, image_ids=self.ids,
                                primary_labels=primary, qcc_labels=qcc)
        protocol = evaluate._prediction_protocol(
            role, name, self.source, self.assets,
            trained=training, source_qcc=self.source_qcc_row if name == "source_p" else None)
        protocol.update({"implementation_fingerprint": self.fingerprint,
                         "implementation_sha256": hashlib.sha256(json.dumps(
                             self.fingerprint, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest(),
                         "prediction_sha256": _sha(npz)})
        _write_json(directory / "protocol.json", protocol)

    def _validate_training(self, arm, **_kwargs):
        return self.training_records[arm]

    def _validate_source_qcc(self, _directory, *, seed):
        if seed != 1:
            raise ValueError("fixture only registers source seed1")
        if (_sha(self.source_qcc_npz) != self.source_qcc_row["prediction_sha256"]
                or _sha(self.source_qcc_protocol) != self.source_qcc_row["protocol_sha256"]):
            raise ValueError("source fixture changed")
        return self.source_qcc_row

    def test_candidate_validator_rejects_wrong_ids_and_primary_label_range(self):
        path = self.prediction_root / "joint_seed1" / "predictions.npz"
        protocol_path = path.with_name("protocol.json")
        with path.open("wb") as stream:
            np.savez_compressed(stream, image_ids=np.arange(320),
                                primary_labels=self.labels, qcc_labels=self.labels)
        protocol = json.loads(protocol_path.read_text())
        protocol["prediction_sha256"] = _sha(path)
        _write_over(protocol_path, protocol)
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            evaluate.validate_prediction("joint", prediction_root=self.prediction_root,
                                         training_root=self.training_root)
        with path.open("wb") as stream:
            bad_primary = self.labels.copy()
            bad_primary[0, 0, 0] = 11
            np.savez_compressed(stream, image_ids=self.ids,
                                primary_labels=bad_primary, qcc_labels=self.labels)
        protocol["prediction_sha256"] = _sha(path)
        _write_over(protocol_path, protocol)
        with self.assertRaisesRegex(ValueError, "primary_labels"):
            evaluate.validate_prediction("joint", prediction_root=self.prediction_root,
                                         training_root=self.training_root)

    def test_candidate_validator_rejects_changed_source_training_lineage(self):
        path = self.prediction_root / "encoder_graph_seed1" / "protocol.json"
        protocol = json.loads(path.read_text())
        protocol["source_training_ids_sha256"] = "0" * 64
        _write_over(path, protocol)
        with self.assertRaisesRegex(ValueError, "invalid SW0150"):
            evaluate.validate_prediction("encoder_graph", prediction_root=self.prediction_root,
                                         training_root=self.training_root)

    def test_real_implementation_fingerprint_resolves_all_evaluation_paths(self):
        patcher = self._patchers[2]
        patcher.stop()
        try:
            fingerprint = evaluate.implementation_fingerprint()
        finally:
            patcher.start()
        self.assertIn("collaborative_test/SW_0150_spike_spatial_mixture/evaluate.py", fingerprint)
        self.assertIn("collaborative_test/SW_0150_spike_spatial_mixture/train.py", fingerprint)
        self.assertIn("collaborative_test/SW_0137_native32_source_qcc/run.py", fingerprint)

    def test_score_uses_production_metrics_and_freezes_all_predictions_before_gt(self):
        output = self.base / "score.json"
        original_file = evaluate.h5py.File

        def checked_file(path, *args, **kwargs):
            for name in (*evaluate.ARMS, "source_p"):
                directory = self.prediction_root / ("source_p_seed1" if name == "source_p"
                                                    else f"{name}_seed1")
                self.assertTrue((directory / "predictions.npz").is_file())
                self.assertTrue((directory / "protocol.json").is_file())
            self.assertTrue(self.source_qcc_npz.is_file())
            self.assertTrue(self.slot_npz.is_file())
            return original_file(path, *args, **kwargs)

        with mock.patch.object(evaluate.h5py, "File", side_effect=checked_file):
            report = evaluate.score_all(output=output, dataset=self.dataset,
                                        prediction_root=self.prediction_root,
                                        training_root=self.training_root)
        self.assertTrue(report["ground_truth_opened_after_all_predictions_validated"])
        self.assertEqual(report["scores"]["joint"]["primary"]["fg_ari"]["valid_count"], 320)
        self.assertEqual(report["bootstrap"]["seed"], 139)
        self.assertFalse(report["may_promote_seed_expansion"])
        self.assertTrue(output.is_file())

    def test_tampered_prediction_fails_before_ground_truth_file_is_opened(self):
        path = self.prediction_root / "joint_seed1" / "protocol.json"
        protocol = json.loads(path.read_text())
        protocol["checkpoint_sha256"] = "0" * 64
        _write_over(path, protocol)
        with mock.patch.object(evaluate.h5py, "File", side_effect=AssertionError("GT opened")):
            with self.assertRaisesRegex(ValueError, "training checkpoint lineage"):
                evaluate.score_all(output=self.base / "bad_score.json", dataset=self.dataset,
                                   prediction_root=self.prediction_root,
                                   training_root=self.training_root)


def _write_over(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    unittest.main()
