from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest
import uuid
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0135_native32_spike_binding.evaluation_contract32 import (
    COUNT, GRID, IMAGE_IDS, METRICS, modal_native32, production_metrics32,
)
from collaborative_test.SW_0140_native32_causal_screen import evaluate

ASSETS = {"encoder_source_sha256": "encoder-sha",
          "feature_preprocessing_sha256": "preprocess-sha",
          "train_cache_sha256": "train-cache-sha",
          "train_manifest_sha256": "train-manifest-sha",
          "validation_cache_sha256": "val-cache-sha",
          "validation_manifest_sha256": "val-manifest-sha"}


def _prediction(seed, condition, labels, root):
    directory = root / f"seed{seed}" / condition
    directory.mkdir(parents=True)
    npz, protocol = directory / "predictions.npz", directory / "protocol.json"
    with npz.open("xb") as stream:
        np.savez_compressed(stream, image_ids=np.asarray(IMAGE_IDS),
                            labels=labels, qcc_labels=labels)
    data = {
        "experiment": "SW0140_native32_causal_endpoint", "status": "predictions_complete",
        "seed": seed, "condition": condition, "image_ids": [1320, 1639],
        "count": 320, "grid_size": [32, 32], "prediction_shape": [320, 32, 32],
        "steps": 1024, "settle": 512, "microbatch_size": 1,
        "readout": {"affinity": "actual-spike positive signed component Pearson product",
                    "threshold": 0.50, "min_group_size": 8,
                    "background": "largest connected component"},
        "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
        "source_core_sha256": "core-sha", "source_manifest_sha256": "manifest-sha",
        "encoder_sha256": "encoder-sha", "feature_preprocessing_sha256": "preprocess-sha",
        "rgb_assets": ASSETS,
        "screen_record_sha256": f"screen{seed}",
        "screen_implementation_fingerprint": {"screen.py": "fp"},
        "all_screen_records_sha256": {"0": "screen0", "1": "screen1", "2": "screen2"},
        "implementation": evaluate.implementation_fingerprint(),
        "prediction_sha256": evaluate.sha256_file(npz),
    }
    protocol.write_text(json.dumps(data), encoding="utf-8")
    return npz, protocol, data


class _FakeH5:
    def __init__(self, mask):
        self.values = {"mask": mask}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __contains__(self, key):
        return key in self.values

    def __getitem__(self, key):
        return self.values[key]


class _MaskSlice:
    shape = (1640, 128, 128, 1)

    def __getitem__(self, item):
        # Give every fixed image a small foreground object so all production
        # metrics have the registered full valid count.
        result = np.zeros((COUNT, 128, 128, 1), dtype=np.uint8)
        result[:, 24:88, 32:96, 0] = 1
        return result


class SW0140EvaluationTests(unittest.TestCase):
    def test_prediction_archive_validator_binds_condition_and_fixed_ids(self):
        root = ROOT / f"_sw0140_eval_test_{uuid.uuid4().hex}"
        root.mkdir()
        try:
            labels = np.zeros((COUNT, GRID, GRID), dtype=np.int64)
            npz, protocol_path, protocol = _prediction(1, "spike_reset_off", labels, root)
            screen_rows = {seed: {"record": {"implementation_fingerprint": protocol[
                "screen_implementation_fingerprint"]}, "sha256": f"screen{seed}"}
                for seed in evaluate.SEEDS}
            source = (Path("core"), Path("manifest"), {}, [], [], "core-sha")
            with (mock.patch.object(evaluate, "implementation_fingerprint", return_value=protocol["implementation"]),
                  mock.patch.object(evaluate, "validate_screen", side_effect=lambda seed, **_kw: screen_rows[seed]),
                  mock.patch.object(evaluate, "_source_contract", return_value=source),
                  mock.patch.object(evaluate, "_validated_rgb_assets", return_value=ASSETS),
                  mock.patch.object(evaluate, "sha256_file", side_effect=lambda p: (
                      protocol["prediction_sha256"] if Path(p) == npz else "manifest-sha"))):
                result = evaluate.validate_prediction(1, "spike_reset_off", output_root=root)
                self.assertEqual(result["labels"].shape, (COUNT, GRID, GRID))
                with self.assertRaises(FileNotFoundError):
                    evaluate.validate_prediction(1, "dendrite_history_off", output_root=root)
                protocol["encoder_sha256"] = "tampered-encoder-sha"
                protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "invalid SW0140 frozen prediction"):
                    evaluate.validate_prediction(1, "spike_reset_off", output_root=root)
        finally:
            for path in (root / "seed1" / "spike_reset_off" / "predictions.npz",
                         root / "seed1" / "spike_reset_off" / "protocol.json"):
                if path.exists():
                    path.unlink()
            if (root / "seed1" / "spike_reset_off").exists():
                (root / "seed1" / "spike_reset_off").rmdir()
            if (root / "seed1").exists():
                (root / "seed1").rmdir()
            root.rmdir()

    def test_scoring_freezes_all_18_predictions_before_opening_ground_truth(self):
        root = ROOT / f"_sw0140_score_test_{uuid.uuid4().hex}"
        root.mkdir()
        try:
            target_px = np.zeros((COUNT, 128, 128, 1), dtype=np.uint8)
            target_px[:, 24:88, 32:96, 0] = 1
            target = modal_native32(target_px)
            sources, slots, predictions = {}, {}, {}
            fake_sha = {}

            def path_record(name, labels):
                archive = root / f"{name}.npz"
                protocol = root / f"{name}.json"
                archive.write_bytes(name.encode())
                protocol.write_bytes((name + "protocol").encode())
                archive_sha, protocol_sha = "sha-" + name, "sha-proto-" + name
                fake_sha[str(archive)] = archive_sha
                fake_sha[str(protocol)] = protocol_sha
                return {"ids": np.asarray(IMAGE_IDS), "labels": labels,
                        "qcc_labels": labels, "archive_path": archive,
                        "protocol_path": protocol, "prediction_sha256": archive_sha,
                        "protocol_sha256": protocol_sha,
                        "protocol": {"source_core_sha256": "core-sha",
                                     "source_manifest_sha256": "manifest-sha",
                                     "encoder_sha256": "encoder-sha",
                                     "feature_preprocessing_sha256": "preprocess-sha",
                                     "rgb_assets": ASSETS}}

            source_labels = np.zeros((COUNT, GRID, GRID), dtype=np.int64)
            slot_labels = target.copy()
            for seed in evaluate.SEEDS:
                sources[seed] = path_record(f"source{seed}", source_labels)
                slots[seed] = path_record(f"slot{seed}", slot_labels)
            for condition in evaluate.CONDITIONS_TO_SCORE:
                for seed in evaluate.SEEDS:
                    predictions[(seed, condition)] = path_record(
                        f"pred{seed}_{condition}", source_labels)
            baseline_path = root / "slot_baseline.json"
            slot_means = {metric: production_metrics32(slot_labels, target)[metric]["mean"]
                          for metric in METRICS}
            baseline = {"scores": {"slot32": {metric: {"three_seed_mean": value}
                                                  for metric, value in slot_means.items()}}}
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            baseline_sha = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
            fake_sha[str(baseline_path)] = baseline_sha
            fake_sha[str(evaluate.HERE / "evaluation_protocol.json")] = "eval-protocol-sha"
            output = root / "summary.json"
            validation_calls = []

            def h5_open(*_args, **_kwargs):
                self.assertEqual(len(validation_calls), 18)
                return _FakeH5(_MaskSlice())

            def validated(seed, condition, **_kwargs):
                validation_calls.append((seed, condition))
                return predictions[(seed, condition)]

            with (mock.patch.object(evaluate, "validate_prediction", side_effect=validated),
                  mock.patch.object(evaluate, "_load_source_reference", side_effect=lambda seed, _root: sources[seed]),
                  mock.patch.object(evaluate, "_load_slot_reference", side_effect=lambda seed, _root: slots[seed]),
                  mock.patch.object(evaluate, "sha256_file", side_effect=lambda p: fake_sha[str(Path(p))]),
                  mock.patch.object(evaluate, "SLOT_BASELINE_PATH", baseline_path),
                  mock.patch.object(evaluate, "SLOT_BASELINE_SHA256", baseline_sha),
                  mock.patch.object(evaluate, "implementation_fingerprint", return_value={"eval.py": "fp"}),
                  mock.patch.object(evaluate.h5py, "File", side_effect=h5_open)):
                report = evaluate.score_all(output=output, dataset=root / "unused.h5")
            self.assertTrue(output.exists())
            self.assertEqual(len(validation_calls), 18)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(set(report["three_seed_means"]), set(evaluate.CONDITIONS_TO_SCORE))
            self.assertEqual(report["paired_bootstrap"]["spike_reset_off"]["source97"]["fg_ari"]["resamples"], 10000)
            self.assertTrue(report["ground_truth_used_for_scoring"])
        finally:
            for path in root.iterdir():
                if path.is_file():
                    path.unlink()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
