from __future__ import annotations

import json
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch
from evaluation_contract32 import (
    COUNT, GRID, IMAGE_IDS, METRICS, PATCH_SIZE, SLOT_CHECKPOINT_SHA256, SLOT_MODEL_SHA256,
    modal_native32, production_metrics32, sha256_file, slot_native_pixels_to32,
    validate_slot_protocol,
)
import score_slot32


@contextlib.contextmanager
def _scratch():
    """A uniquely owned workspace fixture; cleanup touches only its files."""
    root = HERE / f"_test_fixture_{uuid.uuid4().hex}"
    root.mkdir()
    try:
        yield root
    finally:
        for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        root.rmdir()


class Native32ContractTests(unittest.TestCase):
    def test_four_pixel_modal_is_not_repeated_or_eight_pixel_pooling(self):
        masks = np.zeros((COUNT, 128, 128), dtype=np.int64)
        # Four distinct 4x4 instances share one 8x8 block. Native32 retains all
        # four; the registered old16 modal contract would collapse this to ID1.
        masks[0, 0:4, 0:4] = 1
        masks[0, 0:4, 4:8] = 2
        masks[0, 4:8, 0:4] = 3
        masks[0, 4:8, 4:8] = 4
        native32 = modal_native32(masks)
        old16 = score_slot32.clevr_mask_patch(torch.from_numpy(masks), 8)["patch_labels"].numpy()
        self.assertEqual(native32.shape, (COUNT, 32, 32))
        self.assertEqual(native32[0, :2, :2].tolist(), [[1, 2], [3, 4]])
        self.assertEqual(old16[0, 0, 0], 1)  # tie resolves to minimum ID
        self.assertNotEqual(native32[0, 0, 1], old16[0, 0, 0])
        self.assertEqual(PATCH_SIZE, 4)

    def test_minimum_id_tie_includes_background_and_slot_remap_is_preserved(self):
        labels = np.full((COUNT, 128, 128), 9, dtype=np.int64)
        labels[0, 0:2, 0:4] = 0
        labels[0, 2:4, 0:4] = 5
        labels[1, :4, :4] = 7
        converted = slot_native_pixels_to32(labels)
        self.assertEqual(converted.shape, (COUNT, 32, 32))
        self.assertEqual(converted[0, 0, 0], 0)
        self.assertEqual(converted[1, 0, 0], 7)
        self.assertEqual(converted[0, 0, 1], 9)  # no second background remap

    def test_production_metrics_have_all_three_finite_per_image_scores(self):
        target = np.zeros((COUNT, 32, 32), dtype=np.int64)
        target[:, 3:15, 4:16] = 1
        target[:, 15:27, 16:28] = 2
        prediction = target.copy()
        prediction[prediction == 1] = 8
        prediction[prediction == 2] = 4
        scored = production_metrics32(prediction, target)
        self.assertEqual(set(scored), {"fg_ari", "foreground_iou", "matched_object_iou"})
        for row in scored.values():
            self.assertEqual(row["valid_count"], COUNT)
            self.assertEqual(len(row["per_image"]), COUNT)
            self.assertTrue(np.isfinite(row["per_image"]).all())
            self.assertAlmostEqual(row["mean"], 1.0)

    def test_slot_protocol_binds_exact_seed_checkpoint_and_70k_recipe(self):
        with _scratch() as temp:
            prediction = temp / "predictions.npz"
            prediction.write_bytes(b"frozen-prediction-fixture")
            protocol = {
                "seed": 2, "inference_seed": 0, "image_ids": [1320, 1639], "count": 320,
                "resolution": [128, 128], "num_slots": 11,
                "batch_size": 1, "iterations": 3,
                "preprocessing": "full HDF5 RGB image, float32 /127.5 - 1; no crop",
                "checkpoint_sha256": SLOT_CHECKPOINT_SHA256[2],
                "model_sha256": SLOT_MODEL_SHA256,
                "background_rule": "most hard-assigned one-pixel perimeter pixels; smallest slot ID wins ties",
                "label_rule": "slot ID + 1; selected background slot set to 0",
                "ground_truth_used_for_prediction": False,
                "training_protocol": {"seed": 2, "unique_training_images": 70000,
                                       "epochs": 10, "num_slots": 11,
                                       "batch_size": 32, "steps_per_epoch": 2188,
                                       "total_steps": 21880, "iterations": 3,
                                       "learning_rate": 0.0004,
                                       "selected_images_sha256": "c20fa9d537ad25caf6bb6d84012a479971ec69907d0ee3dff1ee8de9cebe8f84",
                                       "ground_truth_used_for_training": False,
                                       "training_id_segments_inclusive": [[0, 999], [1640, 70639]]},
            }
            validate_slot_protocol(protocol, seed=2, prediction_path=prediction)
            protocol["checkpoint_sha256"] = SLOT_CHECKPOINT_SHA256[1]
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                validate_slot_protocol(protocol, seed=2, prediction_path=prediction)

    def test_missing_seed2_prediction_fails_before_dataset_masks_open(self):
        with _scratch() as base:
            native, slot = base / "native", base / "slot"
            # Deliberately provide no predictions. score_all must fail during
            # artifact validation, before opening the HDF5 target dataset.
            native.mkdir()
            slot.mkdir()
            with mock.patch.object(score_slot32.h5py, "File") as hdf_open:
                with self.assertRaises(FileNotFoundError):
                    score_slot32.score_all(dataset=base / "dataset.h5", native_root=native, slot_root=slot)
                hdf_open.assert_not_called()

    def test_slot_only_cli_scores_native32_before_any_native_model_exists(self):
        with _scratch() as base:
            dataset = base / "mini.h5"
            slot_root = base / "slot"
            slot_root.mkdir()
            all_masks = np.zeros((1640, 128, 128), dtype=np.uint8)
            all_masks[1320:1640] = 5
            import h5py
            with h5py.File(dataset, "w") as hdf:
                hdf.create_dataset("image", data=np.zeros((1640, 1), dtype=np.uint8))
                hdf.create_dataset("mask", data=all_masks)

            for seed in range(3):
                folder = slot_root / f"seed{seed}_epoch10"
                folder.mkdir()
                labels = np.ones((COUNT, 128, 128), dtype=np.uint8)
                np.savez_compressed(folder / "predictions.npz", labels=labels,
                                    image_ids=np.asarray(IMAGE_IDS, dtype=np.int64))
                protocol = {
                    "seed": seed, "inference_seed": 0,
                    "image_ids": [1320, 1639], "count": COUNT,
                    "resolution": [128, 128], "num_slots": 11, "iterations": 3,
                    "batch_size": 1,
                    "preprocessing": "full HDF5 RGB image, float32 /127.5 - 1; no crop",
                    "checkpoint_sha256": SLOT_CHECKPOINT_SHA256[seed],
                    "model_sha256": SLOT_MODEL_SHA256,
                    "ground_truth_used_for_prediction": False,
                    "background_rule": "most hard-assigned one-pixel perimeter pixels; smallest slot ID wins ties",
                    "label_rule": "slot ID + 1; selected background slot set to 0",
                    "training_protocol": {
                        "seed": seed, "unique_training_images": 70000, "epochs": 10,
                        "num_slots": 11, "batch_size": 32, "steps_per_epoch": 2188,
                        "total_steps": 21880, "iterations": 3, "learning_rate": 0.0004,
                        "selected_images_sha256": "c20fa9d537ad25caf6bb6d84012a479971ec69907d0ee3dff1ee8de9cebe8f84",
                        "ground_truth_used_for_training": False,
                        "training_id_segments_inclusive": [[0, 999], [1640, 70639]],
                    },
                }
                (folder / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
                prediction16 = torch.ones((COUNT, 16, 16), dtype=torch.int64)
                target16 = torch.full((COUNT, 16, 16), 5, dtype=torch.int64)
                scores = score_slot32.evaluate_patch_masks(prediction16, target16)
                torch.save({"prediction": prediction16, "target": target16,
                            "image_ids": list(IMAGE_IDS), "patch_size": 8}, folder / "patch_masks.pt")
                (folder / "evaluation_summary.json").write_text(json.dumps({
                    "scores": {"mean": {key: float(scores["mean"][key]) for key in METRICS}}
                }), encoding="utf-8")
            fixture_mask_hashes = {
                seed: sha256_file(slot_root / f"seed{seed}_epoch10" / "patch_masks.pt")
                for seed in range(3)
            }

            output = base / "slot32.json"
            stdout = io.StringIO()
            with mock.patch.dict(score_slot32.SLOT_PATCH_MASKS_SHA256, fixture_mask_hashes), \
                    contextlib.redirect_stdout(stdout):
                score_slot32.main(["--slot-only", "--dataset", str(dataset), "--slot-root", str(slot_root),
                                   "--output", str(output)])
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["comparison"], "slot32 baseline only")
            self.assertTrue(result["ground_truth_read_after_all_prediction_sha_and_id_checks"])
            self.assertEqual(result["scores"]["slot32"]["foreground_iou"]["three_seed_mean"], 1.0)
            for seed in range(3):
                self.assertEqual(result["seeds"][str(seed)]["slot32"]["fg_ari"]["valid_count"], COUNT)

            # Paired mode accepts only direct32 native outputs and uses the
            # already-frozen Slot32 arrays without regenerating/interpolating16.
            native_root = base / "native"
            native_root.mkdir()
            for seed in range(3):
                native_dir = native_root / f"seed{seed}"
                native_dir.mkdir()
                native_path = native_dir / "predictions.npz"
                np.savez_compressed(native_path,
                                    labels=np.full((COUNT, GRID, GRID), 5, dtype=np.int64),
                                    image_ids=np.asarray(IMAGE_IDS, dtype=np.int64))
                native_protocol = {
                    "seed": seed, "image_ids": [1320, 1639], "count": COUNT,
                    "grid_size": [GRID, GRID], "prediction_shape": [COUNT, GRID, GRID],
                    "ground_truth_used_for_prediction": False,
                    "prediction_sha256": sha256_file(native_path),
                }
                (native_dir / "protocol.json").write_text(json.dumps(native_protocol), encoding="utf-8")
            paired_output = base / "paired.json"
            with mock.patch.dict(score_slot32.SLOT_PATCH_MASKS_SHA256, fixture_mask_hashes), \
                    contextlib.redirect_stdout(io.StringIO()):
                score_slot32.main(["--dataset", str(dataset), "--native-root", str(native_root),
                                   "--slot-root", str(slot_root), "--output", str(paired_output)])
            paired = json.loads(paired_output.read_text(encoding="utf-8"))
            self.assertEqual(paired["comparison"], "native32 versus Slot32")
            self.assertAlmostEqual(paired["scores"]["native32"]["foreground_iou"]["three_seed_mean"], 1.0)
            self.assertAlmostEqual(paired["scores"]["slot32"]["foreground_iou"]["three_seed_mean"], 1.0)

    def test_script_help_runs_without_pythonpath_configuration(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, str(HERE / "score_slot32.py"), "--help"],
            cwd=tempfile.gettempdir(), env=env, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--native-root", completed.stdout)
        self.assertIn("--slot-root", completed.stdout)


if __name__ == "__main__":
    unittest.main()
