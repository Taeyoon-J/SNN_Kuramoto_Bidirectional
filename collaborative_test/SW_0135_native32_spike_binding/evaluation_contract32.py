"""Shared fixed-320 native32 scoring contract for model and Slot predictions."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks

IMAGE_IDS = tuple(range(1320, 1640))
COUNT = len(IMAGE_IDS)
IMAGE_SIZE = 128
GRID = 32
PATCH_SIZE = IMAGE_SIZE // GRID
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
SLOT_CHECKPOINT_SHA256 = {
    0: "0301a456a586da5e1132c703d095f4736c4351cb86388eabe2aff7f1c96ae091",
    1: "3afadf5958d494241763dcc7ef0600bbeb78611d8489cb4126fce462bf0992b9",
    2: "c665b2b52cb59ee5044b91cebff039a4b5199fa3edc2b5fa769094fcadadb200",
}
SLOT_MODEL_SHA256 = "96c2b12d8b28c22fd2605eccf9027bda304f38bb9332f8c1620898f471c67ec4"
SLOT_PATCH_MASKS_SHA256 = {
    0: "ef1dce0e16bfe0f9340f9c157b0a1400320c7dc2479e4ef68b20a65437f0108b",
    1: "f5c80fceb2506d5356c3aa23ac39556448ebcb4f9454eeebabdc2ef148b1fcbd",
    2: "c23fe2f5b97fe0f2db74c1d39365619bb934d2e7765e73c672a31475acda5ee3",
}
SLOT_SELECTED_TRAINING_IDS_SHA256 = "c20fa9d537ad25caf6bb6d84012a479971ec69907d0ee3dff1ee8de9cebe8f84"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer_labels(value, shape, name):
    array = np.asarray(value)
    if array.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {array.shape}")
    if (not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_)
            or not np.isfinite(array).all() or np.any(array < 0)):
        raise ValueError(f"{name} must contain finite nonnegative integer labels")
    return array.astype(np.int64, copy=False)


def require_fixed_image_ids(image_ids, *, name="image_ids"):
    actual = np.asarray(image_ids)
    expected = np.asarray(IMAGE_IDS, dtype=np.int64)
    if actual.shape != (COUNT,) or not np.issubdtype(actual.dtype, np.integer):
        raise ValueError(f"{name} must be the 320 integer IDs {IMAGE_IDS[0]}..{IMAGE_IDS[-1]}")
    if not np.array_equal(actual, expected):
        raise ValueError(f"{name} does not exactly match the registered 320-row slice")
    return expected


def modal_native32(pixel_masks):
    """Modal/minimum-ID-tie GT conversion directly from native128 masks to32 patches."""
    labels = np.asarray(pixel_masks)
    if labels.shape == (COUNT, IMAGE_SIZE, IMAGE_SIZE, 1):
        labels = labels[..., 0]
    if labels.shape != (COUNT, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("native GT masks must be [320,128,128] or [320,128,128,1]")
    if not np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, np.bool_):
        raise ValueError("native GT masks must contain integer instance IDs")
    if np.any(labels < 0):
        raise ValueError("native GT instance IDs must be nonnegative")
    # This is the production evaluation utility: each 4x4 pixel block votes,
    # and its smallest instance ID wins any tie (including background zero).
    converted = clevr_mask_patch(torch.from_numpy(np.ascontiguousarray(labels)), PATCH_SIZE)
    result = converted["patch_labels"].cpu().numpy()
    if result.shape != (COUNT, GRID, GRID):
        raise AssertionError("production modal converter did not emit32 labels")
    return result.astype(np.int64, copy=False)


def slot_native_pixels_to32(pixel_labels):
    """Apply only the production 4x4 modal score-pooling to stored Slot labels.

    Slot's existing perimeter-background choice/remap happened during inference
    and is intentionally preserved in these labels. This function does not
    select a new background, resize, repeat16-grid labels, or infer masks.
    """
    labels = _integer_labels(pixel_labels, (COUNT, IMAGE_SIZE, IMAGE_SIZE),
                             "Slot native pixel labels")
    converted = clevr_mask_patch(torch.from_numpy(np.ascontiguousarray(labels)), PATCH_SIZE)
    result = converted["patch_labels"].cpu().numpy()
    if result.shape != (COUNT, GRID, GRID):
        raise AssertionError("production modal converter did not emit32 Slot labels")
    return result.astype(np.int64, copy=False)


def validate_native_protocol(protocol, *, seed: int, prediction_path: str | Path):
    if not isinstance(protocol, dict) or protocol.get("seed") != int(seed):
        raise ValueError("native prediction protocol seed mismatch")
    if protocol.get("image_ids") != [IMAGE_IDS[0], IMAGE_IDS[-1]] or protocol.get("count") != COUNT:
        raise ValueError("native prediction protocol is not the fixed320-ID slice")
    if protocol.get("grid_size") != [GRID, GRID] or protocol.get("prediction_shape") != [COUNT, GRID, GRID]:
        raise ValueError("native prediction protocol must identify direct32x32 labels")
    if protocol.get("ground_truth_used_for_prediction") is not False:
        raise ValueError("native predictions must be GT-free")
    digest = sha256_file(prediction_path)
    if protocol.get("prediction_sha256") != digest:
        raise ValueError("native prediction protocol SHA does not match prediction bytes")
    return digest


def validate_slot_protocol(protocol, *, seed: int, prediction_path: str | Path):
    if not isinstance(protocol, dict) or protocol.get("seed") != int(seed):
        raise ValueError("Slot prediction protocol seed mismatch")
    if protocol.get("image_ids") != [IMAGE_IDS[0], IMAGE_IDS[-1]] or protocol.get("count") != COUNT:
        raise ValueError("Slot prediction protocol is not the fixed320-ID slice")
    if protocol.get("inference_seed") != 0:
        raise ValueError("Slot predictions must use the registered fixed inference seed0")
    if protocol.get("batch_size") != 1 or protocol.get("iterations") != 3:
        raise ValueError("Slot inference batch/iteration contract mismatch")
    if protocol.get("preprocessing") != "full HDF5 RGB image, float32 /127.5 - 1; no crop":
        raise ValueError("Slot inference preprocessing contract mismatch")
    if protocol.get("resolution") != [IMAGE_SIZE, IMAGE_SIZE] or protocol.get("num_slots") != 11:
        raise ValueError("Slot protocol must bind native128 inference and 11 slots")
    if protocol.get("checkpoint_sha256") != SLOT_CHECKPOINT_SHA256[int(seed)]:
        raise ValueError("Slot protocol checkpoint is not the registered own70k epoch10 model")
    if protocol.get("model_sha256") != SLOT_MODEL_SHA256:
        raise ValueError("Slot protocol does not bind the registered Slot model implementation")
    if protocol.get("ground_truth_used_for_prediction") is not False:
        raise ValueError("Slot predictions must be GT-free")
    if (protocol.get("background_rule") != "most hard-assigned one-pixel perimeter pixels; smallest slot ID wins ties"
            or protocol.get("label_rule") != "slot ID + 1; selected background slot set to 0"):
        raise ValueError("Slot pixel labels do not bind the registered perimeter-background remap")
    training = protocol.get("training_protocol")
    if not isinstance(training, dict):
        raise ValueError("Slot protocol must include its training provenance")
    if (training.get("seed") != int(seed)
            or training.get("unique_training_images") != 70000
            or training.get("epochs") != 10
            or training.get("batch_size") != 32
            or training.get("steps_per_epoch") != 2188
            or training.get("total_steps") != 21880
            or training.get("iterations") != 3
            or training.get("learning_rate") != 0.0004
            or training.get("selected_images_sha256") != SLOT_SELECTED_TRAINING_IDS_SHA256
            or training.get("num_slots") != 11
            or training.get("ground_truth_used_for_training") is not False
            or training.get("training_id_segments_inclusive") != [[0, 999], [1640, 70639]]):
        raise ValueError("Slot checkpoint training provenance does not match the registered own70k recipe")
    digest = sha256_file(prediction_path)
    return digest


def production_metrics32(prediction, target):
    pred = _integer_labels(prediction, (COUNT, GRID, GRID), "native32 prediction")
    gt = _integer_labels(target, (COUNT, GRID, GRID), "native32 target")
    scored = evaluate_patch_masks(torch.from_numpy(pred), torch.from_numpy(gt))
    result = {}
    for name in METRICS:
        per_image = scored["per_image"][name].detach().cpu().numpy().astype(np.float64)
        mean = float(scored["mean"][name].detach().cpu())
        valid_count = int(scored["valid_count"][name].detach().cpu())
        if per_image.shape != (COUNT,) or valid_count != COUNT or not np.isfinite(per_image).all() or not np.isfinite(mean):
            raise ValueError(f"production metric {name} is not finite for all320 images")
        if abs(float(per_image.mean()) - mean) > 1e-12:
            raise AssertionError(f"production metric {name} mean does not match per-image scores")
        result[name] = {"mean": mean, "valid_count": valid_count,
                        "per_image": per_image.tolist()}
    return result
