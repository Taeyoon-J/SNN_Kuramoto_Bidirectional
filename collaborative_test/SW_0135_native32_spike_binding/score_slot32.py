"""Score direct native32 predictions and own70k Slot pixels on one frozen slice.

Every required prediction artifact is SHA/schema/ID checked before the dataset mask
is opened. Slot masks are pooled directly from saved 128x128 labels at 4x4;
the existing perimeter-background remap is left untouched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import h5py
import torch

from evaluation_contract32 import (
    COUNT, GRID, IMAGE_IDS, METRICS, PATCH_SIZE, SLOT_PATCH_MASKS_SHA256,
    modal_native32, production_metrics32,
    require_fixed_image_ids, sha256_file, slot_native_pixels_to32,
    validate_native_protocol, validate_slot_protocol,
)
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks


def _read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"protocol must be a JSON object: {path}")
    return value


def _load_npz(path: Path, *, prediction_shape, protocol, validator, seed):
    if not path.is_file():
        raise FileNotFoundError(path)
    # Hash first, validate protocol against those exact bytes, then load with
    # pickle disabled. A second hash pass for all six files happens before GT.
    digest = validator(protocol, seed=seed, prediction_path=path)
    with np.load(path, allow_pickle=False) as saved:
        if not {"labels", "image_ids"}.issubset(saved.files):
            raise ValueError(f"prediction NPZ lacks labels/image_ids: {path}")
        ids = require_fixed_image_ids(saved["image_ids"], name=f"{path} image_ids")
        labels = np.asarray(saved["labels"])
    if labels.shape != tuple(prediction_shape):
        raise ValueError(f"{path} labels must have shape {tuple(prediction_shape)}")
    if (not np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, np.bool_)
            or np.any(labels < 0)):
        raise ValueError(f"{path} labels must be nonnegative integers")
    return labels.astype(np.int64, copy=False), ids, digest


def _slot16_audit(slot_dir: Path, seed: int, slot_pixels, ids, target16):
    """Reproduce the saved 16-grid audit from the same native Slot pixels."""
    masks_path = slot_dir / "patch_masks.pt"
    summary_path = slot_dir / "evaluation_summary.json"
    if not masks_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"seed{seed} original 16-grid audit artifacts are required")
    if sha256_file(masks_path) != SLOT_PATCH_MASKS_SHA256[int(seed)]:
        raise ValueError(f"Original Slot 16-grid patch artifact SHA mismatch (seed {seed})")
    try:
        saved = torch.load(masks_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only.
        saved = torch.load(masks_path, map_location="cpu")
    required = {"prediction", "target", "image_ids", "patch_size"}
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError(f"Malformed original Slot patch_masks.pt: {masks_path}")
    saved_ids = require_fixed_image_ids(saved["image_ids"], name=f"{masks_path} image_ids")
    if int(saved["patch_size"]) != 8:
        raise ValueError("Original Slot audit must use its registered 8-pixel/16-grid contract")
    if not np.array_equal(saved_ids, ids):
        raise ValueError("Original Slot 16-grid IDs differ from native prediction IDs")
    saved_prediction = torch.as_tensor(saved["prediction"], dtype=torch.int64)
    saved_target = torch.as_tensor(saved["target"], dtype=torch.int64)
    if saved_prediction.shape != (COUNT, 16, 16) or saved_target.shape != (COUNT, 16, 16):
        raise ValueError("Original Slot patch audit tensors must be [320,16,16]")
    direct16 = clevr_mask_patch(torch.from_numpy(np.ascontiguousarray(slot_pixels)), 8)["patch_labels"]
    if not torch.equal(direct16, saved_prediction):
        raise ValueError(f"Saved Slot 16-grid prediction does not reproduce from native pixels (seed {seed})")
    if not torch.equal(torch.from_numpy(target16), saved_target):
        raise ValueError(f"Saved Slot 16-grid target differs from the current HDF5 slice (seed {seed})")
    scores = evaluate_patch_masks(saved_prediction, saved_target)
    summary = _read_json(summary_path)
    expected = summary.get("scores", {}).get("mean")
    if not isinstance(expected, dict):
        raise ValueError(f"Original Slot evaluation summary has no scores.mean: {summary_path}")
    reproduced = {}
    for metric in METRICS:
        value = float(scores["mean"][metric])
        if not np.isfinite(value) or metric not in expected or abs(value - float(expected[metric])) > 1e-10:
            raise ValueError(f"Original Slot 16-grid {metric} score does not reproduce (seed {seed})")
        reproduced[metric] = value
    return {"prediction_sha256": sha256_file(masks_path),
            "summary_sha256": sha256_file(summary_path), "scores": reproduced,
            "patch_size": 8, "grid": [16, 16]}


def score_all(*, dataset: Path, native_root: Path, slot_root: Path):
    """Validate every frozen prediction, then read GT and calculate metrics."""
    loaded = {}
    for seed in range(3):
        native_dir = native_root / f"seed{seed}"
        native_path, native_protocol_path = native_dir / "predictions.npz", native_dir / "protocol.json"
        native_protocol = _read_json(native_protocol_path)
        native_labels, ids, native_sha = _load_npz(
            native_path, prediction_shape=(COUNT, GRID, GRID), protocol=native_protocol,
            validator=validate_native_protocol, seed=seed)

        slot_dir = slot_root / f"seed{seed}_epoch10"
        slot_path, slot_protocol_path = slot_dir / "predictions.npz", slot_dir / "protocol.json"
        slot_protocol = _read_json(slot_protocol_path)
        slot_pixels, slot_ids, slot_sha = _load_npz(
            slot_path, prediction_shape=(COUNT, 128, 128), protocol=slot_protocol,
            validator=validate_slot_protocol, seed=seed)
        if not np.array_equal(ids, slot_ids):
            raise ValueError(f"seed{seed} native and Slot prediction rows differ")
        loaded[seed] = {
            "native_labels": native_labels, "slot_pixels": slot_pixels,
            "slot32_labels": slot_native_pixels_to32(slot_pixels), "ids": ids,
            "native_path": native_path,
            "native_protocol_path": native_protocol_path,
            "native_sha256": native_sha, "native_protocol_sha256": sha256_file(native_protocol_path),
            "slot_path": slot_path, "slot_protocol_path": slot_protocol_path,
            "slot_sha256": slot_sha, "slot_protocol_sha256": sha256_file(slot_protocol_path),
            "slot_protocol": slot_protocol, "slot_dir": slot_dir,
        }

    return _score_loaded(dataset, loaded, include_native=True)


def score_slot_baseline(*, dataset: Path, slot_root: Path):
    """Score the already-frozen Slot baseline before native32 outputs exist."""
    loaded = {}
    for seed in range(3):
        slot_dir = slot_root / f"seed{seed}_epoch10"
        slot_path, slot_protocol_path = slot_dir / "predictions.npz", slot_dir / "protocol.json"
        slot_protocol = _read_json(slot_protocol_path)
        slot_pixels, ids, slot_sha = _load_npz(
            slot_path, prediction_shape=(COUNT, 128, 128), protocol=slot_protocol,
            validator=validate_slot_protocol, seed=seed)
        # Pool/freeze all three direct4x4 label arrays before opening GT.
        slot32 = slot_native_pixels_to32(slot_pixels)
        loaded[seed] = {
            "slot_pixels": slot_pixels, "slot32_labels": slot32, "ids": ids,
            "slot_path": slot_path, "slot_protocol_path": slot_protocol_path, "slot_sha256": slot_sha,
            "slot_protocol_sha256": sha256_file(slot_protocol_path),
            "slot_protocol": slot_protocol, "slot_dir": slot_dir,
        }
    return _score_loaded(dataset, loaded, include_native=False)


def _score_loaded(dataset: Path, loaded, *, include_native: bool):
    # Rehash every prediction after loading/pooling and immediately before GT.
    for seed, row in loaded.items():
        if include_native and sha256_file(row["native_path"]) != row["native_sha256"]:
            raise ValueError(f"seed{seed} native prediction changed during validation")
        if sha256_file(row["slot_path"]) != row["slot_sha256"]:
            raise ValueError(f"seed{seed} Slot prediction changed during validation")
        if include_native and sha256_file(row["native_protocol_path"]) != row["native_protocol_sha256"]:
            raise ValueError(f"seed{seed} native prediction protocol changed during validation")
        if sha256_file(row["slot_protocol_path"]) != row["slot_protocol_sha256"]:
            raise ValueError(f"seed{seed} Slot prediction protocol changed during validation")

    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    with h5py.File(dataset, "r") as source:
        if "mask" not in source or "image" not in source or source["image"].shape[0] < IMAGE_IDS[-1] + 1:
            raise ValueError("HDF5 does not contain the registered image/mask rows through ID 1639")
        pixel_gt = np.asarray(source["mask"][IMAGE_IDS[0]:IMAGE_IDS[-1] + 1])
    gt32 = modal_native32(pixel_gt)
    gt16 = clevr_mask_patch(torch.from_numpy(np.ascontiguousarray(pixel_gt)), 8)["patch_labels"].numpy()

    for seed, row in loaded.items():
        row["slot16_audit"] = _slot16_audit(
            row["slot_dir"], seed, row["slot_pixels"], row["ids"], gt16)
        row["scores"] = {"slot32": production_metrics32(row["slot32_labels"], gt32)}
        if include_native:
            row["scores"]["native32"] = production_metrics32(row["native_labels"], gt32)
    return loaded, gt32


def _means(loaded):
    output = {}
    arms = ("native32", "slot32") if "native32" in loaded[0]["scores"] else ("slot32",)
    for arm in arms:
        output[arm] = {}
        for metric in METRICS:
            per_seed = {str(seed): loaded[seed]["scores"][arm][metric]["mean"] for seed in range(3)}
            output[arm][metric] = {"per_seed": per_seed,
                                   "three_seed_mean": float(np.mean(list(per_seed.values())))}
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="HDF5 containing native128 mask/image rows")
    parser.add_argument("--native-root", type=Path,
                        help="Directory with seed0/seed1/seed2/{predictions.npz,protocol.json}; direct32 labels")
    parser.add_argument("--slot-root", type=Path, required=True,
                        help="SW0092 Slot output root with seed{seed}_epoch10 artifacts")
    parser.add_argument("--output", type=Path, required=True, help="Create-once JSON result path")
    parser.add_argument("--slot-only", action="store_true",
                        help="Score the registered Slot32 baseline before native32 model predictions exist")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {args.output}")
    if args.slot_only:
        loaded, _ = score_slot_baseline(dataset=args.dataset, slot_root=args.slot_root)
    else:
        if args.native_root is None:
            parser.error("--native-root is required unless --slot-only is selected")
        loaded, _ = score_all(dataset=args.dataset, native_root=args.native_root, slot_root=args.slot_root)
    result = {
        "status": "complete",
        "comparison": "slot32 baseline only" if args.slot_only else "native32 versus Slot32",
        "evaluation_contract": {"image_ids": [IMAGE_IDS[0], IMAGE_IDS[-1], COUNT],
                                "native_image_size": [128, 128], "patch_grid": [GRID, GRID],
                                "patch_pixels": PATCH_SIZE, "metrics": list(METRICS),
                                "native_prediction": "direct32 labels; no conversion from 16-grid outputs",
                                "slot_prediction": "saved native128 labels modal-pooled by production4x4 converter; existing perimeter-background remap preserved"},
        "ground_truth_read_after_all_prediction_sha_and_id_checks": True,
        "slot_original16_audit": {str(seed): loaded[seed]["slot16_audit"] for seed in range(3)},
        "scores": _means(loaded),
        "seeds": {str(seed): {
            "image_ids": [IMAGE_IDS[0], IMAGE_IDS[-1]],
            "slot_prediction_sha256": row["slot_sha256"],
            "slot_protocol_sha256": row["slot_protocol_sha256"],
            "slot_checkpoint_sha256": row["slot_protocol"]["checkpoint_sha256"],
            "slot_model_sha256": row["slot_protocol"]["model_sha256"],
            **({"native_prediction_sha256": row["native_sha256"],
                "native_protocol_sha256": row["native_protocol_sha256"],
                "native32": row["scores"]["native32"]} if "native32" in row["scores"] else {}),
            "slot32": row["scores"]["slot32"],
        } for seed, row in loaded.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "means": result["scores"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
