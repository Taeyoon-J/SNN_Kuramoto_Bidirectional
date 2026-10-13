"""Fixed320 Slot-only native64 score; validates all predictions before GT access."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, ROOT / "collaborative_test"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h5py
import numpy as np
import torch
from snn_kuramoto_bidirectional.evaluation import evaluate_patch_masks

from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0154_native64_resolution_compare.resolution64 import GRID, modal_native64

IMAGE_IDS = contract.IMAGE_IDS
COUNT = contract.COUNT
METRICS = contract.METRICS


def implementation_fingerprint():
    """Bind scorer and the exact frozen Slot/metric/modal dependencies it uses."""
    paths = (
        HERE / "score_slot64.py", HERE / "resolution64.py",
        ROOT / "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
        ROOT / "snn_kuramoto_bidirectional/evaluation.py",
    )
    result = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"scoring dependency missing: {path}")
        result[path.relative_to(ROOT).as_posix()] = contract.sha256_file(path)
    return result


def _read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected protocol JSON object: {path}")
    return value


def _load_slot(slot_root: Path, seed: int):
    folder = slot_root / f"seed{seed}_epoch10"
    npz_path, protocol_path = folder / "predictions.npz", folder / "protocol.json"
    if not npz_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError(f"missing registered Slot seed{seed} prediction artifacts")
    protocol = _read_json(protocol_path)
    digest = contract.validate_slot_protocol(protocol, seed=seed, prediction_path=npz_path)
    with np.load(npz_path, allow_pickle=False) as saved:
        if not {"labels", "image_ids"}.issubset(saved.files):
            raise ValueError(f"Slot NPZ lacks labels/image_ids: {npz_path}")
        ids = contract.require_fixed_image_ids(saved["image_ids"], name=f"seed{seed} Slot IDs")
        labels = np.asarray(saved["labels"])
    if labels.shape != (COUNT, 128, 128):
        raise ValueError(f"Slot labels must be native128 fixed320, got {labels.shape}")
    if (not np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, np.bool_)
            or np.any(labels < 0)):
        raise ValueError("Slot labels must be nonnegative integer IDs")
    labels64 = modal_native64(labels)
    return {"seed": seed, "folder": folder, "npz_path": npz_path,
            "protocol_path": protocol_path, "protocol": protocol, "ids": ids,
            "labels64": labels64, "prediction_sha256": digest,
            "protocol_sha256": contract.sha256_file(protocol_path)}


def prepare_all_slot_predictions(slot_root: Path):
    """Validate three prediction/protocol pairs and pool to64 before any GT read."""
    loaded = {seed: _load_slot(slot_root, seed) for seed in range(3)}
    reference = loaded[0]["ids"]
    for seed, row in loaded.items():
        if not np.array_equal(row["ids"], reference):
            raise ValueError(f"Slot seed{seed} prediction order differs")
    # Recheck every frozen input immediately before returning to the GT caller.
    for seed, row in loaded.items():
        if (contract.sha256_file(row["npz_path"]) != row["prediction_sha256"]
                or contract.sha256_file(row["protocol_path"]) != row["protocol_sha256"]):
            raise ValueError(f"Slot seed{seed} prediction changed during validation")
    return loaded


def _production_metrics64(prediction, target):
    pred = np.asarray(prediction)
    gt = np.asarray(target)
    if pred.shape != (COUNT, GRID, GRID) or gt.shape != pred.shape:
        raise ValueError("native64 prediction and target must be [320,64,64]")
    for name, labels in (("prediction", pred), ("target", gt)):
        if (not np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, np.bool_)
                or np.any(labels < 0)):
            raise ValueError(f"{name} must contain nonnegative integer labels")
    scored = evaluate_patch_masks(
        torch.from_numpy(pred.astype(np.int64, copy=False)), torch.from_numpy(gt.astype(np.int64, copy=False)))
    metrics = {}
    for name in METRICS:
        values = scored["per_image"][name].detach().cpu().numpy().astype(np.float64)
        mean = float(scored["mean"][name].detach().cpu())
        valid_count = int(scored["valid_count"][name].detach().cpu())
        if values.shape != (COUNT,) or valid_count != COUNT or not np.isfinite(values).all() or not np.isfinite(mean):
            raise ValueError(f"native64 production metric {name} is invalid")
        if abs(mean - float(values.mean())) > 1e-12:
            raise AssertionError(f"native64 {name} mean mismatch")
        metrics[name] = {"mean": mean, "valid_count": valid_count,
                         "per_image": values.tolist()}
    return metrics


def score_slot_baseline(*, dataset: Path, slot_root: Path):
    # All prediction arrays, protocol schemas, IDs, source hashes, and pooled64
    # labels are fully validated before the first HDF5 mask read.
    loaded = prepare_all_slot_predictions(slot_root)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    with h5py.File(dataset, "r") as source:
        if "mask" not in source or source["mask"].shape[0] <= IMAGE_IDS[-1]:
            raise ValueError("HDF5 mask rows do not cover the fixed320 evaluation IDs")
        pixel_masks = np.asarray(source["mask"][IMAGE_IDS[0]:IMAGE_IDS[-1] + 1])
    target64 = modal_native64(pixel_masks)
    for row in loaded.values():
        row["scores"] = _production_metrics64(row["labels64"], target64)
    means = {metric: {"per_seed": {str(seed): loaded[seed]["scores"][metric]["mean"] for seed in range(3)},
                      "three_seed_mean": float(np.mean([loaded[s]["scores"][metric]["mean"] for s in range(3)]))}
             for metric in METRICS}
    return {"status": "complete", "experiment": "SW0154_native64_resolution_compare",
            "evaluation_contract": {"image_ids": [IMAGE_IDS[0], IMAGE_IDS[-1], COUNT],
                                    "source_prediction": "Slot saved native128 IDs directly pooled by 2x2 modal; no resize or 32-grid intermediate",
                                    "target": "native128 instance masks directly pooled by production2x2 modal",
                                    "grid": [GRID, GRID], "metrics": list(METRICS)},
            "ground_truth_read_after_all_prediction_validation": True,
            "implementation_fingerprint": implementation_fingerprint(),
            "predictions": {str(seed): {"prediction_sha256": loaded[seed]["prediction_sha256"],
                                       "protocol_sha256": loaded[seed]["protocol_sha256"],
                                       "checkpoint_sha256": loaded[seed]["protocol"]["checkpoint_sha256"],
                                       "labels64_shape": list(loaded[seed]["labels64"].shape),
                                       "scores": loaded[seed]["scores"]}
                            for seed in range(3)},
            "scores": means,
            "interpretation_limit": "Slot native64 score alone is a baseline readout; it does not establish native64 model dynamics or compare model quality."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--slot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"preserve existing result: {args.output}")
    report = score_slot_baseline(dataset=args.dataset, slot_root=args.slot_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"status": "complete", "ground_truth_read_after_all_prediction_validation": True,
                      "output": str(args.output)}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
