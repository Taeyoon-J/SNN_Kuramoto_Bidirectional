"""Fixed320 native64-vs-Slot64 score; validate every prediction before GT access."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PATCH_V2_REPO", HERE.parents[1])).resolve()
for path in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h5py
import numpy as np

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0150_spike_spatial_mixture import preflight as sw150
from collaborative_test.SW_0154_native64_resolution_compare import predict_native64, score_slot64
from collaborative_test.SW_0154_native64_resolution_compare.resolution64 import GRID, modal_native64

SEEDS = (0, 1, 2)
IMAGE_IDS = tuple(contract.IMAGE_IDS)
COUNT = contract.COUNT
DATASET = Path(sw130.source97.DATASET)
PREDICTION_ROOT = HERE / "predictions_native64"
SLOT_ROOT = ROOT / "trained_models/SW0092_slot_our70000_eval"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    files = (HERE / "evaluate_native64.py", HERE / "predict_native64.py",
        HERE / "score_slot64.py", HERE / "resolution64.py",
        ROOT / "collaborative_test/SW_0130_phase_state_integration/run.py",
        ROOT / "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
        ROOT / "collaborative_test/SW_0150_spike_spatial_mixture/preflight.py",
        ROOT / "snn_kuramoto_bidirectional/evaluation.py")
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("native64 evaluator dependency missing")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in files}


def _read_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_native_prediction(seed, *, prediction_root=PREDICTION_ROOT,
                               expected_fingerprint=None):
    """Validate a source prediction and its lineage without opening GT masks."""
    if seed not in SEEDS:
        raise ValueError("only registered source seeds0/1/2 are scored")
    folder = Path(prediction_root) / f"seed{seed}"
    npz_path, protocol_path = folder / "predictions.npz", folder / "protocol.json"
    if not npz_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError(f"missing native64 fixed320 seed{seed} prediction")
    protocol = _read_json(protocol_path)
    checkpoint, manifest_path, manifest, pool, ids, source_sha = sw130.source_contract(seed)
    manifest_sha = sha256_file(manifest_path)
    assets = sw130.validate_rgb_assets()
    parity_path = ROOT / "collaborative_test/SW_0135_native32_spike_binding/results_archive" / (
        f"native32_zero_adapter_parity_seed{seed}_b4.json")
    parity, parity_sha = sw150.load_and_validate_parity(
        parity_path, source_sha=source_sha, source_manifest_sha=manifest_sha,
        training_ids=ids, asset_validation=assets)
    del parity
    fingerprint = predict_native64.implementation_fingerprint()
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError("native64 prediction implementation changed after queue registration")
    digest = sha256_file(npz_path)
    expected_ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    rgb_cache = Path(sw130.VAL_RGB)
    if not rgb_cache.is_file():
        raise FileNotFoundError("registered fixed320 RGB cache is missing")
    expected = {
        "experiment": "SW0154_native64_resolution_compare",
        "status": "prediction_complete", "stage": "native64_fixed320_source_prediction",
        "seed": seed, "image_ids": [IMAGE_IDS[0], IMAGE_IDS[-1]], "count": COUNT,
        "grid_size": [64, 64], "prediction_shape": [COUNT, 64, 64],
        "prediction_sha256": digest, "ground_truth_used_for_prediction": False,
        "source_checkpoint_sha256": source_sha, "source_manifest_sha256": manifest_sha,
        "source_training_ids_sha256": expected_ids_sha,
        "rgb_asset_validation": assets,
        "validation_rgb_cache_sha256": sha256_file(rgb_cache),
        "saved_native32_parity_sha256": parity_sha,
        "model_initialization_seed": predict_native64.MODEL_INIT_SEED,
        "inference_seed": predict_native64.INFERENCE_SEED,
        "steps": 1024, "settle": 512, "live_tail_steps": 0,
        "readout": {"kind": "actual emitted-spike QCC", "synchrony_threshold": .50,
                    "min_group_size": 32, "background": "largest_component",
                    "affinity_mode": "spike"},
        "implementation_fingerprint": fingerprint,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"native64 seed{seed} protocol mismatch at {key}")
    if protocol.get("model_state_sha256_before_after") is None or len(
            protocol["model_state_sha256_before_after"]) != 64:
        raise ValueError("native64 model-state immutability digest is missing")
    dataset_identity = protocol.get("canonical_dataset")
    if (not isinstance(dataset_identity, dict)
            or not isinstance(dataset_identity.get("path"), str)
            or not Path(dataset_identity["path"]).is_absolute()
            or not isinstance(dataset_identity.get("size"), int)
            or dataset_identity["size"] <= 0
            or not isinstance(dataset_identity.get("mtime_ns"), int)
            or dataset_identity["mtime_ns"] <= 0):
        raise ValueError("native64 prediction lacks canonical RGB dataset identity")
    work_dir = predict_native64.WORK_ROOT / f"seed{seed}"
    if (protocol["work_ledger"].get("directory") != str(work_dir.resolve())
            or protocol["work_ledger"].get("completed_image_records") != COUNT
            or len(protocol["work_ledger"].get("sha256", [])) != COUNT):
        raise ValueError("native64 image ledger is incomplete")
    ledger_sha = protocol["work_ledger"]["sha256"]
    current_ledger_sha = [sha256_file(work_dir / f"image_{image_id}.npz")
                          for image_id in IMAGE_IDS]
    if ledger_sha != current_ledger_sha:
        raise ValueError("native64 completed image ledger changed after prediction")
    with np.load(npz_path, allow_pickle=False) as saved:
        if not {"image_ids", "labels"}.issubset(saved.files):
            raise ValueError("native64 prediction NPZ is missing required arrays")
        image_ids = contract.require_fixed_image_ids(saved["image_ids"], name=f"native64 seed{seed} IDs")
        labels = np.asarray(saved["labels"])
    if (labels.shape != (COUNT, GRID, GRID) or labels.dtype != np.int64
            or np.any(labels < 0)):
        raise ValueError("native64 labels must be nonnegative int64 [320,64,64]")
    return {"seed": seed, "ids": image_ids, "labels": labels,
            "npz_path": npz_path, "protocol_path": protocol_path,
            "prediction_sha256": digest, "protocol_sha256": sha256_file(protocol_path),
            "protocol": protocol}


def _paired_bootstrap(left, right, draws):
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if delta.shape != (COUNT,) or not np.isfinite(delta).all():
        raise ValueError("paired native64 score requires320 finite per-image values")
    low, high = np.quantile(delta[draws].mean(axis=1), [.025, .975])
    return {"mean_difference": float(delta.mean()), "ci95": [float(low), float(high)]}


def score_all(*, dataset=DATASET, prediction_root=PREDICTION_ROOT, slot_root=SLOT_ROOT):
    """Freeze and validate six predictions before opening the target masks."""
    fingerprint = implementation_fingerprint()
    expected_prediction_fp = predict_native64.implementation_fingerprint()
    own = {seed: validate_native_prediction(seed, prediction_root=prediction_root,
        expected_fingerprint=expected_prediction_fp) for seed in SEEDS}
    slots = score_slot64.prepare_all_slot_predictions(Path(slot_root))
    if set(own) != set(slots):
        raise AssertionError("native64/Slot seed sets differ")
    for seed in SEEDS:
        if not np.array_equal(own[seed]["ids"], slots[seed]["ids"]):
            raise ValueError(f"native64 and Slot seed{seed} fixed320 IDs differ")
    # Rehash every input after loading arrays and immediately before target access.
    for row in own.values():
        if (sha256_file(row["npz_path"]) != row["prediction_sha256"]
                or sha256_file(row["protocol_path"]) != row["protocol_sha256"]):
            raise ValueError("native64 prediction changed before GT access")
    for row in slots.values():
        if (sha256_file(row["npz_path"]) != row["prediction_sha256"]
                or sha256_file(row["protocol_path"]) != row["protocol_sha256"]):
            raise ValueError("Slot prediction changed before GT access")
    dataset = Path(dataset)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    stat_before = dataset.stat()
    resolved_dataset = str(dataset.resolve())
    for seed in SEEDS:
        recorded = own[seed]["protocol"]["canonical_dataset"]
        if (recorded.get("path") != resolved_dataset
                or recorded.get("size") != stat_before.st_size
                or recorded.get("mtime_ns") != stat_before.st_mtime_ns):
            raise ValueError(f"native64 seed{seed} prediction was made from a different dataset version")
    with h5py.File(dataset, "r") as source:
        if "mask" not in source or source["mask"].shape != (100000, 128, 128, 1):
            raise ValueError("canonical GT mask HDF5 schema changed")
        target_pixels = np.asarray(source["mask"][IMAGE_IDS[0]:IMAGE_IDS[-1] + 1])
    if target_pixels.shape != (COUNT, 128, 128, 1):
        raise ValueError("fixed320 GT slice has unexpected shape")
    target = modal_native64(target_pixels)
    scores = {}
    for seed in SEEDS:
        scores[str(seed)] = {
            "native64": score_slot64._production_metrics64(own[seed]["labels"], target),
            "slot64": score_slot64._production_metrics64(slots[seed]["labels64"], target),
        }
    means = {model: {metric: {
        "per_seed": {str(seed): scores[str(seed)][model][metric]["mean"] for seed in SEEDS},
        "three_seed_mean": float(np.mean([scores[str(seed)][model][metric]["mean"] for seed in SEEDS]))
    } for metric in contract.METRICS} for model in ("native64", "slot64")}
    draws = np.random.RandomState(139).randint(0, COUNT, size=(10000, COUNT))
    paired = {str(seed): {metric: _paired_bootstrap(
        scores[str(seed)]["native64"][metric]["per_image"],
        scores[str(seed)]["slot64"][metric]["per_image"], draws)
        for metric in contract.METRICS} for seed in SEEDS}
    pooled = {metric: _paired_bootstrap(
        np.mean([scores[str(seed)]["native64"][metric]["per_image"] for seed in SEEDS], axis=0),
        np.mean([scores[str(seed)]["slot64"][metric]["per_image"] for seed in SEEDS], axis=0), draws)
        for metric in contract.METRICS}
    stat_after = dataset.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
        raise ValueError("canonical dataset changed during GT scoring")
    return {"experiment": "SW0154_native64_resolution_compare", "status": "complete",
        "seed_scope": list(SEEDS),
        "evaluation_contract": {"image_ids": [IMAGE_IDS[0], IMAGE_IDS[-1], COUNT],
            "grid": [64, 64], "metrics": list(contract.METRICS),
            "native64": "direct actual emitted-spike source QCC; no upsampled32 labels",
            "slot64": "saved native128 IDs pooled directly by production 2x2 modal",
            "target": "canonical native128 mask labels pooled by production 2x2 modal"},
        "ground_truth_opened_after_all_six_predictions_validated": True,
        "implementation_fingerprint": fingerprint,
        "prediction_provenance": {str(seed): {
            "native64_prediction_sha256": own[seed]["prediction_sha256"],
            "native64_protocol_sha256": own[seed]["protocol_sha256"],
            "native64_source_checkpoint_sha256": own[seed]["protocol"]["source_checkpoint_sha256"],
            "native64_model_initialization_seed": own[seed]["protocol"]["model_initialization_seed"],
            "slot64_prediction_sha256": slots[seed]["prediction_sha256"],
            "slot64_protocol_sha256": slots[seed]["protocol_sha256"],
            "slot64_checkpoint_sha256": slots[seed]["protocol"]["checkpoint_sha256"]}
            for seed in SEEDS},
        "scores": scores, "means": means,
        "paired_native64_minus_slot64": paired,
        "paired_three_seed_per_image_mean_difference": pooled,
        "bootstrap": {"samples": 10000, "rng": "numpy RandomState", "seed": 139,
                      "paired_common_image_indices": True},
        "canonical_dataset_stat": {"size": stat_after.st_size,
                                    "mtime_ns": stat_after.st_mtime_ns},
        "interpretation_limit": "This fixed320 score compares one source seed per registered seed on a native64 grid; it is not a fresh64 training or scaling result and does not establish superiority."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--prediction-root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument("--slot-root", type=Path, default=SLOT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"preserve existing native64 score: {args.output}")
    report = score_all(dataset=args.dataset, prediction_root=args.prediction_root,
                       slot_root=args.slot_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "ground_truth_opened_after_all_six_predictions_validated": True},
                     allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
