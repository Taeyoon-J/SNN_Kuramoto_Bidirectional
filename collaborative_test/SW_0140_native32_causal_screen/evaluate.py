"""Frozen native32 causal-intervention predictions and GT-last scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding import score_slot32
from collaborative_test.SW_0135_native32_spike_binding.evaluation_contract32 import (
    COUNT, GRID, IMAGE_IDS, METRICS, modal_native32, production_metrics32,
    require_fixed_image_ids, sha256_file, slot_native_pixels_to32, validate_slot_protocol,
)
from collaborative_test.SW_0135_native32_spike_binding.foundation import load_native32_foundation
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136_eval
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0136_native32_transfer.transfer import state_dict_sha256
from collaborative_test.SW_0140_native32_causal_screen import screen as sw140_screen
from collaborative_test.SW_0140_native32_causal_screen.causal_rollout import (
    CONDITIONS, INTERVENTIONS, TRACE_FIELDS, causal_rollout,
)

SEEDS = (0, 1, 2)
CONDITIONS_TO_SCORE = INTERVENTIONS
PREDICTION_ROOT = HERE / "results_archive" / "causal_predictions"
SUMMARY_PATH = HERE / "results_archive" / "causal_evaluation.json"
DATASET = Path(sw130.source97.DATASET)
SLOT_ROOT = ROOT / "trained_models/SW0092_slot_our70000_eval"
SOURCE_BASELINE_ROOT = sw137.OUTPUT_ROOT
SLOT_BASELINE_PATH = sw137.SLOT_BASELINE
SLOT_BASELINE_SHA256 = sw137.SLOT_BASELINE_SHA256


def implementation_fingerprint():
    paths = (
        HERE / "evaluation_protocol.json", HERE / "causal_rollout.py",
        HERE / "screen.py", HERE / "evaluate.py",
        ROOT / "collaborative_test/SW_0140_native32_causal_screen/protocol.json",
        ROOT / "collaborative_test/SW_0130_phase_state_integration/run.py",
        ROOT / "collaborative_test/SW_0130_phase_state_integration/model.py",
        ROOT / "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        ROOT / "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        ROOT / "collaborative_test/SW_0135_native32_spike_binding/resolution.py",
        ROOT / "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
        ROOT / "collaborative_test/SW_0135_native32_spike_binding/score_slot32.py",
        ROOT / "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        ROOT / "collaborative_test/SW_0137_native32_source_qcc/run.py",
        ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
        ROOT / "snn_kuramoto_bidirectional/graph_generator.py",
        ROOT / "snn_kuramoto_bidirectional/dendric_layer.py",
        ROOT / "snn_kuramoto_bidirectional/membrane_layer.py",
        ROOT / "snn_kuramoto_bidirectional/kuramoto_layer.py",
        ROOT / "snn_kuramoto_bidirectional/sinusoidal_gating.py",
        ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
        ROOT / "snn_kuramoto_bidirectional/evaluation.py",
        ROOT / "snn_kuramoto_bidirectional/gamma_initializer.py",
        ROOT / "snn_kuramoto_bidirectional/hyperparameter.py",
        ROOT / "snn_kuramoto_bidirectional/training/train_gamma_initializer.py",
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0140 evaluation dependency missing: {missing[0]}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def screen_path(seed):
    return HERE / "results_archive" / f"causal_screen_seed{int(seed)}.json"


@lru_cache(maxsize=3)
def _source_contract(seed):
    return sw130.source_contract(int(seed))


@lru_cache(maxsize=1)
def _validated_rgb_assets():
    """Bind this endpoint to the same validated encoder/RGB inputs as SW0137."""
    return sw130.validate_rgb_assets()


def prediction_dir(seed, condition, output_root=PREDICTION_ROOT):
    if int(seed) not in SEEDS or condition not in CONDITIONS_TO_SCORE:
        raise ValueError("unregistered SW0140 seed or condition")
    return Path(output_root) / f"seed{int(seed)}" / condition


def _write_once_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _write_once_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def validate_screen(seed, *, screen_file=None, foundation=None):
    seed = int(seed)
    path = screen_path(seed) if screen_file is None else Path(screen_file)
    before_sha = sha256_file(path)
    row = json.loads(path.read_text(encoding="utf-8"))
    checkpoint, manifest_path, manifest, pool, ids, source_sha = _source_contract(seed)
    expected_fp = (sw140_screen.screen_fingerprint(foundation.provenance["implementation_fingerprint"])
                   if foundation is not None else None)
    fp = row.get("implementation_fingerprint")
    local_fp_valid = (isinstance(fp, dict) and all(
        fp.get((HERE / name).relative_to(ROOT).as_posix()) == sha256_file(HERE / name)
        for name in ("causal_rollout.py", "screen.py", "protocol.json")))
    if (row.get("experiment") != "SW0140_native32_causal_screen"
            or row.get("status") != "screen_complete" or row.get("seed") != seed
            or row.get("steps") != 1024 or row.get("images") != 4
            or row.get("ground_truth_used") is not False or row.get("optimizer_updates") != 0
            or row.get("source_core_sha256") != source_sha
            or row.get("source_manifest_sha256") != sha256_file(manifest_path)
            or row.get("image_ids") != [int(v) for v in ids[:4]]
            or row.get("pool_rows") != [int(v) for v in pool[:4]]
            or row.get("conditions") != list(CONDITIONS)
            or (expected_fp is not None and fp != expected_fp)
            or not local_fp_valid):
        raise ValueError(f"SW0140 seed{seed} causal screen is missing, stale, or invalid")
    for batch in row.get("batches", []):
        if (batch.get("conditions", {}).get("none", {}).get("parity") is not True
                or set(batch.get("conditions", {})) != set(CONDITIONS)):
            raise ValueError("SW0140 causal screen lacks exact NONE parity or all registered conditions")
    if len(row.get("batches", [])) != 4 or sha256_file(path) != before_sha:
        raise ValueError("SW0140 causal screen changed or has incomplete batch records")
    return {"record": row, "path": path, "sha256": before_sha,
            "source_core_path": checkpoint, "source_sha256": source_sha}


def _validate_trace(trace, batch=1):
    expected = {
        "theta": (batch, 1024, 1024, 4),
        "membrane": (batch, 1024, 1024),
        "spikes": (batch, 1024, 1024),
        "component_membrane": (batch, 4, 1024, 1024),
        "component_spikes": (batch, 4, 1024, 1024),
        "component_gates": (batch, 4, 1024, 1024),
    }
    checks = []
    for name, shape in expected.items():
        value = trace.get(name)
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"SW0140 trace {name} must have shape {shape}")
        checks.append(torch.isfinite(value).all())
    if not bool(torch.stack(checks).all().item()):
        raise FloatingPointError("SW0140 intervention trace contains nonfinite values")


@torch.no_grad()
def predict_condition(seed, condition, *, device="cuda:0", output_root=PREDICTION_ROOT,
                      foundation_loader=None, val_cache=None):
    seed, condition = int(seed), str(condition)
    if seed not in SEEDS or condition not in CONDITIONS_TO_SCORE:
        raise ValueError("unregistered SW0140 seed or condition")
    out_dir = prediction_dir(seed, condition, output_root)
    if out_dir.exists():
        raise FileExistsError(f"preserve existing SW0140 prediction: {out_dir}")
    device_obj = torch.device(device)
    if device_obj.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0140 fixed endpoint prediction requires an exclusive CUDA device")
    torch.cuda.set_device(device_obj)
    torch.cuda.init()
    loader = load_native32_foundation if foundation_loader is None else foundation_loader
    foundation = loader(seed, device_obj, verify_rgb_assets=True)
    foundation.wrapped.eval().requires_grad_(False)
    foundation.encoder.eval().requires_grad_(False)
    screen_refs = {screen_seed: validate_screen(
        screen_seed, foundation=foundation if screen_seed == seed else None)
        for screen_seed in SEEDS}
    screen_ref = screen_refs[seed]
    if val_cache is None:
        val_cache = np.load(sw130.VAL_RGB, mmap_mode="r")
    if (tuple(val_cache.shape) != (COUNT, 128, 128, 3)
            or val_cache.dtype != np.uint8):
        raise ValueError("SW0140 endpoint requires the ordered uint8 native128 validation RGB cache")
    ids = np.asarray(IMAGE_IDS, dtype=np.int64)
    state_before = state_dict_sha256((("wrapped", foundation.wrapped),
                                     ("encoder", foundation.encoder)))
    labels = np.empty((COUNT, GRID, GRID), dtype=np.int64)
    qcc_labels = np.empty_like(labels)
    for index in range(COUNT):
        rgb = torch.from_numpy(np.asarray(val_cache[index:index + 1]).copy()).to(device_obj)
        gamma = foundation.encode(rgb)
        if tuple(gamma.shape) != (1, 8, 1024) or not bool(torch.isfinite(gamma).all().item()):
            raise ValueError("SW0140 native32 validation gamma is invalid")
        trace = causal_rollout(foundation.wrapped, gamma, intervention=condition,
                               total_steps=1024)
        _validate_trace(trace)
        qcc = sw136_eval._qcc_labels(trace)
        if torch.is_tensor(qcc):
            qcc = qcc.detach().cpu().numpy()
        if tuple(qcc.shape) != (1, GRID, GRID) or not np.isfinite(qcc).all():
            raise ValueError("SW0140 actual-spike QCC produced invalid native32 labels")
        labels[index] = qcc[0]
        qcc_labels[index] = qcc[0]
        if index % 40 == 39:
            print(json.dumps({"stage": "causal_prediction", "seed": seed,
                              "condition": condition, "predicted": index + 1,
                              "count": COUNT}), flush=True)
    state_after = state_dict_sha256((("wrapped", foundation.wrapped),
                                    ("encoder", foundation.encoder)))
    if state_after != state_before:
        raise AssertionError("SW0140 frozen source/encoder state changed during evaluation")
    protocol_path = out_dir / "protocol.json"
    archive_path = out_dir / "predictions.npz"
    _write_once_npz(archive_path, image_ids=ids, labels=labels, qcc_labels=qcc_labels)
    protocol = {
        "experiment": "SW0140_native32_causal_endpoint",
        "status": "predictions_complete", "seed": seed, "condition": condition,
        "image_ids": [IMAGE_IDS[0], IMAGE_IDS[-1]], "count": COUNT,
        "grid_size": [GRID, GRID], "prediction_shape": [COUNT, GRID, GRID],
        "steps": 1024, "settle": 512, "microbatch_size": 1,
        "readout": {"affinity": "actual-spike positive signed component Pearson product",
                    "threshold": 0.50, "min_group_size": 8,
                    "background": "largest connected component"},
        "prediction_sha256": sha256_file(archive_path),
        "screen_record_sha256": screen_ref["sha256"],
        "screen_implementation_fingerprint": screen_ref["record"]["implementation_fingerprint"],
        "all_screen_records_sha256": {str(s): screen_refs[s]["sha256"] for s in SEEDS},
        "source_core_sha256": foundation.provenance["source_core_sha256"],
        "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
        "source_native32_state_sha256": state_before,
        "encoder_sha256": foundation.provenance["encoder_sha256"],
        "feature_preprocessing_sha256": foundation.provenance["feature_preprocessing_sha256"],
        "rgb_assets": foundation.provenance["rgb_asset_validation"],
        "implementation": implementation_fingerprint(),
        "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
    }
    _write_once_json(protocol_path, protocol)
    return {"seed": seed, "condition": condition, "prediction_sha256": protocol["prediction_sha256"],
            "protocol_sha256": sha256_file(protocol_path), "status": protocol["status"]}


def validate_prediction(seed, condition, *, output_root=PREDICTION_ROOT):
    directory = prediction_dir(seed, condition, output_root)
    archive_path, protocol_path = directory / "predictions.npz", directory / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    screen_refs = {screen_seed: validate_screen(screen_seed) for screen_seed in SEEDS}
    screen_ref = screen_refs[int(seed)]
    source = _source_contract(int(seed))
    assets = _validated_rgb_assets()
    if (protocol.get("experiment") != "SW0140_native32_causal_endpoint"
            or protocol.get("status") != "predictions_complete"
            or protocol.get("seed") != int(seed) or protocol.get("condition") != condition
            or protocol.get("image_ids") != [IMAGE_IDS[0], IMAGE_IDS[-1]]
            or protocol.get("count") != COUNT or protocol.get("grid_size") != [GRID, GRID]
            or protocol.get("prediction_shape") != [COUNT, GRID, GRID]
            or protocol.get("steps") != 1024 or protocol.get("settle") != 512
            or protocol.get("microbatch_size") != 1
            or protocol.get("ground_truth_used_for_prediction") is not False
            or protocol.get("optimizer_updates") != 0
            or protocol.get("source_core_sha256") != source[5]
            or protocol.get("source_manifest_sha256") != sha256_file(source[1])
            or protocol.get("encoder_sha256") != assets["encoder_source_sha256"]
            or protocol.get("feature_preprocessing_sha256") != assets["feature_preprocessing_sha256"]
            or protocol.get("rgb_assets") != assets
            or protocol.get("screen_record_sha256") != screen_ref["sha256"]
            or protocol.get("screen_implementation_fingerprint")
                != screen_ref["record"].get("implementation_fingerprint")
            or protocol.get("all_screen_records_sha256")
                != {str(s): screen_refs[s]["sha256"] for s in SEEDS}
            or protocol.get("implementation") != implementation_fingerprint()
            or protocol.get("readout") != {"affinity": "actual-spike positive signed component Pearson product",
                                            "threshold": 0.50, "min_group_size": 8,
                                            "background": "largest connected component"}
            or sha256_file(archive_path) != protocol.get("prediction_sha256")):
        raise ValueError("invalid SW0140 frozen prediction/protocol binding")
    with np.load(archive_path, allow_pickle=False) as saved:
        if not {"image_ids", "labels", "qcc_labels"}.issubset(saved.files):
            raise ValueError("SW0140 prediction archive lacks required arrays")
        ids = require_fixed_image_ids(saved["image_ids"])
        labels = np.asarray(saved["labels"])
        qcc = np.asarray(saved["qcc_labels"])
    for array, name in ((labels, "labels"), (qcc, "qcc_labels")):
        if (array.shape != (COUNT, GRID, GRID) or not np.issubdtype(array.dtype, np.integer)
                or np.issubdtype(array.dtype, np.bool_) or not np.isfinite(array).all()
                or np.any(array < 0) or np.any(array > 1024)):
            raise ValueError(f"SW0140 {name} violates the fixed320 native32 schema")
    if (sha256_file(archive_path) != protocol["prediction_sha256"]
            or hashlib.sha256(protocol_path.read_bytes()).hexdigest() != protocol_sha):
        raise ValueError("SW0140 prediction bytes changed while validating")
    if not np.array_equal(labels, qcc):
        raise ValueError("SW0140 labels and QCC labels must be the same frozen readout")
    return {"seed": int(seed), "condition": condition, "ids": ids, "labels": labels.astype(np.int64),
            "qcc_labels": qcc.astype(np.int64), "protocol": protocol,
            "prediction_sha256": protocol["prediction_sha256"], "protocol_sha256": protocol_sha,
            "archive_path": archive_path, "protocol_path": protocol_path}


def _load_source_reference(seed, output_root=SOURCE_BASELINE_ROOT):
    return sw137.validate_prediction(Path(output_root) / f"seed{seed}", seed=seed)


def _load_slot_reference(seed, slot_root=SLOT_ROOT):
    directory = Path(slot_root) / f"seed{seed}_epoch10"
    archive, protocol_path = directory / "predictions.npz", directory / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    digest = validate_slot_protocol(protocol, seed=seed, prediction_path=archive)
    with np.load(archive, allow_pickle=False) as saved:
        if not {"image_ids", "labels"}.issubset(saved.files):
            raise ValueError("Slot native128 archive lacks labels or image_ids")
        ids = require_fixed_image_ids(saved["image_ids"], name="Slot image IDs")
        native = np.asarray(saved["labels"])
    labels = slot_native_pixels_to32(native)
    if (sha256_file(archive) != digest or hashlib.sha256(protocol_path.read_bytes()).hexdigest() != protocol_sha):
        raise ValueError("Slot reference changed while being converted")
    return {"ids": ids, "labels": labels, "archive_path": archive,
            "protocol_path": protocol_path, "prediction_sha256": digest,
            "protocol_sha256": protocol_sha, "protocol": protocol}


def _bootstrap_ci(per_image_delta, indices):
    values = np.asarray(per_image_delta, dtype=np.float64)
    if values.shape != (COUNT,) or not np.isfinite(values).all():
        raise ValueError("paired bootstrap requires 320 finite per-image differences")
    boot = values[indices].mean(axis=1)
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def score_all(*, output=SUMMARY_PATH, prediction_root=PREDICTION_ROOT,
              source_root=SOURCE_BASELINE_ROOT, slot_root=SLOT_ROOT, dataset=DATASET):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0140 scoring result: {output}")
    # Load and validate every prediction and both registered reference families
    # before the first target-mask access.
    predictions = {(seed, condition): validate_prediction(seed, condition,
                    output_root=prediction_root)
                   for seed in SEEDS for condition in CONDITIONS_TO_SCORE}
    sources = {seed: _load_source_reference(seed, source_root) for seed in SEEDS}
    slots = {seed: _load_slot_reference(seed, slot_root) for seed in SEEDS}
    # The causal arms and their source97 reference must share the exact trained
    # encoder, preprocessing, RGB-cache manifests, core, and source manifest.
    for seed in SEEDS:
        source_protocol = sources[seed]["protocol"]
        for condition in CONDITIONS_TO_SCORE:
            prediction_protocol = predictions[(seed, condition)]["protocol"]
            for field in ("source_core_sha256", "source_manifest_sha256", "encoder_sha256",
                          "feature_preprocessing_sha256", "rgb_assets"):
                if prediction_protocol.get(field) != source_protocol.get(field):
                    raise ValueError(f"SW0140 {condition} seed{seed} differs from SW0137 source {field}")
    for seed in SEEDS:
        for reference in (sources[seed], slots[seed]):
            if not np.array_equal(reference["ids"], np.asarray(IMAGE_IDS, dtype=np.int64)):
                raise ValueError("SW0140 reference IDs differ from the fixed320 evaluation slice")
    # Rehash every prediction, reference and immutable endpoint contract after
    # validation and immediately before reading ground truth.
    frozen_sha = {}
    for key, row in predictions.items():
        for label, path, expected in (("npz", row["archive_path"], row["prediction_sha256"]),
                                      ("protocol", row["protocol_path"], row["protocol_sha256"])):
            if sha256_file(path) != expected:
                raise ValueError("SW0140 prediction changed before target access")
            frozen_sha[f"seed{key[0]}_{key[1]}_{label}"] = expected
    for family, rows in (("source", sources), ("slot", slots)):
        for seed, row in rows.items():
            if (sha256_file(row["archive_path"]) != row["prediction_sha256"]
                    or sha256_file(row["protocol_path"]) != row["protocol_sha256"]):
                raise ValueError(f"SW0140 {family} reference changed before target access")
            frozen_sha[f"{family}_seed{seed}_npz"] = row["prediction_sha256"]
            frozen_sha[f"{family}_seed{seed}_protocol"] = row["protocol_sha256"]
    if sha256_file(SLOT_BASELINE_PATH) != SLOT_BASELINE_SHA256:
        raise ValueError("registered Slot32 baseline changed before target access")
    frozen_sha["slot32_baseline"] = SLOT_BASELINE_SHA256

    with h5py.File(dataset, "r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0] <= IMAGE_IDS[-1]:
            raise ValueError("registered target masks do not contain fixed320 IDs")
        pixel_gt = np.asarray(h5["mask"][IMAGE_IDS[0]:IMAGE_IDS[-1] + 1])
    target = modal_native32(pixel_gt)
    result = {"status": "complete", "experiment": "SW0140_native32_causal_endpoint",
              "scores": {}, "three_seed_means": {}, "paired_bootstrap": {},
              "frozen_prediction_and_reference_sha256": frozen_sha}
    baseline_rows = {seed: sources[seed]["labels"] for seed in SEEDS}
    slot_rows = {seed: slots[seed]["labels"] for seed in SEEDS}
    for seed in SEEDS:
        result["scores"][f"source97_seed{seed}"] = {
            "qcc": production_metrics32(sources[seed]["labels"], target)}
        result["scores"][f"slot32_seed{seed}"] = {
            "slot32": production_metrics32(slots[seed]["labels"], target)}
    for condition in CONDITIONS_TO_SCORE:
        for seed in SEEDS:
            result["scores"][f"{condition}_seed{seed}"] = {
                "qcc": production_metrics32(predictions[(seed, condition)]["qcc_labels"], target)}
        result["three_seed_means"][condition] = {
            metric: float(np.mean([result["scores"][f"{condition}_seed{s}"]["qcc"][metric]["mean"]
                                   for s in SEEDS])) for metric in METRICS}

    rng = np.random.RandomState(140)
    bootstrap_indices = rng.randint(0, COUNT, size=(10000, COUNT))
    for condition in CONDITIONS_TO_SCORE:
        result["paired_bootstrap"][condition] = {}
        for reference_name, reference_rows in (("source97", baseline_rows), ("slot32", slot_rows)):
            result["paired_bootstrap"][condition][reference_name] = {}
            for metric in METRICS:
                diffs = []
                for seed in SEEDS:
                    left = np.asarray(result["scores"][f"{condition}_seed{seed}"]["qcc"][metric]["per_image"])
                    right_key = "qcc" if reference_name == "source97" else "slot32"
                    right = np.asarray(result["scores"][f"{reference_name}_seed{seed}"][right_key][metric]["per_image"])
                    diffs.append(left - right)
                per_image = np.mean(np.stack(diffs, axis=0), axis=0)
                result["paired_bootstrap"][condition][reference_name][metric] = {
                    "three_seed_mean_difference": float(per_image.mean()),
                    "ci95": _bootstrap_ci(per_image, bootstrap_indices),
                    "resamples": 10000, "rng": "RandomState(140)",
                }
    slot_baseline = json.loads(SLOT_BASELINE_PATH.read_text(encoding="utf-8"))
    result["slot32_registered_baseline_sha256"] = SLOT_BASELINE_SHA256
    result["slot32_recomputed_three_seed_mean"] = {
        metric: float(np.mean([result["scores"][f"slot32_seed{s}"]["slot32"][metric]["mean"]
                              for s in SEEDS])) for metric in METRICS}
    result["slot32_saved_baseline_three_seed_mean"] = {
        metric: float(slot_baseline["scores"]["slot32"][metric]["three_seed_mean"])
        for metric in METRICS}
    for metric in METRICS:
        if abs(result["slot32_recomputed_three_seed_mean"][metric]
               - result["slot32_saved_baseline_three_seed_mean"][metric]) > 1e-10:
            raise ValueError(f"Slot32 {metric} failed registered baseline reproduction")
    result.update({
        "seed_scope": list(SEEDS), "conditions": list(CONDITIONS_TO_SCORE),
        "image_ids": [IMAGE_IDS[0], IMAGE_IDS[-1]], "count": COUNT,
        "grid_size": [GRID, GRID], "metrics": list(METRICS),
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
        "optimizer_updates": 0, "implementation": implementation_fingerprint(),
        "evaluation_protocol_sha256": sha256_file(HERE / "evaluation_protocol.json"),
        "interpretation": "Frozen causal-ablation scores only. These results do not establish intervention benefit, gate necessity, or a promotion decision.",
    })
    _write_once_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("predict", "score"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--condition", choices=CONDITIONS_TO_SCORE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prediction-root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_BASELINE_ROOT)
    parser.add_argument("--slot-root", type=Path, default=SLOT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)
    if args.stage == "predict":
        if args.seed is None or args.condition is None:
            parser.error("--seed and --condition are required for prediction")
        result = predict_condition(args.seed, args.condition, device=args.device,
                                  output_root=args.prediction_root)
    else:
        result = score_all(output=args.output, prediction_root=args.prediction_root,
                           source_root=args.source_root, slot_root=args.slot_root,
                           dataset=args.dataset)
    print(json.dumps({"stage": args.stage, "status": result.get("status"),
                      "seed": args.seed, "condition": args.condition,
                      "ground_truth_used_for_prediction": False, "optimizer_updates": 0},
                     allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
