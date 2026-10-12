"""Frozen native32 selective geodesic comparison; inference only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"), str(ROOT / "snn_kuramoto_bidirectional")]

import h5py
import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0135_native32_spike_binding import foundation
from collaborative_test.SW_0135_native32_spike_binding import score_slot32
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136_eval
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_operator
ARMS = ("R", "C", "T")
SEEDS = (0, 1, 2)
ARCHIVE = HERE / "results_archive"
PREDICTIONS = ARCHIVE / "predictions"
PREFLIGHTS = ARCHIVE / "preflight"
SUMMARY = ARCHIVE / "evaluation.json"
DATASET = Path(sw130.source97.DATASET)
SLOT_ROOT = ROOT / "trained_models/SW0092_slot_our70000_eval"
SLOT_BASELINE = ROOT / "collaborative_test/SW_0135_native32_spike_binding/results_archive/slot70k_native32_baseline.json"
SLOT_BASELINE_SHA256 = "4dc12cb54c3114375b8428cd250c013ff94015c4826915751f2fcb0a2204d753"
TRACE_FIELDS = ("theta", "membrane", "spikes", "component_membrane", "component_spikes", "component_gates")
BOOTSTRAP_SEED = 139
BOOTSTRAP_REPLICATES = 10_000


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_once(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def _write_npz_once(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def implementation_fingerprint():
    merged = {}
    merged.update(foundation.implementation_fingerprint())
    merged.update(sw137.implementation_fingerprint())
    paths = [HERE / "protocol.json", HERE / "run.py", HERE / "geodesic_operator.py",
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
             ROOT / "snn_kuramoto_bidirectional/training/train_gamma_initializer.py"]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0143 dependency missing: {missing[0]}")
    own = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}
    merged.update(own)
    return merged


def _arm_dir(arm, seed, output_root=PREDICTIONS):
    if arm not in ARMS or int(seed) not in SEEDS:
        raise ValueError("unregistered SW0143 arm/seed")
    return Path(output_root) / arm / f"seed{int(seed)}"


def _preflight_path(seed, output_root=PREFLIGHTS):
    if int(seed) not in SEEDS:
        raise ValueError("unregistered SW0143 seed")
    return Path(output_root) / f"seed{int(seed)}.json"


def _arm_spec(arm):
    return {"R": {"effective_contrast": "source learned", "dissimilarity_power": 1},
            "C": {"effective_contrast": 8.0, "dissimilarity_power": 1},
            "T": {"effective_contrast": 8.0, "dissimilarity_power": 2}}[arm]


def _native_source_record(found):
    return {key: found.provenance[key] for key in (
        "source_core_sha256", "source_manifest_sha256", "source_training_ids_sha256",
        "encoder_sha256", "feature_preprocessing_sha256", "rgb_asset_validation")}


def _load_rgb_rows(cache_path, rows):
    cache = np.load(cache_path, mmap_mode="r")
    if cache.dtype != np.uint8 or cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered native128 RGB TRAIN cache schema mismatch")
    return torch.from_numpy(np.asarray(cache[np.asarray(rows, dtype=np.int64)]).copy())


def trace_max_abs(left, right):
    """Descriptive cross-batch trace differences; never an admission test."""
    result = {}
    for key in TRACE_FIELDS:
        a, b = left[key], right[key]
        if a.shape[1:] != b.shape[1:]:
            raise ValueError(f"cannot compare trace shapes for {key}: {tuple(a.shape)} vs {tuple(b.shape)}")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise FloatingPointError(f"nonfinite trace in descriptive comparison: {key}")
        result[key] = float((a.detach().double() - b.detach().double()).abs().max().cpu())
    return result


def assert_same_batch_trace(reference, candidate, label):
    """Require exact parity for identically shaped production rollouts."""
    parity = {}
    for key in TRACE_FIELDS:
        a, b = reference[key], candidate[key]
        if a.shape != b.shape:
            raise AssertionError(f"{label} trace shape differs at {key}: {tuple(a.shape)} != {tuple(b.shape)}")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise FloatingPointError(f"nonfinite {label} trace: {key}")
        parity[key] = torch.equal(a, b)
        if not parity[key]:
            raise AssertionError(f"{label} trace differs from the frozen SW0135 path at {key}")
    return parity


@torch.no_grad()
def preflight_seed(seed, *, device="cuda:0", output_root=PREFLIGHTS):
    """GT-free source-order TRAIN structural and B1/B4 parity check."""
    seed = int(seed); destination = _preflight_path(seed, output_root)
    if destination.exists():
        raise FileExistsError(f"preserve existing SW0143 preflight: {destination}")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0143 full native32 preflight requires exclusive CUDA")
    torch.cuda.set_device(device); torch.cuda.init()
    found = foundation.load_native32_foundation(seed, device, verify_rgb_assets=True)
    found.encoder.eval(); found.wrapped.eval()
    checkpoint_graph_method = found.wrapped.core.graph_generator._geodesic_distance
    graph = found.wrapped.core.graph_generator
    rgb = _load_rgb_rows(sw130.TRAIN_RGB, found.pool_indices[:4]).to(device)
    gamma4 = found.encode(rgb)
    graph_before = graph(gamma4)
    torch.cuda.reset_peak_memory_stats(device)
    geodesic_operator.install_operator(graph, "R")
    graph_after = graph(gamma4)
    if not torch.equal(graph_before, graph_after):
        raise AssertionError("R operator changed the exact SW0135 native graph output")
    # Freeze original traces at both batch shapes. Cross-batch variation is
    # descriptive only; it is not a scientific admission threshold.
    graph._geodesic_distance = checkpoint_graph_method
    native_trace4 = late_rollout(found.wrapped, gamma4, total_steps=1024, live_tail_steps=0)
    native_trace = late_rollout(found.wrapped, gamma4[:1], total_steps=1024, live_tail_steps=0)
    geodesic_operator.install_operator(graph, "R")
    trace4 = late_rollout(found.wrapped, gamma4, total_steps=1024, live_tail_steps=0)
    trace1 = late_rollout(found.wrapped, gamma4[:1], total_steps=1024, live_tail_steps=0)
    source_parity = {
        "B1": assert_same_batch_trace(native_trace, trace1, "R B1"),
        "B4": assert_same_batch_trace(native_trace4, trace4, "R B4"),
    }
    cross_batch_native = trace_max_abs(
        {key: value[:1] for key, value in native_trace4.items()}, native_trace)
    cross_batch_r = trace_max_abs({key: value[:1] for key, value in trace4.items()}, trace1)
    # Verify both registered intervention operators can execute a full B4
    # native32 training-shaped forward before any fixed320 predictions.
    intervention_shapes = {}
    for arm in ("C", "T"):
        geodesic_operator.install_operator(graph, arm)
        trace = late_rollout(found.wrapped, gamma4, total_steps=1024, live_tail_steps=0)
        for key in TRACE_FIELDS:
            if not torch.isfinite(trace[key]).all():
                raise FloatingPointError(f"nonfinite B4 {arm} preflight trace: {key}")
        intervention_shapes[arm] = {key: list(trace[key].shape) for key in TRACE_FIELDS}
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    source = sw130.source_contract(seed)
    source_labels = sw137.validate_prediction(sw137._prediction_dir(seed), seed=seed)
    # A B1 TRAIN QCC call exercises the unchanged native readout and confirms
    # the runtime mixed tensor/NumPy return contract before validation inference.
    train_qcc = sw136_eval._predict_one(found, None, gamma4[:1], primary=False)[1]
    if tuple(np.asarray(train_qcc).shape) != (1, 32, 32) or not np.isfinite(train_qcc).all():
        raise ValueError("native TRAIN QCC preflight has the wrong array shape or nonfinite labels")
    record = {
        "experiment": "SW0143_native32_selective_geodesic", "status": "preflight_complete",
        "seed": seed, "implementation": implementation_fingerprint(),
        "source": _native_source_record(found),
        "training_first4_pool_rows": [int(x) for x in found.pool_indices[:4]],
        "training_first4_image_ids": [int(x) for x in found.image_ids[:4]],
        "source_r_method_exact_graph_output": True,
        "source_r_native_trace_equal": source_parity,
        "native_b1_b4_trace_max_abs_descriptive": cross_batch_native,
        "r_b1_b4_trace_max_abs_descriptive": cross_batch_r,
        "intervention_b4_trace_shapes": intervention_shapes,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "train_qcc_shape": list(np.asarray(train_qcc).shape),
        "source137_prediction_sha256": source_labels["prediction_sha256"],
        "source137_protocol_sha256": source_labels["protocol_sha256"],
        "source_core_checkpoint_sha256": source[5], "ground_truth_used": False,
        "optimizer_updates": 0,
    }
    _write_json_once(destination, record)
    return record


@torch.no_grad()
def predict_reference(seed, *, device="cuda:0", output_root=PREDICTIONS, preflight_root=PREFLIGHTS):
    """Re-run original R on all320 and require exact per-label SW0137 agreement."""
    seed = int(seed)
    preflight = json.loads(_preflight_path(seed, preflight_root).read_text(encoding="utf-8"))
    if (preflight.get("status") != "preflight_complete"
            or preflight.get("implementation") != implementation_fingerprint()
            or preflight.get("ground_truth_used") is not False
            or preflight.get("optimizer_updates") != 0):
        raise ValueError("SW0143 source R preflight is stale or invalid")
    reference = sw137.validate_prediction(sw137._prediction_dir(seed), seed=seed)
    if (reference["prediction_sha256"] != preflight.get("source137_prediction_sha256")
            or reference["protocol_sha256"] != preflight.get("source137_protocol_sha256")):
        raise ValueError("SW0137 R source labels changed since SW0143 preflight")
    target = _arm_dir("R", seed, output_root)
    if target.exists():
        raise FileExistsError(f"preserve existing SW0143 R output: {target}")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0143 R regression replay requires exclusive CUDA")
    torch.cuda.set_device(device); torch.cuda.init()
    found = foundation.load_native32_foundation(seed, device, verify_rgb_assets=True)
    found.encoder.eval(); found.wrapped.eval()
    geodesic_operator.install_operator(found.wrapped.core.graph_generator, "R")
    cache = np.load(sw130.VAL_RGB, mmap_mode="r")
    if cache.shape != (320, 128, 128, 3) or cache.dtype != np.uint8:
        raise ValueError("registered fixed320 RGB cache schema mismatch")
    ids = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
    labels = np.empty((320, 32, 32), dtype=np.int64)
    for index in range(320):
        rgb = torch.from_numpy(np.asarray(cache[index:index + 1]).copy()).to(device)
        gamma = found.encode(rgb)
        _sample, qcc = sw136_eval._predict_one(found, None, gamma, primary=False)
        qcc = np.asarray(qcc)
        if qcc.shape != (1, 32, 32) or not np.isfinite(qcc).all():
            raise ValueError(f"R QCC output invalid at fixed image index {index}")
        labels[index] = qcc[0]
        if index % 40 == 39:
            print(json.dumps({"stage": "predict_R_regression", "seed": seed,
                              "predicted": index + 1, "count": 320}), flush=True)
    if not np.array_equal(ids, reference["ids"]):
        raise ValueError("SW0137 R reference IDs changed")
    if not np.array_equal(labels, reference["labels"]):
        mismatch = int(np.count_nonzero(labels != reference["labels"]))
        raise AssertionError(f"R full320 labels differ from immutable SW0137 source ({mismatch} patches)")
    _write_npz_once(target / "predictions.npz", image_ids=ids, qcc_labels=labels)
    digest = sha256_file(target / "predictions.npz")
    protocol = {
        "experiment": "SW0143_native32_selective_geodesic", "status": "complete",
        "seed": seed, "arm": "R", "arm_spec": _arm_spec("R"),
        "image_ids": [1320, 1639], "count": 320, "grid_size": [32, 32],
        "prediction_shape": [320, 32, 32], "steps": 1024, "settle": 512,
        "microbatch_size": 1, "readout": "SW0137 source97 actual-spike QCC unchanged",
        "implementation": implementation_fingerprint(),
        "source": preflight["source"], "source137_prediction_sha256": reference["prediction_sha256"],
        "source137_protocol_sha256": reference["protocol_sha256"],
        "prediction_sha256": digest, "ground_truth_used_for_prediction": False,
        "optimizer_updates": 0, "reference_reused": False,
        "full320_exact_sw0137_label_regression": True,
    }
    _write_json_once(target / "protocol.json", protocol)
    return {"status": "reference_complete", "seed": seed, "prediction_sha256": digest,
            "exact_sw0137_labels": True}


@torch.no_grad()
def predict_arm(seed, arm, *, device="cuda:0", output_root=PREDICTIONS,
                preflight_root=PREFLIGHTS):
    seed = int(seed)
    if arm not in ("C", "T"):
        raise ValueError("only C/T require a new inference prediction; R is the frozen source reference")
    target = _arm_dir(arm, seed, output_root)
    if target.exists():
        raise FileExistsError(f"preserve existing SW0143 prediction: {target}")
    preflight = json.loads(_preflight_path(seed, preflight_root).read_text(encoding="utf-8"))
    if (preflight.get("status") != "preflight_complete"
            or preflight.get("implementation") != implementation_fingerprint()
            or preflight.get("ground_truth_used") is not False
            or preflight.get("optimizer_updates") != 0):
        raise ValueError("SW0143 preflight missing/stale; refuse altered inference")
    # Validate all source-R artifacts for the seed before a contrast intervention.
    ref = sw137.validate_prediction(sw137._prediction_dir(seed), seed=seed)
    r143 = validate_arm_prediction(_arm_dir("R", seed, output_root), seed=seed,
                                   predictions_root=output_root, preflight_root=preflight_root)
    if not np.array_equal(ref["labels"], r143["labels"]):
        raise ValueError("R source labels are not exact SW0137 regression before C/T inference")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0143 native32 prediction requires exclusive CUDA")
    torch.cuda.set_device(device); torch.cuda.init()
    found = foundation.load_native32_foundation(seed, device, verify_rgb_assets=True)
    found.encoder.eval(); found.wrapped.eval()
    source_record = _native_source_record(found)
    graph = found.wrapped.core.graph_generator
    geodesic_operator.install_operator(graph, arm)
    cache = np.load(sw130.VAL_RGB, mmap_mode="r")
    if cache.shape != (320, 128, 128, 3) or cache.dtype != np.uint8:
        raise ValueError("registered fixed320 RGB cache schema mismatch")
    ids = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
    labels = np.empty((320, 32, 32), dtype=np.int64)
    for index in range(320):
        rgb = torch.from_numpy(np.asarray(cache[index:index + 1]).copy()).to(device)
        gamma = found.encode(rgb)
        _sample, qcc = sw136_eval._predict_one(found, None, gamma, primary=False)
        qcc = np.asarray(qcc)
        if qcc.shape != (1, 32, 32) or not np.isfinite(qcc).all():
            raise ValueError(f"SW0143 QCC output invalid at fixed image index {index}")
        labels[index] = qcc[0]
        if index % 40 == 39:
            print(json.dumps({"stage": "predict", "arm": arm, "seed": seed,
                              "predicted": index + 1, "count": 320}), flush=True)
    if not np.array_equal(ids, ref["ids"]):
        raise ValueError("fixed validation IDs changed during inference")
    _write_npz_once(target / "predictions.npz", image_ids=ids, qcc_labels=labels)
    digest = sha256_file(target / "predictions.npz")
    protocol = {
        "experiment": "SW0143_native32_selective_geodesic", "status": "complete",
        "seed": seed, "arm": arm, "arm_spec": _arm_spec(arm),
        "image_ids": [1320, 1639], "count": 320, "grid_size": [32, 32],
        "prediction_shape": [320, 32, 32], "steps": 1024, "settle": 512,
        "microbatch_size": 1,
        "readout": {"affinity": "unchanged positive-clamped signed Pearson component-product on actual spikes",
                    "threshold": .50, "min_group_size": 8, "background": "largest component"},
        "implementation": implementation_fingerprint(), "source": source_record,
        "preflight_sha256": sha256_file(_preflight_path(seed, preflight_root)),
        "r_reference_prediction_sha256": r143["prediction_sha256"],
        "prediction_sha256": digest, "ground_truth_used_for_prediction": False,
        "optimizer_updates": 0,
    }
    _write_json_once(target / "protocol.json", protocol)
    return {"status": "prediction_complete", "seed": seed, "arm": arm,
            "prediction_sha256": digest}


def validate_arm_prediction(directory, *, seed, predictions_root=PREDICTIONS, preflight_root=PREFLIGHTS):
    directory = Path(directory)
    npz_path, protocol_path = directory / "predictions.npz", directory / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    if (protocol.get("experiment") != "SW0143_native32_selective_geodesic"
            or protocol.get("status") != "complete" or protocol.get("seed") != int(seed)
            or protocol.get("arm") not in ARMS
            or protocol.get("implementation") != implementation_fingerprint()
            or protocol.get("image_ids") != [1320, 1639] or protocol.get("count") != 320
            or protocol.get("grid_size") != [32, 32]
            or protocol.get("prediction_shape") != [320, 32, 32]
            or protocol.get("steps") != 1024 or protocol.get("settle") != 512
            or protocol.get("microbatch_size") != 1
            or protocol.get("ground_truth_used_for_prediction") is not False
            or protocol.get("optimizer_updates") != 0
            or protocol.get("arm_spec") != _arm_spec(protocol.get("arm"))
            or protocol.get("prediction_sha256") != sha256_file(npz_path)):
        raise ValueError(f"invalid SW0143 prediction/protocol for seed{seed}: {directory}")
    if protocol["arm"] == "R":
        ref = sw137.validate_prediction(sw137._prediction_dir(seed), seed=seed)
        source_contract = sw130.source_contract(int(seed))
        expected_id_sha = hashlib.sha256(np.asarray(source_contract[4], dtype="<i8").tobytes()).hexdigest()
        expected_source = {
            "source_core_sha256": source_contract[5],
            "source_manifest_sha256": sha256_file(source_contract[1]),
            "source_training_ids_sha256": expected_id_sha,
            "encoder_sha256": ref["protocol"]["encoder_sha256"],
            "feature_preprocessing_sha256": ref["protocol"]["feature_preprocessing_sha256"],
            "rgb_asset_validation": ref["protocol"]["rgb_assets"],
        }
        if (protocol.get("reference_reused") is not False
                or protocol.get("full320_exact_sw0137_label_regression") is not True
                or protocol.get("source137_prediction_sha256") != ref["prediction_sha256"]
                or protocol.get("source137_protocol_sha256") != ref["protocol_sha256"]
                or protocol.get("source") != expected_source):
            raise ValueError("R protocol no longer binds its SW0137 source reference")
    else:
        preflight = _preflight_path(seed, preflight_root)
        if protocol.get("preflight_sha256") != sha256_file(preflight):
            raise ValueError("intervention protocol has stale seed preflight")
        ref = validate_arm_prediction(_arm_dir("R", seed, predictions_root), seed=seed,
                                      predictions_root=predictions_root, preflight_root=preflight_root)
        if (protocol.get("r_reference_prediction_sha256") != ref["prediction_sha256"]
                or protocol.get("source") != ref["protocol"].get("source")):
            raise ValueError("intervention protocol has stale R reference")
    with np.load(npz_path, allow_pickle=False) as archive:
        if not {"image_ids", "qcc_labels"}.issubset(archive.files):
            raise ValueError("SW0143 NPZ must contain image_ids and qcc_labels")
        ids = contract.require_fixed_image_ids(archive["image_ids"])
        labels = np.asarray(archive["qcc_labels"])
    if labels.shape != (320, 32, 32) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("SW0143 labels do not match direct native32 integer contract")
    if (sha256_file(npz_path) != protocol["prediction_sha256"]
            or sha256_file(protocol_path) != protocol_sha):
        raise ValueError("SW0143 prediction/protocol changed during validation")
    if protocol["arm"] == "R" and not np.array_equal(labels, ref["labels"]):
        raise ValueError("saved R labels are not exact SW0137 reference labels")
    return {"seed": int(seed), "arm": protocol["arm"], "ids": ids,
            "labels": labels.astype(np.int64, copy=False), "protocol": protocol,
            "npz_path": npz_path, "protocol_path": protocol_path,
            "prediction_sha256": protocol["prediction_sha256"],
            "protocol_sha256": protocol_sha}


def _bootstrap_paired(seed_rows_a, seed_rows_b):
    # Each seed contains one score per fixed image. Use identical resampled
    # image indices across paired arms and seeds; seed means remain unweighted.
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    draw = rng.randint(0, 320, size=(BOOTSTRAP_REPLICATES, 320))
    per_seed_delta = np.stack([a - b for a, b in zip(seed_rows_a, seed_rows_b)], axis=0)
    samples = per_seed_delta[:, draw].mean(axis=(0, 2))
    return {"mean": float(per_seed_delta.mean()),
            "ci95": [float(x) for x in np.percentile(samples, [2.5, 97.5])],
            "seed_means": [float(row.mean()) for row in per_seed_delta]}


def score_all(*, dataset=DATASET, slot_root=SLOT_ROOT, predictions_root=PREDICTIONS,
              preflight_root=PREFLIGHTS,
              output=SUMMARY):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0143 score: {output}")
    loaded = {arm: {} for arm in ARMS}
    for seed in SEEDS:
        for arm in ARMS:
            loaded[arm][seed] = validate_arm_prediction(
                _arm_dir(arm, seed, predictions_root), seed=seed,
                predictions_root=predictions_root, preflight_root=preflight_root)
    slot_rows = {}
    for seed in SEEDS:
        directory = Path(slot_root) / f"seed{seed}_epoch10"
        path, protocol_path = directory / "predictions.npz", directory / "protocol.json"
        protocol = score_slot32._read_json(protocol_path)
        contract.validate_slot_protocol(protocol, seed=seed, prediction_path=path)
        with np.load(path, allow_pickle=False) as saved:
            if not {"labels", "image_ids"}.issubset(saved.files):
                raise ValueError("Slot archive must contain native128 labels and image_ids")
            ids = contract.require_fixed_image_ids(saved["image_ids"], name="Slot IDs")
            pixels = np.asarray(saved["labels"])
        labels32 = contract.slot_native_pixels_to32(pixels)
        slot_rows[seed] = {"ids": ids, "labels": labels32, "npz": path,
                           "protocol_path": protocol_path,
                           "npz_sha": sha256_file(path), "protocol_sha": sha256_file(protocol_path),
                           "protocol": protocol}
    if sha256_file(SLOT_BASELINE) != SLOT_BASELINE_SHA256:
        raise ValueError("registered Slot32 baseline SHA changed")
    baseline = json.loads(SLOT_BASELINE.read_text(encoding="utf-8"))
    for seed in SEEDS:
        ids = loaded["R"][seed]["ids"]
        for arm in ARMS:
            if not np.array_equal(ids, loaded[arm][seed]["ids"]):
                raise ValueError("paired SW0143 arm IDs differ")
        if not np.array_equal(ids, slot_rows[seed]["ids"]):
            raise ValueError("Slot and SW0143 prediction IDs differ")
    # Validate all arms, all seeds, and Slot files before opening any target mask.
    for arm in ARMS:
        for seed in SEEDS:
            row = loaded[arm][seed]
            if (sha256_file(row["npz_path"]) != row["prediction_sha256"]
                    or sha256_file(row["protocol_path"]) != row["protocol_sha256"]):
                raise ValueError("frozen SW0143 prediction changed before GT access")
    for row in slot_rows.values():
        if sha256_file(row["npz"]) != row["npz_sha"] or sha256_file(row["protocol_path"]) != row["protocol_sha"]:
            raise ValueError("frozen Slot predictions changed before GT access")
    if sha256_file(SLOT_BASELINE) != SLOT_BASELINE_SHA256:
        raise ValueError("Slot baseline changed before GT access")
    with h5py.File(dataset, "r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0] <= 1639:
            raise ValueError("HDF5 lacks the registered fixed320 target-mask rows")
        pixels_gt = np.asarray(h5["mask"][1320:1640])
    gt = contract.modal_native32(pixels_gt)
    scores = {arm: {seed: contract.production_metrics32(loaded[arm][seed]["labels"], gt)
                    for seed in SEEDS} for arm in ARMS}
    slot_scores = {seed: contract.production_metrics32(slot_rows[seed]["labels"], gt) for seed in SEEDS}
    means = {}
    for arm in ARMS:
        means[arm] = {metric: float(np.mean([scores[arm][seed][metric]["mean"] for seed in SEEDS])
                              ) for metric in contract.METRICS}
    slot_mean = {metric: float(np.mean([slot_scores[seed][metric]["mean"] for seed in SEEDS]))
                 for metric in contract.METRICS}
    for metric in contract.METRICS:
        expected = float(baseline["scores"]["slot32"][metric]["three_seed_mean"])
        if abs(slot_mean[metric] - expected) > 1e-10:
            raise ValueError(f"registered Slot {metric} baseline does not reproduce exactly")
    paired = {}
    for left, right in (("T", "C"), ("C", "R"), ("T", "R"), ("T", "Slot32")):
        paired[f"{left}-{right}"] = {}
        for metric in contract.METRICS:
            a = ([np.asarray(scores[left][s][metric]["per_image"]) for s in SEEDS]
                 if left != "Slot32" else [np.asarray(slot_scores[s][metric]["per_image"]) for s in SEEDS])
            b = ([np.asarray(scores[right][s][metric]["per_image"]) for s in SEEDS]
                 if right != "Slot32" else [np.asarray(slot_scores[s][metric]["per_image"]) for s in SEEDS])
            paired[f"{left}-{right}"][metric] = _bootstrap_paired(a, b)
    gate = {
        "T_FG_mean_above_R": means["T"]["fg_ari"] > means["R"]["fg_ari"],
        "T_FG_mean_above_Slot32": means["T"]["fg_ari"] > slot_mean["fg_ari"],
        "T_minus_R_FG_CI_lower_positive": paired["T-R"]["fg_ari"]["ci95"][0] > 0,
        "T_IoUs_above_Slot32": all(means["T"][m] > slot_mean[m] for m in contract.METRICS[1:]),
        "T_minus_Slot32_FG_CI_lower_positive": paired["T-Slot32"]["fg_ari"]["ci95"][0] > 0,
    }
    report = {
        "experiment": "SW0143_native32_selective_geodesic", "status": "complete",
        "implementation": implementation_fingerprint(), "seed_scope": list(SEEDS),
        "prediction_contract": {"image_ids": [1320, 1639, 320], "grid": [32, 32],
                                "steps": 1024, "settle": 512, "ground_truth_used_for_prediction": False},
        "scores": {arm: {str(seed): scores[arm][seed] for seed in SEEDS} for arm in ARMS},
        "slot_scores": {str(seed): slot_scores[seed] for seed in SEEDS},
        "three_seed_means": {**means, "Slot32": slot_mean}, "paired_bootstrap": paired,
        "bootstrap": {"rng": "numpy RandomState", "seed": BOOTSTRAP_SEED,
                      "replicates": BOOTSTRAP_REPLICATES, "common_image_resamples": True},
        "registered_gate": gate, "all_registered_gate_conditions_pass": all(gate.values()),
        "provenance": {"slot_baseline_sha256": SLOT_BASELINE_SHA256,
            "slot": {str(seed): {"prediction_sha256": slot_rows[seed]["npz_sha"],
                                  "protocol_sha256": slot_rows[seed]["protocol_sha"],
                                  "checkpoint_sha256": slot_rows[seed]["protocol"]["checkpoint_sha256"]}
                      for seed in SEEDS},
            "predictions": {arm: {str(seed): loaded[arm][seed]["prediction_sha256"] for seed in SEEDS}
                            for arm in ARMS}},
        "ground_truth_opened_after_all_predictions_validated": True,
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
        "optimizer_updates": 0,
        "scientific_limits": "Frozen inference only. No fresh native32 training, scaling, gate-necessity, or unconditional superiority claim.",
    }
    _write_json_once(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "reference", "predict", "score"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--predictions-root", type=Path, default=PREDICTIONS)
    parser.add_argument("--preflight-root", type=Path, default=PREFLIGHTS)
    parser.add_argument("--output", type=Path, default=SUMMARY)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--slot-root", type=Path, default=SLOT_ROOT)
    args = parser.parse_args(argv)
    if args.stage != "score" and args.seed is None:
        parser.error("--seed is required except for score")
    if args.stage == "predict" and args.arm not in {"C", "T"}:
        parser.error("--arm C or T required for predict")
    if args.stage == "preflight":
        result = preflight_seed(args.seed, device=args.device, output_root=args.preflight_root)
    elif args.stage == "reference":
        result = predict_reference(args.seed, device=args.device, output_root=args.predictions_root,
                                   preflight_root=args.preflight_root)
    elif args.stage == "predict":
        result = predict_arm(args.seed, args.arm, device=args.device,
                             output_root=args.predictions_root, preflight_root=args.preflight_root)
    else:
        result = score_all(dataset=args.dataset, slot_root=args.slot_root,
                           predictions_root=args.predictions_root, preflight_root=args.preflight_root,
                           output=args.output)
    print(json.dumps({"stage": args.stage, "status": result.get("status", "complete"),
                      "seed": args.seed, "arm": args.arm,
                      "ground_truth_used_for_prediction": False}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
