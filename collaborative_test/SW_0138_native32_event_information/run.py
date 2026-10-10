"""Frozen native32 QCC event-information screen with a retained-gate readout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

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
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0135_native32_spike_binding.foundation import (
    load_native32_foundation, sha256_file,
)
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0130_phase_state_integration.model import BASE_THRESHOLD, COMPONENTS
from snn_kuramoto_bidirectional.membrane_layer import act_fun_adp

SEEDS = (0, 1, 2)
IMAGE_IDS = contract.IMAGE_IDS
COUNT, GRID = contract.COUNT, contract.GRID
ARCHIVE = HERE / "results_archive"
PREDICTION_ROOT = ARCHIVE / "event_predictions"
SUMMARY_DIR = ARCHIVE / "evaluation"
DATASET = Path(sw130.source97.DATASET)


def implementation_fingerprint():
    paths = [HERE / "protocol.json", HERE / "run.py",
             ROOT / "collaborative_test/SW_0130_phase_state_integration/run.py",
             ROOT / "collaborative_test/SW_0130_phase_state_integration/model.py",
             ROOT / "collaborative_test/SW_0134_native_spike_binding/rollout.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/resolution.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
             ROOT / "collaborative_test/SW_0137_native32_source_qcc/run.py",
             ROOT / "collaborative_test/SW_0136_native32_transfer/evaluate.py",
             ROOT / "snn_kuramoto_bidirectional/membrane_layer.py",
             ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
             ROOT / "snn_kuramoto_bidirectional/evaluation.py"]
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def _write_json_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_npz_once(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def prediction_dir(seed, root=PREDICTION_ROOT):
    if int(seed) not in SEEDS:
        raise ValueError("SW0138 only registers source seeds0,1,2")
    return Path(root) / f"seed{int(seed)}"


def _validate_trace(trace, batch):
    shapes = {
        "component_membrane": (batch, 4, GRID * GRID, 1024),
        "component_spikes": (batch, 4, GRID * GRID, 1024),
        "component_gates": (batch, 4, GRID * GRID, 1024),
        "membrane": (batch, GRID * GRID, 1024),
        "spikes": (batch, GRID * GRID, 1024),
        "theta": (batch, 1024, GRID * GRID, 4),
    }
    for key, expected in shapes.items():
        value = trace.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != expected:
            raise ValueError(f"native32 trace {key} must have shape {expected}")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"native32 trace {key} is nonfinite")


def native_events_from_equation(component_membrane, v_th):
    """Evaluate the native binary threshold equation directly, without S/g division."""
    if component_membrane.ndim != 4:
        raise ValueError("component membrane must be [B,4,N,T]")
    batch, components, nodes, steps = component_membrane.shape
    threshold = torch.as_tensor(v_th, device=component_membrane.device,
                                dtype=component_membrane.dtype)
    if threshold.shape == (batch * components, nodes):
        threshold = threshold.reshape(batch, components, nodes)
    if threshold.shape == (batch, components, nodes):
        threshold = threshold.unsqueeze(-1)
    if threshold.shape not in {(batch, components, nodes, 1), (1, 1, 1, 1)}:
        raise ValueError("native membrane threshold does not match folded source layout")
    events = act_fun_adp(component_membrane - threshold)
    if not torch.all((events == 0) | (events == 1)):
        raise AssertionError("native threshold equation did not produce binary events")
    return events


def assert_spike_gate_identity(component_spikes, events, component_gates):
    if component_spikes.shape != events.shape or component_spikes.shape != component_gates.shape:
        raise ValueError("S, E and g must have the same folded native component shape")
    if not torch.equal(component_spikes, events * component_gates):
        raise AssertionError("native emitted spikes do not equal the directly captured E*g")
    return True


def assert_late_rollout_parity(left, right):
    """Require exact equality for all registered frozen rollout trajectories."""
    fields = ("component_membrane", "component_spikes", "component_gates",
              "membrane", "spikes", "theta")
    for field in fields:
        if field not in left or field not in right or not torch.equal(left[field], right[field]):
            raise AssertionError(f"SW0138 rollout differs from frozen late_rollout at {field}")
    return True


def qcc_pair(trace, gates):
    """Run the exact SW0137 QCC helper on emitted S and, separately, native g."""
    if gates.shape != trace["component_spikes"].shape:
        raise ValueError("gate counterfactual must preserve the native [B,4,N,T] layout")
    actual = sw137.sw136_eval._qcc_labels(trace)
    gate_trace = {"component_spikes": gates, "spikes": gates.mean(dim=1)}
    gate = sw137.sw136_eval._qcc_labels(gate_trace)
    for name, labels in (("actual", actual), ("gate", gate)):
        if not isinstance(labels, np.ndarray) or labels.shape != (trace["spikes"].shape[0], GRID, GRID):
            raise ValueError(f"{name} QCC readout must return native32 NumPy labels")
        if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise ValueError(f"{name} QCC labels must be nonnegative integers")
    return actual.astype(np.int64, copy=False), gate.astype(np.int64, copy=False)


def _first_train_row(seed):
    source = sw130.source_contract(int(seed))
    pool, ids = source[3], source[4]
    if len(pool) != 4096 or len(ids) != 4096:
        raise ValueError("source97 first-TRAIN preflight requires registered4096 order")
    return int(pool[0]), int(ids[0])


def _one_rollout(foundation, rgb):
    gamma = foundation.encode(rgb)
    trace = late_rollout(foundation.wrapped, gamma, total_steps=1024, live_tail_steps=0)
    _validate_trace(trace, batch=rgb.shape[0])
    threshold = _native_threshold(foundation.wrapped, trace["component_membrane"].shape[0],
                                  trace["component_membrane"].shape[2])
    events = native_events_from_equation(trace["component_membrane"], threshold)
    assert_spike_gate_identity(trace["component_spikes"], events, trace["component_gates"])
    actual, gate = qcc_pair(trace, trace["component_gates"])
    return gamma, trace, events, actual, gate


def _native_threshold(wrapped, batch, nodes):
    """Match SW0134's registered threshold expression from wrapped component b."""
    if tuple(wrapped.b.shape) != (COMPONENTS,):
        raise ValueError("native SW0130 source threshold requires four b scalars")
    component_ids = torch.arange(COMPONENTS, device=wrapped.b.device).repeat(batch)
    return (BASE_THRESHOLD * torch.exp(wrapped.b[component_ids])).reshape(
        batch, COMPONENTS, 1, 1).expand(batch, COMPONENTS, nodes, 1)


def _preflight_parity(foundation, rgb):
    gamma = foundation.encode(rgb)
    first = late_rollout(foundation.wrapped, gamma, total_steps=1024, live_tail_steps=0)
    second = late_rollout(foundation.wrapped, gamma, total_steps=1024, live_tail_steps=0)
    _validate_trace(first, batch=rgb.shape[0])
    _validate_trace(second, batch=rgb.shape[0])
    assert_late_rollout_parity(first, second)
    events = native_events_from_equation(
        first["component_membrane"], _native_threshold(
            foundation.wrapped, first["component_membrane"].shape[0],
            first["component_membrane"].shape[2]))
    assert_spike_gate_identity(first["component_spikes"], events, first["component_gates"])
    return gamma, events, first


def _source_record(foundation):
    source = sw130.source_contract(foundation.seed)
    return {
        "source_core_sha256": source[5],
        "source_manifest_sha256": sha256_file(source[1]),
        "source_native32_state_sha256": sw137.state_dict_sha256((
            ("wrapped", foundation.wrapped), ("encoder", foundation.encoder))),
        "encoder_sha256": foundation.provenance["encoder_sha256"],
        "feature_preprocessing_sha256": foundation.provenance["feature_preprocessing_sha256"],
        "rgb_asset_validation": foundation.provenance["rgb_asset_validation"],
        "source_training_ids_sha256": foundation.provenance["source_training_ids_sha256"],
    }


def _write_preflight_once(out_dir, record):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    path = out_dir / "preflight.json"
    _write_json_once(path, record)
    return sha256_file(path)


def _write_prediction(out_dir, seed, ids, actual, gate, provenance, sw137_row):
    out_dir = Path(out_dir)
    if out_dir.exists():
        if {p.name for p in out_dir.iterdir()} != {"preflight.json"}:
            raise FileExistsError(f"preserve existing SW0138 attempt output: {out_dir}")
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
    ids = contract.require_fixed_image_ids(ids)
    for name, value in (("actual", actual), ("gate", gate)):
        if value.shape != (COUNT, GRID, GRID) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{name} prediction violates the fixed native32 label contract")
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"{name} prediction contains invalid labels")
    if not np.array_equal(actual, sw137_row["labels"]):
        raise AssertionError("SW0138 actual-spike QCC labels do not exactly reproduce SW0137")
    prediction_path = out_dir / "predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez_compressed(stream, image_ids=ids, actual_qcc_labels=actual,
                            gate_qcc_labels=gate)
        stream.flush(); os.fsync(stream.fileno())
    protocol = {
        "experiment": "SW0138_native32_event_information", "status": "complete",
        "seed": int(seed), "image_ids": [IMAGE_IDS[0], IMAGE_IDS[-1]], "count": COUNT,
        "grid_size": [GRID, GRID], "prediction_shape": [COUNT, GRID, GRID],
        "steps": 1024, "settle": 512, "microbatch_size": 1,
        "readout": {"affinity": "SW0137 product-positive component Pearson QCC",
                    "threshold": 0.50, "minimum_component_size": 8,
                    "background": "largest connected component"},
        "event_identity": "direct act_fun_adp(m-v_th) event; exact emitted S=E*g asserted",
        "counterfactual": "retained native g supplied only as readout activity; recurrence unchanged",
        "preflight": provenance["preflight"],
        "preflight_sha256": provenance["preflight_sha256"],
        "source": {k: v for k, v in provenance.items() if k != "preflight"},
        "sw0137_reference": {"prediction_sha256": sw137_row["prediction_sha256"],
                             "protocol_sha256": sw137_row["protocol_sha256"],
                             "labels_exactly_reproduced": True},
        "prediction_sha256": sha256_file(prediction_path),
        "implementation": implementation_fingerprint(),
        "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
    }
    _write_json_once(out_dir / "protocol.json", protocol)
    return protocol


@torch.no_grad()
def predict_seed(seed, *, device="cuda:0", output_root=PREDICTION_ROOT,
                sw137_root=sw137.OUTPUT_ROOT):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("SW0138 only registers source seeds0,1,2")
    out_dir = prediction_dir(seed, output_root)
    if out_dir.exists():
        raise FileExistsError(f"preserve existing SW0138 output: {out_dir}")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full SW0138 inference requires an exclusive CUDA device")
    torch.cuda.set_device(device); torch.cuda.init()

    reference = sw137.validate_prediction(Path(sw137_root) / f"seed{seed}", seed=seed)
    foundation = load_native32_foundation(seed, device, verify_rgb_assets=True)
    foundation.encoder.eval(); foundation.wrapped.eval()
    provenance = _source_record(foundation)
    if provenance["source_core_sha256"] != sw130.source97.EXPECTED_SOURCE_SHAS[seed]:
        raise ValueError("loaded model is not the registered same-seed SW0097 source")
    before_state = sw137.state_dict_sha256((("wrapped", foundation.wrapped),
                                             ("encoder", foundation.encoder)))

    train_row, train_image_id = _first_train_row(seed)
    train_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r")
    if train_cache.dtype != np.uint8 or train_cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered full-resolution TRAIN RGB cache mismatch")
    train_rgb = torch.from_numpy(np.asarray(train_cache[train_row:train_row + 1]).copy()).to(device)
    gamma, events, trace = _preflight_parity(foundation, train_rgb)
    actual, gate = qcc_pair(trace, trace["component_gates"])
    preflight = {
        "split": "TRAIN", "image_id": train_image_id, "pool_row": train_row,
        "gamma_shape": list(gamma.shape), "qcc_shape": list(actual.shape),
        "trajectory_parity": "exact all six frozen late_rollout fields on repeated state-reset run",
        "event_identity_exact": True,
        "gate_zero_positions": int((trace["component_gates"] == 0).sum().item()),
        "event_one_at_zero_gate": int(((events == 1) & (trace["component_gates"] == 0)).sum().item()),
        "ground_truth_used": False,
    }
    provenance["preflight"] = preflight
    provenance["preflight_sha256"] = _write_preflight_once(
        out_dir, {**provenance, "preflight": preflight,
                  "implementation": implementation_fingerprint(),
                  "status": "source_event_preflight_complete"})
    del train_cache, train_rgb, gamma, events, trace, actual, gate

    cache = np.load(sw130.VAL_RGB, mmap_mode="r")
    if cache.dtype != np.uint8 or cache.shape != (COUNT, 128, 128, 3):
        raise ValueError("registered fixed320 RGB validation cache mismatch")
    actual_labels = np.empty((COUNT, GRID, GRID), dtype=np.int64)
    gate_labels = np.empty_like(actual_labels)
    for index in range(COUNT):
        rgb = torch.from_numpy(np.asarray(cache[index:index + 1]).copy()).to(device)
        _gamma, _trace, _events, actual_one, gate_one = _one_rollout(foundation, rgb)
        actual_labels[index] = actual_one[0]
        gate_labels[index] = gate_one[0]
        if not np.array_equal(actual_labels[index], reference["labels"][index]):
            raise AssertionError(f"actual-spike QCC differs from completed SW0137 at image {IMAGE_IDS[index]}")
        if index % 40 == 39:
            print(json.dumps({"stage": "event_information_prediction", "seed": seed,
                              "predicted": index + 1, "count": COUNT}), flush=True)
    after_state = sw137.state_dict_sha256((("wrapped", foundation.wrapped),
                                            ("encoder", foundation.encoder)))
    if after_state != before_state:
        raise AssertionError("frozen source or encoder changed during SW0138 inference")
    provenance["preflight"] = preflight
    protocol = _write_prediction(out_dir, seed, np.asarray(IMAGE_IDS, dtype=np.int64),
                                 actual_labels, gate_labels, provenance, reference)
    return {"status": "prediction_complete", "seed": seed,
            "prediction_sha256": protocol["prediction_sha256"]}


def validate_prediction(out_dir, *, seed, sw137_root=sw137.OUTPUT_ROOT):
    out_dir = Path(out_dir)
    archive_path, protocol_path = out_dir / "predictions.npz", out_dir / "protocol.json"
    raw = protocol_path.read_bytes()
    protocol_sha = hashlib.sha256(raw).hexdigest()
    protocol = json.loads(raw.decode("utf-8"))
    source = sw130.source_contract(int(seed))
    assets = sw137._validated_rgb_assets()
    source_record = protocol.get("source")
    reference = sw137.validate_prediction(Path(sw137_root) / f"seed{seed}", seed=seed)
    preflight = protocol.get("preflight")
    preflight_path = out_dir / "preflight.json"
    preflight_raw = preflight_path.read_bytes()
    preflight_record = json.loads(preflight_raw.decode("utf-8"))
    expected_ids_sha = hashlib.sha256(np.asarray(source[4], dtype="<i8").tobytes()).hexdigest()
    if (protocol.get("experiment") != "SW0138_native32_event_information"
            or protocol.get("status") != "complete" or protocol.get("seed") != int(seed)
            or protocol.get("image_ids") != [1320, 1639] or protocol.get("count") != COUNT
            or protocol.get("grid_size") != [GRID, GRID]
            or protocol.get("prediction_shape") != [COUNT, GRID, GRID]
            or protocol.get("steps") != 1024 or protocol.get("settle") != 512
            or protocol.get("microbatch_size") != 1
            or protocol.get("ground_truth_used_for_prediction") is not False
            or protocol.get("optimizer_updates") != 0
            or not isinstance(source_record, dict)
            or source_record.get("source_core_sha256") != source[5]
            or source_record.get("source_manifest_sha256") != sha256_file(source[1])
            or source_record.get("encoder_sha256") != assets["encoder_source_sha256"]
            or source_record.get("feature_preprocessing_sha256") != assets["feature_preprocessing_sha256"]
            or source_record.get("rgb_asset_validation") != assets
            or source_record.get("source_training_ids_sha256") != expected_ids_sha
            or not source_record.get("source_native32_state_sha256")
            or source_record.get("source_native32_state_sha256")
                != reference["protocol"].get("source_native32_state_sha256")
            or source_record.get("source_manifest_sha256")
                != reference["protocol"].get("source_manifest_sha256")
            or source_record.get("encoder_sha256")
                != reference["protocol"].get("encoder_sha256")
            or source_record.get("feature_preprocessing_sha256")
                != reference["protocol"].get("feature_preprocessing_sha256")
            or source_record.get("rgb_asset_validation")
                != reference["protocol"].get("rgb_assets")
            or not isinstance(preflight_record, dict)
            or protocol.get("sw0137_reference", {}).get("prediction_sha256") != reference["prediction_sha256"]
            or protocol.get("sw0137_reference", {}).get("protocol_sha256") != reference["protocol_sha256"]
            or protocol.get("sw0137_reference", {}).get("labels_exactly_reproduced") is not True
            or protocol.get("preflight_sha256") != sha256_file(out_dir / "preflight.json")
            or preflight_record.get("implementation") != implementation_fingerprint()
            or preflight_record.get("status") != "source_event_preflight_complete"
            or preflight_record.get("preflight") != preflight
            or preflight_record.get("source_core_sha256") != source[5]
            or not isinstance(preflight, dict)
            or preflight.get("event_identity_exact") is not True
            or preflight.get("image_id") != int(source[4][0])
            or preflight.get("pool_row") != int(source[3][0])
            or preflight.get("gamma_shape") != [1, 8, 1024]
            or preflight.get("qcc_shape") != [1, 32, 32]
            or preflight.get("ground_truth_used") is not False
            or protocol.get("implementation") != implementation_fingerprint()
            or protocol.get("prediction_sha256") != sha256_file(archive_path)):
        raise ValueError(f"invalid SW0138 source event prediction for seed{seed}")
    with np.load(archive_path, allow_pickle=False) as saved:
        if not {"image_ids", "actual_qcc_labels", "gate_qcc_labels"}.issubset(saved.files):
            raise ValueError("SW0138 archive is missing paired QCC arrays")
        ids = contract.require_fixed_image_ids(saved["image_ids"])
        actual, gate = saved["actual_qcc_labels"], saved["gate_qcc_labels"]
    for name, labels in (("actual", actual), ("gate", gate)):
        if (labels.shape != (COUNT, GRID, GRID) or not np.issubdtype(labels.dtype, np.integer)
                or np.issubdtype(labels.dtype, np.bool_) or np.any(labels < 0)
                or np.any(labels > 1024)):
            raise ValueError(f"invalid {name} QCC labels in SW0138 archive")
    if not np.array_equal(actual, reference["labels"]):
        raise ValueError("SW0138 actual QCC array no longer matches completed SW0137")
    if (sha256_file(archive_path) != protocol["prediction_sha256"]
            or hashlib.sha256(protocol_path.read_bytes()).hexdigest() != protocol_sha):
        raise ValueError("SW0138 prediction bytes changed while validating")
    return {"seed": int(seed), "ids": ids, "actual": actual.astype(np.int64, copy=False),
            "gate": gate.astype(np.int64, copy=False), "protocol": protocol,
            "archive_path": archive_path, "protocol_path": protocol_path,
            "prediction_sha256": protocol["prediction_sha256"],
            "protocol_sha256": protocol_sha}


def _paired_bootstrap(gate_rows, actual_rows, *, seed=138, replicates=10000):
    output = {}
    rng = np.random.RandomState(seed)
    index_draws = rng.randint(0, COUNT, size=(replicates, COUNT))
    for metric in contract.METRICS:
        # Average seed-wise paired differences per image, then resample the same
        # image indices for the three-seed mean contrast.
        per_image = np.mean(np.stack([
            np.asarray(gate_rows[s][metric]["per_image"], dtype=np.float64)
            - np.asarray(actual_rows[s][metric]["per_image"], dtype=np.float64)
            for s in SEEDS], axis=0), axis=0)
        samples = per_image[index_draws].mean(axis=1)
        output[metric] = {"gate_minus_actual_mean": float(per_image.mean()),
                          "paired_ci95": [float(np.quantile(samples, .025)),
                                          float(np.quantile(samples, .975))],
                          "per_image_three_seed_differences": per_image.tolist()}
    return output


def score(*, output_root=PREDICTION_ROOT, dataset=DATASET, output=SUMMARY_DIR,
          sw137_root=sw137.OUTPUT_ROOT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0138 score output: {output}")
    rows = {seed: validate_prediction(prediction_dir(seed, output_root), seed=seed,
                                      sw137_root=sw137_root) for seed in SEEDS}
    if any(not np.array_equal(rows[0]["ids"], rows[s]["ids"]) for s in (1, 2)):
        raise ValueError("SW0138 seeds do not share the fixed ordered image IDs")
    # Freeze and reverify all six arrays/protocols and their SW0137 dependencies
    # before opening the target-mask dataset.
    for seed, row in rows.items():
        ref = sw137.validate_prediction(Path(sw137_root) / f"seed{seed}", seed=seed)
        if (sha256_file(row["archive_path"]) != row["prediction_sha256"]
                or sha256_file(row["protocol_path"]) != row["protocol_sha256"]
                or sha256_file(ref["archive_path"]) != ref["prediction_sha256"]
                or sha256_file(ref["protocol_path"]) != ref["protocol_sha256"]
                or not np.array_equal(row["actual"], ref["labels"])):
            raise ValueError("frozen event/source QCC payload changed before target access")
    with h5py.File(dataset, "r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0] <= IMAGE_IDS[-1]:
            raise ValueError("fixed320 target mask slice is absent")
        pixel_gt = np.asarray(h5["mask"][IMAGE_IDS[0]:IMAGE_IDS[-1] + 1])
    gt = contract.modal_native32(pixel_gt)
    actual_scores, gate_scores = {}, {}
    for seed in SEEDS:
        actual_scores[seed] = contract.production_metrics32(rows[seed]["actual"], gt)
        gate_scores[seed] = contract.production_metrics32(rows[seed]["gate"], gt)
    actual_mean = {metric: float(np.mean([actual_scores[s][metric]["mean"] for s in SEEDS]))
                   for metric in contract.METRICS}
    gate_mean = {metric: float(np.mean([gate_scores[s][metric]["mean"] for s in SEEDS]))
                 for metric in contract.METRICS}
    report = {
        "experiment": "SW0138_native32_event_information", "status": "complete",
        "seed_scope": list(SEEDS), "actual_spike_scores": actual_scores,
        "retained_gate_readout_scores": gate_scores,
        "actual_spike_three_seed_mean": actual_mean,
        "retained_gate_three_seed_mean": gate_mean,
        "paired_bootstrap_gate_minus_actual": _paired_bootstrap(
            gate_scores, actual_scores),
        "prediction_provenance": {str(s): {
            "prediction_sha256": rows[s]["prediction_sha256"],
            "protocol_sha256": rows[s]["protocol_sha256"],
            "sw0137_prediction_sha256": rows[s]["protocol"]["sw0137_reference"]["prediction_sha256"]}
            for s in SEEDS},
        "evaluation_contract": {"image_ids": [1320, 1639, 320], "grid_size": [32, 32],
                                "metrics": list(contract.METRICS),
                                "ground_truth_after_all_six_prediction_arrays": True},
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
        "optimizer_updates": 0,
        "interpretation": "Frozen readout-information comparison only; gate dynamics, feedback, and reset remain unchanged. No gate-necessity or promotion claim.",
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_json_once(output / "evaluation.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("predict", "score"), required=True)
    parser.add_argument("--seed", choices=("0", "1", "2"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument("--sw137-root", type=Path, default=sw137.OUTPUT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=SUMMARY_DIR)
    args = parser.parse_args(argv)
    if args.stage == "predict":
        if args.seed is None:
            parser.error("--seed required for predict")
        result = predict_seed(int(args.seed), device=args.device,
                              output_root=args.output_root, sw137_root=args.sw137_root)
    else:
        result = score(output_root=args.output_root, dataset=args.dataset,
                       output=args.output, sw137_root=args.sw137_root)
    print(json.dumps({"status": result.get("status", "complete"),
                      "stage": args.stage, "seed": args.seed,
                      "ground_truth_used_for_prediction": False,
                      "optimizer_updates": 0}, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
