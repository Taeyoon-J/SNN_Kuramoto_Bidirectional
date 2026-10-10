"""Prediction-only native32 transfer plus explicit GT-last scoring stage."""
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
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"), str(ROOT / "snn_kuramoto_bidirectional")]

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding import score_slot32
from collaborative_test.SW_0135_native32_spike_binding.evaluation_contract32 import (
    COUNT, GRID, IMAGE_IDS, METRICS, production_metrics32, modal_native32,
    require_fixed_image_ids, sha256_file, slot_native_pixels_to32,
    validate_slot_protocol,
)
from collaborative_test.SW_0135_native32_spike_binding.foundation import (
    load_native32_foundation, sha256_file as file_sha,
)
from collaborative_test.SW_0135_native32_spike_binding.binder import (
    NativeSpikeSlotBinder, canonical_hard_labels,
)
from collaborative_test.SW_0136_native32_transfer.transfer import (
    ARMS, load_transferred_model, state_dict_sha256,
)
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components

CHECKPOINTS = {
    "actual_joint": "7bb7efcaf3640c6de3ca4ac5b8f8f265f16f6e75a2ffb24b0515862930e49beb",
    "actual_frozen": "361b426179a4eb3f2d39169398557e0b898c995afab62c68cdcb025677b783ed",
}
SLOT_BASELINE_SHA256 = "4dc12cb54c3114375b8428cd250c013ff94015c4826915751f2fcb0a2204d753"
SLOT_BASELINE_PATH = ROOT / "collaborative_test/SW_0135_native32_spike_binding/results_archive/slot70k_native32_baseline.json"
ARCHIVE = HERE / "results_archive"
PREDICTION_ROOT = ARCHIVE / "transfer_predictions_seed1"
DATASET = Path(sw130.source97.DATASET)
SLOT_ROOT = ROOT / "trained_models/SW0092_slot_our70000_eval"


def _write_once_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def _write_npz_once(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _qcc_labels(trace):
    groups = spike_synchrony_components(
        trace["spikes"].detach().cpu(), synchrony_threshold=.50,
        min_group_size=8, settle=512,
        components=trace["component_spikes"].detach().cpu(),
        background="largest_component", affinity_mode="spike", spatial_grid_size=32)
    labels = spatial_components_to_patch_labels(groups, 32, device="cpu")
    if tuple(labels.shape) != (trace["spikes"].shape[0], 32, 32):
        raise ValueError("native32 QCC did not emit direct32 labels")
    return labels.numpy().astype(np.int64, copy=False)


def _validate_native32_trace(trace, batch):
    expected = {
        "component_spikes": (batch, 4, 1024, 1024),
        "component_membrane": (batch, 4, 1024, 1024),
        "component_gates": (batch, 4, 1024, 1024),
        "membrane": (batch, 1024, 1024),
        "spikes": (batch, 1024, 1024),
        "theta": (batch, 1024, 1024, 4),
    }
    for name, shape in expected.items():
        value = trace.get(name)
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"transferred rollout field {name} must have shape {shape}")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"transferred rollout field {name} is nonfinite")


@torch.no_grad()
def _predict_one(foundation, binder, gamma, *, primary):
    trace = late_rollout(foundation.wrapped, gamma, total_steps=1024, live_tail_steps=0)
    _validate_native32_trace(trace, gamma.shape[0])
    qcc = _qcc_labels(trace)
    if primary:
        assignments, _slots, _patch = binder(trace["component_spikes"][..., 512:])
        labels = canonical_hard_labels(assignments).cpu().numpy().astype(np.int64, copy=False)
        if tuple(assignments.shape) != (gamma.shape[0], 1024, 11):
            raise ValueError("transferred binder did not emit native32 P")
        if not torch.isfinite(assignments).all() or not torch.allclose(
                assignments.sum(-1), torch.ones_like(assignments[..., 0]), atol=1e-5, rtol=1e-5):
            raise ValueError("transferred native32 assignment matrix is invalid")
    else:
        labels = None
    return labels, qcc


def _save_predictions(directory, name, *, ids, labels, qcc_labels, transfer_record, source_reference=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / "predictions.npz"
    protocol_path = directory / "protocol.json"
    labels = np.asarray(labels, dtype=np.int64)
    qcc_labels = np.asarray(qcc_labels, dtype=np.int64)
    if labels.shape != (COUNT, GRID, GRID) or qcc_labels.shape != (COUNT, GRID, GRID):
        raise ValueError("prediction arrays do not match fixed320 native32 schema")
    _write_npz_once(npz_path, image_ids=np.asarray(ids, dtype=np.int64),
                    labels=labels, qcc_labels=qcc_labels)
    protocol = {
        "experiment": "SW0136_native32_transfer", "stage": "prediction_complete",
        "status": "complete", "arm": name, "seed": 1,
        "image_ids": [IMAGE_IDS[0], IMAGE_IDS[-1]], "count": COUNT,
        "prediction_shape": [COUNT, GRID, GRID], "grid_size": [GRID, GRID],
        "microbatch_size": 1, "steps": 1024, "settle": 512,
        "primary": ("mapped source97 QCC reference only" if name == "source97_qcc"
                    else "native32 transferred binder P argmax / canonical largest-slot background"),
        "secondary": "QCC threshold .50, min8, largest-component background",
        "prediction_sha256": sha256_file(npz_path),
        "transfer": transfer_record,
        "source_reference": source_reference,
        "ground_truth_used_for_prediction": False,
        "optimizer_updates": 0,
        "implementation": implementation_fingerprint(),
    }
    _write_once_json(protocol_path, protocol)
    return {"directory": str(directory), "prediction_path": str(npz_path),
            "protocol_path": str(protocol_path), "prediction_sha256": protocol["prediction_sha256"],
            "protocol_sha256": sha256_file(protocol_path), "protocol": protocol}


def implementation_fingerprint():
    paths = [HERE / "protocol.json", HERE / "transfer.py", HERE / "evaluate.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/resolution.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/binder.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/score_slot32.py",
             ROOT / "collaborative_test/SW_0134_native_spike_binding/train.py",
             ROOT / "collaborative_test/SW_0134_native_spike_binding/run.py",
             ROOT / "collaborative_test/SW_0134_native_spike_binding/binder.py",
             ROOT / "collaborative_test/SW_0134_native_spike_binding/pilot_queue.py",
             ROOT / "collaborative_test/SW_0134_native_spike_binding/rollout.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
             ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
             ROOT / "snn_kuramoto_bidirectional/evaluation.py"]
    return {p.relative_to(ROOT).as_posix(): file_sha(p) for p in paths}


def validate_prediction(directory, *, arm):
    directory = Path(directory)
    npz, protocol_path = directory / "predictions.npz", directory / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    if (protocol.get("experiment") != "SW0136_native32_transfer"
            or protocol.get("status") != "complete" or protocol.get("arm") != arm
            or protocol.get("seed") != 1 or protocol.get("image_ids") != [1320, 1639]
            or protocol.get("count") != 320 or protocol.get("grid_size") != [32, 32]
            or protocol.get("prediction_shape") != [320, 32, 32]
            or protocol.get("ground_truth_used_for_prediction") is not False
            or protocol.get("optimizer_updates") != 0
            or protocol.get("implementation") != implementation_fingerprint()
            or sha256_file(npz) != protocol.get("prediction_sha256")):
        raise ValueError(f"invalid/fingerprint-mismatched SW0136 {arm} prediction")
    transfer = protocol.get("transfer")
    if not isinstance(transfer, dict):
        raise ValueError("prediction is missing transfer/checkpoint provenance")
    if arm in ARMS:
        if (transfer.get("arm") != arm or transfer.get("seed") != 1
                or transfer.get("checkpoint_sha256") != CHECKPOINTS[arm]
                or transfer.get("source_core_sha256") != sw130.source97.EXPECTED_SOURCE_SHAS[1]
                or transfer.get("strict_transfer") is not True
                or transfer.get("ground_truth_used") is not False
                or transfer.get("optimizer_updates") != 0):
            raise ValueError("transferred arm checkpoint/provenance binding is invalid")
    elif (transfer.get("reference_role") != "mapped SW0097 native32 QCC only"
          or transfer.get("source_core_sha256") != sw130.source97.EXPECTED_SOURCE_SHAS[1]
          or not transfer.get("converted_state_sha256")):
        raise ValueError("source97 QCC reference provenance is invalid")
    with np.load(npz, allow_pickle=False) as saved:
        if not {"image_ids", "labels", "qcc_labels"}.issubset(saved.files):
            raise ValueError("prediction archive missing required arrays")
        ids = require_fixed_image_ids(saved["image_ids"])
        labels, qcc = saved["labels"], saved["qcc_labels"]
    if labels.shape != (320, 32, 32) or qcc.shape != (320, 32, 32):
        raise ValueError("prediction arrays have incorrect geometry")
    for x in (labels, qcc):
        if not np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.bool_) or np.any(x < 0):
            raise ValueError("prediction labels must be nonnegative integers")
    if np.any(labels > 10) or np.any(qcc > 1024):
        raise ValueError("prediction labels exceed the registered native32 readout cardinality")
    if sha256_file(npz) != protocol["prediction_sha256"] or sha256_file(protocol_path) != protocol_sha:
        raise ValueError("prediction bytes changed while being validated")
    return {"labels": labels.astype(np.int64, copy=False), "qcc_labels": qcc.astype(np.int64, copy=False),
            "image_ids": ids, "protocol": protocol, "prediction_sha256": protocol["prediction_sha256"],
            "protocol_sha256": protocol_sha, "prediction_path": npz,
            "protocol_path": protocol_path}


@torch.no_grad()
def _source_qcc_prediction(foundation, cache, ids):
    n = len(ids)
    qcc = np.empty((n, 32, 32), dtype=np.int64)
    for start in range(n):
        row = np.asarray(cache[start:start + 1]).copy()
        rgb = torch.from_numpy(row).to(next(foundation.encoder.parameters()).device)
        gamma = foundation.encode(rgb)
        _primary, qcc[start:start + 1] = _predict_one(foundation, None, gamma, primary=False)
    return qcc


def _first_registered_train_image():
    _checkpoint, _manifest_path, _manifest, pool_indices, ids, _sha = sw130.source_contract(1)
    if len(pool_indices) != 4096 or len(ids) != 4096:
        raise ValueError("source97 seed1 training order must contain the registered4096 IDs")
    return int(pool_indices[0]), int(ids[0])


def predict_arm(arm, *, device="cuda:0", output_root=PREDICTION_ROOT,
                include_source_reference=False):
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0136 prediction requires an available exclusive CUDA device")
    torch.cuda.set_device(device); torch.cuda.init()
    output_root = Path(output_root)
    out_dir = output_root / arm
    if out_dir.exists():
        raise FileExistsError(f"preserve existing transfer output: {out_dir}")
    foundation = load_native32_foundation(1, device, verify_rgb_assets=True)
    cache = np.load(sw130.VAL_RGB, mmap_mode="r")
    if cache.shape[0] != COUNT:
        raise ValueError("registered validation RGB cache must contain exactly320 ordered images")
    ids = list(range(1320, 1640))

    source_record = None
    if include_source_reference:
        source_dir = output_root / "source97_qcc_seed1"
        if source_dir.exists():
            raise FileExistsError(f"preserve existing source reference: {source_dir}")
        # Use the still-source-loaded foundation so candidate and source weights never coexist on GPU.
        foundation.encoder.eval(); foundation.wrapped.eval()
        train_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r")
        train_row, source_preflight_id = _first_registered_train_image()
        with torch.no_grad():
            source_rgb = torch.from_numpy(np.asarray(train_cache[train_row:train_row + 1]).copy()).to(device)
            source_gamma = foundation.encode(source_rgb)
            _source_p, source_preflight_qcc = _predict_one(
                foundation, None, source_gamma, primary=False)
        if (source_preflight_qcc.shape != (1, 32, 32)
                or not bool(torch.isfinite(source_gamma).all())):
            raise ValueError("source97 native32 TRAIN structural preflight failed")
        source_preflight = {"image_id": source_preflight_id, "pool_row": train_row,
                            "gamma_shape": list(source_gamma.shape),
                            "qcc_shape": list(source_preflight_qcc.shape),
                            "finite": True, "ground_truth_used": False}
        del train_cache, source_rgb, source_gamma, source_preflight_qcc
        source_qcc = _source_qcc_prediction(foundation, cache, ids)
        source_record = {"seed": 1,
                         "source_core_sha256": foundation.provenance["source_core_sha256"],
                         "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
                         "converted_state_sha256": state_dict_sha256((("wrapped", foundation.wrapped),)),
                         "structural_preflight": source_preflight,
                         "reference_role": "mapped SW0097 native32 QCC only"}
        source_labels = np.zeros_like(source_qcc)
        _save_predictions(source_dir, "source97_qcc", ids=ids, labels=source_labels,
                          qcc_labels=source_qcc, transfer_record=source_record)
        del source_qcc, source_labels

    # Strictly restore every completed SW0134 tensor only after the source reference is frozen.
    foundation, binder, decoder, transfer = load_transferred_model(arm, device, foundation=foundation)

    # GT-free structural preflight on the first source-ordered TRAIN example.
    train_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r")
    train_row, preflight_image_id = _first_registered_train_image()
    if train_cache.shape[0] != 70000:
        raise ValueError("registered training RGB cache must contain exactly70000 rows")
    with torch.no_grad():
        pre_rgb = torch.from_numpy(np.asarray(train_cache[train_row:train_row + 1]).copy()).to(device)
        pre_gamma = foundation.encode(pre_rgb)
        pre_labels, pre_qcc = _predict_one(foundation, binder, pre_gamma, primary=True)
    if pre_labels.shape != (1, 32, 32) or pre_qcc.shape != (1, 32, 32):
        raise ValueError("single-example transfer preflight produced invalid native32 labels")
    preflight = {"status": "finite_native32_transfer", "arm": arm, "seed": 1,
                 "image_id": preflight_image_id, "pool_row": train_row,
                 "split": "registered first seed1 TRAIN image", "gamma_shape": list(pre_gamma.shape),
                 "labels_shape": list(pre_labels.shape), "qcc_shape": list(pre_qcc.shape),
                 "finite": bool(torch.isfinite(pre_gamma).all()),
                 "ground_truth_used": False, "optimizer_updates": 0}
    out_dir.mkdir(parents=True, exist_ok=False)
    _write_once_json(out_dir / "preflight.json", preflight)
    del train_cache, pre_rgb, pre_gamma, pre_labels, pre_qcc

    del cache

    cache = np.load(sw130.VAL_RGB, mmap_mode="r")
    primary = np.empty((COUNT, 32, 32), dtype=np.int64)
    qcc = np.empty_like(primary)
    with torch.no_grad():
        for index in range(COUNT):
            rgb = torch.from_numpy(np.asarray(cache[index:index + 1]).copy()).to(device)
            gamma = foundation.encode(rgb)
            p, q = _predict_one(foundation, binder, gamma, primary=True)
            primary[index] = p[0]
            qcc[index] = q[0]
            if index % 40 == 39:
                print(json.dumps({"arm": arm, "predicted": index + 1, "count": COUNT}), flush=True)
    del cache
    directory = _save_predictions(out_dir, arm, ids=ids, labels=primary,
                                  qcc_labels=qcc, transfer_record=transfer,
                                  source_reference=source_record)
    return {"status": "prediction_complete", "arm": arm, "seed": 1,
            "preflight": preflight, "prediction": directory}


def score_transfer(*, output_root=PREDICTION_ROOT, dataset=DATASET,
                   slot_root=SLOT_ROOT, output=None):
    """Validate and hash every prediction first; only then read GT and score."""
    output = Path(output or (ARCHIVE / "transfer_evaluation_seed1"))
    if output.exists():
        raise FileExistsError(f"preserve existing SW0136 score directory: {output}")
    output_root = Path(output_root)
    loaded = {arm: validate_prediction(output_root / arm, arm=arm) for arm in ARMS}
    source = validate_prediction(output_root / "source97_qcc_seed1", arm="source97_qcc")
    if not np.array_equal(source["image_ids"], loaded["actual_joint"]["image_ids"]):
        raise ValueError("source97 QCC IDs differ from transferred predictions")
    source_transfer = source["protocol"].get("transfer", {})
    _source_checkpoint, source_manifest_path, _source_manifest = sw130.source97.source_paths(1)
    if (source_transfer.get("reference_role") != "mapped SW0097 native32 QCC only"
            or source_transfer.get("source_core_sha256") != sw130.source97.EXPECTED_SOURCE_SHAS[1]
            or source_transfer.get("source_manifest_sha256") != sha256_file(source_manifest_path)
            or source_transfer.get("converted_state_sha256") is None):
        raise ValueError("source QCC report does not identify the mapped SW0097 source model")
    if not np.array_equal(loaded["actual_joint"]["image_ids"], loaded["actual_frozen"]["image_ids"]):
        raise ValueError("transferred arm image IDs differ")
    if source["protocol"].get("source_reference") is not None:
        raise ValueError("source QCC protocol must not recursively reference another source")

    # Validate the fixed Slot predictions and pool their saved native pixels before GT access.
    slot_rows = {}
    for seed in range(3):
        folder = Path(slot_root) / f"seed{seed}_epoch10"
        npz_path, protocol_path = folder / "predictions.npz", folder / "protocol.json"
        protocol = score_slot32._read_json(protocol_path)
        slot_sha = validate_slot_protocol(protocol, seed=seed, prediction_path=npz_path)
        with np.load(npz_path, allow_pickle=False) as saved:
            if not {"labels", "image_ids"}.issubset(saved.files):
                raise ValueError("Slot prediction archive schema invalid")
            slot_ids = require_fixed_image_ids(saved["image_ids"], name="Slot image IDs")
            slot_native = np.asarray(saved["labels"])
        slot32 = slot_native_pixels_to32(slot_native)
        slot_rows[seed] = {"folder": folder, "npz": npz_path, "protocol_path": protocol_path,
                           "protocol": protocol, "native": slot_native,
                           "labels32": slot32, "ids": slot_ids, "sha": slot_sha,
                           "protocol_sha": sha256_file(protocol_path)}
    baseline_path = SLOT_BASELINE_PATH
    if sha256_file(baseline_path) != SLOT_BASELINE_SHA256:
        raise ValueError("registered Slot32 baseline summary bytes changed")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    # Freeze/recheck every prediction and the baseline immediately before the only GT open.
    for row in loaded.values():
        if sha256_file(row["prediction_path"]) != row["prediction_sha256"] or sha256_file(
                row["protocol_path"]) != row["protocol_sha256"]:
            raise ValueError("frozen transferred prediction changed before GT scoring")
    if (sha256_file(source["prediction_path"]) != source["prediction_sha256"]
            or sha256_file(source["protocol_path"]) != source["protocol_sha256"]):
        raise ValueError("frozen source97 prediction changed before GT scoring")
    if sha256_file(baseline_path) != SLOT_BASELINE_SHA256:
        raise ValueError("registered Slot32 baseline changed before GT scoring")
    for row in slot_rows.values():
        if sha256_file(row["npz"]) != row["sha"] or sha256_file(row["protocol_path"]) != row["protocol_sha"]:
            raise ValueError("frozen Slot prediction changed before GT scoring")
    if not np.array_equal(loaded["actual_joint"]["image_ids"], np.asarray(IMAGE_IDS)):
        raise ValueError("transferred predictions do not match fixed320 IDs")

    with h5py.File(dataset, "r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0] < IMAGE_IDS[-1] + 1:
            raise ValueError("registered HDF5 masks missing rows through1639")
        pixel_gt = np.asarray(h5["mask"][IMAGE_IDS[0]:IMAGE_IDS[-1] + 1])
    gt32 = modal_native32(pixel_gt)
    gt16 = None
    scores = {}
    for name, row in loaded.items():
        scores[name] = {"primary": production_metrics32(row["labels"], gt32),
                        "qcc": production_metrics32(row["qcc_labels"], gt32)}
    scores["source97_qcc"] = {"qcc": production_metrics32(source["qcc_labels"], gt32)}
    slot16_audits = {}
    for seed, row in slot_rows.items():
        scores[f"slot_seed{seed}"] = {"slot32": production_metrics32(row["labels32"], gt32)}
        # Preserve the existing eight-pixel audit and prove the modal32 path uses same source.
        if gt16 is None:
            import torch as _torch
            from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
            gt16 = clevr_mask_patch(_torch.from_numpy(np.ascontiguousarray(pixel_gt)), 8)["patch_labels"].numpy()
        slot16_audits[str(seed)] = score_slot32._slot16_audit(
            row["folder"], seed, row["native"], row["ids"], gt16)
    slot32_means = {metric: float(np.mean([scores[f"slot_seed{s}"]["slot32"][metric]["mean"]
                                           for s in range(3)])) for metric in METRICS}
    recorded_means = baseline["scores"]["slot32"]
    for metric in METRICS:
        if abs(slot32_means[metric] - float(recorded_means[metric]["three_seed_mean"])) > 1e-10:
            raise ValueError(f"fixed Slot32 {metric} mean did not reproduce")
    report = {"experiment": "SW0136_native32_transfer", "status": "complete",
              "seed": 1, "arms": [*ARMS, "source97_qcc"],
              "evaluation_contract": {"image_ids": [1320, 1639, 320],
                                      "grid_size": [32, 32], "patch_pixels": 4,
                                      "metrics": list(METRICS), "ground_truth_after_prediction_hashes": True},
              "prediction_provenance": {name: {"prediction_sha256": row["prediction_sha256"],
                                                 "protocol_sha256": row["protocol_sha256"]}
                                        for name, row in loaded.items()},
              "source97_prediction_provenance": {"prediction_sha256": source["prediction_sha256"],
                                                 "protocol_sha256": source["protocol_sha256"]},
              "slot_baseline_sha256": SLOT_BASELINE_SHA256,
              "slot32_three_seed_means": slot32_means,
              "slot32_seed1_means": {m: scores["slot_seed1"]["slot32"][m]["mean"] for m in METRICS},
              "slot_original16_audits": slot16_audits,
              "scores": scores,
              "ground_truth_used_for_prediction": False,
              "ground_truth_used_for_scoring": True,
              "optimizer_updates": 0,
              "interpretation": "Two completed SW0134 seed1 models transferred to native32; evaluation-only, no training or promotion claim."}
    output.mkdir(parents=True, exist_ok=False)
    _write_once_json(output / "evaluation.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("predict", "score"), required=True)
    parser.add_argument("--arm", choices=(*ARMS,), help="Required for prediction stage")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--slot-root", type=Path, default=SLOT_ROOT)
    parser.add_argument("--output", type=Path, help="Score directory for --stage score")
    args = parser.parse_args(argv)
    if args.stage == "predict":
        if args.arm is None:
            parser.error("--arm is required for predict")
        result = predict_arm(args.arm, device=args.device, output_root=args.output_root,
                             include_source_reference=args.arm == "actual_joint")
    else:
        result = score_transfer(output_root=args.output_root, dataset=args.dataset,
                                slot_root=args.slot_root, output=args.output)
    print(json.dumps({"status": result.get("status", "complete"), "stage": args.stage,
                      "arm": args.arm, "ground_truth_used_for_prediction": False,
                      "optimizer_updates": 0}, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
