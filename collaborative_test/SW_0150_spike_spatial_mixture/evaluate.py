"""GT-last SW0150 fixed320 native32 predictions and seed1 evaluation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
               str(ROOT / "snn_kuramoto_bidirectional")]

import h5py
import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0136_native32_transfer.transfer import state_dict_sha256
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0150_spike_spatial_mixture import integration, train
from collaborative_test.SW_0150_spike_spatial_mixture.likelihood import make_decoder

NAME = train.NAME
SEED = 1
ARMS = train.ARMS
COUNT, GRID = contract.COUNT, contract.GRID
IMAGE_IDS = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
PREDICTION_ROOT = HERE / "predictions"
SUMMARY = HERE / "results_archive" / "evaluation_seed1.json"
DATASET = Path(sw130.source97.DATASET)
SLOT_ROOT = ROOT / "trained_models/SW0092_slot_our70000_eval"
SLOT_BASELINE = ROOT / "collaborative_test/SW_0135_native32_spike_binding/results_archive/slot70k_native32_baseline.json"
SLOT_BASELINE_SHA256 = "4dc12cb54c3114375b8428cd250c013ff94015c4826915751f2fcb0a2204d753"
PRIMARY_RULE = "argmax SW0150 native32 P; column0 is background"
SECONDARY_RULE = "unchanged production SW0136 QCC from same T1024 trace"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    result = dict(train.implementation_fingerprint())
    paths = (HERE / "evaluate.py", HERE / "protocol.json",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/score_slot32.py",
             ROOT / "collaborative_test/SW_0136_native32_transfer/evaluate.py",
             ROOT / "collaborative_test/SW_0137_native32_source_qcc/run.py")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0150 evaluation dependency missing: {missing[0]}")
    result.update({path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths})
    return result


def implementation_sha256():
    payload = json.dumps(implementation_fingerprint(), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_record():
    checkpoint, manifest_path, manifest, rows, ids, core_sha = sw130.source_contract(SEED)
    ids = [int(value) for value in ids]
    rows = np.asarray(rows, dtype=np.int64)
    if len(ids) != 4096 or len(set(ids)) != 4096 or len(rows) != 4096:
        raise ValueError("source97 seed1 lineage must retain its fixed 4096 TRAIN order")
    return {"checkpoint": Path(checkpoint), "manifest_path": Path(manifest_path),
            "manifest": manifest, "rows": rows, "ids": ids, "core_sha256": core_sha}


def _fresh_foundation(device):
    # Use the exact source initialization seed/fork-RNG path used by the
    # registered preflight and trainer, then verify the full source contract.
    foundation = train._load_foundation(device)
    source = _source_record()
    assets = sw130.validate_rgb_assets()
    if (foundation.provenance.get("source_core_sha256") != source["core_sha256"]
            or foundation.provenance.get("source_manifest_sha256") != sha256_file(source["manifest_path"])
            or foundation.provenance.get("source_training_ids_sha256")
            != hashlib.sha256(np.asarray(source["ids"], dtype="<i8").tobytes()).hexdigest()
            or foundation.provenance.get("rgb_asset_validation") != assets
            or list(map(int, foundation.image_ids)) != source["ids"]
            or not np.array_equal(foundation.pool_indices, source["rows"])):
        raise ValueError("fresh SW0150 evaluation foundation differs from source97 seed1 lineage")
    return foundation, source, assets


def _load_trained_foundation(arm, device):
    trained = train.validate_training(arm)
    foundation, source, assets = _fresh_foundation(device)
    if hasattr(foundation.wrapped, "spatial_decoder"):
        raise ValueError("source foundation unexpectedly contains SW0150 decoder")
    decoder = make_decoder(8, seed=150001).to(device)
    foundation.wrapped.add_module("spatial_decoder", decoder)
    try:
        payload = torch.load(trained["checkpoint"], map_location=device, weights_only=False)
        if not isinstance(payload, dict) or set(payload) != {"wrapped", "encoder"}:
            raise ValueError("completed SW0150 checkpoint has an unexpected payload schema")
        foundation.wrapped.load_state_dict(payload["wrapped"], strict=True)
        foundation.encoder.load_state_dict(payload["encoder"], strict=True)
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        raise ValueError(f"cannot strictly load completed SW0150 {arm} checkpoint") from exc
    foundation.wrapped.eval()
    foundation.encoder.eval()
    foundation.wrapped.requires_grad_(False)
    foundation.encoder.requires_grad_(False)
    return foundation, source, assets, trained


def _validate_prediction_arrays(npz_path):
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {"image_ids", "primary_labels", "qcc_labels"}
        if not required.issubset(archive.files):
            raise ValueError("SW0150 prediction NPZ lacks required fixed arrays")
        ids = contract.require_fixed_image_ids(archive["image_ids"])
        primary = np.asarray(archive["primary_labels"])
        qcc = np.asarray(archive["qcc_labels"])
    for name, labels, maximum in (("primary_labels", primary, 10),
                                  ("qcc_labels", qcc, 1024)):
        if (labels.shape != (COUNT, GRID, GRID)
                or not np.issubdtype(labels.dtype, np.integer)
                or np.issubdtype(labels.dtype, np.bool_)
                or not np.isfinite(labels).all()
                or np.any(labels < 0) or np.any(labels > maximum)):
            raise ValueError(f"SW0150 {name} violates the fixed native32 integer schema")
    return ids, primary.astype(np.int64, copy=False), qcc.astype(np.int64, copy=False)


def _write_prediction(out_dir, arrays, protocol):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"preserve existing SW0150 prediction: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    npz = out_dir / "predictions.npz"
    with npz.open("xb") as stream:
        np.savez_compressed(stream, image_ids=IMAGE_IDS,
                            primary_labels=arrays["primary_labels"],
                            qcc_labels=arrays["qcc_labels"])
        stream.flush(); os.fsync(stream.fileno())
    protocol["prediction_sha256"] = sha256_file(npz)
    protocol["implementation_fingerprint"] = implementation_fingerprint()
    protocol["implementation_sha256"] = implementation_sha256()
    path = out_dir / "protocol.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    return {"npz": npz, "protocol": path, "protocol_sha256": sha256_file(path),
            "prediction_sha256": protocol["prediction_sha256"]}


@torch.no_grad()
def predict_arm(arm, *, device="cuda:0", output_root=PREDICTION_ROOT):
    if arm not in ARMS:
        raise ValueError("unregistered SW0150 candidate arm")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0150 fixed320 prediction requires its assigned CUDA device")
    torch.cuda.set_device(device)
    foundation, source, assets, trained = _load_trained_foundation(arm, device)
    initial_state_sha = state_dict_sha256((("wrapped", foundation.wrapped),
                                           ("encoder", foundation.encoder)))
    cache = np.load(sw130.VAL_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (COUNT, 128, 128, 3):
        raise ValueError("registered fixed320 validation RGB cache schema changed")
    primary, qcc = [], []
    for index in range(COUNT):
        rgb = torch.from_numpy(np.asarray(cache[index:index + 1]).copy()).to(device)
        result = integration.forward_assignment(foundation, rgb, live_tail_steps=0)
        if tuple(result["p"].shape[:2]) != (1, 1024) or result["p"].shape[2] > 11:
            raise ValueError("SW0150 P prediction has malformed native32 assignment shape")
        primary.append(result["primary_labels"][0].detach().cpu().numpy().astype(np.int64))
        qcc.append(result["qcc"][0].detach().cpu().numpy().astype(np.int64))
        if index % 32 == 31:
            print(json.dumps({"stage": "predict", "arm": arm, "images": index + 1,
                              "total": COUNT}, allow_nan=False), flush=True)
    arrays = {"primary_labels": np.stack(primary), "qcc_labels": np.stack(qcc)}
    ids, _p, _q = _validate_prediction_arrays_for_batch(arrays)
    if not np.array_equal(ids, IMAGE_IDS):
        raise AssertionError("SW0150 prediction order left the fixed320 range")
    final_state_sha = state_dict_sha256((("wrapped", foundation.wrapped),
                                         ("encoder", foundation.encoder)))
    if initial_state_sha != final_state_sha:
        raise AssertionError("SW0150 inference changed trained model parameters")
    protocol = _prediction_protocol("trained", arm, source, assets, trained=trained)
    return _write_prediction(Path(output_root) / f"{arm}_seed1", arrays, protocol)


@torch.no_grad()
def predict_source(*, device="cuda:0", output_root=PREDICTION_ROOT):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0150 source P prediction requires its assigned CUDA device")
    torch.cuda.set_device(device)
    foundation, source, assets = _fresh_foundation(device)
    foundation.wrapped.eval(); foundation.encoder.eval()
    before = state_dict_sha256((("wrapped", foundation.wrapped), ("encoder", foundation.encoder)))
    reference = sw137.validate_prediction(sw137.OUTPUT_ROOT / "seed1", seed=SEED)
    cache = np.load(sw130.VAL_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (COUNT, 128, 128, 3):
        raise ValueError("registered fixed320 validation RGB cache schema changed")
    primary, qcc = [], []
    for index in range(COUNT):
        rgb = torch.from_numpy(np.asarray(cache[index:index + 1]).copy()).to(device)
        result = integration.forward_assignment(foundation, rgb, live_tail_steps=0)
        primary.append(result["primary_labels"][0].cpu().numpy().astype(np.int64))
        qcc.append(result["qcc"][0].cpu().numpy().astype(np.int64))
        if index % 32 == 31:
            print(json.dumps({"stage": "predict_source", "images": index + 1,
                              "total": COUNT}, allow_nan=False), flush=True)
    arrays = {"primary_labels": np.stack(primary), "qcc_labels": np.stack(qcc)}
    ids, _p, source_qcc = _validate_prediction_arrays_for_batch(arrays)
    if not np.array_equal(ids, IMAGE_IDS) or not np.array_equal(source_qcc, reference["labels"]):
        raise ValueError("untrained source P QCC labels differ from frozen SW0137 QCC")
    after = state_dict_sha256((("wrapped", foundation.wrapped), ("encoder", foundation.encoder)))
    if before != after:
        raise AssertionError("untrained source P inference changed frozen source state")
    protocol = _prediction_protocol("source_p", "source_p", source, assets,
                                    source_qcc=reference)
    return _write_prediction(Path(output_root) / "source_p_seed1", arrays, protocol)


def _validate_prediction_arrays_for_batch(arrays):
    primary, qcc = np.asarray(arrays["primary_labels"]), np.asarray(arrays["qcc_labels"])
    if (primary.shape != (COUNT, GRID, GRID) or qcc.shape != (COUNT, GRID, GRID)
            or not np.issubdtype(primary.dtype, np.integer)
            or not np.issubdtype(qcc.dtype, np.integer)
            or np.any(primary < 0) or np.any(primary > 10)
            or np.any(qcc < 0) or np.any(qcc > 1024)):
        raise ValueError("generated SW0150 labels violate fixed320 native32 schema")
    return IMAGE_IDS, primary, qcc


def _prediction_protocol(role, name, source, assets, *, trained=None, source_qcc=None):
    row = {"experiment": NAME, "status": "complete", "role": role, "arm": name,
           "seed": SEED, "image_ids": [1320, 1639], "count": COUNT,
           "grid_size": [GRID, GRID], "prediction_shape": [COUNT, GRID, GRID],
           "primary_readout": PRIMARY_RULE,
           "secondary_readout": SECONDARY_RULE,
           "steps": 1024, "settle": 512, "live_tail": 0, "microbatch_size": 1,
           "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
           "source_core_sha256": source["core_sha256"],
           "source_manifest_sha256": sha256_file(source["manifest_path"]),
           "source_training_ids_sha256": train.ids_sha(source["ids"]),
           "rgb_asset_validation": assets,
           "training_manifest_sha256": trained["manifest_sha256"] if trained else None,
           "checkpoint_sha256": trained["manifest"]["checkpoint_sha256"] if trained else None,
           "preflight_sha256": trained["manifest"]["preflight_sha256"] if trained else None,
           "warm_sha256": trained["manifest"]["preflight_warm_sha256"] if trained else None,
           "source_qcc_prediction_sha256": source_qcc["prediction_sha256"] if source_qcc else None,
           "source_qcc_protocol_sha256": source_qcc["protocol_sha256"] if source_qcc else None}
    return row


def validate_prediction(name, *, prediction_root=PREDICTION_ROOT, training_root=train.TRAIN):
    if name not in (*ARMS, "source_p"):
        raise ValueError("unregistered SW0150 prediction name")
    directory = Path(prediction_root) / ("source_p_seed1" if name == "source_p"
                                       else f"{name}_seed1")
    archive_path, protocol_path = directory / "predictions.npz", directory / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes.decode("utf-8-sig"))
    source = _source_record()
    assets = sw130.validate_rgb_assets()
    source_manifest_sha = sha256_file(source["manifest_path"])
    if name == "source_p":
        trained = None
        expected_role = "source_p"
        source_qcc = sw137.validate_prediction(sw137.OUTPUT_ROOT / "seed1", seed=SEED)
    else:
        trained = train.validate_training(name, output_root=training_root)
        expected_role = "trained"
        source_qcc = None
    if (protocol.get("experiment") != NAME or protocol.get("status") != "complete"
            or protocol.get("role") != expected_role or protocol.get("arm") != name
            or protocol.get("seed") != SEED or protocol.get("image_ids") != [1320, 1639]
            or protocol.get("count") != COUNT or protocol.get("grid_size") != [GRID, GRID]
            or protocol.get("prediction_shape") != [COUNT, GRID, GRID]
            or protocol.get("primary_readout") != PRIMARY_RULE
            or protocol.get("secondary_readout") != SECONDARY_RULE
            or protocol.get("steps") != 1024 or protocol.get("settle") != 512
            or protocol.get("live_tail") != 0 or protocol.get("microbatch_size") != 1
            or protocol.get("ground_truth_used_for_prediction") is not False
            or protocol.get("optimizer_updates") != 0
            or protocol.get("source_core_sha256") != source["core_sha256"]
            or protocol.get("source_manifest_sha256") != source_manifest_sha
            or protocol.get("source_training_ids_sha256") != train.ids_sha(source["ids"])
            or protocol.get("rgb_asset_validation") != assets
            or protocol.get("implementation_fingerprint") != implementation_fingerprint()
            or protocol.get("implementation_sha256") != implementation_sha256()
            or protocol.get("prediction_sha256") != sha256_file(archive_path)):
        raise ValueError(f"invalid SW0150 {name} fixed320 prediction lineage")
    if trained is not None:
        manifest = trained["manifest"]
        if (protocol.get("training_manifest_sha256") != trained["manifest_sha256"]
                or protocol.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
                or protocol.get("preflight_sha256") != manifest["preflight_sha256"]
                or protocol.get("warm_sha256") != manifest["preflight_warm_sha256"]):
            raise ValueError("SW0150 prediction does not bind validated training checkpoint lineage")
    else:
        if (protocol.get("training_manifest_sha256") is not None
                or protocol.get("checkpoint_sha256") is not None
                or protocol.get("source_qcc_prediction_sha256") != source_qcc["prediction_sha256"]
                or protocol.get("source_qcc_protocol_sha256") != source_qcc["protocol_sha256"]):
            raise ValueError("source P prediction does not bind the immutable source QCC reference")
    ids, primary, qcc = _validate_prediction_arrays(archive_path)
    if source_qcc is not None and not np.array_equal(qcc, source_qcc["labels"]):
        raise ValueError("source P archive secondary QCC labels differ from SW0137")
    if (sha256_file(archive_path) != protocol["prediction_sha256"]
            or sha256_file(protocol_path) != protocol_sha):
        raise ValueError("SW0150 prediction bytes changed during validation")
    return {"name": name, "ids": ids, "primary_labels": primary, "qcc_labels": qcc,
            "protocol": protocol, "archive_path": archive_path,
            "protocol_path": protocol_path, "prediction_sha256": protocol["prediction_sha256"],
            "protocol_sha256": protocol_sha, "trained": trained}


def _slot_reference():
    directory = SLOT_ROOT / "seed1_epoch10"
    npz, protocol_path = directory / "predictions.npz", directory / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes.decode("utf-8-sig"))
    digest = contract.validate_slot_protocol(protocol, seed=SEED, prediction_path=npz)
    with np.load(npz, allow_pickle=False) as archive:
        if not {"labels", "image_ids"}.issubset(archive.files):
            raise ValueError("registered Slot native pixel archive lacks labels/image_ids")
        ids = contract.require_fixed_image_ids(archive["image_ids"], name="Slot image_ids")
        pixels = np.asarray(archive["labels"])
    labels = contract.slot_native_pixels_to32(pixels)
    if sha256_file(npz) != digest or sha256_file(protocol_path) != protocol_sha:
        raise ValueError("Slot prediction bytes changed during validation")
    return {"ids": ids, "labels": labels, "npz": npz,
            "protocol_path": protocol_path, "prediction_sha256": digest,
            "protocol_sha256": protocol_sha}


def _bootstrap_delta(left, right, draws):
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if delta.shape != (COUNT,) or not np.isfinite(delta).all():
        raise ValueError("paired SW0150 bootstrap requires 320 finite per-image scores")
    low, high = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
    return {"mean_difference": float(delta.mean()), "ci95": [float(low), float(high)]}


def score_all(*, output=SUMMARY, dataset=DATASET, prediction_root=PREDICTION_ROOT,
              training_root=train.TRAIN):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0150 score output: {output}")
    # Freeze every candidate and reference before the first target-mask read.
    own = {name: validate_prediction(name, prediction_root=prediction_root,
                                     training_root=training_root)
           for name in (*ARMS, "source_p")}
    source_qcc = sw137.validate_prediction(sw137.OUTPUT_ROOT / "seed1", seed=SEED)
    slot = _slot_reference()
    if not all(np.array_equal(own[name]["ids"], IMAGE_IDS) for name in own):
        raise ValueError("SW0150 prediction IDs do not exactly match the fixed320 range")
    if (not np.array_equal(source_qcc["ids"], IMAGE_IDS)
            or not np.array_equal(slot["ids"], IMAGE_IDS)
            or not np.array_equal(own["source_p"]["qcc_labels"], source_qcc["labels"])):
        raise ValueError("SW0150/source97/Slot fixed320 prediction arrays are not aligned")
    if sha256_file(SLOT_BASELINE) != SLOT_BASELINE_SHA256:
        raise ValueError("registered Slot32 three-seed baseline changed")
    baseline = json.loads(SLOT_BASELINE.read_text(encoding="utf-8"))
    expected_baseline_contract = {"image_ids": [1320, 1639, 320],
                                  "native_image_size": [128, 128],
                                  "patch_grid": [32, 32], "patch_pixels": 4,
                                  "metrics": list(contract.METRICS)}
    if (baseline.get("status") != "complete"
            or baseline.get("comparison") != "slot32 baseline only"
            or baseline.get("evaluation_contract") != expected_baseline_contract):
        raise ValueError("registered Slot32 baseline contract/schema changed")
    slot_means = {metric: float(baseline["scores"]["slot32"][metric]["three_seed_mean"])
                  for metric in contract.METRICS}
    if any(not np.isfinite(value) for value in slot_means.values()):
        raise ValueError("registered Slot32 comparison means are nonfinite")
    paths = [row[key] for row in own.values() for key in ("archive_path", "protocol_path")]
    paths.extend((source_qcc["archive_path"], source_qcc["protocol_path"], slot["npz"],
                  slot["protocol_path"], SLOT_BASELINE))
    frozen_hashes = {str(path): sha256_file(path) for path in paths}
    if (any(frozen_hashes[str(row["archive_path"])] != row["prediction_sha256"]
            or frozen_hashes[str(row["protocol_path"])] != row["protocol_sha256"]
            for row in own.values())
            or frozen_hashes[str(source_qcc["archive_path"])] != source_qcc["prediction_sha256"]
            or frozen_hashes[str(source_qcc["protocol_path"])] != source_qcc["protocol_sha256"]
            or frozen_hashes[str(slot["npz"])] != slot["prediction_sha256"]
            or frozen_hashes[str(slot["protocol_path"])] != slot["protocol_sha256"]
            or frozen_hashes[str(SLOT_BASELINE)] != SLOT_BASELINE_SHA256):
        raise ValueError("a frozen SW0150/source/Slot prediction changed before target access")

    with h5py.File(dataset, "r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0] <= int(IMAGE_IDS[-1]):
            raise ValueError("registered HDF5 native128 target masks lack fixed320 range")
        native_masks = np.asarray(h5["mask"][int(IMAGE_IDS[0]):int(IMAGE_IDS[-1]) + 1])
    target = contract.modal_native32(native_masks)
    scores = {
        "encoder_graph": {
            "primary": contract.production_metrics32(own["encoder_graph"]["primary_labels"], target),
            "qcc": contract.production_metrics32(own["encoder_graph"]["qcc_labels"], target)},
        "joint": {
            "primary": contract.production_metrics32(own["joint"]["primary_labels"], target),
            "qcc": contract.production_metrics32(own["joint"]["qcc_labels"], target)},
        "source_p": {"primary": contract.production_metrics32(own["source_p"]["primary_labels"], target)},
        "source97_qcc": {"qcc": contract.production_metrics32(source_qcc["labels"], target)},
        "slot32_seed1": {"slot32": contract.production_metrics32(slot["labels"], target)},
    }
    metric_means = lambda result: {metric: float(result[metric]["mean"])
                                   for metric in contract.METRICS}
    means = {name: {readout: metric_means(values)
                    for readout, values in readouts.items()}
             for name, readouts in scores.items()}
    draws = np.random.RandomState(139).randint(0, COUNT, size=(10000, COUNT))
    paired = {}
    for reference, score_key, readout in (("source_qcc", "source97_qcc", "primary"),
                                          ("control_primary", "encoder_graph", "primary")):
        reference_readout = "qcc" if reference == "source_qcc" else readout
        paired[f"joint_primary_minus_{reference}"] = {
            metric: _bootstrap_delta(scores["joint"]["primary"][metric]["per_image"],
                                     scores[score_key][reference_readout][metric]["per_image"], draws)
            for metric in contract.METRICS}
    for arm in ARMS:
        paired[f"{arm}_primary_minus_source_primary"] = {
            metric: _bootstrap_delta(scores[arm]["primary"][metric]["per_image"],
                                     scores["source_p"]["primary"][metric]["per_image"], draws)
            for metric in contract.METRICS}
    paired["source_primary_minus_source_qcc"] = {
        metric: _bootstrap_delta(scores["source_p"]["primary"][metric]["per_image"],
                                 scores["source97_qcc"]["qcc"][metric]["per_image"], draws)
        for metric in contract.METRICS}
    gate = {
        "joint_primary_fg_at_least_source_qcc_plus_0_01":
            means["joint"]["primary"]["fg_ari"] >= means["source97_qcc"]["qcc"]["fg_ari"] + 0.01,
        "joint_primary_fg_at_least_encoder_graph_primary_plus_0_01":
            means["joint"]["primary"]["fg_ari"] >= means["encoder_graph"]["primary"]["fg_ari"] + 0.01,
        "joint_source_qcc_fg_ci_lower_positive":
            paired["joint_primary_minus_source_qcc"]["fg_ari"]["ci95"][0] > 0,
        "joint_control_fg_ci_lower_positive":
            paired["joint_primary_minus_control_primary"]["fg_ari"]["ci95"][0] > 0,
        "joint_primary_foreground_iou_above_slot_three_seed_plus_0_05":
            means["joint"]["primary"]["foreground_iou"] > slot_means["foreground_iou"] + 0.05,
        "joint_primary_matched_object_iou_above_slot_three_seed_plus_0_05":
            means["joint"]["primary"]["matched_object_iou"] > slot_means["matched_object_iou"] + 0.05,
    }
    report = {
        "experiment": NAME, "status": "complete", "seed_scope": [SEED],
        "scores": scores, "means": means,
        "registered_slot32_three_seed_means": slot_means,
        "paired_bootstrap": paired,
        "bootstrap": {"samples": 10000, "rng": "numpy RandomState", "seed": 139,
                      "common_image_indices": True},
        "promotion_gate": gate, "may_promote_seed_expansion": all(gate.values()),
        "prediction_provenance": {
            name: {"prediction_sha256": row["prediction_sha256"],
                  "protocol_sha256": row["protocol_sha256"],
                  "training_manifest_sha256": row["protocol"].get("training_manifest_sha256"),
                  "checkpoint_sha256": row["protocol"].get("checkpoint_sha256")}
            for name, row in own.items()},
        "reference_provenance": {"source97_qcc_prediction_sha256": source_qcc["prediction_sha256"],
                                 "source97_qcc_protocol_sha256": source_qcc["protocol_sha256"],
                                 "slot32_seed1_prediction_sha256": slot["prediction_sha256"],
                                 "slot32_seed1_protocol_sha256": slot["protocol_sha256"],
                                 "slot32_three_seed_baseline_sha256": SLOT_BASELINE_SHA256},
        "ground_truth_opened_after_all_predictions_validated": True,
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
        "limits": "Fixed seed1 native32 pilot only; no fresh 70k native32 training, scaling, or universal superiority claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("predict", "predict-source", "score"), required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.stage == "predict":
        if args.arm is None:
            parser.error("--arm is required for candidate prediction")
        result = predict_arm(args.arm, device=args.device)
    elif args.stage == "predict-source":
        result = predict_source(device=args.device)
    else:
        result = score_all()
    print(json.dumps({"status": result.get("status", "complete"),
                      "stage": args.stage, "arm": args.arm,
                      "ground_truth_used_for_prediction": False,
                      "ground_truth_used_for_scoring": args.stage == "score"},
                     allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
