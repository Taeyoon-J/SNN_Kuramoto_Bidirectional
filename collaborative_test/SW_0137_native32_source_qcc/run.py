"""Frozen source97 native32 actual-spike QCC prediction and GT-last scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from functools import lru_cache

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
    COUNT, GRID, IMAGE_IDS, METRICS, modal_native32, production_metrics32,
    require_fixed_image_ids, sha256_file, slot_native_pixels_to32, validate_slot_protocol,
)
from collaborative_test.SW_0135_native32_spike_binding.foundation import (
    load_native32_foundation,
)
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136_eval
from collaborative_test.SW_0136_native32_transfer.transfer import state_dict_sha256
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components

ARCHIVE = HERE / "results_archive"
OUTPUT_ROOT = ARCHIVE / "source_predictions"
SUMMARY_DIR = ARCHIVE / "source_qcc_evaluation"
DATASET = Path(sw130.source97.DATASET)
SLOT_ROOT = ROOT / "trained_models/SW0092_slot_our70000_eval"
SLOT_BASELINE = ROOT / "collaborative_test/SW_0135_native32_spike_binding/results_archive/slot70k_native32_baseline.json"
SLOT_BASELINE_SHA256 = "4dc12cb54c3114375b8428cd250c013ff94015c4826915751f2fcb0a2204d753"
SEEDS = (0, 1, 2)


@lru_cache(maxsize=1)
def _validated_rgb_assets():
    return sw130.validate_rgb_assets()


def _write_json_once(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def _write_npz_once(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def implementation_fingerprint():
    files = [HERE / "protocol.json", HERE / "run.py",
             ROOT / "collaborative_test/SW_0130_phase_state_integration/run.py",
             ROOT / "collaborative_test/SW_0130_phase_state_integration/model.py",
             ROOT / "collaborative_test/SW_0134_native_spike_binding/rollout.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/resolution.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/score_slot32.py",
             ROOT / "collaborative_test/SW_0136_native32_transfer/evaluate.py",
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
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0137 implementation dependency missing: {missing[0]}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in files}


def _prediction_dir(seed, output_root=OUTPUT_ROOT):
    if int(seed) not in SEEDS:
        raise ValueError("SW0137 only registers source seeds 0, 1 and 2")
    return Path(output_root) / f"seed{int(seed)}"


def _read_npz_labels(path):
    with np.load(path, allow_pickle=False) as archive:
        if not {"image_ids", "qcc_labels"}.issubset(archive.files):
            raise ValueError("source QCC archive is missing image_ids or qcc_labels")
        ids = require_fixed_image_ids(archive["image_ids"])
        labels = np.asarray(archive["qcc_labels"])
    if (labels.shape != (COUNT, GRID, GRID) or not np.issubdtype(labels.dtype, np.integer)
            or np.issubdtype(labels.dtype, np.bool_) or np.any(labels < 0) or np.any(labels > 1024)):
        raise ValueError("source QCC labels violate the fixed native32 integer schema")
    return ids, labels.astype(np.int64, copy=False)


def validate_prediction(directory, *, seed):
    directory = Path(directory)
    archive_path, protocol_path = directory / "predictions.npz", directory / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    source_contract = sw130.source_contract(int(seed))
    source_manifest_sha = sha256_file(source_contract[1])
    assets = _validated_rgb_assets()
    preflight = protocol.get("preflight")
    if (protocol.get("experiment") != "SW0137_native32_source_qcc"
            or protocol.get("status") != "complete" or protocol.get("seed") != int(seed)
            or protocol.get("image_ids") != [1320, 1639] or protocol.get("count") != 320
            or protocol.get("grid_size") != [32, 32]
            or protocol.get("prediction_shape") != [320, 32, 32]
            or protocol.get("steps") != 1024 or protocol.get("settle") != 512
            or protocol.get("microbatch_size") != 1
            or protocol.get("ground_truth_used_for_prediction") is not False
            or protocol.get("optimizer_updates") != 0
            or protocol.get("source_core_sha256") != sw130.source97.EXPECTED_SOURCE_SHAS[int(seed)]
            or protocol.get("source_manifest_sha256") != source_manifest_sha
            or protocol.get("encoder_sha256") != assets["encoder_source_sha256"]
            or protocol.get("feature_preprocessing_sha256") != assets["feature_preprocessing_sha256"]
            or protocol.get("rgb_assets") != assets
            or not protocol.get("source_native32_state_sha256")
            or not protocol.get("source_wrapped_state_sha256")
            or not isinstance(preflight, dict)
            or preflight.get("split") not in {"TRAIN", "registered first seed1 TRAIN image"}
            or preflight.get("ground_truth_used") is not False
            or preflight.get("finite") is not True
            or preflight.get("image_id") != int(source_contract[4][0])
            or preflight.get("qcc_shape") != [1, 32, 32]
            or protocol.get("implementation") != implementation_fingerprint()
            or protocol.get("prediction_sha256") != sha256_file(archive_path)):
        raise ValueError(f"invalid SW0137 source prediction/provenance for seed{seed}")
    ids, labels = _read_npz_labels(archive_path)
    if (sha256_file(archive_path) != protocol["prediction_sha256"]
            or sha256_file(protocol_path) != protocol_sha):
        raise ValueError("source QCC prediction changed during validation")
    return {"seed": int(seed), "ids": ids, "labels": labels, "protocol": protocol,
            "archive_path": archive_path, "protocol_path": protocol_path,
            "prediction_sha256": protocol["prediction_sha256"], "protocol_sha256": protocol_sha}


def _write_prediction(directory, seed, labels, source_record, *, reused=None):
    directory = Path(directory)
    if labels.shape != (COUNT, GRID, GRID) or not np.isfinite(labels).all():
        raise ValueError("source prediction is not finite fixed320 native32 labels")
    if directory.exists():
        existing = {path.name for path in directory.iterdir()}
        if existing != {"preflight.json"}:
            raise FileExistsError(f"preserve existing SW0137 source result: {directory}")
    else:
        directory.mkdir(parents=True, exist_ok=False)
    path = directory / "predictions.npz"
    _write_npz_once(path, image_ids=np.asarray(IMAGE_IDS, dtype=np.int64),
                    qcc_labels=np.asarray(labels, dtype=np.int64))
    protocol = {
        "experiment": "SW0137_native32_source_qcc", "status": "complete", "seed": int(seed),
        "image_ids": [IMAGE_IDS[0], IMAGE_IDS[-1]], "count": COUNT,
        "grid_size": [GRID, GRID], "prediction_shape": [COUNT, GRID, GRID],
        "steps": 1024, "settle": 512, "microbatch_size": 1,
        "readout": {"affinity": "product of positive signed component Pearson correlations on actual spikes",
                    "threshold": 0.50, "min_group_size": 8,
                    "background": "largest connected component"},
        "source_core_sha256": source_record["source_core_sha256"],
        "source_manifest_sha256": source_record["source_manifest_sha256"],
        "source_native32_state_sha256": source_record["source_native32_state_sha256"],
        "source_wrapped_state_sha256": source_record["source_wrapped_state_sha256"],
        "encoder_sha256": source_record["encoder_sha256"],
        "feature_preprocessing_sha256": source_record["feature_preprocessing_sha256"],
        "rgb_assets": source_record["rgb_assets"],
        "preflight": source_record.get("preflight"),
        "prediction_sha256": sha256_file(path), "implementation": implementation_fingerprint(),
        "reused_sw0136_seed1": reused,
        "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
    }
    _write_json_once(directory / "protocol.json", protocol)
    return protocol


def _write_preflight_once(directory, preflight):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    _write_json_once(directory / "preflight.json", preflight)


@torch.no_grad()
def _predict_source_qcc(foundation, cache, *, device):
    ids = np.asarray(IMAGE_IDS, dtype=np.int64)
    labels = np.empty((COUNT, GRID, GRID), dtype=np.int64)
    for index in range(COUNT):
        rgb = torch.from_numpy(np.asarray(cache[index:index + 1]).copy()).to(device)
        gamma = foundation.encode(rgb)
        _, qcc = sw136_eval._predict_one(foundation, None, gamma, primary=False)
        labels[index] = qcc[0]
        if index % 40 == 39:
            print(json.dumps({"stage": "source_qcc_prediction", "predicted": index + 1,
                              "count": COUNT}), flush=True)
    return ids, labels


def _validate_train_preflight_output(gamma, qcc):
    """Validate the mixed Torch/NumPy readout returned by the frozen SW0136 API."""
    if tuple(qcc.shape) != (1, 32, 32) or not bool(torch.isfinite(gamma).all()) \
            or not bool(np.isfinite(np.asarray(qcc)).all()):
        raise ValueError("source97 native32 TRAIN structural preflight failed")


def _source_record(foundation):
    return {
        "source_core_sha256": foundation.provenance["source_core_sha256"],
        "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
        "source_native32_state_sha256": state_dict_sha256((("wrapped", foundation.wrapped),
                                                            ("encoder", foundation.encoder))),
        "source_wrapped_state_sha256": state_dict_sha256((("wrapped", foundation.wrapped),)),
        "encoder_sha256": foundation.provenance["encoder_sha256"],
        "feature_preprocessing_sha256": foundation.provenance["feature_preprocessing_sha256"],
        "rgb_assets": foundation.provenance["rgb_asset_validation"],
    }


def _normalize_sw136_train_preflight(preflight, *, image_id, pool_row):
    """Bind SW0136's native fields to the same first registered TRAIN image."""
    if (not isinstance(preflight, dict)
            or preflight.get("image_id") != int(image_id)
            or preflight.get("pool_row") != int(pool_row)
            or preflight.get("ground_truth_used") is not False
            or preflight.get("finite") is not True
            or preflight.get("qcc_shape") != [1, 32, 32]
            or preflight.get("gamma_shape") != [1, 8, 1024]):
        raise ValueError("SW0136 preflight is not bound to the registered first TRAIN row")
    return {**preflight, "split": "TRAIN"}


def _sw136_source_path():
    return sw136_eval.PREDICTION_ROOT / "source97_qcc_seed1"


def _sw136_source_process_live():
    """Require live /proc argv evidence; a stale queue state alone is insufficient."""
    state_path = sw136_eval.ARCHIVE / "transfer_queue_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    task = state.get("tasks", {}).get("sw0136_actual_joint_seed1", {})
    status = task.get("status")
    pid = task.get("child_pid")
    if pid is not None:
        cmdline_path = Path(f"/proc/{int(pid)}/cmdline")
        try:
            argv = cmdline_path.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            argv = ""
        if ("collaborative_test.SW_0136_native32_transfer.evaluate" in argv
                and "--arm actual_joint" in argv):
            return True
    # Before a child starts, wait only when the exact supervisor is still alive.
    supervisor = state.get("supervisor_pid")
    if status in {"queued", "waiting_for_exclusive_gpu", "reserved", "launching"} and supervisor:
        try:
            argv = Path(f"/proc/{int(supervisor)}/cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace")
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            argv = ""
        return "SW_0136_native32_transfer.transfer_queue" in argv
    return False


def _reuse_sw136_seed1(output_dir):
    """Reuse the frozen SW0136 source row only after code, bytes, IDs and source assets validate."""
    from collaborative_test.SW_0136_native32_transfer import transfer as sw136_transfer
    source_dir = _sw136_source_path()
    try:
        reused = sw136_eval.validate_prediction(source_dir, arm="source97_qcc")
    except FileNotFoundError:
        return None
    protocol = reused["protocol"]
    expected_source = sw130.source_contract(1)
    foundation = load_native32_foundation(1, "cpu", verify_rgb_assets=True)
    current_record = _source_record(foundation)
    transfer = protocol.get("transfer", {})
    upstream_preflight = transfer.get("structural_preflight", {})
    expected_pool_row = int(expected_source[3][0])
    expected_image_id = int(expected_source[4][0])
    # SW0136 records the actual fields (no synthetic ``split`` key). Normalize
    # only after binding both the global TRAIN ID and its registered pool row.
    reused_preflight = _normalize_sw136_train_preflight(
        upstream_preflight, image_id=expected_image_id, pool_row=expected_pool_row)
    # The mapped checkpoint is identical, but a device-specific construction
    # can alter serialized non-parameter buffers. In that case perform the
    # registered seed1 inference here instead of treating the valid upstream
    # prediction as reusable or weakening its provenance checks.
    if transfer.get("converted_state_sha256") != current_record["source_wrapped_state_sha256"]:
        return None
    if (protocol.get("steps") != 1024 or protocol.get("settle") != 512
            or protocol.get("microbatch_size") != 1
            or protocol.get("ground_truth_used_for_prediction") is not False
            or protocol.get("optimizer_updates") != 0
            or protocol.get("primary") != "mapped source97 QCC reference only"
            or transfer.get("reference_role") != "mapped SW0097 native32 QCC only"
            or transfer.get("source_core_sha256") != expected_source[5]
            or transfer.get("source_manifest_sha256") != sha256_file(expected_source[1])
            or protocol.get("implementation") != sw136_eval.implementation_fingerprint()):
        raise ValueError("existing SW0136 seed1 source reference does not meet reuse requirements")
    if not current_record["rgb_assets"] or current_record["source_core_sha256"] != expected_source[5]:
        raise ValueError("current seed1 source assets/core fail validation before reusing prediction")
    reused_record = {
        "origin": "SW0136 actual_joint source97 QCC artifact",
        "origin_prediction_sha256": reused["prediction_sha256"],
        "origin_protocol_sha256": reused["protocol_sha256"],
        "origin_implementation": protocol["implementation"],
        "preflight": reused_preflight,
        **current_record,
    }
    _, labels = _read_npz_labels(reused["prediction_path"])
    _write_prediction(output_dir, 1, labels, reused_record, reused=reused_record)
    return {"status": "prediction_complete", "seed": 1, "reused": True,
            "origin_prediction_sha256": reused["prediction_sha256"]}


def wait_and_reuse_sw136_seed1(output_dir):
    """Reuse a complete SW0136 source result, waiting only for a verified live producer."""
    source_dir = _sw136_source_path()
    prediction = source_dir / "predictions.npz"
    protocol = source_dir / "protocol.json"
    while _sw136_source_process_live() and not (prediction.is_file() and protocol.is_file()):
        time.sleep(5)
    if prediction.is_file() and protocol.is_file():
        return _reuse_sw136_seed1(output_dir)
    return None


@torch.no_grad()
def predict_seed(seed, *, device="cuda:0", output_root=OUTPUT_ROOT, allow_seed1_reuse=True):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("SW0137 only registers source seeds 0, 1 and 2")
    output_dir = _prediction_dir(seed, output_root)
    if output_dir.exists():
        raise FileExistsError(f"preserve existing SW0137 prediction: {output_dir}")
    if seed == 1 and allow_seed1_reuse:
        reused = wait_and_reuse_sw136_seed1(output_dir)
        if reused is not None:
            return reused

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0137 full source evaluation requires an exclusive CUDA device")
    torch.cuda.set_device(device); torch.cuda.init()
    foundation = load_native32_foundation(seed, device, verify_rgb_assets=True)
    foundation.encoder.eval(); foundation.wrapped.eval()
    record = _source_record(foundation)
    source_state_sha_before = record["source_native32_state_sha256"]
    if record["source_core_sha256"] != sw130.source97.EXPECTED_SOURCE_SHAS[seed]:
        raise ValueError("native32 foundation did not load the registered source97 checkpoint")
    _checkpoint, _manifest_path, manifest, pool, ids, _source_sha = sw130.source_contract(seed)
    if len(pool) != 4096 or len(ids) != 4096:
        raise ValueError("registered source97 training-order contract is invalid")
    row = int(pool[0]); image_id = int(ids[0])

    # One GT-free source-ordered TRAIN example checks the real32 path before the fixed VAL sweep.
    train_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r")
    if train_cache.dtype != np.uint8 or train_cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered source RGB TRAIN cache has unexpected shape/dtype")
    sample = torch.from_numpy(np.asarray(train_cache[row:row + 1]).copy()).to(device)
    gamma = foundation.encode(sample)
    _sample, sample_qcc = sw136_eval._predict_one(foundation, None, gamma, primary=False)
    _validate_train_preflight_output(gamma, sample_qcc)
    preflight = {"split": "TRAIN", "image_id": image_id, "pool_row": row,
                 "gamma_shape": list(gamma.shape), "qcc_shape": list(sample_qcc.shape),
                 "finite": True, "ground_truth_used": False}
    _write_preflight_once(output_dir, preflight)
    del train_cache, sample, gamma, sample_qcc, _sample

    cache = np.load(sw130.VAL_RGB, mmap_mode="r")
    if cache.shape != (COUNT, 128, 128, 3) or cache.dtype != np.uint8:
        raise ValueError("registered RGB validation cache does not match fixed320 native128 contract")
    ids_array, labels = _predict_source_qcc(foundation, cache, device=device)
    if not np.array_equal(ids_array, np.asarray(IMAGE_IDS, dtype=np.int64)):
        raise AssertionError("source QCC loop left the fixed ordered validation slice")
    source_state_sha_after = state_dict_sha256((("wrapped", foundation.wrapped),
                                                ("encoder", foundation.encoder)))
    if source_state_sha_after != source_state_sha_before:
        raise AssertionError("read-only source prediction changed native model/encoder state")
    record["preflight"] = preflight
    _write_prediction(output_dir, seed, labels, record)
    return {"status": "prediction_complete", "seed": seed, "reused": False,
            "preflight": preflight, "prediction_path": str(output_dir / "predictions.npz")}


def score_sources(*, output_root=OUTPUT_ROOT, dataset=DATASET, slot_root=SLOT_ROOT,
                  output=SUMMARY_DIR):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0137 scoring output: {output}")
    source_rows = {seed: validate_prediction(_prediction_dir(seed, output_root), seed=seed)
                   for seed in SEEDS}
    slot_rows = {}
    for seed in SEEDS:
        folder = Path(slot_root) / f"seed{seed}_epoch10"
        npz, protocol_path = folder / "predictions.npz", folder / "protocol.json"
        protocol = score_slot32._read_json(protocol_path)
        digest = validate_slot_protocol(protocol, seed=seed, prediction_path=npz)
        with np.load(npz, allow_pickle=False) as archive:
            if not {"labels", "image_ids"}.issubset(archive.files):
                raise ValueError("Slot native128 prediction archive missing labels/image_ids")
            slot_ids = require_fixed_image_ids(archive["image_ids"], name="Slot image IDs")
            pixels = np.asarray(archive["labels"])
        slot32 = slot_native_pixels_to32(pixels)
        slot_rows[seed] = {"labels": slot32, "ids": slot_ids, "npz": npz, "protocol_path": protocol_path,
                           "prediction_sha256": digest, "protocol_sha256": sha256_file(protocol_path),
                           "native128": pixels, "folder": folder}
    if sha256_file(SLOT_BASELINE) != SLOT_BASELINE_SHA256:
        raise ValueError("registered Slot32 baseline changed before scoring")
    baseline = json.loads(SLOT_BASELINE.read_text(encoding="utf-8"))
    if not np.array_equal(source_rows[0]["ids"], source_rows[1]["ids"]) or not np.array_equal(
            source_rows[1]["ids"], source_rows[2]["ids"]):
        raise ValueError("source97 seed prediction IDs are not paired")

    # Reverify every source/Slot payload and protocol before the first HDF5 target read.
    for row in source_rows.values():
        if (sha256_file(row["archive_path"]) != row["prediction_sha256"]
                or sha256_file(row["protocol_path"]) != row["protocol_sha256"]):
            raise ValueError("frozen source97 QCC prediction changed before target access")
    for row in slot_rows.values():
        if (sha256_file(row["npz"]) != row["prediction_sha256"]
                or sha256_file(row["protocol_path"]) != row["protocol_sha256"]):
            raise ValueError("frozen Slot32 source changed before target access")
    if sha256_file(SLOT_BASELINE) != SLOT_BASELINE_SHA256:
        raise ValueError("registered Slot32 baseline changed before target access")
    if not np.array_equal(source_rows[0]["ids"], np.asarray(IMAGE_IDS, dtype=np.int64)):
        raise ValueError("source predictions do not match the fixed320 ID range")

    with h5py.File(dataset, "r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0] <= IMAGE_IDS[-1]:
            raise ValueError("registered HDF5 target masks are missing the fixed320 range")
        pixel_gt = np.asarray(h5["mask"][IMAGE_IDS[0]:IMAGE_IDS[-1] + 1])
    gt32 = modal_native32(pixel_gt)
    scores = {f"source97_seed{seed}": {"qcc": production_metrics32(row["labels"], gt32)}
              for seed, row in source_rows.items()}
    scores.update({f"slot_seed{seed}": {"slot32": production_metrics32(row["labels"], gt32)}
                   for seed, row in slot_rows.items()})
    source_means = {metric: float(np.mean([scores[f"source97_seed{s}"]["qcc"][metric]["mean"]
                                          for s in SEEDS])) for metric in METRICS}
    slot_means = {metric: float(np.mean([scores[f"slot_seed{s}"]["slot32"][metric]["mean"]
                                         for s in SEEDS])) for metric in METRICS}
    baseline_means = baseline["scores"]["slot32"]
    for metric in METRICS:
        if abs(slot_means[metric] - float(baseline_means[metric]["three_seed_mean"])) > 1e-10:
            raise ValueError(f"saved Slot32 {metric} baseline failed exact three-seed reproduction")
    report = {
        "experiment": "SW0137_native32_source_qcc", "status": "complete",
        "seed_scope": list(SEEDS), "scores": scores,
        "source97_three_seed_mean": source_means,
        "slot32_three_seed_mean": slot_means,
        "slot32_registered_baseline_sha256": SLOT_BASELINE_SHA256,
        "prediction_provenance": {f"source97_seed{s}": {
            "prediction_sha256": source_rows[s]["prediction_sha256"],
            "protocol_sha256": source_rows[s]["protocol_sha256"],
            "source_core_sha256": source_rows[s]["protocol"]["source_core_sha256"],
            "source_manifest_sha256": source_rows[s]["protocol"]["source_manifest_sha256"],
            "reused_sw0136_seed1": bool(source_rows[s]["protocol"].get("reused_sw0136_seed1"))}
            for s in SEEDS},
        "evaluation_contract": {"image_ids": [1320, 1639, 320], "grid_size": [32, 32],
                               "metrics": list(METRICS), "ground_truth_after_all_prediction_hashes": True},
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
        "optimizer_updates": 0,
        "interpretation": "Three-seed frozen-source native32 QCC evaluation against the fixed Slot32 baseline; no training, gate attribution, scaling, or promotion claim.",
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_json_once(output / "evaluation.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("predict", "score"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--slot-root", type=Path, default=SLOT_ROOT)
    parser.add_argument("--output", type=Path, default=SUMMARY_DIR)
    parser.add_argument("--no-reuse-seed1", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "predict":
        if args.seed is None:
            parser.error("--seed is required for prediction stage")
        result = predict_seed(args.seed, device=args.device, output_root=args.output_root,
                              allow_seed1_reuse=not args.no_reuse_seed1)
    else:
        result = score_sources(output_root=args.output_root, dataset=args.dataset,
                               slot_root=args.slot_root, output=args.output)
    print(json.dumps({"status": result.get("status", "complete"), "stage": args.stage,
                      "seed": args.seed, "ground_truth_used_for_prediction": False,
                      "optimizer_updates": 0}, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
