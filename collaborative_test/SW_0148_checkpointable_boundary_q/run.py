"""SW0148 hash-bound native32 prediction and GT-last scoring stages."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
               str(ROOT / "snn_kuramoto_bidirectional")]
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136_eval
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0146_native32_boundary_q_pilot import run as sw146
from collaborative_test.SW_0148_checkpointable_boundary_q import train as sw148_train

NAME = sw148_train.NAME
ARMS = sw146.ARMS
SEED = sw146.SEED
GRID, COUNT, STEPS, SETTLE, TAIL = sw146.GRID, sw146.COUNT, sw146.STEPS, sw146.SETTLE, sw146.TAIL
PRED = HERE / "predictions"
ARCHIVE = HERE / "results_archive"
SOURCE = sw137.OUTPUT_ROOT / "seed1"
SLOT_BASELINE = sw146.SLOT_BASELINE
SLOT_BASELINE_SHA = sw146.SLOT_BASELINE_SHA
DATASET = sw130.source97.DATASET


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def implementation_fingerprint():
    result = dict(sw146.fingerprint())
    for path in (HERE / "run.py", HERE / "train.py", HERE / "checkpoint.py",
                 HERE / "protocol.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
        result[path.relative_to(ROOT).as_posix()] = sha(path)
    return result


def implementation_sha():
    raw = json.dumps(implementation_fingerprint(), sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _prediction_dir(arm, root=PRED):
    if arm not in ARMS:
        raise ValueError("unregistered SW0148 arm")
    return Path(root) / f"{arm}_seed1"


def _foundation(device):
    f, src = sw146._source_foundation(device)
    ids, rows, _ = sw146.ordered_train()
    expected_assets = sw146.validated_assets()
    if (f.provenance.get("source_core_sha256") != src[5]
            or f.provenance.get("source_manifest_sha256") != sw146.sha(src[1])
            or list(map(int, f.image_ids)) != list(map(int, src[4]))
            or not np.array_equal(f.pool_indices, np.asarray(src[3], dtype=np.int64))
            or not np.array_equal(ids, list(map(int, src[4]))[:len(ids)])
            or rows.shape[0] != len(ids)
            or f.provenance.get("rgb_asset_validation") != expected_assets):
        raise ValueError("SW0148 native32 source foundation mismatch")
    return f


def predict_arm(arm: str, *, device="cuda:0", training_root=sw148_train.TRAIN,
                output_root=PRED) -> dict:
    trained = sw148_train.validate_training(arm, output_root=training_root)
    out = _prediction_dir(arm, output_root)
    if out.exists():
        raise FileExistsError(f"preserve existing SW0148 prediction directory: {out}")
    f = _foundation(device)
    state = torch.load(trained["checkpoint"], map_location=device, weights_only=True)
    f.wrapped.load_state_dict(state["wrapped"], strict=True)
    f.encoder.load_state_dict(state["encoder"], strict=True)
    f.wrapped.eval()
    f.encoder.eval()
    cache = np.load(sw130.VAL_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape[0] < COUNT or cache.shape[1:] != (128, 128, 3):
        raise ValueError("registered fixed320 validation RGB cache mismatch")
    labels = np.empty((COUNT, GRID, GRID), dtype=np.int64)
    with torch.no_grad():
        for i in range(COUNT):
            rgb = torch.from_numpy(np.asarray(cache[i:i + 1]).copy()).to(device)
            gamma = f.encode(rgb)
            _native, qcc = sw136_eval._predict_one(f, None, gamma, primary=False)
            qcc = np.asarray(qcc)
            if qcc.shape != (1, GRID, GRID) or not np.isfinite(qcc).all():
                raise ValueError("invalid fixed320 actual-spike QCC prediction")
            labels[i] = qcc[0]
            if (i + 1) % 40 == 0:
                print(json.dumps({"stage": "predict", "arm": arm, "count": i + 1}), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".{out.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    npz = staging / "predictions.npz"
    image_ids = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
    with npz.open("xb") as stream:
        np.savez_compressed(stream, image_ids=image_ids, qcc_labels=labels)
    protocol = {
        "experiment": NAME, "status": "complete", "seed": SEED, "arm": arm,
        "image_ids": [1320, 1639], "count": COUNT,
        "grid_size": [GRID, GRID], "prediction_shape": [COUNT, GRID, GRID],
        "steps": STEPS, "settle": SETTLE, "live_tail": TAIL,
        "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
        "training_manifest_sha256": trained["manifest_sha256"],
        "checkpoint_sha256": trained["manifest"]["checkpoint_sha256"],
        "source_core_sha256": f.provenance["source_core_sha256"],
        "source_manifest_sha256": f.provenance["source_manifest_sha256"],
        "source_training_ids_sha256": f.provenance["source_training_ids_sha256"],
        "rgb_asset_validation": f.provenance["rgb_asset_validation"],
        "prediction_sha256": sha(npz),
        "implementation_fingerprint": implementation_fingerprint(),
        "implementation_sha256": implementation_sha(),
    }
    sw146.write_once(staging / "protocol.json", protocol)
    os.rename(staging, out)
    return protocol


def validate_prediction(arm, *, prediction_root=PRED, training_root=sw148_train.TRAIN):
    directory = _prediction_dir(arm, prediction_root)
    npz, protocol_path = directory / "predictions.npz", directory / "protocol.json"
    trained = sw148_train.validate_training(arm, output_root=training_root)
    raw = protocol_path.read_bytes()
    protocol_sha = hashlib.sha256(raw).hexdigest()
    protocol = json.loads(raw.decode("utf-8"))
    src = sw146.source(SEED)
    current_assets = sw146.validated_assets()
    expected_id_sha = hashlib.sha256(
        np.asarray(src[4], dtype="<i8").tobytes()).hexdigest()
    if (protocol.get("experiment") != NAME or protocol.get("status") != "complete"
            or protocol.get("seed") != SEED or protocol.get("arm") != arm
            or protocol.get("image_ids") != [1320, 1639]
            or protocol.get("count") != COUNT
            or protocol.get("grid_size") != [GRID, GRID]
            or protocol.get("prediction_shape") != [COUNT, GRID, GRID]
            or protocol.get("steps") != STEPS or protocol.get("settle") != SETTLE
            or protocol.get("live_tail") != TAIL
            or protocol.get("ground_truth_used_for_prediction") is not False
            or protocol.get("optimizer_updates") != 0
            or protocol.get("training_manifest_sha256") != trained["manifest_sha256"]
            or protocol.get("checkpoint_sha256") != trained["manifest"]["checkpoint_sha256"]
            or protocol.get("source_core_sha256") != src[5]
            or protocol.get("source_manifest_sha256") != sw146.sha(src[1])
            or protocol.get("source_training_ids_sha256") != expected_id_sha
            or protocol.get("rgb_asset_validation") != current_assets
            or protocol.get("implementation_fingerprint") != implementation_fingerprint()
            or protocol.get("implementation_sha256") != implementation_sha()
            or protocol.get("prediction_sha256") != sha(npz)):
        raise ValueError(f"invalid SW0148 {arm} prediction contract")
    with np.load(npz, allow_pickle=False) as archive:
        if not {"image_ids", "qcc_labels"}.issubset(archive.files):
            raise ValueError("SW0148 prediction missing fixed arrays")
        ids = contract.require_fixed_image_ids(archive["image_ids"])
        labels = np.asarray(archive["qcc_labels"])
    if (labels.shape != (COUNT, GRID, GRID) or not np.issubdtype(labels.dtype, np.integer)
            or np.issubdtype(labels.dtype, np.bool_) or np.any(labels < 0) or np.any(labels > 1024)):
        raise ValueError("invalid SW0148 QCC label array")
    if sha(npz) != protocol["prediction_sha256"] or sha(protocol_path) != protocol_sha:
        raise ValueError("SW0148 prediction bytes changed during validation")
    return {"ids": ids, "labels": labels.astype(np.int64, copy=False),
            "protocol": protocol, "npz": npz, "npz_sha256": sha(npz),
            "protocol_path": protocol_path, "protocol_sha256": protocol_sha}


def _bootstrap_delta(left, right, draws):
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    lo, hi = np.quantile(delta[draws].mean(axis=1), [.025, .975])
    return {"mean_difference": float(delta.mean()), "ci95": [float(lo), float(hi)]}


def score_all(*, output=ARCHIVE / "evaluation_seed1.json", dataset=DATASET):
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserve existing SW0148 score")
    # Validate and freeze all candidate and reference bytes before GT access.
    own = {arm: validate_prediction(arm) for arm in ARMS}
    source = sw137.validate_prediction(sw137.OUTPUT_ROOT / "seed1", seed=SEED)
    slot = sw146._slot_reference()
    for row in own.values():
        if not np.array_equal(row["ids"], source["ids"]) or not np.array_equal(row["ids"], slot["ids"]):
            raise ValueError("candidate/source/Slot32 fixed image IDs differ")
    if sha(SLOT_BASELINE) != SLOT_BASELINE_SHA:
        raise ValueError("registered Slot32 baseline changed")
    baseline = json.loads(SLOT_BASELINE.read_text(encoding="utf-8"))
    for row in own.values():
        if sha(row["npz"]) != row["npz_sha256"] or sha(row["protocol_path"]) != row["protocol_sha256"]:
            raise ValueError("candidate bytes changed before GT")
    if sha(source["archive_path"]) != source["prediction_sha256"] or sha(source["protocol_path"]) != source["protocol_sha256"]:
        raise ValueError("source97 reference changed before GT")
    if sha(slot["npz"]) != slot["npz_sha256"] or sha(slot["protocol_path"]) != slot["protocol_sha256"]:
        raise ValueError("Slot reference changed before GT")
    if sha(SLOT_BASELINE) != SLOT_BASELINE_SHA:
        raise ValueError("registered Slot32 baseline changed immediately before GT")
    with sw146.h5py.File(dataset, "r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0] <= 1639:
            raise ValueError("registered fixed320 target masks unavailable")
        pixels = np.asarray(h5["mask"][1320:1640])
    gt = contract.modal_native32(pixels)
    scores = {arm: contract.production_metrics32(own[arm]["labels"], gt) for arm in ARMS}
    scores["source97_seed1"] = contract.production_metrics32(source["labels"], gt)
    scores["Slot32_seed1"] = contract.production_metrics32(slot["labels"], gt)
    slot_means = {metric: float(baseline["scores"]["slot32"][metric]["three_seed_mean"])
                  for metric in contract.METRICS}
    draws = np.random.RandomState(139).randint(0, COUNT, size=(10000, COUNT))
    paired = {}
    for reference, key in (("source97", "source97_seed1"), ("encoder_graph", "encoder_graph")):
        paired[f"joint_minus_{reference}"] = {
            metric: _bootstrap_delta(scores["joint"][metric]["per_image"],
                                     scores[key][metric]["per_image"], draws)
            for metric in contract.METRICS}
    means = {name: {metric: float(scores[name][metric]["mean"]) for metric in contract.METRICS}
             for name in (*ARMS, "source97_seed1", "Slot32_seed1")}
    gate = {
        "joint_fg_at_least_source_plus_0_01": means["joint"]["fg_ari"] >= means["source97_seed1"]["fg_ari"] + .01,
        "joint_fg_at_least_control_plus_0_01": means["joint"]["fg_ari"] >= means["encoder_graph"]["fg_ari"] + .01,
        "joint_control_fg_ci_lower_positive": paired["joint_minus_encoder_graph"]["fg_ari"]["ci95"][0] > 0,
        "joint_source_fg_ci_lower_positive": paired["joint_minus_source97"]["fg_ari"]["ci95"][0] > 0,
        "joint_foreground_iou_above_slot_three_seed_plus_0_05": means["joint"]["foreground_iou"] > slot_means["foreground_iou"] + .05,
        "joint_matched_object_iou_above_slot_three_seed_plus_0_05": means["joint"]["matched_object_iou"] > slot_means["matched_object_iou"] + .05,
    }
    result = {"experiment": NAME, "status": "complete", "seed": SEED,
              "scores": scores, "means": means, "registered_slot32_three_seed_means": slot_means,
              "paired_bootstrap": paired,
              "bootstrap": {"samples": 10000, "rng": "numpy RandomState", "seed": 139,
                            "common_image_indices": True},
              "promotion_gate": gate, "may_promote_seed_expansion": all(gate.values()),
              "provenance": {"candidate_predictions": {arm: own[arm]["npz_sha256"] for arm in ARMS},
                             "source97_prediction": source["prediction_sha256"],
                             "slot_prediction": slot["npz_sha256"],
                             "slot_baseline_sha256": SLOT_BASELINE_SHA},
              "ground_truth_opened_after_all_predictions_validated": True,
              "ground_truth_used_for_prediction": False,
              "ground_truth_used_for_scoring": True,
              "limits": "Paired seed1 recovery pilot only; does not establish scaling or universal gate claims."}
    sw146.write_once(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "predict", "score"), required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.stage != "score" and args.arm is None:
        parser.error("--arm is required for train/predict")
    if args.stage == "train":
        record = sw148_train.train_arm(args.arm, device=args.device)
    elif args.stage == "predict":
        record = predict_arm(args.arm, device=args.device)
    else:
        record = score_all(output=args.output or ARCHIVE / "evaluation_seed1.json")
    print(json.dumps({"experiment": NAME, "stage": args.stage,
                      "status": record.get("status"), "arm": record.get("arm")},
                     allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
