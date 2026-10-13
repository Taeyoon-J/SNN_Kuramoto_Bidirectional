"""SW0149 GT-free prediction and frozen320 GT-last paired scoring."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

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
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136_eval
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0149_spike_graph_modularity import train

NAME = train.NAME
ARMS = train.ARMS
PREDICTIONS = HERE / "predictions"
ARCHIVE = HERE / "results_archive"
SOURCE137 = sw137.OUTPUT_ROOT / "seed1"
SLOT_ROOT = ROOT / "trained_models/SW0092_slot_our70000_eval"
SLOT_BASELINE = ROOT / "collaborative_test/SW_0135_native32_spike_binding/results_archive/slot70k_native32_baseline.json"
SLOT_BASELINE_SHA256 = "4dc12cb54c3114375b8428cd250c013ff94015c4826915751f2fcb0a2204d753"
DATASET = Path(sw130.source97.DATASET)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    result = dict(train.implementation_fingerprint())
    paths = (
        HERE / "evaluate.py", HERE / "protocol.json",
        ROOT / "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
        ROOT / "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        ROOT / "collaborative_test/SW_0137_native32_source_qcc/run.py",
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        result[path.relative_to(ROOT).as_posix()] = sha(path)
    return result


def implementation_sha():
    data = json.dumps(implementation_fingerprint(), sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _prediction_dir(arm, root=PREDICTIONS):
    if arm not in ARMS:
        raise ValueError("unregistered SW0149 arm")
    return Path(root) / f"{arm}_seed1"


def _write_json_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"preserving existing result: {path}")
    else:
        temporary.unlink()


def _write_npz_once(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def predict_arm(arm, *, device="cuda:0", output_root=PREDICTIONS):
    trained = train.validate_training(arm)
    out = _prediction_dir(arm, output_root)
    if out.exists():
        raise FileExistsError(f"preserving existing SW0149 prediction: {out}")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0149 registered prediction requires CUDA")
    torch.cuda.set_device(dev)
    foundation = sw135.load_native32_foundation(train.SEED, dev, verify_rgb_assets=True)
    state = torch.load(trained["checkpoint"], map_location=dev, weights_only=True)
    if not isinstance(state, dict) or set(state) != {"wrapped", "encoder"}:
        raise ValueError("SW0149 checkpoint has an unexpected state schema")
    foundation.wrapped.load_state_dict(state["wrapped"], strict=True)
    foundation.encoder.load_state_dict(state["encoder"], strict=True)
    foundation.wrapped.eval()
    foundation.encoder.eval()
    cache = np.load(sw130.VAL_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape[0] < contract.COUNT or cache.shape[1:] != (128, 128, 3):
        raise ValueError("registered native128 RGB validation cache changed schema")
    labels = np.empty((contract.COUNT, contract.GRID, contract.GRID), dtype=np.int64)
    with torch.no_grad():
        for index in range(contract.COUNT):
            rgb_np = np.asarray(cache[index:index + 1]).copy()
            rgb = torch.from_numpy(rgb_np).to(dev)
            gamma = foundation.encode(rgb)
            _unused_primary, qcc = sw136_eval._predict_one(foundation, None, gamma, primary=False)
            qcc = np.asarray(qcc)
            if (qcc.shape != (1, contract.GRID, contract.GRID)
                    or not np.issubdtype(qcc.dtype, np.integer)
                    or not np.isfinite(qcc).all() or np.any(qcc < 0)
                    or np.any(qcc > 1024)):
                raise ValueError("SW0149 actual-spike QCC returned invalid direct32 labels")
            labels[index] = qcc[0]
            del rgb_np, rgb, gamma, qcc
            if (index + 1) % 40 == 0:
                print(json.dumps({"stage": "predict", "arm": arm,
                                  "images": index + 1, "total": contract.COUNT}), flush=True)
    staging = out.parent / f".{out.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        archive = staging / "predictions.npz"
        _write_npz_once(archive, image_ids=np.asarray(contract.IMAGE_IDS, dtype=np.int64),
                        qcc_labels=labels)
        protocol = {
            "experiment": NAME, "status": "complete", "stage": "prediction_complete",
            "arm": arm, "seed": train.SEED,
            "image_ids": [contract.IMAGE_IDS[0], contract.IMAGE_IDS[-1]],
            "count": contract.COUNT, "grid_size": [contract.GRID, contract.GRID],
            "prediction_shape": [contract.COUNT, contract.GRID, contract.GRID],
            "steps": train.STEPS, "settle": train.SETTLE,
            "live_tail": 0, "readout": "unchanged production actual-spike QCC .50/min8/largest component",
            "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
            "training_manifest_sha256": trained["manifest_sha256"],
            "checkpoint_sha256": trained["checkpoint_sha256"],
            "source_core_sha256": foundation.provenance["source_core_sha256"],
            "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
            "source_training_ids_sha256": foundation.provenance["source_training_ids_sha256"],
            "rgb_asset_validation": foundation.provenance["rgb_asset_validation"],
            "prediction_sha256": sha(archive),
            "implementation_fingerprint": implementation_fingerprint(),
            "implementation_sha256": implementation_sha(),
        }
        _write_json_once(staging / "protocol.json", protocol)
        os.rename(staging, out)
        return protocol
    except BaseException:
        # Keep the uniquely named incomplete staging directory for inspection.
        raise


def validate_prediction(arm, *, root=PREDICTIONS):
    directory = _prediction_dir(arm, root)
    archive_path, protocol_path = directory / "predictions.npz", directory / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    trained = train.validate_training(arm)
    source_contract = sw130.source_contract(train.SEED)
    assets = sw130.validate_rgb_assets()
    expected_ids_sha = hashlib.sha256(
        np.asarray(source_contract[4], dtype="<i8").tobytes()).hexdigest()
    if (protocol.get("experiment") != NAME or protocol.get("status") != "complete"
            or protocol.get("stage") != "prediction_complete" or protocol.get("arm") != arm
            or protocol.get("seed") != train.SEED
            or protocol.get("image_ids") != [1320, 1639] or protocol.get("count") != contract.COUNT
            or protocol.get("grid_size") != [contract.GRID, contract.GRID]
            or protocol.get("prediction_shape") != [contract.COUNT, contract.GRID, contract.GRID]
            or protocol.get("steps") != train.STEPS or protocol.get("settle") != train.SETTLE
            or protocol.get("live_tail") != 0
            or protocol.get("readout") != "unchanged production actual-spike QCC .50/min8/largest component"
            or protocol.get("ground_truth_used_for_prediction") is not False
            or protocol.get("optimizer_updates") != 0
            or protocol.get("training_manifest_sha256") != trained["manifest_sha256"]
            or protocol.get("checkpoint_sha256") != trained["checkpoint_sha256"]
            or protocol.get("source_core_sha256") != sw130.source97.EXPECTED_SOURCE_SHAS[train.SEED]
            or protocol.get("source_manifest_sha256") != sw135.sha256_file(source_contract[1])
            or protocol.get("source_training_ids_sha256") != expected_ids_sha
            or protocol.get("rgb_asset_validation") != assets
            or protocol.get("implementation_fingerprint") != implementation_fingerprint()
            or protocol.get("implementation_sha256") != implementation_sha()
            or protocol.get("prediction_sha256") != sha(archive_path)):
        raise ValueError(f"invalid frozen SW0149 {arm} prediction contract")
    with np.load(archive_path, allow_pickle=False) as saved:
        if not {"image_ids", "qcc_labels"}.issubset(saved.files):
            raise ValueError("SW0149 prediction archive missing arrays")
        ids = contract.require_fixed_image_ids(saved["image_ids"])
        labels = np.asarray(saved["qcc_labels"])
    if (labels.shape != (contract.COUNT, contract.GRID, contract.GRID)
            or not np.issubdtype(labels.dtype, np.integer)
            or np.issubdtype(labels.dtype, np.bool_) or np.any(labels < 0)
            or np.any(labels > 1024)):
        raise ValueError("SW0149 prediction labels violate the fixed native32 contract")
    if sha(archive_path) != protocol["prediction_sha256"] or sha(protocol_path) != protocol_sha:
        raise ValueError("SW0149 prediction bytes changed while validating")
    return {"ids": ids, "labels": labels.astype(np.int64, copy=False),
            "protocol": protocol, "archive_path": archive_path,
            "protocol_path": protocol_path, "prediction_sha256": protocol["prediction_sha256"],
            "protocol_sha256": protocol_sha}


def _slot_reference():
    slot_dir = SLOT_ROOT / "seed1_epoch10"
    archive_path = slot_dir / "predictions.npz"
    protocol_path = slot_dir / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes.decode("utf-8-sig"))
    archive_sha = contract.validate_slot_protocol(protocol, seed=1,
                                                  prediction_path=archive_path)
    with np.load(archive_path, allow_pickle=False) as saved:
        if not {"image_ids", "labels"}.issubset(saved.files):
            raise ValueError("registered Slot archive missing pixel labels/IDs")
        ids = contract.require_fixed_image_ids(saved["image_ids"])
        pixel_labels = np.asarray(saved["labels"])
    labels = contract.slot_native_pixels_to32(pixel_labels)
    if sha(archive_path) != archive_sha or sha(protocol_path) != protocol_sha:
        raise ValueError("Slot prediction bytes changed during validation")
    return {"ids": ids, "labels": labels, "archive_path": archive_path,
            "protocol_path": protocol_path, "prediction_sha256": archive_sha,
            "protocol_sha256": protocol_sha, "protocol": protocol}


def _bootstrap_delta(left, right, draws):
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    low, high = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
    return {"mean_difference": float(delta.mean()),
            "ci95": [float(low), float(high)]}


def score_all(*, output=ARCHIVE / "evaluation_seed1.json", dataset=DATASET):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0149 score output: {output}")
    # Freeze and validate every prediction/reference before the single GT read.
    own = {arm: validate_prediction(arm) for arm in ARMS}
    source = sw137.validate_prediction(SOURCE137, seed=1)
    slot = _slot_reference()
    for arm in ARMS:
        if not (np.array_equal(own[arm]["ids"], source["ids"])
                and np.array_equal(own[arm]["ids"], slot["ids"])):
            raise ValueError("SW0149/source/Slot fixed320 image IDs differ")
    if sha(SLOT_BASELINE) != SLOT_BASELINE_SHA256:
        raise ValueError("registered Slot32 three-seed baseline bytes changed")
    baseline = json.loads(SLOT_BASELINE.read_text(encoding="utf-8"))
    slot_means = {metric: float(baseline["scores"]["slot32"][metric]["three_seed_mean"])
                  for metric in contract.METRICS}
    for row in own.values():
        if (sha(row["archive_path"]) != row["prediction_sha256"]
                or sha(row["protocol_path"]) != row["protocol_sha256"]):
            raise ValueError("SW0149 prediction changed before GT access")
    if (sha(source["archive_path"]) != source["prediction_sha256"]
            or sha(source["protocol_path"]) != source["protocol_sha256"]):
        raise ValueError("source97 reference changed before GT access")
    if (sha(slot["archive_path"]) != slot["prediction_sha256"]
            or sha(slot["protocol_path"]) != slot["protocol_sha256"]
            or sha(SLOT_BASELINE) != SLOT_BASELINE_SHA256):
        raise ValueError("Slot reference/baseline changed before GT access")
    with h5py.File(dataset, "r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0] <= contract.IMAGE_IDS[-1]:
            raise ValueError("registered GT masks do not contain fixed320 image IDs")
        pixels = np.asarray(h5["mask"][contract.IMAGE_IDS[0]:contract.IMAGE_IDS[-1] + 1])
    target = contract.modal_native32(pixels)
    scores = {arm: contract.production_metrics32(own[arm]["labels"], target)
              for arm in ARMS}
    scores["source97_seed1"] = contract.production_metrics32(source["labels"], target)
    scores["Slot32_seed1"] = contract.production_metrics32(slot["labels"], target)
    means = {name: {metric: float(row[metric]["mean"]) for metric in contract.METRICS}
             for name, row in scores.items()}
    draws = np.random.RandomState(139).randint(0, contract.COUNT,
                                              size=(10000, contract.COUNT))
    paired = {}
    for ref in ("encoder_graph", "source97_seed1"):
        paired[f"joint_minus_{ref}"] = {
            metric: _bootstrap_delta(scores["joint"][metric]["per_image"],
                                     scores[ref][metric]["per_image"], draws)
            for metric in contract.METRICS}
    gate = {
        "joint_fg_at_least_source_plus_0_01": means["joint"]["fg_ari"] >= means["source97_seed1"]["fg_ari"] + 0.01,
        "joint_fg_at_least_encoder_graph_plus_0_01": means["joint"]["fg_ari"] >= means["encoder_graph"]["fg_ari"] + 0.01,
        "joint_source_fg_ci_lower_positive": paired["joint_minus_source97_seed1"]["fg_ari"]["ci95"][0] > 0,
        "joint_encoder_graph_fg_ci_lower_positive": paired["joint_minus_encoder_graph"]["fg_ari"]["ci95"][0] > 0,
        "joint_foreground_iou_above_slot_three_seed_plus_0_05": means["joint"]["foreground_iou"] > slot_means["foreground_iou"] + 0.05,
        "joint_matched_object_iou_above_slot_three_seed_plus_0_05": means["joint"]["matched_object_iou"] > slot_means["matched_object_iou"] + 0.05,
    }
    result = {
        "status": "complete", "experiment": NAME, "seed": 1,
        "scores": scores, "means": means,
        "registered_slot32_three_seed_means": slot_means,
        "paired_bootstrap": paired,
        "bootstrap": {"replicates": 10000, "rng": "numpy RandomState",
                      "seed": 139, "common_image_indices": True},
        "promotion_gate": gate, "may_promote_seed_expansion": all(gate.values()),
        "provenance": {
            "predictions": {arm: {"npz_sha256": own[arm]["prediction_sha256"],
                                  "protocol_sha256": own[arm]["protocol_sha256"],
                                  "training_manifest_sha256": own[arm]["protocol"]["training_manifest_sha256"]}
                            for arm in ARMS},
            "source97_npz_sha256": source["prediction_sha256"],
            "slot_npz_sha256": slot["prediction_sha256"],
            "slot_baseline_sha256": SLOT_BASELINE_SHA256,
        },
        "ground_truth_opened_after_all_predictions_validated": True,
        "ground_truth_used_for_prediction": False,
        "ground_truth_used_for_scoring": True,
        "limits": "Paired seed1 pilot only; no semantic source-graph or scaling claim.",
    }
    _write_json_once(output, result)
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("predict", "score"), required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "predict":
        if args.arm is None:
            parser.error("--arm is required for prediction")
        result = predict_arm(args.arm, device=args.device)
    else:
        result = score_all(output=args.output or ARCHIVE / "evaluation_seed1.json")
    print(json.dumps({"status": result.get("status", "prediction_complete"),
                      "stage": args.stage, "arm": args.arm,
                      "ground_truth_used_for_prediction": False,
                      "ground_truth_used_for_scoring": args.stage == "score"},
                     allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
