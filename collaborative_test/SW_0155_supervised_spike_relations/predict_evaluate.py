"""Fixed320 prediction and GT-last scoring for the SW0155 supervised pilot."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

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
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143
from collaborative_test.SW_0150_spike_spatial_mixture.integration import forward_assignment
from collaborative_test.SW_0152_source_preserving_partition_equivariance import training as source_training
from collaborative_test.SW_0155_supervised_spike_relations import artifactvalidator, train

EXPERIMENT = train.EXPERIMENT
ARMS = train.ARMS
SEED = train.SEED
IMAGE_IDS = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
COUNT, GRID = contract.COUNT, contract.GRID
PREDICTION_ROOT = HERE / "predictions"
EVALUATION_OUTPUT = HERE / "results_archive" / "supervised_fixed320_evaluation.json"
DATASET = train.HDF5_PATH
BASELINE_T = sw143._arm_dir("T", SEED)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    paths = [HERE / name for name in ("predict_evaluate.py", "protocol.json", "train.py", "objective.py",
        "artifactvalidator.py", "preflight.py", "preflight_migration.py")]
    paths.extend(ROOT / rel for rel in (
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
        "collaborative_test/SW_0137_native32_source_qcc/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/run.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0150_spike_spatial_mixture/integration.py",
        "collaborative_test/SW_0152_source_preserving_partition_equivariance/deep_encoder.py",
        "collaborative_test/SW_0152_source_preserving_partition_equivariance/training.py",
        "snn_kuramoto_bidirectional/evaluation.py"))
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0155 evaluation dependency missing")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _source_lineage():
    source = train.source_contract()
    assets = sw130.validate_rgb_assets()
    ref = sw137.validate_prediction(sw137.OUTPUT_ROOT / "seed1", seed=SEED)
    if not np.array_equal(ref["ids"], IMAGE_IDS):
        raise ValueError("source QCC reference IDs differ from fixed320 evaluation slice")
    return source, assets, ref


def _load_candidate(arm, device):
    task = {"arm": arm, "output": str(train.TRAIN_ROOT / f"{arm}_seed1")}
    source = train.source_contract()
    assets = sw130.validate_rgb_assets()
    if not artifactvalidator.validate_training_artifact(task, source=source, assets=assets):
        raise ValueError(f"SW0155 {arm} training artifact is not a validated completion")
    manifest_path = Path(task["output"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    foundation = train._load_foundation(device)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(train.RESIDUAL_INIT_SEED)
        source_training.install_deep_encoder(foundation)
    payload = torch.load(Path(task["output"]) / "checkpoint.pt", map_location=device,
                         weights_only=False)
    if not isinstance(payload, dict) or set(payload) != {"wrapped", "encoder"}:
        raise ValueError("SW0155 checkpoint schema mismatch")
    foundation.wrapped.load_state_dict(payload["wrapped"], strict=True)
    foundation.encoder.load_state_dict(payload["encoder"], strict=True)
    foundation.wrapped.eval().requires_grad_(False)
    foundation.encoder.eval().requires_grad_(False)
    return foundation, source, assets, manifest, sha256_file(manifest_path)


def _atomic_prediction(directory, ids, labels, protocol):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"preserve existing SW0155 prediction: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.with_name(directory.name + f".{os.getpid()}.staging")
    staging.mkdir(exist_ok=False)
    try:
        npz = staging / "predictions.npz"
        with npz.open("xb") as stream:
            np.savez_compressed(stream, image_ids=ids, qcc_labels=labels)
            stream.flush(); os.fsync(stream.fileno())
        protocol["prediction_sha256"] = sha256_file(npz)
        protocol["implementation_fingerprint"] = implementation_fingerprint()
        pp = staging / "protocol.json"
        with pp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(protocol, stream, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        protocol["protocol_sha256"] = sha256_file(pp)
        # The protocol's self-hash is stored in queue state, not inside itself.
        if directory.exists():
            raise FileExistsError(f"preserve concurrently-created prediction: {directory}")
        os.rename(staging, directory)
        return {"prediction_sha256": protocol["prediction_sha256"],
                "protocol_sha256": sha256_file(directory / "protocol.json")}
    except BaseException:
        # Leave the unique staging directory as evidence; never erase partial output.
        raise


@torch.no_grad()
def predict(arm, *, device="cuda:0", output_root=PREDICTION_ROOT):
    if arm not in ARMS:
        raise ValueError("SW0155 predicts only the two registered supervised arms")
    output = Path(output_root) / f"{arm}_seed1"
    if output.exists():
        raise FileExistsError(f"preserve existing SW0155 prediction: {output}")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("fixed320 candidate prediction requires assigned CUDA")
    torch.cuda.set_device(dev)
    foundation, source, assets, manifest, manifest_sha = _load_candidate(arm, dev)
    source_qcc = sw137.validate_prediction(sw137.OUTPUT_ROOT / "seed1", seed=SEED)
    if not np.array_equal(source_qcc["ids"], IMAGE_IDS):
        raise ValueError("immutable SW0137 source QCC reference IDs changed")
    hdf = Path(train.HDF5_PATH)
    hdf_stat = hdf.stat()
    bindings = manifest["bindings"]
    if (bindings.get("hdf5_path") != str(hdf.resolve())
            or bindings.get("hdf5_size") != hdf_stat.st_size
            or bindings.get("hdf5_mtime_ns") != hdf_stat.st_mtime_ns):
        raise ValueError("prediction HDF5 identity differs from the completed TRAIN artifact")
    model_state = lambda: source_training.state_sha({
        **{"wrapped." + name: value for name, value in foundation.wrapped.state_dict().items()},
        **{"encoder." + name: value for name, value in foundation.encoder.state_dict().items()}})
    before_model = model_state()
    frozen_base = source_training.state_sha(foundation.encoder.base_encoder)
    if frozen_base != manifest.get("frozen_base_encoder_sha256_before_after"):
        raise ValueError("restored pretrained base encoder differs from frozen TRAIN lineage")
    cache = np.load(sw130.VAL_RGB, mmap_mode="r", allow_pickle=False)
    if cache.shape != (COUNT, 128, 128, 3) or cache.dtype != np.uint8:
        raise ValueError("fixed320 validation RGB cache contract changed")
    labels = np.empty((COUNT, GRID, GRID), dtype=np.int64)
    for index in range(COUNT):
        rgb = torch.from_numpy(np.asarray(cache[index:index + 1]).copy()).to(dev)
        result = forward_assignment(foundation, rgb, live_tail_steps=0)
        qcc = result["qcc"].detach().cpu().numpy()
        if qcc.shape != (1, GRID, GRID) or not np.isfinite(qcc).all():
            raise ValueError(f"invalid QCC readout at fixed image {index}")
        labels[index] = qcc[0].astype(np.int64, copy=False)
        if index % 32 == 31:
            print(json.dumps({"stage": "predict", "arm": arm,
                              "predicted": index + 1, "count": COUNT}), flush=True)
    protocol = {"experiment": EXPERIMENT, "status": "complete", "arm": arm,
        "seed": SEED, "image_ids": [1320, 1639], "count": COUNT,
        "grid_size": [GRID, GRID], "prediction_shape": [COUNT, GRID, GRID],
        "readout": "unchanged production native32 actual-spike QCC from T1024 trace",
        "steps": train.STEPS, "settle": train.SETTLE, "live_tail": 0,
        "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
        "source_core_sha256": source["source_sha256"],
        "source_manifest_sha256": sha256_file(source["manifest_path"]),
        "source_training_ids_sha256": train.ids_sha(source["image_ids"]),
        "source_qcc_prediction_sha256": source_qcc["prediction_sha256"],
        "source_qcc_protocol_sha256": source_qcc["protocol_sha256"],
        "rgb_asset_validation": assets, "training_manifest_sha256": manifest_sha,
        "training_checkpoint_sha256": manifest["checkpoint_sha256"],
        "training_implementation_fingerprint": manifest["implementation_fingerprint"],
        "training_hdf5_path": bindings["hdf5_path"], "training_hdf5_size": bindings["hdf5_size"],
        "training_hdf5_mtime_ns": bindings["hdf5_mtime_ns"],
        "frozen_base_encoder_sha256": frozen_base,
        "prediction_model_state_sha256_before": before_model,
        "prediction_model_state_sha256_after": model_state()}
    if protocol["prediction_model_state_sha256_before"] != protocol["prediction_model_state_sha256_after"]:
        raise AssertionError("inference changed the restored SW0155 model state")
    return _atomic_prediction(output, IMAGE_IDS, labels, protocol)


def validate_prediction(arm, *, prediction_root=PREDICTION_ROOT):
    if arm not in ARMS:
        raise ValueError("unregistered SW0155 prediction arm")
    directory = Path(prediction_root) / f"{arm}_seed1"
    npz, pp = directory / "predictions.npz", directory / "protocol.json"
    protocol_bytes = pp.read_bytes()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    source, assets, source_qcc = _source_lineage()
    task = {"arm": arm, "output": str(train.TRAIN_ROOT / f"{arm}_seed1")}
    if not artifactvalidator.validate_training_artifact(task, source=source, assets=assets):
        raise ValueError(f"SW0155 {arm} training artifact no longer validates")
    manifest_path = Path(task["output"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (protocol.get("experiment") != EXPERIMENT or protocol.get("status") != "complete"
            or protocol.get("arm") != arm or protocol.get("seed") != SEED
            or protocol.get("image_ids") != [1320, 1639] or protocol.get("count") != COUNT
            or protocol.get("grid_size") != [GRID, GRID]
            or protocol.get("prediction_shape") != [COUNT, GRID, GRID]
            or protocol.get("steps") != train.STEPS or protocol.get("settle") != train.SETTLE
            or protocol.get("live_tail") != 0 or protocol.get("optimizer_updates") != 0
            or protocol.get("ground_truth_used_for_prediction") is not False
            or protocol.get("source_core_sha256") != source["source_sha256"]
            or protocol.get("source_manifest_sha256") != sha256_file(source["manifest_path"])
            or protocol.get("source_training_ids_sha256") != train.ids_sha(source["image_ids"])
            or protocol.get("source_qcc_prediction_sha256") != source_qcc["prediction_sha256"]
            or protocol.get("source_qcc_protocol_sha256") != source_qcc["protocol_sha256"]
            or protocol.get("rgb_asset_validation") != assets
            or protocol.get("training_manifest_sha256") != sha256_file(manifest_path)
            or protocol.get("training_checkpoint_sha256") != manifest.get("checkpoint_sha256")
            or protocol.get("training_implementation_fingerprint") != manifest.get("implementation_fingerprint")
            or protocol.get("training_hdf5_path") != manifest.get("bindings", {}).get("hdf5_path")
            or protocol.get("training_hdf5_size") != manifest.get("bindings", {}).get("hdf5_size")
            or protocol.get("training_hdf5_mtime_ns") != manifest.get("bindings", {}).get("hdf5_mtime_ns")
            or protocol.get("frozen_base_encoder_sha256") != manifest.get("frozen_base_encoder_sha256_before_after")
            or protocol.get("prediction_model_state_sha256_before") != protocol.get("prediction_model_state_sha256_after")
            or protocol.get("implementation_fingerprint") != implementation_fingerprint()
            or protocol.get("prediction_sha256") != sha256_file(npz)):
        raise ValueError(f"SW0155 {arm} prediction provenance does not match current lineage")
    with np.load(npz, allow_pickle=False) as saved:
        ids = contract.require_fixed_image_ids(saved["image_ids"])
        labels = np.asarray(saved["qcc_labels"])
    if labels.shape != (COUNT, GRID, GRID) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("SW0155 QCC labels violate native32 schema")
    if sha256_file(npz) != protocol["prediction_sha256"] or sha256_file(pp) != protocol_sha:
        raise ValueError("SW0155 protocol changed during validation")
    return {"ids": ids, "labels": labels.astype(np.int64, copy=False),
            "protocol": protocol, "npz_path": npz, "protocol_path": pp,
            "prediction_sha256": sha256_file(npz), "protocol_sha256": sha256_file(pp)}


def _validate_source_prediction():
    row = sw137.validate_prediction(sw137.OUTPUT_ROOT / "seed1", seed=SEED)
    if not np.array_equal(row["ids"], IMAGE_IDS):
        raise ValueError("fixed source prediction IDs mismatch")
    return row


def _validate_t_prediction():
    row = sw143.validate_arm_prediction(BASELINE_T, seed=SEED)
    if row["arm"] != "T" or not np.array_equal(row["ids"], IMAGE_IDS):
        raise ValueError("SW0143T seed1 reference mismatch")
    return row


def _paired_bootstrap(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != (COUNT,) or right.shape != (COUNT,):
        raise ValueError("paired bootstrap requires fixed320 per-image arrays")
    rng = np.random.RandomState(139)
    draws = rng.randint(0, COUNT, size=(10000, COUNT))
    delta = left - right
    values = delta[draws].mean(axis=1)
    return {"mean": float(delta.mean()),
            "ci95": [float(v) for v in np.percentile(values, [2.5, 97.5])]}


def score_all(*, output=EVALUATION_OUTPUT, dataset=DATASET,
              prediction_root=PREDICTION_ROOT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0155 evaluation: {output}")
    # All predictions and reference provenance are frozen before the first GT read.
    predictions = {arm: validate_prediction(arm, prediction_root=prediction_root) for arm in ARMS}
    source = _validate_source_prediction()
    incumbent = _validate_t_prediction()
    if (not np.array_equal(source["ids"], IMAGE_IDS)
            or not np.array_equal(incumbent["ids"], IMAGE_IDS)):
        raise ValueError("fixed320 source/incumbent IDs do not align")
    frozen = {}
    for arm, row in predictions.items():
        frozen[f"{arm}:npz"] = sha256_file(row["npz_path"])
        frozen[f"{arm}:protocol"] = sha256_file(row["protocol_path"])
        if frozen[f"{arm}:npz"] != row["prediction_sha256"] or frozen[f"{arm}:protocol"] != row["protocol_sha256"]:
            raise ValueError("SW0155 prediction changed before GT access")
    frozen.update({"source:npz": sha256_file(source["archive_path"]),
                   "source:protocol": sha256_file(source["protocol_path"]),
                   "incumbent_T:npz": sha256_file(incumbent["npz_path"]),
                   "incumbent_T:protocol": sha256_file(incumbent["protocol_path"])})
    if frozen["source:npz"] != source["prediction_sha256"] or frozen["source:protocol"] != source["protocol_sha256"]:
        raise ValueError("source97 prediction changed before GT access")
    if frozen["incumbent_T:npz"] != incumbent["prediction_sha256"] or frozen["incumbent_T:protocol"] != incumbent["protocol_sha256"]:
        raise ValueError("SW0143T prediction changed before GT access")
    dataset = Path(dataset)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    dataset_stat = dataset.stat()
    for row in predictions.values():
        protocol = row["protocol"]
        if (protocol.get("training_hdf5_path") != str(dataset.resolve())
                or protocol.get("training_hdf5_size") != dataset_stat.st_size
                or protocol.get("training_hdf5_mtime_ns") != dataset_stat.st_mtime_ns):
            raise ValueError("scoring dataset differs from the completed TRAIN mask source")
    for arm, row in predictions.items():
        if sha256_file(row["npz_path"]) != frozen[f"{arm}:npz"] or sha256_file(row["protocol_path"]) != frozen[f"{arm}:protocol"]:
            raise ValueError("candidate prediction changed immediately before GT access")
    if (sha256_file(source["archive_path"]) != frozen["source:npz"]
            or sha256_file(source["protocol_path"]) != frozen["source:protocol"]
            or sha256_file(incumbent["npz_path"]) != frozen["incumbent_T:npz"]
            or sha256_file(incumbent["protocol_path"]) != frozen["incumbent_T:protocol"]):
        raise ValueError("reference prediction changed immediately before GT access")
    with h5py.File(dataset, "r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0] <= 1639:
            raise ValueError("canonical HDF5 does not cover fixed320 evaluation masks")
        target = contract.modal_native32(np.asarray(h5["mask"][1320:1640]))
    score = {arm: contract.production_metrics32(row["labels"], target)
             for arm, row in predictions.items()}
    score["source97"] = contract.production_metrics32(source["labels"], target)
    score["SW0143_T"] = contract.production_metrics32(incumbent["labels"], target)
    means = {name: {metric: float(row[metric]["mean"]) for metric in contract.METRICS}
             for name, row in score.items()}
    paired = {}
    for arm in ARMS:
        for ref_name in ("source97", "SW0143_T"):
            for metric in contract.METRICS:
                paired[f"{arm}_minus_{ref_name}_{metric}"] = _paired_bootstrap(
                    score[arm][metric]["per_image"], score[ref_name][metric]["per_image"])
    for metric in contract.METRICS:
        paired[f"joint_minus_encoder_graph_{metric}"] = _paired_bootstrap(
            score["joint"][metric]["per_image"], score["encoder_graph"][metric]["per_image"])
    report = {"experiment": EXPERIMENT, "status": "complete", "seed_scope": [SEED],
        "implementation_fingerprint": implementation_fingerprint(),
        "leaderboard": "supervised SW0155 seed1 pilot; separate from unsupervised leaderboard",
        "scores": score, "means": means, "paired_bootstrap": paired,
        "bootstrap": {"rng": "numpy RandomState", "seed": 139, "replicates": 10000,
                       "common_image_indices": True},
        "prediction_contract": {"image_ids": [1320, 1639, COUNT], "grid": [GRID, GRID],
                                "readout": "unchanged production native32 actual-spike QCC"},
        "frozen_base_encoder": "The pretrained RGB base encoder remained frozen during SW0155; only the exact-zero residual branch was trainable.",
        "ground_truth_opened_after_all_predictions_validated": True,
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
        "scientific_limits": "One supervised seed1 pilot; not a supervised Slot comparison, full-dataset retraining result, scale claim, or universal superiority claim.",
        "artifact_sha256_before_gt": frozen,
        "scoring_dataset": {"path": str(dataset.resolve()), "size": dataset_stat.st_size,
                            "mtime_ns": dataset_stat.st_mtime_ns},
        "prediction_provenance": {arm: {"prediction_sha256": row["prediction_sha256"],
            "protocol_sha256": row["protocol_sha256"],
            "training_manifest_sha256": row["protocol"]["training_manifest_sha256"],
            "training_checkpoint_sha256": row["protocol"]["training_checkpoint_sha256"]}
            for arm, row in predictions.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, output)
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("predict", "score"), required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.stage == "predict":
        if args.arm is None:
            parser.error("--arm required for predict")
        result = predict(args.arm, device=args.device)
    else:
        result = score_all()
    print(json.dumps({"stage": args.stage, "status": result.get("status", "prediction_complete"),
        "arm": args.arm, "ground_truth_used_for_prediction": False,
        "ground_truth_used_for_scoring": args.stage == "score"}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
