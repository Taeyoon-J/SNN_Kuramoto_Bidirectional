"""GT-free fixed320 predictions for both SW0187 shared-prefix branches."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import torch

_visible_before_import = os.environ.get("CUDA_VISIBLE_DEVICES")
try:
    train = importlib.import_module("collaborative_test.SW_0187_joint_encoder_unique_scaling.train")
    native = importlib.import_module("collaborative_test.SW_0179_qce_seed_replication.predict_evaluate")
    sw130 = importlib.import_module("collaborative_test.SW_0130_phase_state_integration.run")
    contract = importlib.import_module("collaborative_test.SW_0135_native32_spike_binding.evaluation_contract32")
finally:
    if _visible_before_import is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else: os.environ["CUDA_VISIBLE_DEVICES"] = _visible_before_import

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (0, 1, 2)
ARMS = ("control512", "candidate1024")
ENDPOINTS = ("conditional", "standalone")
COUNT, GRID = int(contract.COUNT), int(contract.GRID)
IMAGE_IDS = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
EXPERIMENT = "SW0187_shared_prefix_unique_scaling_fixed320_prediction"
QUEUE_MODULE = "collaborative_test.SW_0187_joint_encoder_unique_scaling.prediction_queue"
PREDICTION_ROOT = HERE / "results_archive" / "predictions_20261017"
WORK_ROOT = HERE / "results_archive" / "prediction_work_20261017"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def implementation_fingerprint():
    files = (
        "collaborative_test/SW_0187_joint_encoder_unique_scaling/DESIGN_KO.md",
        "collaborative_test/SW_0187_joint_encoder_unique_scaling/train.py",
        "collaborative_test/SW_0187_joint_encoder_unique_scaling/training_queue.py",
        "collaborative_test/SW_0187_joint_encoder_unique_scaling/predict_evaluate.py",
        "collaborative_test/SW_0187_joint_encoder_unique_scaling/prediction_queue.py",
        "collaborative_test/SW_0187_joint_encoder_unique_scaling/test_predict_evaluate.py",
        "collaborative_test/SW_0187_joint_encoder_unique_scaling/test_prediction_queue.py",
        "collaborative_test/SW_0179_qce_seed_replication/predict_evaluate.py",
        "collaborative_test/SW_0165_foreground_partition_replication/predict_evaluate.py",
        "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
    )
    result = {name: sha256_file(ROOT / name) for name in files if (ROOT / name).is_file()}
    if len(result) != len(files):
        raise FileNotFoundError("SW0187 prediction dependency missing")
    return {"experiment": EXPERIMENT, "files": result, "seeds": list(SEEDS),
        "arms": list(ARMS), "image_ids": IMAGE_IDS.tolist(), "validation_data_read": False}


def _queue_state(seed):
    queue = train._module(train.QUEUE_MODULE)
    path = queue.state_path(seed)
    if not path.is_file():
        raise FileNotFoundError(f"SW0187 training queue state missing for seed {seed}")
    return queue, path, json.loads(path.read_text(encoding="utf-8"))


def _artifact(seed, arm):
    return train._paths(train.OUTPUT_ROOT, arm, seed)[1]


def _load_model(seed, arm, prepared, device):
    dev = torch.device(device)
    if dev.type == "cpu" and torch.cuda.is_initialized():
        raise RuntimeError("SW0187 CPU model loading must precede CUDA initialization")
    if dev.type == "cuda" and not torch.cuda.is_initialized():
        raise RuntimeError("SW0187 GPU model loading requires initialized worker CUDA")
    result = prepared["artifacts"][arm]
    manifest = result["manifest"]
    found, _optimizer, _families, factory_binding, setup = train.source._make_fresh(
        "product_q_ce", int(seed), prepared["training_inputs"], dev)
    base_encoder = setup[2]
    frozen_factory_components = setup[3]
    if (train.source._state_sha(found.wrapped, found.encoder) != manifest.get("bindings", {}).get("initial_model_state_sha256")
            or factory_binding != manifest.get("bindings", {}).get("factory_binding")):
        raise ValueError("SW0187 fresh seed-specific SourceT factory differs from branch lineage")
    checkpoint_path = result["path"] / "checkpoint.pt"
    if sha256_file(checkpoint_path) != manifest.get("checkpoint_sha256"):
        raise ValueError("SW0187 checkpoint bytes changed after artifact validation")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (payload.get("experiment") != train.EXPERIMENT or payload.get("arm") != arm
            or payload.get("seed") != int(seed) or payload.get("bindings") != manifest.get("bindings")
            or payload.get("model_state_sha256") != manifest.get("model_state_sha256")
            or train.source._state_sha_from_payload(payload) != manifest.get("model_state_sha256")):
        raise ValueError("SW0187 checkpoint payload identity/state digest is invalid")
    found.wrapped.load_state_dict(payload["wrapped"], strict=True)
    found.encoder.load_state_dict(payload["encoder"], strict=True)
    found.wrapped.eval(); found.encoder.eval(); found.encoder.base_encoder.eval(); found.patcher.eval()
    train._assert_trained_encoder_manifest(manifest,payload["encoder"])
    actual = train.source._state_sha(found.wrapped, found.encoder)
    if actual != manifest.get("model_state_sha256"):
        raise ValueError("SW0187 loaded model tensor digest mismatch")
    util = train.source._module(train.source.SOURCE_UTIL_MODULE)
    if util._frozen_source_components(found, base_encoder) != frozen_factory_components:
        raise ValueError("SW0187 trained model changed frozen source/preprocessing components")
    if sha256_file(result["path"] / "optimizer.pt") != manifest.get("optimizer_sha256"):
        raise ValueError("SW0187 optimizer bytes differ from completed manifest")
    return found, actual


def prepare_prediction_inputs_cpu(seed):
    seed = int(seed)
    if seed not in SEEDS or torch.cuda.is_initialized():
        raise RuntimeError("SW0187 prediction preparation requires a registered seed before CUDA init")
    queue, state_path, state = _queue_state(seed)
    if state.get("experiment") != "SW0187_training_queue" or state.get("seed") != seed:
        raise ValueError("SW0187 training state identity mismatch")
    if state.get("status") == "complete_rejected":
        raise RuntimeError(f"SW0187 seed{seed} has a terminal scientific rejection")
    if state.get("status") != "complete" or state.get("validation_data_read") is not False:
        raise RuntimeError(f"SW0187 seed{seed} training is not terminal complete")
    inputs = train.prepare_training_inputs_cpu()
    prefix_path = train._paths(train.OUTPUT_ROOT, "shared_prefix", seed)[1]
    prefix = train.validate_prefix_artifact(prefix_path, seed=seed,
        expected_fingerprint=train.fingerprint_sha256())
    artifacts = {}
    for arm in ARMS:
        path = _artifact(seed, arm)
        row = train.validate_training_artifact(arm, path, seed=seed)
        if state.get("arm_manifests", {}).get(arm) != row["manifest_sha256"]:
            raise ValueError(f"SW0187 queue state does not bind {arm} manifest")
        artifacts[arm] = {**row, "path": path}
    queue_identity = queue._cpu_admission(seed, train)
    queue_payload, queue_sha = queue._queue_fingerprint(train, seed, queue_identity)
    if (state.get("queue_fingerprint_sha256") != queue_sha
            or state.get("prefix_manifest_sha256") != sha256_file(prefix_path / "manifest.json")
            or state.get("prefix_state_sha256") != prefix["prefix_state_sha256"]
            or state.get("proposal_count") != 96):
        raise ValueError("SW0187 queue receipt does not bind complete prefix and branch artifacts")
    refs = native._references(seed)
    rgb_path = Path(sw130.VAL_RGB)
    meta_path = rgb_path.with_suffix(rgb_path.suffix + ".complete.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    ids_sha = hashlib.sha256(IMAGE_IDS.astype("<i8").tobytes()).hexdigest()
    if (meta.get("status") != "complete" or meta.get("cache_shape") != [COUNT, 128, 128, 3]
            or meta.get("ids_mapping_sha256") != ids_sha or rgb.dtype != np.uint8
            or tuple(rgb.shape) != (COUNT, 128, 128, 3)):
        raise ValueError("SW0187 fixed320 RGB cache schema/mapping changed")
    rgb_hashes = [hashlib.sha256(np.ascontiguousarray(rgb[i]).tobytes()).hexdigest() for i in range(COUNT)]
    dataset = Path(sw130.source97.DATASET).resolve(); stat = dataset.stat()
    prepared = {"seed": seed, "training_inputs": inputs, "artifacts": artifacts,
        "prefix_manifest_sha256": sha256_file(prefix_path / "manifest.json"),
        "prefix_state_sha256": prefix["prefix_state_sha256"], "queue": queue,
        "queue_state": state, "queue_state_path": state_path, "queue_state_sha256": sha256_file(state_path),
        "queue_fingerprint_sha256": queue_sha, "queue_payload": queue_payload,
        "references": refs, "rgb": rgb, "rgb_path": rgb_path, "rgb_meta_path": meta_path,
        "rgb_meta_sha256": sha256_file(meta_path), "rgb_hashes": rgb_hashes,
        "dataset": dataset, "dataset_size_bytes": stat.st_size, "dataset_mtime_ns": stat.st_mtime_ns}
    model_states = {}
    for arm in ARMS:
        model, model_sha = _load_model(seed, arm, prepared, "cpu")
        model_states[arm] = model_sha
        del model
    prepared["model_state_sha256"] = model_states
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0187 CPU prediction validation initialized CUDA")
    return prepared


def _bindings(prepared, arm):
    manifest = prepared["artifacts"][arm]["manifest"]
    refs = prepared["references"]
    return {"experiment": EXPERIMENT, "seed": prepared["seed"], "arm": arm,
        "training_manifest_sha256": prepared["artifacts"][arm]["manifest_sha256"],
        "training_state_sha256": prepared["queue_state_sha256"],
        "training_queue_fingerprint_sha256": prepared["queue_fingerprint_sha256"],
        "training_model_state_sha256": manifest["model_state_sha256"],
        "training_history_sha256": manifest["history_sha256"],
        "shared_prefix_manifest_sha256": prepared["prefix_manifest_sha256"],
        "shared_prefix_state_sha256": prepared["prefix_state_sha256"],
        "source_binding": manifest["bindings"]["source_binding"],
        "source_T_protocol_sha256": refs["source_T"]["protocol_sha256"],
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "dataset_arrays_sha256": manifest["dataset_arrays_sha256"],
        "ordered_train_ids_sha256": _sha_json(manifest["bindings"]["ordered_train_ids"]),
        "implementation_fingerprint": implementation_fingerprint()}


def _out(seed, arm, prediction_root=None):
    root = PREDICTION_ROOT if prediction_root is None else Path(prediction_root)
    return root / f"{arm}_seed{int(seed)}"


def _work(seed, arm, work_root=None):
    root = WORK_ROOT if work_root is None else Path(work_root)
    return root / f"{arm}_seed{int(seed)}"


def _write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if path.exists(): raise FileExistsError(path)
        os.link(temp, path)
    finally:
        if temp.exists(): temp.unlink()


@torch.no_grad()
def predict_arm(seed, arm, *, device="cuda:0", prediction_root=None, work_root=None, prepared=None):
    seed = int(seed)
    if arm not in ARMS: raise ValueError("unknown SW0187 prediction arm")
    prepared = prepare_prediction_inputs_cpu(seed) if prepared is None else prepared
    if prepared.get("seed") != seed: raise ValueError("prepared prediction inputs belong to another seed")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0187 fixed320 inference requires leased CUDA")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None or torch.cuda.device_count() != 1:
        raise RuntimeError("SW0187 prediction worker must see exactly its leased GPU")
    torch.cuda.set_device(dev)
    found, before = _load_model(seed, arm, prepared, dev)
    binding = _bindings(prepared, arm); binding_sha = _sha_json(binding)
    work = _work(seed, arm, work_root); work.mkdir(parents=True, exist_ok=True)
    binding_wrapper = {"bindings": binding, "bindings_sha256": binding_sha}
    binding_path = work / "bindings.json"
    if binding_path.exists():
        if json.loads(binding_path.read_text(encoding="utf-8")) != binding_wrapper:
            raise ValueError("prediction work directory is bound to another model")
    else: _write_json(binding_path, binding_wrapper)
    for index, image_id in enumerate(IMAGE_IDS.tolist()):
        folder = native._rowdir(work, index)
        if folder.exists():
            native._read_row(work, index, int(image_id), prepared["rgb_hashes"][index], binding_sha)
            continue
        image = torch.from_numpy(np.ascontiguousarray(prepared["rgb"][index:index + 1])).to(dev)
        torch.cuda.reset_peak_memory_stats(dev); started = time.perf_counter()
        row = native._predict_one(found, image, prepared["references"], index)
        native._predict_row(work, index, int(image_id), prepared["rgb_hashes"][index], binding_sha,
            row, seconds=time.perf_counter() - started, peak_bytes=torch.cuda.max_memory_allocated(dev))
        del image, row
    after = train.source._state_sha(found.wrapped, found.encoder)
    if after != before: raise RuntimeError("SW0187 model changed during fixed320 prediction")
    ledger = native._ledger_sha(work)
    source_state = {"experiment": EXPERIMENT, "seed": seed, "arm": arm,
        "model_state_sha256_before": before, "model_state_sha256_after": after,
        "model_unchanged": True, "bindings_sha256": binding_sha, "row_commits_sha256": ledger}
    state_file = work / "source_state.json"
    if state_file.exists():
        if json.loads(state_file.read_text(encoding="utf-8")) != source_state:
            raise ValueError("SW0187 immutable model state/row ledger changed")
    else: _write_json(state_file, source_state)
    labels, support_rows = {ep: [] for ep in ENDPOINTS}, []
    for index, image_id in enumerate(IMAGE_IDS.tolist()):
        row_labels, support, _ = native._read_row(work, index, int(image_id), prepared["rgb_hashes"][index], binding_sha)
        for ep in ENDPOINTS: labels[ep].append(row_labels[ep])
        support_rows.append(support)
    labels = {key: np.stack(value).astype(np.int32) for key, value in labels.items()}
    support = np.stack(support_rows).astype(np.int32)
    labels, support = native.validate_prediction_arrays(labels, support,
        prepared["references"]["source_T"]["labels"] > 0)
    output = _out(seed, arm, prediction_root)
    if output.exists():
        return validate_prediction(seed, arm, prediction_root=prediction_root,
            work_root=work_root, prepared=prepared)
    stage = output.with_name(output.name + f".stage-{uuid.uuid4().hex}"); stage.mkdir(parents=True)
    archive = stage / "predictions.npz"
    with archive.open("xb") as stream:
        np.savez_compressed(stream, image_ids=IMAGE_IDS, conditional=labels["conditional"],
            standalone=labels["standalone"], standalone_support=support)
        stream.flush(); os.fsync(stream.fileno())
    protocol = {"experiment": EXPERIMENT, "status": "complete", "seed": seed, "arm": arm,
        "image_ids": IMAGE_IDS.tolist(), "count": COUNT, "prediction_shape": [COUNT, GRID, GRID],
        "endpoints": list(ENDPOINTS), "bindings": binding, "bindings_sha256": binding_sha,
        "implementation_fingerprint": implementation_fingerprint(),
        "prediction_sha256": sha256_file(archive), "row_commits_sha256": ledger,
        "source_state_sha256": sha256_file(state_file), "ground_truth_used_for_prediction": False,
        "validation_data_read": False, "optimizer_updates": 0, "steps": 1024,
        "settle": 512, "live_tail": 0,
        "conditional_foreground_contract": "exact SourceT membership",
        "standalone_foreground_contract": "exact native production QCC membership"}
    _write_json(stage / "protocol.json", protocol); os.rename(stage, output)
    return validate_prediction(seed, arm, prediction_root=prediction_root,
        work_root=work_root, prepared=prepared)


def validate_prediction(seed, arm, *, prediction_root=None, work_root=None, prepared=None):
    seed = int(seed); prepared = prepare_prediction_inputs_cpu(seed) if prepared is None else prepared
    if arm not in ARMS: raise ValueError("unknown SW0187 prediction arm")
    binding = _bindings(prepared, arm); binding_sha = _sha_json(binding)
    work, output = _work(seed, arm, work_root), _out(seed, arm, prediction_root)
    state = json.loads((work / "source_state.json").read_text(encoding="utf-8"))
    ledger = native._ledger_sha(work)
    expected_model = prepared["model_state_sha256"][arm]
    if (state.get("experiment") != EXPERIMENT or state.get("seed") != seed or state.get("arm") != arm
            or state.get("model_unchanged") is not True or state.get("bindings_sha256") != binding_sha
            or state.get("row_commits_sha256") != ledger
            or state.get("model_state_sha256_before") != expected_model
            or state.get("model_state_sha256_after") != expected_model):
        raise ValueError("SW0187 prediction source-state/ledger receipt invalid")
    archive = output / "predictions.npz"; protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = {"experiment": EXPERIMENT, "status": "complete", "seed": seed, "arm": arm,
        "image_ids": IMAGE_IDS.tolist(), "count": COUNT, "prediction_shape": [COUNT, GRID, GRID],
        "endpoints": list(ENDPOINTS), "bindings": binding, "bindings_sha256": binding_sha,
        "implementation_fingerprint": implementation_fingerprint(), "row_commits_sha256": ledger,
        "source_state_sha256": sha256_file(work / "source_state.json"),
        "ground_truth_used_for_prediction": False, "validation_data_read": False,
        "optimizer_updates": 0, "steps": 1024, "settle": 512, "live_tail": 0,
        "conditional_foreground_contract": "exact SourceT membership",
        "standalone_foreground_contract": "exact native production QCC membership"}
    if any(protocol.get(k) != v for k, v in expected.items()) or protocol.get("prediction_sha256") != sha256_file(archive):
        raise ValueError("SW0187 prediction protocol/archive digest mismatch")
    with np.load(archive, allow_pickle=False) as data:
        contract.require_fixed_image_ids(data["image_ids"], name="SW0187 fixed320 IDs")
        labels = {ep: np.asarray(data[ep]) for ep in ENDPOINTS}; support = np.asarray(data["standalone_support"])
    labels, support = native.validate_prediction_arrays(labels, support,
        prepared["references"]["source_T"]["labels"] > 0)
    for i, image_id in enumerate(IMAGE_IDS.tolist()):
        rows, row_support, _ = native._read_row(work, i, int(image_id), prepared["rgb_hashes"][i], binding_sha)
        if any(not np.array_equal(rows[ep], labels[ep][i]) for ep in ENDPOINTS) or not np.array_equal(row_support, support[i]):
            raise ValueError("SW0187 published predictions differ from committed row ledger")
    return {"seed": seed, "arm": arm, "ids": IMAGE_IDS.copy(), "labels": labels,
        "standalone_support": support, "prediction_sha256": protocol["prediction_sha256"],
        "protocol_sha256": sha256_file(protocol_path), "protocol": protocol, "directory": output}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args(argv)
    if args.worker:
        if args.physical_gpu not in range(4) or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
            parser.error("worker physical GPU must match its leased CUDA visibility")
        prepared = prepare_prediction_inputs_cpu(args.source_seed)
        found = {arm: predict_arm(args.source_seed, arm, prepared=prepared) for arm in ARMS}
    else:
        parser.error("prediction evaluator runs only inside its owned queue worker")
    print(json.dumps({"seed": args.source_seed, "status": "complete",
        "predictions": {arm: row["prediction_sha256"] for arm, row in found.items()}}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
