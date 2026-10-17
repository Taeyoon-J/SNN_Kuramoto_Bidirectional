"""GT-free fixed320 predictions from the completed SW0186 proposal runs."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from collaborative_test.SW_0186_full_encoder_qce import train
from collaborative_test.SW_0179_qce_seed_replication import predict_evaluate as native_eval
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARM = "product_q_ce"
SEEDS = (0, 1, 2)
COUNT, GRID = int(contract.COUNT), int(contract.GRID)
IMAGE_IDS = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
ENDPOINTS = ("conditional", "standalone")
EXPERIMENT = "SW0186_full_encoder_qce_fixed320_prediction"
QUEUE_MODULE = "collaborative_test.SW_0186_full_encoder_qce.prediction_queue"
PREDICTION_ROOT = HERE / "results_archive" / "predictions_20261016"
WORK_ROOT = HERE / "results_archive" / "prediction_work_20261016"


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
    parent = native_eval.implementation_fingerprint()
    relpaths = (
        "collaborative_test/SW_0186_full_encoder_qce/DESIGN_KO.md",
        "collaborative_test/SW_0186_full_encoder_qce/train.py",
        "collaborative_test/SW_0186_full_encoder_qce/training_queue.py",
        "collaborative_test/SW_0186_full_encoder_qce/predict_evaluate.py",
        "collaborative_test/SW_0186_full_encoder_qce/test_predict_evaluate.py",
    )
    result = {rel: sha256_file(ROOT / rel) for rel in relpaths
              if (ROOT / rel).is_file()}
    if len(result) != len(relpaths):
        raise FileNotFoundError("SW0186 prediction fingerprint dependency is missing")
    return {"experiment": EXPERIMENT, "source_evaluator_fingerprint": parent,
            "files": result, "seeds": list(SEEDS), "image_ids": IMAGE_IDS.tolist(),
            "validation_data_read": False}


def _queue_state(seed):
    queue = train._module(train.QUEUE_MODULE)
    path = queue._state_path(seed)
    if not path.is_file():
        raise FileNotFoundError(f"SW0186 training state missing for source seed {seed}")
    return queue, path, json.loads(path.read_text(encoding="utf-8"))


def _load_model(seed, prepared, device):
    manifest = prepared["manifest"]
    artifact = prepared["artifact"]
    dev = torch.device(device)
    if dev.type == "cpu" and torch.cuda.is_initialized():
        raise RuntimeError("SW0186 CPU model load must precede CUDA initialization")
    if dev.type == "cuda" and not torch.cuda.is_initialized():
        raise RuntimeError("SW0186 GPU model load requires the worker CUDA context")
    found, _opt, _families, fresh_binding, setup = train.base._make_fresh(
        "product_q_ce", int(seed), prepared["training_inputs"], dev)
    base_encoder = setup[2]
    frozen_factory_components = setup[3]
    initial_sha = train.base._state_sha(found.wrapped, found.encoder)
    bindings = manifest.get("bindings", {})
    if (initial_sha != manifest.get("initial_model_state_sha256")
            or fresh_binding != bindings.get("factory_binding")):
        raise ValueError("SW0186 fresh source factory differs from the admitted training lineage")
    checkpoint_path = artifact / "checkpoint.pt"
    if sha256_file(checkpoint_path) != manifest.get("checkpoint_sha256"):
        raise ValueError("SW0186 checkpoint file changed after artifact validation")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (payload.get("experiment") != train.EXPERIMENT or payload.get("arm") != ARM
            or payload.get("seed") != int(seed) or payload.get("bindings") != bindings
            or payload.get("model_state_sha256") != manifest.get("final_model_state_sha256")
            or train.base._state_sha_from_payload(payload) != manifest.get("final_model_state_sha256")):
        raise ValueError("SW0186 checkpoint payload identity/state digest is invalid")
    found.wrapped.load_state_dict(payload["wrapped"], strict=True)
    found.encoder.load_state_dict(payload["encoder"], strict=True)
    found.wrapped.eval(); found.encoder.eval(); found.encoder.base_encoder.eval(); found.patcher.eval()
    final_sha = train.base._state_sha(found.wrapped, found.encoder)
    if final_sha != manifest.get("final_model_state_sha256"):
        raise ValueError("SW0186 loaded checkpoint tensor digest mismatch")
    util = train._module(train.base.SOURCE_UTIL_MODULE)
    if (util._frozen_source_components(found, base_encoder) != frozen_factory_components
            or manifest.get("frozen_source_components_after") != frozen_factory_components):
        raise ValueError("SW0186 trained model changed frozen source/preprocessing components")
    if train.base._base_encoder_sha(found) != manifest.get("source_base_encoder_final_sha256"):
        raise ValueError("SW0186 loaded model lost its trained source RGB encoder")
    if sha256_file(artifact / "optimizer.pt") != manifest.get("optimizer_sha256"):
        raise ValueError("SW0186 optimizer bytes differ from the completed training manifest")
    return found, final_sha


def prepare_prediction_inputs_cpu(seed):
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError("SW0186 prediction seed must be 0, 1, or 2")
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0186 prediction preparation must happen before CUDA initialization")
    queue, state_path, state = _queue_state(seed)
    if state.get("experiment") != "SW0186_training_queue" or state.get("seed") != seed or state.get("arm") != ARM:
        raise ValueError("SW0186 training queue state identity mismatch")
    if state.get("status") == "complete_rejected":
        raise RuntimeError(f"SW0186 seed{seed} is terminally scientifically rejected")
    if state.get("status") != "complete" or state.get("validation_data_read") is not False:
        raise RuntimeError(f"SW0186 seed{seed} training is not complete")
    preflight_receipt = train._preflight_receipt(seed)
    inputs, identity = queue._cpu_admission(seed, train, preflight_receipt)
    factory_proof = identity["source_factory_proof"]
    artifact = train.OUTPUT_ROOT / f"{ARM}_seed{seed}"
    manifest = train.validate_training_artifact(ARM, artifact, seed=seed)
    manifest_path = artifact / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    _queue_payload, queue_sha = queue._queue_fingerprint(train, seed, identity)
    if (state.get("queue_fingerprint_sha256") != queue_sha
            or state.get("manifest_sha256") != manifest_sha
            or state.get("artifact_sha256") != manifest_sha
            or state.get("proposal_cursor") != train.PROPOSALS
            or manifest.get("accepted_updates") != state.get("accepted_updates")):
        raise ValueError("SW0186 final training artifact does not match its queue receipt")
    refs = native_eval._references(seed)
    rgb_path = Path(sw130.VAL_RGB)
    meta_path = rgb_path.with_suffix(rgb_path.suffix + ".complete.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    expected_ids_sha = hashlib.sha256(IMAGE_IDS.astype("<i8").tobytes()).hexdigest()
    if (meta.get("status") != "complete" or meta.get("cache_shape") != [COUNT, 128, 128, 3]
            or meta.get("ids_mapping_sha256") != expected_ids_sha or rgb.dtype != np.uint8
            or tuple(rgb.shape) != (COUNT, 128, 128, 3)):
        raise ValueError("SW0186 fixed320 RGB cache schema/mapping changed")
    rgb_hashes = [hashlib.sha256(np.ascontiguousarray(rgb[i]).tobytes()).hexdigest()
                  for i in range(COUNT)]
    dataset = Path(sw130.source97.DATASET).resolve()
    stat = dataset.stat()
    prepared = {"seed": seed, "training_inputs": inputs, "artifact": artifact,
        "manifest": manifest, "manifest_sha256": manifest_sha,
        "training_state": state, "training_state_path": state_path,
        "training_state_sha256": sha256_file(state_path), "queue_fingerprint_sha256": queue_sha,
        "factory_proof": factory_proof, "references": refs,
        "rgb": rgb, "rgb_path": rgb_path, "rgb_meta_path": meta_path,
        "rgb_meta_sha256": sha256_file(meta_path), "rgb_hashes": rgb_hashes,
        "dataset": dataset, "dataset_size_bytes": stat.st_size, "dataset_mtime_ns": stat.st_mtime_ns}
    found, model_sha = _load_model(seed, prepared, "cpu")
    del found
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0186 CPU model validation initialized CUDA")
    prepared["model_state_sha256"] = model_sha
    return prepared


def _out(seed, prediction_root=None):
    root = PREDICTION_ROOT if prediction_root is None else Path(prediction_root)
    return root / f"product_q_ce_seed{int(seed)}"


def _work(seed, work_root=None):
    root = WORK_ROOT if work_root is None else Path(work_root)
    return root / f"product_q_ce_seed{int(seed)}"


def _write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.link(temp, path)
    finally:
        if temp.exists(): temp.unlink()


def _bindings(prepared):
    return {"experiment": EXPERIMENT, "seed": prepared["seed"], "arm": ARM,
        "training_manifest_sha256": prepared["manifest_sha256"],
        "training_state_sha256": prepared["training_state_sha256"],
        "training_queue_fingerprint_sha256": prepared["queue_fingerprint_sha256"],
        "model_state_sha256": prepared["model_state_sha256"],
        "source_binding": prepared["manifest"]["bindings"]["source_binding"],
        "source_T_protocol_sha256": prepared["references"]["source_T"]["protocol_sha256"],
        "dataset_manifest_sha256": prepared["training_inputs"]["dataset_manifest_sha256"],
        "dataset_arrays_sha256": prepared["training_inputs"]["dataset_arrays_sha256"],
        "implementation_fingerprint": implementation_fingerprint()}


@torch.no_grad()
def predict_seed(seed, *, device="cuda:0", prediction_root=None, work_root=None, prepared=None):
    seed = int(seed)
    prepared = prepare_prediction_inputs_cpu(seed) if prepared is None else prepared
    if prepared.get("seed") != seed:
        raise ValueError("SW0186 prepared prediction inputs belong to another seed")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0186 fixed320 inference requires a leased CUDA device")
    torch.cuda.set_device(dev)
    found, model_before = _load_model(seed, prepared, dev)
    binding = _bindings(prepared); binding_sha = _sha_json(binding)
    work = _work(seed, work_root); work.mkdir(parents=True, exist_ok=True)
    binding_path = work / "bindings.json"
    wrapper = {"bindings": binding, "bindings_sha256": binding_sha}
    if binding_path.exists():
        if json.loads(binding_path.read_text(encoding="utf-8")) != wrapper:
            raise ValueError("existing SW0186 prediction rows use different bindings")
    else:
        _write_json(binding_path, wrapper)
    for index, image_id in enumerate(IMAGE_IDS.tolist()):
        folder = native_eval._rowdir(work, index)
        if folder.exists():
            native_eval._read_row(work, index, int(image_id), prepared["rgb_hashes"][index], binding_sha)
            continue
        image = torch.from_numpy(np.ascontiguousarray(prepared["rgb"][index:index + 1])).to(dev)
        torch.cuda.reset_peak_memory_stats(dev); started = time.perf_counter()
        row = native_eval._predict_one(found, image, prepared["references"], index)
        native_eval._predict_row(work, index, int(image_id), prepared["rgb_hashes"][index],
            binding_sha, row, seconds=time.perf_counter() - started,
            peak_bytes=torch.cuda.max_memory_allocated(dev))
        del image, row
    model_after = train.base._state_sha(found.wrapped, found.encoder)
    if model_after != model_before:
        raise RuntimeError("SW0186 trained model changed during prediction")
    ledger = native_eval._ledger_sha(work)
    source_state = {"experiment": EXPERIMENT, "seed": seed, "arm": ARM,
        "model_state_sha256_before": model_before, "model_state_sha256_after": model_after,
        "model_unchanged": True, "bindings_sha256": binding_sha, "row_commits_sha256": ledger}
    state_file = work / "source_state.json"
    if state_file.exists():
        if json.loads(state_file.read_text(encoding="utf-8")) != source_state:
            raise ValueError("SW0186 immutable source state/row ledger changed")
    else:
        _write_json(state_file, source_state)
    labels, supports = {endpoint: [] for endpoint in ENDPOINTS}, []
    for index, image_id in enumerate(IMAGE_IDS.tolist()):
        row_labels, support, _row = native_eval._read_row(
            work, index, int(image_id), prepared["rgb_hashes"][index], binding_sha)
        for endpoint in ENDPOINTS: labels[endpoint].append(row_labels[endpoint])
        supports.append(support)
    labels = {key: np.stack(values).astype(np.int32) for key, values in labels.items()}
    support = np.stack(supports).astype(np.int32)
    native_eval.validate_prediction_arrays(labels, support,
        prepared["references"]["source_T"]["labels"] > 0)
    output = _out(seed, prediction_root)
    if output.exists():
        return validate_prediction(seed, prediction_root=prediction_root, work_root=work_root,
                                   prepared=prepared)
    stage = output.with_name(output.name + f".stage-{uuid.uuid4().hex}")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        archive = stage / "predictions.npz"
        with archive.open("xb") as stream:
            np.savez_compressed(stream, image_ids=IMAGE_IDS, conditional=labels["conditional"],
                standalone=labels["standalone"], standalone_support=support)
            stream.flush(); os.fsync(stream.fileno())
        protocol = {"experiment": EXPERIMENT, "status": "complete", "seed": seed,
            "arm": ARM, "image_ids": IMAGE_IDS.tolist(), "count": COUNT,
            "prediction_shape": [COUNT, GRID, GRID], "endpoints": list(ENDPOINTS),
            "bindings": binding, "bindings_sha256": binding_sha,
            "implementation_fingerprint": implementation_fingerprint(),
            "prediction_sha256": sha256_file(archive),
            "row_commits_sha256": ledger, "source_state_sha256": sha256_file(state_file),
            "ground_truth_used_for_prediction": False, "validation_data_read": False,
            "optimizer_updates": 0, "steps": 1024, "settle": 512, "live_tail": 0,
            "conditional_foreground_contract": "exact SourceT membership",
            "standalone_foreground_contract": "exact native production QCC membership"}
        _write_json(stage / "protocol.json", protocol)
        os.rename(stage, output)
    except BaseException:
        raise
    return validate_prediction(seed, prediction_root=prediction_root, work_root=work_root,
                               prepared=prepared)


def validate_prediction(seed, *, prediction_root=None, work_root=None, prepared=None):
    seed = int(seed)
    prepared = prepare_prediction_inputs_cpu(seed) if prepared is None else prepared
    binding = _bindings(prepared); binding_sha = _sha_json(binding)
    work, output = _work(seed, work_root), _out(seed, prediction_root)
    state = json.loads((work / "source_state.json").read_text(encoding="utf-8"))
    ledger = native_eval._ledger_sha(work)
    if (state.get("experiment") != EXPERIMENT or state.get("seed") != seed
            or state.get("arm") != ARM or state.get("model_unchanged") is not True
            or state.get("bindings_sha256") != binding_sha
            or state.get("row_commits_sha256") != ledger
            or state.get("model_state_sha256_before") != prepared["model_state_sha256"]
            or state.get("model_state_sha256_after") != prepared["model_state_sha256"]):
        raise ValueError("SW0186 prediction source-state receipt/ledger invalid")
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    archive = output / "predictions.npz"
    expected = {"experiment": EXPERIMENT, "status": "complete", "seed": seed,
        "arm": ARM, "image_ids": IMAGE_IDS.tolist(), "count": COUNT,
        "prediction_shape": [COUNT, GRID, GRID], "endpoints": list(ENDPOINTS),
        "bindings": binding, "bindings_sha256": binding_sha,
        "implementation_fingerprint": implementation_fingerprint(),
        "row_commits_sha256": ledger, "source_state_sha256": sha256_file(work / "source_state.json"),
        "ground_truth_used_for_prediction": False, "validation_data_read": False,
        "optimizer_updates": 0, "steps": 1024, "settle": 512, "live_tail": 0,
        "conditional_foreground_contract": "exact SourceT membership",
        "standalone_foreground_contract": "exact native production QCC membership"}
    if any(protocol.get(key) != value for key, value in expected.items()) \
            or protocol.get("prediction_sha256") != sha256_file(archive):
        raise ValueError("SW0186 prediction protocol/binding/archive digest mismatch")
    with np.load(archive, allow_pickle=False) as data:
        contract.require_fixed_image_ids(data["image_ids"], name="SW0186 fixed320 IDs")
        labels = {key: np.asarray(data[key]) for key in ENDPOINTS}
        support = np.asarray(data["standalone_support"])
    labels, support = native_eval.validate_prediction_arrays(labels, support,
        prepared["references"]["source_T"]["labels"] > 0)
    for index, image_id in enumerate(IMAGE_IDS.tolist()):
        row_labels, row_support, _ = native_eval._read_row(
            work, index, int(image_id), prepared["rgb_hashes"][index], binding_sha)
        if any(not np.array_equal(row_labels[key], labels[key][index]) for key in ENDPOINTS):
            raise ValueError("SW0186 published predictions differ from committed row ledger")
        if not np.array_equal(row_support, support[index]):
            raise ValueError("SW0186 standalone supports differ from row ledger")
    return {"seed": seed, "arm": ARM, "ids": IMAGE_IDS.copy(), "labels": labels,
        "standalone_support": support, "prediction_sha256": protocol["prediction_sha256"],
        "protocol_sha256": sha256_file(output / "protocol.json"), "protocol": protocol,
        "directory": output}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--physical-gpu", type=int, choices=range(4), required=True)
    args = parser.parse_args(argv)
    if not args.worker:
        parser.error("prediction worker is owned by prediction_queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu) or torch.cuda.is_initialized():
        raise RuntimeError("SW0186 prediction worker does not match its exclusive GPU lease")
    prepared = prepare_prediction_inputs_cpu(args.source_seed)
    result = predict_seed(args.source_seed, device="cuda:0", prepared=prepared)
    print(json.dumps({"status": "complete", "seed": args.source_seed,
        "prediction_sha256": result["prediction_sha256"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
