"""SW161 GT-free fixed320 prediction and strict GT-last evaluation."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"), str(ROOT / "snn_kuramoto_bidirectional")]

import h5py
import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0135_native32_spike_binding.binder import NativeSpikeSlotBinder, canonical_hard_labels
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143, geodesic_operator
from collaborative_test.SW_0155_supervised_spike_relations import train as sw155
from collaborative_test.SW_0161_supervised_spike_partition import train

ARMS = train.ARMS
SEED = train.SEED
IMAGE_IDS = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
COUNT, GRID = contract.COUNT, contract.GRID
PREDICTION_ROOT = HERE / "results_archive" / "predictions_20261014"
WORK_ROOT = HERE / "results_archive" / "prediction_work_20261014"
SCORE_PATH = HERE / "results_archive" / "evaluation_seed1.json"
DATASET = Path(sw130.source97.DATASET)
SLOT_ROOT = ROOT / "trained_models/SW0092_slot_our70000_eval"
SLOT_PATH = SLOT_ROOT / "seed1_epoch10" / "predictions.npz"
SLOT_PROTOCOL = SLOT_ROOT / "seed1_epoch10" / "protocol.json"
BASELINE_T = Path(sw143._arm_dir("T", SEED))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def implementation_fingerprint():
    paths = [HERE / n for n in ("predict_evaluate.py", "protocol.json", "DESIGN_KO.md", "objective.py", "train.py", "cache.py", "preflight.py")]
    paths += [ROOT / p for p in (
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
        "collaborative_test/SW_0135_native32_spike_binding/binder.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0137_native32_source_qcc/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
        "snn_kuramoto_bidirectional/evaluation.py")]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("SW161 prediction dependency is missing")
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def _source_refs():
    source = sw137.validate_prediction(sw137.OUTPUT_ROOT / "seed1", seed=SEED)
    tpred = sw143.validate_arm_prediction(BASELINE_T, seed=SEED)
    if (not np.array_equal(source["ids"], IMAGE_IDS) or not np.array_equal(tpred["ids"], IMAGE_IDS)
            or tpred.get("arm") != "T"):
        raise ValueError("SW161 source/T references do not match fixed320 IDs")
    slot_protocol = json.loads(SLOT_PROTOCOL.read_text(encoding="utf-8"))
    slot_sha = contract.validate_slot_protocol(slot_protocol, seed=SEED, prediction_path=SLOT_PATH)
    with np.load(SLOT_PATH, allow_pickle=False) as data:
        slot_ids = contract.require_fixed_image_ids(data["image_ids"], name="Slot image_ids")
        pixel = np.asarray(data["labels"] if "labels" in data.files else data["pixel_labels"])
    if not np.array_equal(slot_ids, IMAGE_IDS):
        raise ValueError("Slot reference IDs changed")
    slot_labels = contract.slot_native_pixels_to32(pixel)
    return source, tpred, {"ids": slot_ids, "labels": slot_labels, "prediction_sha256": slot_sha,
        "protocol_path": SLOT_PROTOCOL, "protocol_sha256": sha256_file(SLOT_PROTOCOL), "archive_path": SLOT_PATH}


def _training_artifact(arm, training_root=train.TRAIN_ROOT):
    if arm not in ARMS:
        raise ValueError("unregistered SW161 arm")
    folder = Path(training_root) / f"{arm}_seed1"
    manifest = train.validate_training_artifact(arm, directory=folder)
    manifest_path, checkpoint_path = folder / "manifest.json", folder / "checkpoint.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if set(payload) != {"binder", "binder_state_sha256", "completed_updates"}:
        raise ValueError("SW161 endpoint checkpoint schema changed")
    state_sha = sw155.source_training.state_sha(payload["binder"])
    if state_sha != manifest["binder_state_sha256"] or payload["completed_updates"] != train.UPDATES:
        raise ValueError("SW161 endpoint checkpoint differs from its validated manifest")
    return {"folder": folder, "manifest": manifest, "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_path": checkpoint_path, "checkpoint_sha256": sha256_file(checkpoint_path),
        "model_sha256": state_sha}


def _load_predictor(arm, device, training_root=train.TRAIN_ROOT):
    artifact = _training_artifact(arm, training_root)
    foundation = sw155._load_foundation(device)
    geodesic_operator.install_operator(foundation.wrapped.core.graph_generator, "T")
    binder = NativeSpikeSlotBinder(seed=train.HEAD_SEED).to(device)
    payload = torch.load(artifact["checkpoint_path"], map_location=device, weights_only=False)
    binder.load_state_dict(payload["binder"], strict=True)
    binder.eval().requires_grad_(False)
    loaded_sha = sw155.source_training.state_sha(binder.state_dict())
    if loaded_sha != artifact["model_sha256"]:
        raise ValueError("loaded SW161 binder differs from final checkpoint")
    foundation.wrapped.eval().requires_grad_(False)
    if hasattr(foundation, "encoder"):
        foundation.encoder.eval().requires_grad_(False)
    return foundation, binder, artifact


def _prediction_bindings(arm, artifact, source, tpred, slot, rgb, dataset=DATASET, cache_manifest=None):
    dataset = Path(dataset)
    st = dataset.stat()
    source_contract = sw155.source_contract()
    cache_assets = sw130.validate_rgb_assets()
    if rgb.dtype != np.uint8 or rgb.shape != (COUNT, 128, 128, 3):
        raise ValueError("SW161 validation RGB cache must be fixed320 uint8 NHWC")
    current_cache = cache_manifest if cache_manifest is not None else train.validate_cache()
    cache_bind = current_cache["bindings"]
    if (str(dataset.resolve()) != cache_bind.get("hdf5_path") or st.st_size != cache_bind.get("hdf5_size")
            or st.st_mtime_ns != cache_bind.get("hdf5_mtime_ns")
            or current_cache.get("bindings_sha256") != artifact["manifest"].get("cache_bindings_sha256")):
        raise ValueError("SW161 validation dataset differs from admitted training HDF5")
    return {"experiment": train.EXPERIMENT, "arm": arm, "seed": SEED,
        "implementation_fingerprint": implementation_fingerprint(),
        "training_manifest_sha256": artifact["manifest_sha256"],
        "training_checkpoint_sha256": artifact["checkpoint_sha256"],
        "training_model_state_sha256": artifact["model_sha256"],
        "training_implementation_fingerprint": artifact["manifest"]["implementation_fingerprint"],
        "cache_manifest_sha256": artifact["manifest"]["cache_manifest_sha256"],
        "cache_bindings_sha256": artifact["manifest"]["cache_bindings_sha256"],
        "source_model_state_sha256": current_cache["source_model_state_before_sha256"],
        "source_checkpoint_sha256": source_contract["source_sha256"],
        "source_manifest_sha256": sha256_file(source_contract["manifest_path"]),
        "source_training_ids_sha256": sw155.ids_sha(source_contract["image_ids"]),
        "rgb_asset_validation": cache_assets,
        "source_qcc_prediction_sha256": source["prediction_sha256"],
        "source_qcc_protocol_sha256": source["protocol_sha256"],
        "source_T_prediction_sha256": tpred["prediction_sha256"],
        "source_T_protocol_sha256": tpred["protocol_sha256"],
        "slot_prediction_sha256": slot["prediction_sha256"], "slot_protocol_sha256": slot["protocol_sha256"],
        "dataset_path": str(dataset.resolve()), "dataset_size_bytes": st.st_size, "dataset_mtime_ns": st.st_mtime_ns,
        "validation_rgb_shape": list(rgb.shape), "validation_rgb_dtype": str(rgb.dtype),
        "image_ids_sha256": hashlib.sha256(IMAGE_IDS.astype("<i8").tobytes()).hexdigest(),
        "steps": 1024, "settle": 512, "live_tail_steps": 64,
        "readout": "SW0135 NativeSpikeSlotBinder argmax(P) canonical_hard_labels"}


def prepare_prediction_cpu(arm, *, training_root=train.TRAIN_ROOT, dataset=DATASET):
    """Validate expensive dependencies before CUDA allocation; queue and child share this contract."""
    artifact = _training_artifact(arm, training_root)
    cache_manifest_path = Path(train.CACHE_ROOT) / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if (sha256_file(cache_manifest_path) != artifact["manifest"].get("cache_manifest_sha256")
            or cache_manifest.get("status") != "complete"):
        raise ValueError("SW161 source cache changed after strict training artifact validation")
    source, tpred, slot = _source_refs()
    rgb = np.load(sw130.VAL_RGB, mmap_mode="r", allow_pickle=False)
    bindings = _prediction_bindings(arm, artifact, source, tpred, slot, rgb, dataset, cache_manifest)
    return {"artifact": artifact, "cache_manifest": cache_manifest, "source": source,
            "source_T": tpred, "slot": slot, "rgb": rgb, "bindings": bindings,
            "bindings_sha256": _sha_json(bindings)}


def _row_dir(work, i): return Path(work) / "rows" / f"{i:04d}"


def _write_row(work, i, image_id, rgb_sha, bind_sha, labels):
    target = _row_dir(work, i)
    if target.exists(): raise FileExistsError(f"preserving SW161 row {i}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(target.name + f".staging-{os.getpid()}-{uuid.uuid4().hex}")
    stage.mkdir()
    arr = np.asarray(labels, dtype=np.int64)
    with (stage / "row.npz").open("xb") as f:
        np.savez_compressed(f, image_index=np.asarray(i, np.int64), image_id=np.asarray(image_id, np.int64), labels=arr)
        f.flush(); os.fsync(f.fileno())
    row = {"image_index": i, "image_id": image_id, "rgb_sha256": rgb_sha, "bindings_sha256": bind_sha,
        "labels_sha256": hashlib.sha256(arr.astype("<i8", copy=False).tobytes()).hexdigest(),
        "npz_sha256": sha256_file(stage / "row.npz")}
    with (stage / "row.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(row, f, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.rename(stage, target)
    return row


def _load_row(work, i, image_id, rgb_sha, bind_sha):
    folder = _row_dir(work, i); record = json.loads((folder / "row.json").read_text(encoding="utf-8")); npz = folder / "row.npz"
    expected = {"image_index": i, "image_id": image_id, "rgb_sha256": rgb_sha, "bindings_sha256": bind_sha}
    if any(record.get(k) != v for k, v in expected.items()) or record.get("npz_sha256") != sha256_file(npz):
        raise ValueError(f"SW161 row {i} has invalid identity or digest")
    with np.load(npz, allow_pickle=False) as data:
        labels = np.asarray(data["labels"])
        if (int(data["image_index"]) != i or int(data["image_id"]) != image_id or labels.shape != (GRID, GRID)
                or labels.dtype.kind not in "iu" or np.any(labels < 0) or np.unique(labels).size > 11):
            raise ValueError(f"SW161 row {i} labels violate native32 contract")
    labels = labels.astype(np.int64, copy=False)
    if record.get("labels_sha256") != hashlib.sha256(labels.astype("<i8", copy=False).tobytes()).hexdigest():
        raise ValueError(f"SW161 row {i} label checksum mismatch")
    return labels


def _rows_digest(work):
    digests = []
    for i in range(COUNT):
        p = _row_dir(work, i) / "row.json"
        if not p.is_file(): raise ValueError(f"SW161 prediction row {i} is not committed")
        digests.append(sha256_file(p))
    return hashlib.sha256(json.dumps(digests, separators=(",", ":")).encode()).hexdigest()


def _validate_prediction_protocol(protocol, *, arm, bindings, bindings_sha, rows_sha, artifact):
    """One producer/consumer schema bound to freshly validated lineage."""
    expected = {"experiment": train.EXPERIMENT, "status": "complete", "arm": arm, "seed": SEED,
        "image_ids": [int(IMAGE_IDS[0]), int(IMAGE_IDS[-1])], "count": COUNT,
        "grid_size": [GRID, GRID], "prediction_shape": [COUNT, GRID, GRID],
        "steps": 1024, "settle": 512, "live_tail_steps": 64,
        "optimizer_updates": 0, "ground_truth_used_for_prediction": False,
        "ground_truth_used_for_scoring": False, "prediction_bindings": bindings,
        "prediction_bindings_sha256": bindings_sha, "prediction_row_commits_sha256": rows_sha,
        "implementation_fingerprint": implementation_fingerprint(),
        "training_manifest_sha256": artifact["manifest_sha256"],
        "training_checkpoint_sha256": artifact["checkpoint_sha256"],
        "training_model_state_sha256": artifact["model_sha256"],
        "source_model_state_sha256_before": bindings["source_model_state_sha256"],
        "source_model_state_sha256_after": bindings["source_model_state_sha256"],
        "model_state_sha256_before": artifact["model_sha256"],
        "model_state_sha256_after": artifact["model_sha256"]}
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"SW161 prediction provenance mismatch: {key}")
    return True


def _publish_prediction(folder, labels, protocol):
    folder = Path(folder)
    if folder.exists(): raise FileExistsError(f"preserving existing SW161 prediction {folder}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    stage = folder.with_name(folder.name + f".staging-{uuid.uuid4().hex}"); stage.mkdir()
    with (stage / "predictions.npz").open("xb") as f:
        np.savez_compressed(f, image_ids=IMAGE_IDS, labels=np.asarray(labels, dtype=np.int64))
        f.flush(); os.fsync(f.fileno())
    protocol["prediction_sha256"] = sha256_file(stage / "predictions.npz")
    with (stage / "protocol.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(protocol, f, indent=2, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    digest = sha256_file(stage / "protocol.json")
    os.rename(stage, folder)
    return {"prediction_sha256": protocol["prediction_sha256"], "protocol_sha256": digest}


@torch.no_grad()
def predict(arm, *, device="cuda:0", prediction_root=PREDICTION_ROOT, work_root=WORK_ROOT,
            training_root=train.TRAIN_ROOT, dataset=DATASET):
    if arm not in ARMS: raise ValueError("unregistered SW161 arm")
    output = Path(prediction_root) / f"{arm}_seed1"
    if output.exists(): return validate_prediction(arm, prediction_root=prediction_root, work_root=work_root, training_root=training_root, dataset=dataset)
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("SW161 prediction requires assigned CUDA")
    torch.cuda.set_device(dev)
    prepared = prepare_prediction_cpu(arm, training_root=training_root, dataset=dataset)
    artifact, cache, bindings = prepared["artifact"], prepared["rgb"], prepared["bindings"]
    foundation, binder, artifact = _load_predictor(arm, dev, training_root)
    bind_sha = _sha_json(bindings); work = Path(work_root) / f"{arm}_seed1"; marker = work / "bindings.json"
    if work.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != {"bindings": bindings, "bindings_sha256": bind_sha}:
            raise ValueError("preserving partial SW161 rows with changed lineage")
    else:
        work.mkdir(parents=True)
        with marker.open("x", encoding="utf-8", newline="\n") as f:
            json.dump({"bindings": bindings, "bindings_sha256": bind_sha}, f, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    labels = np.empty((COUNT, GRID, GRID), dtype=np.int64)
    source_state_before = train.cache._state_digest(foundation)
    if source_state_before != prepared["cache_manifest"]["source_model_state_before_sha256"]:
        raise ValueError("SW161 source foundation differs from frozen TRAIN cache")
    model_before = sw155.source_training.state_sha(binder.state_dict())
    if model_before != artifact["model_sha256"]: raise ValueError("SW161 binder hash changed before prediction")
    for i, image_id in enumerate(IMAGE_IDS.tolist()):
        rgb = np.asarray(cache[i:i + 1]).copy(); rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        if _row_dir(work, i).exists():
            labels[i] = _load_row(work, i, int(image_id), rgb_sha, bind_sha); continue
        gamma = foundation.encode(torch.from_numpy(rgb).to(dev))
        trace = foundation.rollout(gamma, total_steps=1024, live_tail_steps=64)
        for key in sw143.TRACE_FIELDS:
            value = trace.get(key)
            if not torch.is_tensor(value) or not torch.isfinite(value).all(): raise FloatingPointError(f"invalid trace {key}")
        p, _, _ = binder(trace["component_spikes"][:, :, :, 512:])
        row = canonical_hard_labels(p).cpu().numpy()[0]
        if row.shape != (GRID, GRID) or row.dtype.kind not in "iu" or np.any(row < 0): raise ValueError("invalid SW161 hard labels")
        labels[i] = row; _write_row(work, i, int(image_id), rgb_sha, bind_sha, row)
        if i % 32 == 31: print(json.dumps({"stage":"predict","arm":arm,"predicted":i+1,"count":COUNT}), flush=True)
        del gamma, trace, p
    model_after = sw155.source_training.state_sha(binder.state_dict())
    if model_after != model_before: raise AssertionError("SW161 inference changed binder state")
    source_state_after = train.cache._state_digest(foundation)
    if source_state_after != source_state_before: raise AssertionError("SW161 inference changed source foundation state")
    protocol = {"experiment": train.EXPERIMENT, "status":"complete", "arm":arm,"seed":SEED,
        "image_ids":[int(IMAGE_IDS[0]),int(IMAGE_IDS[-1])],"count":COUNT,"grid_size":[GRID,GRID],
        "prediction_shape":[COUNT,GRID,GRID],"steps":1024,"settle":512,"live_tail_steps":64,
        "readout":"NativeSpikeSlotBinder argmax(P); largest occupied slot background; groups <8 mapped to background; canonical foreground IDs",
        "optimizer_updates":0,"ground_truth_used_for_prediction":False,"ground_truth_used_for_scoring":False,
        "prediction_bindings":bindings,"prediction_bindings_sha256":bind_sha,"prediction_row_commits_sha256":_rows_digest(work),
        "implementation_fingerprint":implementation_fingerprint(),"training_manifest_sha256":artifact["manifest_sha256"],
        "training_checkpoint_sha256":artifact["checkpoint_sha256"],"training_model_state_sha256":model_before,
        "source_model_state_sha256_before":source_state_before,"source_model_state_sha256_after":source_state_after,
        "model_state_sha256_before":model_before,"model_state_sha256_after":model_after}
    return _publish_prediction(output, labels, protocol)


def validate_prediction(arm, *, prediction_root=PREDICTION_ROOT, work_root=WORK_ROOT,
                        training_root=train.TRAIN_ROOT, dataset=DATASET):
    if arm not in ARMS: raise ValueError("unregistered SW161 arm")
    folder = Path(prediction_root) / f"{arm}_seed1"; protocol_path = folder / "protocol.json"; archive_path = folder / "predictions.npz"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    prepared = prepare_prediction_cpu(arm, training_root=training_root, dataset=dataset)
    artifact, cache = prepared["artifact"], prepared["rgb"]
    bindings, bind_sha = prepared["bindings"], prepared["bindings_sha256"]
    _validate_prediction_protocol(protocol, arm=arm, bindings=bindings, bindings_sha=bind_sha,
        rows_sha=_rows_digest(Path(work_root)/f"{arm}_seed1"), artifact=artifact)
    if protocol.get("prediction_sha256") != sha256_file(archive_path): raise ValueError("SW161 prediction archive hash mismatch")
    with np.load(archive_path, allow_pickle=False) as data:
        ids = contract.require_fixed_image_ids(data["image_ids"]); labels = np.asarray(data["labels"])
    if labels.shape != (COUNT,GRID,GRID) or labels.dtype.kind not in "iu" or np.any(labels<0) or np.any(labels>=11):
        raise ValueError("SW161 prediction labels violate fixed native32 schema")
    marker = json.loads((Path(work_root)/f"{arm}_seed1"/"bindings.json").read_text(encoding="utf-8"))
    if marker != {"bindings":bindings,"bindings_sha256":bind_sha}: raise ValueError("SW161 row ledger lineage mismatch")
    for i, image_id in enumerate(IMAGE_IDS.tolist()):
        rgb_sha=hashlib.sha256(np.asarray(cache[i:i+1]).tobytes()).hexdigest()
        if not np.array_equal(_load_row(work_root,i,int(image_id),rgb_sha,bind_sha),labels[i]): raise ValueError(f"SW161 row mismatch {i}")
    return {"ids":ids,"labels":labels.astype(np.int64,copy=False),"protocol":protocol,"npz_path":archive_path,
        "protocol_path":protocol_path,"prediction_sha256":sha256_file(archive_path),"protocol_sha256":sha256_file(protocol_path)}


def _paired_bootstrap(left, right):
    delta=np.asarray(left,dtype=np.float64)-np.asarray(right,dtype=np.float64)
    if delta.shape!=(COUNT,) or not np.isfinite(delta).all(): raise ValueError("paired comparisons require finite fixed320 vectors")
    rng=np.random.RandomState(139); samples=delta[rng.randint(0,COUNT,size=(10000,COUNT))].mean(1)
    return {"mean":float(delta.mean()),"ci95":[float(x) for x in np.percentile(samples,[2.5,97.5])]}


def _registered_gate(means, paired):
    candidate, control = "mask_supervised_P", "pair_supervised_P"
    gate={"primary_metric":"fg_ari","candidate_exceeds_SW0143_T":means[candidate]["fg_ari"]>means["SW0143_T"]["fg_ari"],
      "candidate_exceeds_matched_pair_control":means[candidate]["fg_ari"]>means[control]["fg_ari"],
      "candidate_minus_SW0143_T_fg_ari_ci95_lower_positive":paired[f"{candidate}_minus_SW0143_T_fg_ari"]["ci95"][0]>0,
      "candidate_minus_control_fg_ari_ci95_lower_positive":paired[f"{candidate}_minus_{control}_fg_ari"]["ci95"][0]>0}
    gate["registered_seed1_gate_pass"]=all(gate.values())
    comparisons={f"{arm}_minus_{base}_{metric}_mean_positive":means[arm][metric]>means[base][metric]
      for arm in ARMS for base in ("source97","SW0143_T",*[other for other in ARMS if other!=arm],"Slot70k") for metric in contract.METRICS}
    return gate,comparisons


def score_all(*, output=SCORE_PATH, prediction_root=PREDICTION_ROOT, dataset=DATASET):
    output=Path(output)
    if output.exists(): raise FileExistsError(f"preserving existing SW161 score {output}")
    # Validate both candidates and every registered reference before opening HDF5 masks.
    predictions={arm:validate_prediction(arm,prediction_root=prediction_root,dataset=dataset) for arm in ARMS}
    source,tpred,slot=_source_refs()
    frozen={**predictions,"source97":source,"SW0143_T":tpred,"Slot70k":slot}
    hashes={}
    for name,record in frozen.items():
        ap=record.get("npz_path",record.get("archive_path")); pp=record["protocol_path"]
        hashes[f"{name}:prediction"]=sha256_file(ap); hashes[f"{name}:protocol"]=sha256_file(pp)
        if hashes[f"{name}:prediction"]!=record["prediction_sha256"] or hashes[f"{name}:protocol"]!=record["protocol_sha256"]:
            raise ValueError("frozen SW161 prediction/reference changed before GT")
    dataset=Path(dataset); stat=dataset.stat()
    for arm,rec in predictions.items():
        b=rec["protocol"]["prediction_bindings"]
        if b["dataset_path"]!=str(dataset.resolve()) or b["dataset_size_bytes"]!=stat.st_size or b["dataset_mtime_ns"]!=stat.st_mtime_ns:
            raise ValueError("SW161 dataset identity changed before GT")
    for name,rec in frozen.items():
        if sha256_file(rec.get("npz_path",rec.get("archive_path")))!=hashes[f"{name}:prediction"] or sha256_file(rec["protocol_path"])!=hashes[f"{name}:protocol"]:
            raise ValueError("SW161 frozen artifact changed immediately before GT")
    with h5py.File(dataset,"r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0]<=int(IMAGE_IDS[-1]): raise ValueError("SW161 evaluation HDF5 lacks fixed320 masks")
        target=contract.modal_native32(np.asarray(h5["mask"][int(IMAGE_IDS[0]):int(IMAGE_IDS[-1])+1]))
    scores={name:contract.production_metrics32(rec["labels"],target) for name,rec in frozen.items()}
    means={name:{m:float(data[m]["mean"]) for m in contract.METRICS} for name,data in scores.items()}
    paired={}
    for arm in ARMS:
        for baseline in ("source97","SW0143_T","Slot70k",*[other for other in ARMS if other!=arm]):
            for metric in contract.METRICS:
                paired[f"{arm}_minus_{baseline}_{metric}"]=_paired_bootstrap(scores[arm][metric]["per_image"],scores[baseline][metric]["per_image"])
    gate,comparisons=_registered_gate(means,paired)
    report={"experiment":train.EXPERIMENT,"status":"complete","seed_scope":[SEED],"implementation_fingerprint":implementation_fingerprint(),
      "scores":scores,"means":means,"paired_bootstrap":paired,"registered_seed1_gate":gate,"all_metric_mean_comparisons":comparisons,
      "bootstrap":{"rng":"numpy RandomState","seed":139,"replicates":10000,"common_image_indices":True},
      "ground_truth_opened_after_all_predictions_validated":True,"ground_truth_used_for_prediction":False,"ground_truth_used_for_scoring":True,
      "prediction_hashes_before_gt":hashes,"dataset":{"path":str(dataset.resolve()),"size_bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns},
      "interpretation_limits":["single seed1 supervised pilot","does not establish 70k scaling","no threshold, loss, or training-recipe sweep"]}
    output.parent.mkdir(parents=True,exist_ok=True); temp=output.with_name(output.name+f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temp.open("x",encoding="utf-8",newline="\n") as f: json.dump(report,f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.rename(temp,output); return report


def validate_score(path=SCORE_PATH, *, prediction_root=PREDICTION_ROOT, dataset=DATASET):
    record=json.loads(Path(path).read_text(encoding="utf-8"))
    if (record.get("experiment")!=train.EXPERIMENT or record.get("status")!="complete" or record.get("seed_scope")!=[SEED]
        or record.get("ground_truth_opened_after_all_predictions_validated") is not True or record.get("ground_truth_used_for_prediction") is not False
        or record.get("ground_truth_used_for_scoring") is not True): raise ValueError("SW161 score violates GT-last contract")
    scratch=Path(path).with_name(Path(path).name+f".verify-{uuid.uuid4().hex}")
    try:
        fresh=score_all(output=scratch,prediction_root=prediction_root,dataset=dataset)
        if fresh!=record: raise ValueError("SW161 score differs from recomputed fixed320 metrics")
    finally:
        try: scratch.unlink()
        except FileNotFoundError: pass
    return record


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage",choices=("predict","score"),required=True)
    parser.add_argument("--arm",choices=ARMS); parser.add_argument("--device",default="cuda:0"); args=parser.parse_args(argv)
    if args.stage=="predict":
        if args.arm is None: parser.error("--arm is required for prediction")
        result=predict(args.arm,device=args.device); stage="predict"
    else: result=score_all(); stage="score"
    print(json.dumps({"stage":stage,"status":result.get("status","prediction_complete"),"arm":args.arm,
      "ground_truth_used_for_prediction":False,"ground_truth_used_for_scoring":stage=="score"},allow_nan=False),flush=True); return 0


if __name__=="__main__": raise SystemExit(main())
