"""Fixed-320 inference and GT-last evaluation for frozen SW0167 heads."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import h5py
import numpy as np
import torch

from collaborative_test.SW_0167_structured_relation_energy import (
    feature_modes, structured_energy, train,
)
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143, geodesic_operator
from collaborative_test.SW_0161_supervised_spike_partition import cache as source_cache
from collaborative_test.SW_0162_connected_signed_partition import preflight as source_capture
from collaborative_test.SW_0164_foreground_preserving_partition import (
    predict_evaluate as sw164_eval,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 1
IMAGE_IDS = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
COUNT = int(contract.COUNT)
GRID = 32
ARMS = (*train.ARMS, "source_T_control")
PREDICTION_ROOT = HERE / "results_archive" / "predictions_20261015"
WORK_ROOT = HERE / "results_archive" / "prediction_work_20261015"
SCORE_PATH = HERE / "results_archive" / "evaluation_seed1_20261015.json"
TRAIN_ROOT = train.TRAIN_ROOT
BOOTSTRAP_SEED = 167
BOOTSTRAP_REPLICATES = 10_000
SW0164_SEED1_SCORE_SHA256 = "978a9d97658837eb19f414a7fb7dbb35d86c6135cb1213e12e63e2837fd65df6"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode()).hexdigest()


def _fingerprint_paths():
    rels = (
        "collaborative_test/SW_0167_structured_relation_energy/DESIGN_KO.md",
        "collaborative_test/SW_0167_structured_relation_energy/feature_modes.py",
        "collaborative_test/SW_0167_structured_relation_energy/structured_energy.py",
        "collaborative_test/SW_0167_structured_relation_energy/training_objective.py",
        "collaborative_test/SW_0167_structured_relation_energy/train.py",
        "collaborative_test/SW_0167_structured_relation_energy/predict_evaluate.py",
        "collaborative_test/SW_0161_supervised_spike_partition/cache.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0137_native32_source_qcc/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0162_connected_signed_partition/preflight.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/foreground_signed.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/predict_evaluate.py",
    )
    return [ROOT / rel for rel in rels]


def implementation_fingerprint():
    paths = _fingerprint_paths()
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def _out(arm, root=PREDICTION_ROOT):
    if arm not in ARMS:
        raise ValueError("unregistered SW0167 prediction arm")
    return Path(root) / f"{arm}_seed1"


def _rowdir(work, index):
    return Path(work) / "rows" / f"{index:04d}"


def _work(root=WORK_ROOT):
    return Path(root) / "shared_source_seed1"


def _references():
    qcc = sw137.validate_prediction(sw137.OUTPUT_ROOT / "seed1", seed=SEED)
    source_t = sw143.validate_arm_prediction(sw143._arm_dir("T", SEED), seed=SEED)
    if (not np.array_equal(qcc["ids"], IMAGE_IDS) or not np.array_equal(source_t["ids"], IMAGE_IDS)
            or source_t.get("arm") != "T"):
        raise ValueError("registered source-QCC/Source-T fixed320 references mismatch")
    slot = sw164_eval._references()["slot"]
    sw164 = sw164_eval.validate_prediction("foreground_induced")
    if not np.array_equal(sw164["ids"], IMAGE_IDS):
        raise ValueError("registered SW0164 fixed320 prediction IDs mismatch")
    sw164["prediction_path"] = Path(sw164["directory"]) / "predictions.npz"
    sw164["protocol_path"] = Path(sw164["directory"]) / "protocol.json"
    return {"source_qcc": qcc, "source_T": source_t, "slot": slot, "SW0164": sw164}


def prepare_prediction_inputs_cpu(*, dataset=None):
    """Validate heads, fixed references and RGB/HDF identity without reading VAL masks."""
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0167 CPU prediction admission must precede CUDA initialization")
    manifests = {}
    for arm in train.ARMS:
        manifests[arm] = train.validate_training_artifact(arm)
    first = manifests[train.ARMS[0]]["bindings"]
    second = manifests[train.ARMS[1]]["bindings"]
    for key in ("source_checkpoint_sha256", "source_manifest_sha256", "source_cache_manifest_sha256",
                "context_cache_manifest_sha256", "source_state_sha256", "preflight_report_sha256",
                "ordered_train_ids_sha256"):
        if first.get(key) != second.get(key):
            raise ValueError(f"SW0167 trained arms do not share frozen source provenance: {key}")
    # Rebind current immutable training cache roots to each final artifact.
    s_path = source_cache.CACHE_ROOT / "manifest.json"
    c_path = train.audit_train_cache.context_cache.CACHE_ROOT / "manifest.json"
    if (sha256_file(s_path) != first.get("source_cache_manifest_sha256")
            or sha256_file(c_path) != first.get("context_cache_manifest_sha256")):
        raise ValueError("SW0167 source/context cache changed after training")
    refs = _references()
    rgb_path = Path(sw130.VAL_RGB)
    rgb_meta_path = rgb_path.with_suffix(rgb_path.suffix + ".complete.json")
    meta = json.loads(rgb_meta_path.read_text(encoding="utf-8-sig"))
    expected_map = hashlib.sha256(np.arange(1320, 1640, dtype="<i8").tobytes()).hexdigest()
    if (meta.get("status") != "complete" or meta.get("cache_shape") != [320, 128, 128, 3]
            or meta.get("ids_mapping_sha256") != expected_map):
        raise ValueError("registered fixed320 validation RGB mapping changed")
    rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    if rgb.dtype != np.uint8 or tuple(rgb.shape) != (320, 128, 128, 3):
        raise ValueError("fixed320 RGB cache schema changed")
    rgb_hashes = [hashlib.sha256(np.ascontiguousarray(rgb[i]).tobytes()).hexdigest()
                  for i in range(COUNT)]
    dataset_path = Path(dataset) if dataset is not None else Path(sw130.source97.DATASET)
    if dataset_path.resolve() != Path(sw130.source97.DATASET).resolve() or not dataset_path.is_file():
        raise ValueError("scoring dataset is not the registered canonical HDF5")
    st = dataset_path.stat()
    train_path, train_manifest, _manifest, _pool, _ids, source_sha = sw130.source_contract(SEED)
    if source_sha != first["source_checkpoint_sha256"] or sha256_file(train_manifest) != first["source_manifest_sha256"]:
        raise ValueError("trained head source lineage differs from the current registered source")
    return {"manifests": manifests, "references": refs, "rgb": rgb,
            "rgb_path": rgb_path, "rgb_manifest_sha256": sha256_file(rgb_meta_path),
            "rgb_hashes": rgb_hashes, "dataset": dataset_path,
            "dataset_path": str(dataset_path.resolve()), "dataset_size_bytes": st.st_size,
            "dataset_mtime_ns": st.st_mtime_ns, "source_checkpoint_path": train_path,
            "source_manifest_path": train_manifest, "source_checkpoint_sha256": source_sha}


def _bindings(p):
    stat = p["rgb_path"].stat()
    refs = p["references"]
    return {
        "experiment": "SW0167_structured_relation_energy_prediction",
        "seed": SEED, "image_ids": IMAGE_IDS.tolist(),
        "implementation_fingerprint": implementation_fingerprint(),
        "training_manifest_sha256": {a: sha256_file(TRAIN_ROOT / f"{a}_seed1" / "manifest.json") for a in train.ARMS},
        "training_checkpoint_sha256": {a: sha256_file(TRAIN_ROOT / f"{a}_seed1" / "checkpoint.pt") for a in train.ARMS},
        "training_bindings_sha256": {a: _json_sha(p["manifests"][a]["bindings"]) for a in train.ARMS},
        "preflight_report_sha256": p["manifests"][train.ARMS[0]]["bindings"]["preflight_report_sha256"],
        "source_checkpoint_sha256": p["source_checkpoint_sha256"],
        "source_manifest_sha256": sha256_file(p["source_manifest_path"]),
        "source_state_sha256": p["manifests"][train.ARMS[0]]["bindings"]["source_state_sha256"],
        "source_state_digest_family": "SW0161 cache._state_digest (parameters, buffers, encoder stats and clip)",
        "source_T_prediction_sha256": refs["source_T"]["prediction_sha256"],
        "source_T_protocol_sha256": refs["source_T"]["protocol_sha256"],
        "source_qcc_prediction_sha256": refs["source_qcc"]["prediction_sha256"],
        "source_qcc_protocol_sha256": refs["source_qcc"]["protocol_sha256"],
        "SW0164_prediction_sha256": refs["SW0164"]["prediction_sha256"],
        "SW0164_protocol_sha256": refs["SW0164"]["protocol_sha256"],
        "slot_prediction_sha256": refs["slot"]["prediction_sha256"],
        "slot_protocol_sha256": refs["slot"]["protocol_sha256"],
        "slot_baseline_sha256": refs["slot"]["baseline_sha256"],
        "rgb_manifest_sha256": p["rgb_manifest_sha256"],
        "rgb_size_bytes": stat.st_size, "rgb_mtime_ns": stat.st_mtime_ns,
        "rgb_hashes_sha256": _json_sha(p["rgb_hashes"]),
        "dataset_path": p["dataset_path"], "dataset_size_bytes": p["dataset_size_bytes"],
        "dataset_mtime_ns": p["dataset_mtime_ns"],
        "steps": 1024, "settle": 512, "live_tail": 0, "optimizer_updates": 0,
        "feature_mode": "rho from actual B1 emitted spikes; source Q/A from same B1 actual rollout",
        "ground_truth_used_for_prediction": False,
    }


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.link(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def _load_heads(p, device):
    heads = {}
    for arm in train.ARMS:
        artifact = train.TRAIN_ROOT / f"{arm}_seed1"
        train.validate_training_artifact(arm, artifact, expected_bindings=p["manifests"][arm]["bindings"])
        payload = torch.load(artifact / "checkpoint.pt", map_location="cpu", weights_only=False)
        if payload.get("arm") != arm or payload.get("bindings") != p["manifests"][arm]["bindings"]:
            raise ValueError(f"trained {arm} checkpoint binding mismatch")
        head = structured_energy.PairRelationMLP().to(device)
        head.load_state_dict(payload["head"], strict=True)
        head.eval()
        for parameter in head.parameters(): parameter.requires_grad_(False)
        heads[arm] = head
    return heads


@torch.no_grad()
def _predict_one(found, heads, image, refs, index, gain):
    gamma = found.encode(image)
    trace, a = source_capture.capture_source_coupling(
        found.wrapped.core.kuramoto,
        lambda: late_rollout(found.wrapped, gamma, total_steps=1024, live_tail_steps=0))
    source_capture._trace_check(trace)
    source_capture._validate_actual_A(a)
    q = source_capture._actual_q(trace)
    qcc_batch = sw136._qcc_labels(trace)
    qcc = np.asarray(qcc_batch[0], dtype=np.int32)
    if qcc.shape != (32, 32) or not np.array_equal(qcc, refs["source_T"]["labels"][index]):
        raise ValueError(f"fresh native B1 Source-T control differs at image {index}")
    rho = feature_modes.component_rho(trace["component_spikes"][0, :, :, 512:])
    q0 = q[0]
    a0 = a[0]
    feature = feature_modes.features_from_rho_and_context_q(rho, q0, a0, gain)
    e0, wa = structured_energy.energy_components(q0, a0, gain)
    labels = {}
    merge_counts = {}
    for arm, head in heads.items():
        energy, _ = head(feature, e0, wa)
        result = structured_energy.solve_energy(energy.detach().cpu().numpy(), qcc,
                                                grid_shape=(32, 32))
        prediction = np.asarray(result["labels"], dtype=np.int32)
        if (not np.array_equal(prediction > 0, qcc > 0)
                or prediction.max(initial=0) > 1024):
            raise ValueError(f"{arm} readout changed fixed foreground or exceeded native capacity")
        labels[arm] = prediction
        merge_counts[arm] = int(result["merge_count"])
    q_np = q0.detach().cpu().numpy()
    a_np = a0.detach().cpu().numpy()
    zero_head = structured_energy.solve_energy(
        structured_energy.base_energy(q_np, a_np, gain), qcc, grid_shape=(32, 32))
    if not np.array_equal(zero_head["labels"], refs["SW0164"]["labels"][index]):
        raise ValueError(f"zero-residual SW0167 energy differs from archived SW0164 row {index}")
    return {"labels": labels, "control": qcc,
            "q_sha256": hashlib.sha256(np.ascontiguousarray(q_np).tobytes()).hexdigest(),
            "a_sha256": hashlib.sha256(np.ascontiguousarray(a_np).tobytes()).hexdigest(),
            "gain": float(gain), "merge_counts": merge_counts,
            "trace_sha256": {key: source_capture._tensor_sha(trace[key]) for key in source_capture.TRACE_FIELDS},
            "control_qcc": qcc}


def _read_row(work, index, image_id, rgb_sha, bindings_sha):
    folder = _rowdir(work, index)
    row_path, npz_path = folder / "row.json", folder / "row.npz"
    record = json.loads(row_path.read_text(encoding="utf-8"))
    if (record.get("index") != index or record.get("image_id") != image_id
            or record.get("rgb_sha256") != rgb_sha or record.get("bindings_sha256") != bindings_sha
            or record.get("npz_sha256") != sha256_file(npz_path)):
        raise ValueError(f"SW0167 prediction row {index} provenance/hash mismatch")
    trace_hashes = record.get("trace_sha256")
    if (not isinstance(trace_hashes, dict) or set(trace_hashes) != set(source_capture.TRACE_FIELDS)
            or any(not _is_sha256(value) for value in trace_hashes.values())
            or not _is_sha256(record.get("q_sha256"))
            or not _is_sha256(record.get("actual_a_sha256"))
            or not np.isfinite(record.get("gain", np.nan)) or record["gain"] <= 0
            or not np.isfinite(record.get("rollout_seconds", np.nan)) or record["rollout_seconds"] <= 0
            or type(record.get("cuda_peak_allocated_bytes")) is not int
            or record["cuda_peak_allocated_bytes"] < 0
            or record.get("foreground_support_exact") is not True):
        raise ValueError("SW0167 prediction row diagnostics/digests are invalid")
    with np.load(npz_path, allow_pickle=False) as row:
        expected = {"structured_hinge", "local_margin", "source_T_control"}
        if set(row.files) != expected:
            raise ValueError("SW0167 per-image archive schema changed")
        arrays = {name: np.asarray(row[name]) for name in expected}
    for arm, labels in arrays.items():
        if labels.dtype != np.int32 or labels.shape != (GRID, GRID) or labels.min() < 0:
            raise ValueError(f"invalid {arm} prediction row geometry/dtype")
        maximum = 1024 if arm in train.ARMS else 128
        if labels.max(initial=0) > maximum:
            raise ValueError(f"{arm} prediction exceeds registered label capacity")
        expected_hash = record.get("labels_sha256", {}).get(arm)
        if expected_hash != hashlib.sha256(labels.astype("<i4", copy=False).tobytes()).hexdigest():
            raise ValueError(f"{arm} row label digest mismatch")
    if not np.array_equal(arrays["structured_hinge"] > 0, arrays["source_T_control"] > 0):
        raise ValueError("structured-hinge readout changed source foreground membership")
    if not np.array_equal(arrays["local_margin"] > 0, arrays["source_T_control"] > 0):
        raise ValueError("local-margin readout changed source foreground membership")
    return arrays, record


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _ledger_sha(work):
    records = []
    for index in range(COUNT):
        path = _rowdir(work, index) / "row.json"
        if not path.is_file(): raise ValueError("SW0167 fixed320 prediction ledger is incomplete")
        records.append(sha256_file(path))
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


@torch.no_grad()
def predict_all(*, device="cuda:0", prediction_root=PREDICTION_ROOT,
                work_root=WORK_ROOT, prepared=None):
    p = prepared or prepare_prediction_inputs_cpu()
    bindings = _bindings(p); bindings_sha = _json_sha(bindings)
    work = _work(work_root); work.mkdir(parents=True, exist_ok=True)
    binding_path = work / "bindings.json"
    wrapper = {"bindings": bindings, "bindings_sha256": bindings_sha}
    if binding_path.exists():
        if json.loads(binding_path.read_text(encoding="utf-8")) != wrapper:
            raise ValueError("existing SW0167 row work has different bindings; preserving it")
    else:
        _atomic_json(binding_path, wrapper)
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0167 fixed320 inference requires the owner-leased CUDA device")
    torch.cuda.set_device(device)
    from collaborative_test.SW_0135_native32_spike_binding.foundation import load_native32_foundation
    found = load_native32_foundation(SEED, device, verify_rgb_assets=False)
    found.encoder.eval(); found.wrapped.eval()
    geodesic_operator.install_operator(found.wrapped.core.graph_generator, "T")
    if getattr(found.wrapped.core, "kuramoto_backend", None) != "factorized":
        raise ValueError("SW0167 inference requires registered factorized backend")
    before = source_cache._state_digest(found)
    if before != bindings["source_state_sha256"]:
        raise ValueError("source foundation differs from immutable SW0161 cache lineage")
    heads = _load_heads(p, device)
    # The registered checkpoint stores the learned native graph gain; it is
    # frozen by construction and checked against the gain receipt/cache rows.
    gain = float(torch.exp(found.wrapped.core.graph_generator.log_coupling_gain.detach()).cpu())
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("native source gain is nonfinite/nonpositive")
    for index, image_id in enumerate(IMAGE_IDS.tolist()):
        folder = _rowdir(work, index)
        if folder.exists():
            _read_row(work, index, image_id, p["rgb_hashes"][index], bindings_sha)
            continue
        image = torch.from_numpy(np.ascontiguousarray(p["rgb"][index:index+1])).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        row = _predict_one(found, heads, image, p["references"], index, gain)
        labels = {**row["labels"], "source_T_control": row["control"]}
        stage = folder.with_name(folder.name + f".stage-{uuid.uuid4().hex}")
        stage.mkdir(parents=True, exist_ok=False)
        try:
            npz_path = stage / "row.npz"
            with npz_path.open("xb") as stream:
                np.savez_compressed(stream, **labels)
                stream.flush(); os.fsync(stream.fileno())
            label_hashes = {name: hashlib.sha256(value.astype("<i4", copy=False).tobytes()).hexdigest()
                            for name, value in labels.items()}
            record = {"index": index, "image_id": int(image_id),
                "rgb_sha256": p["rgb_hashes"][index], "bindings_sha256": bindings_sha,
                "labels_sha256": label_hashes, "q_sha256": row["q_sha256"],
                "actual_a_sha256": row["a_sha256"], "gain": row["gain"],
                "merge_counts": row["merge_counts"], "trace_sha256": row["trace_sha256"],
                "foreground_support_exact": True,
                "rollout_seconds": float(time.perf_counter()-started),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "npz_sha256": sha256_file(npz_path)}
            with (stage / "row.json").open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            os.rename(stage, folder)
        except BaseException:
            raise
        del image, row, labels
        if (index + 1) % 40 == 0:
            print(json.dumps({"stage": "SW0167_prediction", "done": index+1, "count": COUNT}), flush=True)
    after = source_cache._state_digest(found)
    if after != before:
        raise RuntimeError("frozen SW0167 source foundation changed during fixed320 inference")
    state_path = work / "source_state.json"
    state = {"source_state_sha256_before": before, "source_state_sha256_after": after,
             "source_state_digest_family": "SW0161 cache._state_digest", "source_unchanged": True,
             "bindings_sha256": bindings_sha, "row_commits_sha256": _ledger_sha(work)}
    if state_path.exists():
        if json.loads(state_path.read_text(encoding="utf-8")) != state:
            raise ValueError("existing source immutability receipt differs")
    else:
        _atomic_json(state_path, state)
    return publish_predictions(prediction_root=prediction_root, work_root=work_root, prepared=p)


def publish_predictions(*, prediction_root=PREDICTION_ROOT, work_root=WORK_ROOT, prepared=None):
    p = prepared or prepare_prediction_inputs_cpu()
    bindings = _bindings(p); bsha = _json_sha(bindings); work = _work(work_root)
    state = json.loads((work / "source_state.json").read_text(encoding="utf-8"))
    if (state.get("source_unchanged") is not True
            or state.get("source_state_sha256_before") != bindings["source_state_sha256"]
            or state.get("source_state_sha256_after") != bindings["source_state_sha256"]
            or state.get("bindings_sha256") != bsha or state.get("row_commits_sha256") != _ledger_sha(work)):
        raise ValueError("SW0167 immutable-source/row ledger receipt invalid")
    arrays = {arm: [] for arm in ARMS}
    rows = []
    for i, image_id in enumerate(IMAGE_IDS.tolist()):
        values, rec = _read_row(work, i, image_id, p["rgb_hashes"][i], bsha)
        if not np.array_equal(values["source_T_control"], p["references"]["source_T"]["labels"][i]):
            raise ValueError(f"fresh Source-T control replay mismatch at image {i}")
        for arm in ARMS: arrays[arm].append(values[arm])
        rows.append(rec)
    arrays = {arm: np.stack(value).astype(np.int32) for arm, value in arrays.items()}
    led = _ledger_sha(work)
    for arm in ARMS:
        out = _out(arm, prediction_root)
        if out.exists():
            validate_prediction(arm, prediction_root=prediction_root, work_root=work_root, prepared=p)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        stage = out.with_name(out.name + f".stage-{uuid.uuid4().hex}")
        stage.mkdir(parents=True, exist_ok=False)
        try:
            with (stage / "predictions.npz").open("xb") as stream:
                np.savez_compressed(stream, image_ids=IMAGE_IDS, labels=arrays[arm])
                stream.flush(); os.fsync(stream.fileno())
            prediction_sha = sha256_file(stage / "predictions.npz")
            protocol = {"experiment": "SW0167_structured_relation_energy_prediction",
                "status": "complete", "arm": arm, "seed": SEED,
                "image_ids": IMAGE_IDS.tolist(), "count": COUNT,
                "grid_size": [GRID, GRID], "prediction_shape": [COUNT, GRID, GRID],
                "label_capacity": 128 if arm == "source_T_control" else 1024,
                "implementation_fingerprint": implementation_fingerprint(),
                "bindings": bindings, "bindings_sha256": bsha,
                "prediction_sha256": prediction_sha, "row_commits_sha256": led,
                "source_state_receipt_sha256": sha256_file(work / "source_state.json"),
                "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
                "steps": 1024, "settle": 512, "live_tail": 0,
                "fresh_source_T_replay_exact": True}
            with (stage / "protocol.json").open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(protocol, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            os.rename(stage, out)
        except BaseException:
            raise
    for arm in ARMS:
        validate_prediction(arm, prediction_root=prediction_root, work_root=work_root, prepared=p)
    return {"status": "prediction_complete", "arms": list(ARMS), "count": COUNT}


def validate_prediction(arm, *, prediction_root=PREDICTION_ROOT, work_root=WORK_ROOT, prepared=None):
    if arm not in ARMS:
        raise ValueError("unregistered SW0167 prediction arm")
    p = prepared or prepare_prediction_inputs_cpu()
    bindings = _bindings(p); bsha = _json_sha(bindings); work = _work(work_root); out = _out(arm, prediction_root)
    state = json.loads((work / "source_state.json").read_text(encoding="utf-8"))
    if (state.get("source_unchanged") is not True
            or state.get("source_state_sha256_before") != bindings["source_state_sha256"]
            or state.get("source_state_sha256_after") != bindings["source_state_sha256"]
            or state.get("bindings_sha256") != bsha or state.get("row_commits_sha256") != _ledger_sha(work)):
        raise ValueError("SW0167 prediction source-state receipt invalid")
    proto = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    npz_path = out / "predictions.npz"
    expected = {"experiment": "SW0167_structured_relation_energy_prediction", "status": "complete",
        "arm": arm, "seed": SEED, "image_ids": IMAGE_IDS.tolist(), "count": COUNT,
        "grid_size": [GRID, GRID], "prediction_shape": [COUNT, GRID, GRID],
        "label_capacity": 128 if arm == "source_T_control" else 1024,
        "implementation_fingerprint": implementation_fingerprint(), "bindings": bindings,
        "bindings_sha256": bsha, "row_commits_sha256": state["row_commits_sha256"],
        "source_state_receipt_sha256": sha256_file(work / "source_state.json"),
        "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
        "steps": 1024, "settle": 512, "live_tail": 0, "fresh_source_T_replay_exact": True}
    if any(proto.get(key) != value for key, value in expected.items()) or proto.get("prediction_sha256") != sha256_file(npz_path):
        raise ValueError("SW0167 prediction protocol/bindings/hash mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        ids = contract.require_fixed_image_ids(archive["image_ids"], name="SW0167 fixed320 IDs")
        labels = np.asarray(archive["labels"])
    cap = 128 if arm == "source_T_control" else 1024
    if (labels.shape != (COUNT, GRID, GRID) or labels.dtype != np.int32
            or labels.min() < 0 or labels.max(initial=0) > cap):
        raise ValueError("SW0167 prediction label dtype/shape/capacity invalid")
    for i, image_id in enumerate(IMAGE_IDS.tolist()):
        row, _ = _read_row(work, i, image_id, p["rgb_hashes"][i], bsha)
        if not np.array_equal(row[arm], labels[i]):
            raise ValueError("published SW0167 labels differ from the frozen per-image row")
        if arm == "source_T_control" and not np.array_equal(labels[i], p["references"]["source_T"]["labels"][i]):
            raise ValueError("published fresh Source-T control differs from archive")
    return {"arm": arm, "ids": ids, "labels": labels.astype(np.int64),
            "prediction_sha256": proto["prediction_sha256"],
            "protocol_sha256": sha256_file(out / "protocol.json"),
            "protocol": proto, "directory": out}


def _bootstrap(candidate, baseline):
    d = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if d.shape != (COUNT,) or not np.isfinite(d).all():
        raise ValueError("paired score vectors must be finite fixed320 arrays")
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    indices = rng.randint(0, COUNT, size=(BOOTSTRAP_REPLICATES, COUNT))
    values = d[indices].mean(axis=1)
    return {"mean": float(d.mean()), "ci95": [float(x) for x in np.percentile(values, [2.5, 97.5])],
            "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}


def _metrics(labels_by_arm, gt):
    return {arm: contract.production_metrics32(labels, gt) for arm, labels in labels_by_arm.items()}


def _score_comparisons(scores, baseline_scores):
    """Compute preregistered paired-FG-ARI comparisons and fixed-recipe gate."""
    scores["SW0164_seed1"] = baseline_scores
    means = {name: {metric: float(result[metric]["mean"]) for metric in contract.METRICS}
             for name, result in scores.items()}
    paired = {}
    vectors = {
        "source_T": scores["source_T_control"]["fg_ari"]["per_image"],
        "local_margin": scores["local_margin"]["fg_ari"]["per_image"],
        "SW0164_seed1": scores["SW0164_seed1"]["fg_ari"]["per_image"],
    }
    for arm in train.ARMS:
        for baseline, vector in vectors.items():
            paired[f"{arm}_minus_{baseline}_fg_ari"] = _bootstrap(
                scores[arm]["fg_ari"]["per_image"], vector)
    candidate = means["structured_hinge"]
    fg_exact = np.array_equal(
        scores["structured_hinge"]["foreground_iou"]["per_image"],
        scores["SW0164_seed1"]["foreground_iou"]["per_image"])
    gate = {
        "fg_mean_above_SW0164_seed1": candidate["fg_ari"] > means["SW0164_seed1"]["fg_ari"],
        "fg_paired_ci_lower_above_zero_vs_SW0164":
            paired["structured_hinge_minus_SW0164_seed1_fg_ari"]["ci95"][0] > 0,
        "fg_paired_ci_lower_above_zero_vs_local_margin":
            paired["structured_hinge_minus_local_margin_fg_ari"]["ci95"][0] > 0,
        "object_iou_not_below_SW0164_seed1":
            candidate["matched_object_iou"] >= means["SW0164_seed1"]["matched_object_iou"],
        "foreground_iou_exact_SW0164_seed1": fg_exact,
    }
    gate["seed_expansion_gate_passed"] = all(gate.values())
    return means, paired, gate, fg_exact


def score_all(*, output=SCORE_PATH, prediction_root=PREDICTION_ROOT,
              work_root=WORK_ROOT, prepared=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    p = prepared or prepare_prediction_inputs_cpu()
    # Freeze and validate every prediction and reference before opening the
    # validation mask dataset for the first time.
    frozen = {arm: validate_prediction(arm, prediction_root=prediction_root,
                                        work_root=work_root, prepared=p) for arm in ARMS}
    refs = p["references"]
    ref_records = {"source_T": refs["source_T"], "SW0164_seed1": refs["SW0164"],
                   "Slot70k": refs["slot"]}
    archive_hashes = {}
    archive_paths = {}
    for name, record in ref_records.items():
        path = Path(record.get("prediction_path", record.get("npz_path", record.get("archive_path"))))
        protocol_path = Path(record["protocol_path"])
        pred_sha, proto_sha = record["prediction_sha256"], record["protocol_sha256"]
        if sha256_file(path) != pred_sha or sha256_file(protocol_path) != proto_sha:
            raise ValueError(f"immutable {name} reference changed before GT")
        archive_paths[name] = path; archive_hashes[name] = {"prediction_sha256": pred_sha,
                                                            "protocol_sha256": proto_sha}
    prediction_hashes = {arm: {"prediction_sha256": frozen[arm]["prediction_sha256"],
                               "protocol_sha256": frozen[arm]["protocol_sha256"]} for arm in ARMS}
    # The registered SW0164 score is itself pinned. Its labels and metrics are
    # independently checked below against this scorer's production metrics,
    # avoiding a second full legacy admission/GT pass.
    baseline_path = Path(sw164_eval.SCORE_PATH)
    baseline_sha = sha256_file(baseline_path)
    if baseline_sha != SW0164_SEED1_SCORE_SHA256:
        raise ValueError("registered SW0164 seed-1 comparator score changed")
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    baseline_scores = baseline_report["scores"]["foreground_induced"]
    dataset = p["dataset"]
    st = dataset.stat()
    if st.st_size != p["dataset_size_bytes"] or st.st_mtime_ns != p["dataset_mtime_ns"]:
        raise ValueError("canonical HDF5 identity changed before GT access")
    with h5py.File(dataset, "r") as hdf5:
        if "mask" not in hdf5 or hdf5["mask"].shape[0] <= int(IMAGE_IDS[-1]):
            raise ValueError("canonical validation mask dataset missing")
        raw_masks = np.asarray(hdf5["mask"][int(IMAGE_IDS[0]):int(IMAGE_IDS[-1])+1])
    gt = contract.modal_native32(raw_masks)
    label_sets = {arm: frozen[arm]["labels"] for arm in ARMS}
    label_sets.update({"Source-T": refs["source_T"]["labels"],
                       "SW0164_seed1": refs["SW0164"]["labels"],
                       "Slot70k": refs["slot"]["labels"]})
    scores = _metrics(label_sets, gt)
    for metric in contract.METRICS:
        if not np.array_equal(scores["SW0164_seed1"][metric]["per_image"],
                              np.asarray(baseline_scores[metric]["per_image"])):
            raise ValueError(f"SW0164 score disagrees with its frozen prediction: {metric}")
        if scores["SW0164_seed1"][metric]["mean"] != float(baseline_scores[metric]["mean"]):
            raise ValueError(f"SW0164 score mean disagrees with its frozen prediction: {metric}")
    # The fresh Source-T control must exactly replay the archived Source-T
    # labels and therefore its fixed-foreground IoU vector.
    if not np.array_equal(label_sets["source_T_control"], label_sets["Source-T"]):
        raise ValueError("fresh Source-T control is not an exact fixed320 replay")
    for arm in train.ARMS:
        if not np.array_equal(scores[arm]["foreground_iou"]["per_image"],
                              scores["Source-T"]["foreground_iou"]["per_image"]):
            raise ValueError(f"{arm} foreground IoU differs from exact fixed-support Source-T")
    means, paired, gate, fg_exact = _score_comparisons(scores, baseline_scores)
    # Recheck all frozen predictions/references and dataset stat after GT use.
    for arm in ARMS:
        updated = validate_prediction(arm, prediction_root=prediction_root,
                                      work_root=work_root, prepared=p)
        if updated["prediction_sha256"] != prediction_hashes[arm]["prediction_sha256"]:
            raise ValueError("SW0167 prediction changed during scoring")
    for name, record in ref_records.items():
        if sha256_file(archive_paths[name]) != archive_hashes[name]["prediction_sha256"] or sha256_file(record["protocol_path"]) != archive_hashes[name]["protocol_sha256"]:
            raise ValueError(f"immutable {name} reference changed during scoring")
    if sha256_file(sw164_eval.SCORE_PATH) != baseline_sha:
        raise ValueError("immutable SW0164 seed-1 score changed during scoring")
    st_after = dataset.stat()
    if st_after.st_size != p["dataset_size_bytes"] or st_after.st_mtime_ns != p["dataset_mtime_ns"]:
        raise ValueError("canonical HDF5 identity changed during scoring")
    report = {"experiment": "SW0167_structured_relation_energy_score", "status": "complete",
        "seed_scope": [SEED], "implementation_fingerprint": implementation_fingerprint(),
        "scores": scores, "means": means, "paired_bootstrap": paired,
        "seed1_expansion_gate": gate, "SW0164_seed1_score_sha256": baseline_sha,
        "SW0164_seed1_means": {m: float(baseline_scores[m]["mean"]) for m in contract.METRICS},
        "prediction_hashes_before_gt": prediction_hashes,
        "reference_hashes_before_gt": archive_hashes,
        "ground_truth_opened_after_all_predictions_validated": True,
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
        "dataset": {"path": p["dataset_path"], "size_bytes": p["dataset_size_bytes"],
                    "mtime_ns": p["dataset_mtime_ns"]},
        "bootstrap": {"rng": "numpy RandomState", "seed": BOOTSTRAP_SEED,
                      "replicates": BOOTSTRAP_REPLICATES, "common_paired_image_indices": True},
        "source_T_foreground_iou_exact": fg_exact,
        "interpretation": "fixed seed1 supervised relation-energy readout; no automatic seed expansion"}
    _atomic_json(output, report)
    return report


def validate_score(path=SCORE_PATH, *, prediction_root=PREDICTION_ROOT, work_root=WORK_ROOT, prepared=None):
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    p = prepared or prepare_prediction_inputs_cpu()
    preds = {arm: validate_prediction(arm, prediction_root=prediction_root,
        work_root=work_root, prepared=p) for arm in ARMS}
    expected_hashes = {arm: {"prediction_sha256": preds[arm]["prediction_sha256"],
                             "protocol_sha256": preds[arm]["protocol_sha256"]} for arm in ARMS}
    if (report.get("experiment") != "SW0167_structured_relation_energy_score"
            or report.get("status") != "complete" or report.get("seed_scope") != [SEED]
            or report.get("implementation_fingerprint") != implementation_fingerprint()
            or report.get("prediction_hashes_before_gt") != expected_hashes
            or report.get("ground_truth_used_for_prediction") is not False
            or report.get("ground_truth_used_for_scoring") is not True):
        raise ValueError("SW0167 score provenance/schema mismatch")
    for name, reference in (("source_T", p["references"]["source_T"]),
                            ("SW0164_seed1", p["references"]["SW0164"]),
                            ("Slot70k", p["references"]["slot"])):
        reference_path = Path(reference.get("prediction_path", reference.get("npz_path", reference.get("archive_path"))))
        if (sha256_file(reference_path) != report.get("reference_hashes_before_gt", {}).get(name, {}).get("prediction_sha256")
                or sha256_file(reference["protocol_path"]) != report["reference_hashes_before_gt"][name]["protocol_sha256"]):
            raise ValueError("score references changed")
    baseline_path = Path(sw164_eval.SCORE_PATH)
    baseline_sha = sha256_file(baseline_path)
    if (baseline_sha != SW0164_SEED1_SCORE_SHA256
            or report.get("SW0164_seed1_score_sha256") != baseline_sha):
        raise ValueError("SW0164 baseline score binding changed")
    # Recompute every score field from the immutable prediction labels and
    # canonical GT, including paired intervals and the registered gate.
    scratch = path.with_name(path.name + f".verify-{uuid.uuid4().hex}.json")
    try:
        fresh = score_all(output=scratch, prediction_root=prediction_root,
                          work_root=work_root, prepared=p)
        keys = ("scores", "means", "paired_bootstrap", "seed1_expansion_gate",
                "SW0164_seed1_score_sha256", "SW0164_seed1_means",
                "prediction_hashes_before_gt", "reference_hashes_before_gt",
                "ground_truth_opened_after_all_predictions_validated",
                "ground_truth_used_for_prediction", "ground_truth_used_for_scoring",
                "dataset", "bootstrap")
        if any(report.get(key) != fresh.get(key) for key in keys):
            raise ValueError("independent SW0167 score/gate recomputation mismatch")
    finally:
        if scratch.exists(): scratch.unlink()
    return {"status": "valid", "score_sha256": sha256_file(path),
            "seed1_expansion_gate": report["seed1_expansion_gate"]}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("predict", "score"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "predict":
        result = predict_all(device=args.device)
    else:
        result = score_all(output=args.output or SCORE_PATH)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
