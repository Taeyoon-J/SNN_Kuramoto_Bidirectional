"""Fixed-320 source-T prediction and GT-last SW0162 scoring."""
from __future__ import annotations
import hashlib, json, os, sys, uuid, time
from pathlib import Path
try:
    import resource
except ImportError:  # Windows CPU-only review
    resource = None

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"), str(ROOT / "snn_kuramoto_bidirectional")]
import h5py
import numpy as np
import torch
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0135_native32_spike_binding.foundation import load_native32_foundation
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143, geodesic_operator
from collaborative_test.SW_0162_connected_signed_partition import connected_signed, preflight, preflight_queue

SEED = 1
IMAGE_IDS = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
COUNT, GRID = contract.COUNT, contract.GRID
PREDICTION_ROOT = HERE / "results_archive" / "predictions_20261014"
WORK_ROOT = HERE / "results_archive" / "prediction_work_20261014"
SCORE_PATH = HERE / "results_archive" / "evaluation_seed1_20261014.json"
DATASET = Path(sw130.source97.DATASET)
SLOT_ROOT = ROOT / "trained_models/SW0092_slot_our70000_eval"
SLOT_BASELINE = ROOT / "collaborative_test/SW_0135_native32_spike_binding/results_archive/slot70k_native32_baseline.json"
SLOT_BASELINE_SHA256 = "4dc12cb54c3114375b8428cd250c013ff94015c4826915751f2fcb0a2204d753"
BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES = 162, 10_000
ARMS = ("connected_signed", "source_T_control")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def implementation_fingerprint():
    result = {}
    result.update(preflight.implementation_fingerprint())
    result.update(sw143.implementation_fingerprint())
    result.update(sw137.implementation_fingerprint())
    for path in (HERE / "predict_evaluate.py", HERE / "protocol.json", HERE / "DESIGN_KO.md",
                 HERE / "connected_signed.py", Path(contract.__file__)):
        if not path.is_file():
            raise FileNotFoundError(f"SW0162 dependency missing: {path}")
        result[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return result


def _output_dir(arm, root=PREDICTION_ROOT):
    if arm not in ARMS:
        raise ValueError("unregistered SW0162 prediction arm")
    return Path(root) / f"{arm}_seed1"


def _work_dir(root=WORK_ROOT):
    return Path(root) / "connected_signed_seed1"


def _peak_rss_bytes():
    if resource is None:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if sys.platform.startswith("linux") else value


def _preflight_record():
    report = json.loads(preflight.OUTPUT.read_text(encoding="utf-8"))
    if not preflight_queue.validate_report(preflight.OUTPUT,
            expected_fingerprint=preflight.implementation_fingerprint(),
            expected_source=preflight_queue.source_identity()):
        raise ValueError("SW0162 actual source8 preflight is missing or stale")
    return report, sha256_file(preflight.OUTPUT)


def _source_references():
    source = sw137.validate_prediction(sw137.OUTPUT_ROOT / "seed1", seed=SEED)
    tpred = sw143.validate_arm_prediction(sw143._arm_dir("T", SEED), seed=SEED)
    tpred["archive_path"] = tpred["npz_path"]
    if not np.array_equal(source["ids"], IMAGE_IDS) or not np.array_equal(tpred["ids"], IMAGE_IDS):
        raise ValueError("source QCC/T IDs differ from the registered fixed320 sequence")
    if tpred["arm"] != "T":
        raise ValueError("registered source-T archive has the wrong arm")
    slot_dir = SLOT_ROOT / "seed1_epoch10"
    slot_npz, slot_protocol_path = slot_dir / "predictions.npz", slot_dir / "protocol.json"
    baseline_sha = sha256_file(SLOT_BASELINE)
    if baseline_sha != SLOT_BASELINE_SHA256:
        raise ValueError("registered Slot baseline JSON hash changed")
    slot_protocol = json.loads(slot_protocol_path.read_text(encoding="utf-8"))
    slot_sha = contract.validate_slot_protocol(slot_protocol, seed=SEED, prediction_path=slot_npz)
    with np.load(slot_npz, allow_pickle=False) as saved:
        slot_ids = contract.require_fixed_image_ids(saved["image_ids"], name="Slot IDs")
        slot_labels = contract.slot_native_pixels_to32(saved["labels"])
    if not np.array_equal(slot_ids, IMAGE_IDS):
        raise ValueError("Slot labels do not match fixed320 IDs")
    slot = {"ids": slot_ids, "labels": slot_labels, "prediction_sha256": slot_sha,
        "protocol_sha256": sha256_file(slot_protocol_path), "protocol": slot_protocol,
        "prediction_path": slot_npz, "archive_path": slot_npz, "protocol_path": slot_protocol_path,
        "baseline_sha256": baseline_sha}
    return {"source_qcc": source, "source_T": tpred, "slot": slot}


def prepare_prediction_inputs_cpu(*, dataset=DATASET):
    """Validate source/reference/RGB/HDF identity without requiring preflight completion or reading GT."""
    refs = _source_references()
    rgb_path = Path(sw130.VAL_RGB)
    rgb_meta_path = rgb_path.with_suffix(rgb_path.suffix + ".complete.json")
    if not rgb_path.is_file() or not rgb_meta_path.is_file():
        raise FileNotFoundError("registered fixed320 RGB cache/metadata missing")
    rgb_meta = json.loads(rgb_meta_path.read_text(encoding="utf-8-sig"))
    expected_id_sha = hashlib.sha256(np.arange(1320, 1640, dtype="<i8").tobytes()).hexdigest()
    if (rgb_meta.get("status") != "complete" or rgb_meta.get("cache_shape") != [320, 128, 128, 3]
            or rgb_meta.get("ids_mapping_sha256") != expected_id_sha):
        raise ValueError("registered validation RGB metadata changed")
    rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    if tuple(rgb.shape) != (320, 128, 128, 3) or rgb.dtype != np.uint8:
        raise ValueError("fixed320 source RGB cache schema mismatch")
    rgb_hashes = [hashlib.sha256(np.asarray(rgb[i]).tobytes()).hexdigest() for i in range(COUNT)]
    dataset = Path(dataset)
    if dataset.resolve() != Path(DATASET).resolve() or not dataset.is_file():
        raise ValueError("scoring path differs from canonical HDF5 dataset")
    stat = dataset.stat()
    return {"references": refs,
        "rgb": rgb, "rgb_path": rgb_path, "rgb_manifest_sha256": sha256_file(rgb_meta_path),
        "rgb_hashes": rgb_hashes, "dataset": dataset, "dataset_path": str(dataset.resolve()),
        "dataset_size_bytes": stat.st_size, "dataset_mtime_ns": stat.st_mtime_ns}


def prepare_prediction_cpu(*, dataset=DATASET):
    """Require the validated eight-TRAIN source preflight on top of CPU input checks."""
    prepared = prepare_prediction_inputs_cpu(dataset=dataset)
    pref, pref_sha = _preflight_record()
    prepared.update(preflight=pref, preflight_sha256=pref_sha)
    return prepared


def _prediction_bindings(prepared):
    refs, source = prepared["references"], prepared["preflight"]["source"]
    rgb_stat = prepared["rgb_path"].stat()
    return {"experiment": "SW0162_connected_signed_partition", "seed": SEED,
        "implementation_fingerprint": implementation_fingerprint(),
        "preflight_sha256": prepared["preflight_sha256"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_manifest_sha256": source["manifest_sha256"],
        "source_model_state_sha256": source["source_state_sha256_before"],
        "source_training_ids_sha256": source["ordered_train_ids_sha256"],
        "source_T_prediction_sha256": refs["source_T"]["prediction_sha256"],
        "source_T_protocol_sha256": refs["source_T"]["protocol_sha256"],
        "source_qcc_prediction_sha256": refs["source_qcc"]["prediction_sha256"],
        "source_qcc_protocol_sha256": refs["source_qcc"]["protocol_sha256"],
        "slot_prediction_sha256": refs["slot"]["prediction_sha256"],
        "slot_protocol_sha256": refs["slot"]["protocol_sha256"],
        "slot_baseline_sha256": refs["slot"]["baseline_sha256"],
        "fixed_validation_rgb_manifest_sha256": prepared["rgb_manifest_sha256"],
        "fixed_validation_rgb_size": rgb_stat.st_size, "fixed_validation_rgb_mtime_ns": rgb_stat.st_mtime_ns,
        "image_ids_sha256": hashlib.sha256(IMAGE_IDS.astype("<i8").tobytes()).hexdigest(),
        "dataset_path": prepared["dataset_path"], "dataset_size_bytes": prepared["dataset_size_bytes"],
        "dataset_mtime_ns": prepared["dataset_mtime_ns"],
        "steps": 1024, "settle": 512, "live_tail": 0, "optimizer_updates": 0,
        "ground_truth_used_for_prediction": False}


def _write_json_once(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())


def _sha_tensor(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _row_dir(work, index):
    return Path(work) / "rows" / f"{index:04d}"


def _rename_new_directory(stage, target):
    if target.exists():
        raise FileExistsError(f"preserving existing SW0162 directory: {target}")
    for attempt in range(8):
        try:
            os.rename(stage, target)
            return
        except PermissionError:
            if target.exists() or attempt == 7:
                raise
            time.sleep(0.025 * (attempt + 1))


def _write_row(work, i, image_id, rgb_sha, bind_sha, labels, control_labels,
               q_sha, a_sha, gain, merges, trace_shas, diagnostics):
    target = _row_dir(work, i)
    if target.exists():
        raise FileExistsError(f"preserving existing SW0162 row {i}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(target.name + f".staging-{os.getpid()}-{uuid.uuid4().hex}")
    stage.mkdir()
    labels = np.asarray(labels, dtype=np.int32)
    with (stage / "row.npz").open("xb") as stream:
        np.savez_compressed(stream, index=np.asarray(i, np.int64), image_id=np.asarray(image_id, np.int64),
                            labels=labels, control_labels=np.asarray(control_labels, dtype=np.int32))
        stream.flush(); os.fsync(stream.fileno())
    row = {"index": i, "image_id": image_id, "rgb_sha256": rgb_sha,
        "bindings_sha256": bind_sha, "labels_sha256": hashlib.sha256(labels.astype("<i4").tobytes()).hexdigest(),
        "q_sha256": q_sha, "actual_A_sha256": a_sha, "gain": float(gain),
        "merge_count": int(merges), "npz_sha256": sha256_file(stage / "row.npz"),
        "trace_sha256": trace_shas,
        "control_labels_sha256": hashlib.sha256(np.asarray(control_labels, dtype="<i4").tobytes()).hexdigest(),
        **diagnostics}
    _write_json_once(stage / "row.json", row)
    _rename_new_directory(stage, target)


def _load_row(work, i, image_id, rgb_sha, bind_sha):
    folder = _row_dir(work, i)
    row = json.loads((folder / "row.json").read_text(encoding="utf-8"))
    archive = folder / "row.npz"
    expected = {"index": i, "image_id": image_id, "rgb_sha256": rgb_sha, "bindings_sha256": bind_sha}
    if any(row.get(key) != value for key, value in expected.items()) or row.get("npz_sha256") != sha256_file(archive):
        raise ValueError(f"SW0162 row {i} has invalid identity/hash")
    with np.load(archive, allow_pickle=False) as saved:
        labels = np.asarray(saved["labels"])
        control = np.asarray(saved["control_labels"])
        if (int(saved["index"]) != i or int(saved["image_id"]) != image_id
                or labels.shape != (GRID, GRID) or labels.dtype != np.int32
                or np.any(labels < 0) or np.unique(labels).size > 129
                or control.shape != (GRID, GRID) or control.dtype != np.int32
                or np.any(control < 0) or np.unique(control).size > 129):
            raise ValueError(f"SW0162 row {i} violates native32 variable-K contract")
    if row.get("labels_sha256") != hashlib.sha256(labels.astype("<i4").tobytes()).hexdigest():
        raise ValueError(f"SW0162 row {i} label hash mismatch")
    if row.get("control_labels_sha256") != hashlib.sha256(control.astype("<i4").tobytes()).hexdigest():
        raise ValueError(f"SW0162 row {i} source-QCC label hash mismatch")
    hashes = [row.get("q_sha256"), row.get("actual_A_sha256"), *row.get("trace_sha256", {}).values()]
    if (not isinstance(row.get("trace_sha256"), dict)
            or set(row["trace_sha256"]) != set(preflight.TRACE_FIELDS)
            or any(not isinstance(value, str) or len(value) != 64 for value in hashes)):
        raise ValueError(f"SW0162 row {i} trace/Q/A digests are incomplete")
    if (not np.isfinite(row.get("gain", np.nan)) or row["gain"] <= 0
            or type(row.get("merge_count")) is not int or row["merge_count"] < 0):
        raise ValueError(f"SW0162 row {i} has invalid solver provenance")
    sizes = row.get("raw_group_sizes")
    if (not isinstance(sizes, list) or not sizes or any(type(x) is not int or x <= 0 for x in sizes)
            or sum(sizes) != 1024 or type(row.get("discarded_mass")) is not int
            or row["discarded_mass"] != sum(x for x in sizes if x < 8)
            or type(row.get("retained_group_count")) is not int or row["retained_group_count"] < 1
            or not 0 <= row.get("candidate_foreground_fraction", -1) <= 1
            or not np.isfinite(row.get("rollout_seconds", np.nan)) or row["rollout_seconds"] < 0
            or not np.isfinite(row.get("solver_seconds", np.nan)) or row["solver_seconds"] < 0):
        raise ValueError(f"SW0162 row {i} diagnostic metadata invalid")
    rss = row.get("peak_rss_bytes")
    if rss is not None and (type(rss) is not int or rss <= 0):
        raise ValueError(f"SW0162 row {i} RSS diagnostic invalid")
    return labels


def _load_control_row(work, i, image_id, rgb_sha, bind_sha):
    _load_row(work, i, image_id, rgb_sha, bind_sha)
    folder = _row_dir(work, i)
    row = json.loads((folder / "row.json").read_text(encoding="utf-8"))
    with np.load(folder / "row.npz", allow_pickle=False) as saved:
        labels = np.asarray(saved["control_labels"])
    if row.get("control_labels_sha256") != hashlib.sha256(labels.astype("<i4").tobytes()).hexdigest():
        raise ValueError(f"SW0162 control row {i} label hash mismatch")
    return labels


def _row_ledger_sha(work):
    row_hashes = []
    for i in range(COUNT):
        path = _row_dir(work, i) / "row.json"
        if not path.is_file():
            raise ValueError(f"SW0162 row {i} is not committed")
        row_hashes.append(sha256_file(path))
    return hashlib.sha256(json.dumps(row_hashes, separators=(",", ":")).encode()).hexdigest()


def _validate_trace(trace):
    expected = {"theta": (1, 1024, 1024, 4), "membrane": (1, 1024, 1024),
        "spikes": (1, 1024, 1024), "component_membrane": (1, 4, 1024, 1024),
        "component_spikes": (1, 4, 1024, 1024), "component_gates": (1, 4, 1024, 1024)}
    for key, shape in expected.items():
        tensor = trace.get(key)
        if not torch.is_tensor(tensor) or tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
            raise ValueError(f"SW0162 actual source T trace invalid at {key}")


def _publish_stage(stage, output):
    _rename_new_directory(stage, output)


def _prediction_diagnostics(work):
    rows = [json.loads((_row_dir(work, i) / "row.json").read_text(encoding="utf-8"))
            for i in range(COUNT)]
    rss_values = [x["peak_rss_bytes"] for x in rows if x.get("peak_rss_bytes") is not None]
    return {"per_image": rows,
        "rollout_seconds_total": float(sum(x["rollout_seconds"] for x in rows)),
        "solver_seconds_total": float(sum(x["solver_seconds"] for x in rows)),
        "peak_rss_bytes_max": max(rss_values) if rss_values else None,
        "discarded_mass_total": int(sum(x["discarded_mass"] for x in rows)),
        "candidate_predicted_foreground_fraction_mean": float(np.mean([x["candidate_foreground_fraction"] for x in rows])),
        "diagnostic_only": True}


def _write_prediction_directory(output, arm, labels, bindings, bind_sha, *, work,
                                 source_state_before=None, source_state_after=None,
                                 source_t=None):
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + f".staging-{uuid.uuid4().hex}")
    stage.mkdir()
    with (stage / "predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, image_ids=IMAGE_IDS, labels=np.asarray(labels, dtype=np.int32))
        stream.flush(); os.fsync(stream.fileno())
    protocol = {"experiment": "SW0162_connected_signed_partition", "status": "complete",
        "arm": arm, "seed": SEED, "image_ids": [1320, 1639], "count": COUNT,
        "grid_size": [32, 32], "prediction_shape": [COUNT, 32, 32],
        "implementation_fingerprint": implementation_fingerprint(), "bindings": bindings,
        "bindings_sha256": bind_sha, "prediction_sha256": sha256_file(stage / "predictions.npz"),
        "optimizer_updates": 0, "ground_truth_used_for_prediction": False}
    if arm == "connected_signed":
        protocol.update({"row_commits_sha256": _row_ledger_sha(work),
            "source_state_sha256_before": source_state_before,
            "source_state_sha256_after": source_state_after,
            "diagnostics": _prediction_diagnostics(work),
            "readout": "source-T actual Q/A connected signed greedy partition; variable K; min8/largest BG",
            "steps": 1024, "settle": 512, "live_tail": 0, "reference_reused": False})
    else:
        protocol.update({"row_commits_sha256": _row_ledger_sha(work),
            "source_T_prediction_sha256": source_t["prediction_sha256"],
            "source_T_protocol_sha256": source_t["protocol_sha256"],
            "fresh_same_rollout_qcc_replay": True,
            "predicted_foreground_fraction_mean": float(np.mean(np.asarray(labels) > 0)),
            "diagnostic_only": True})
    _write_json_once(stage / "protocol.json", protocol)
    _publish_stage(stage, output)


def _finalize_rows_without_rollout(*, prepared, bindings, bind_sha, work, prediction_root, work_root):
    expected = prepared["references"]["source_T"]["labels"]
    candidate, control = [], []
    for i, image_id in enumerate(IMAGE_IDS.tolist()):
        candidate.append(_load_row(work, i, image_id, prepared["rgb_hashes"][i], bind_sha))
        control.append(_load_control_row(work, i, image_id, prepared["rgb_hashes"][i], bind_sha))
    candidate, control = np.stack(candidate), np.stack(control)
    if not np.array_equal(control, expected):
        raise ValueError("committed fresh-QCC control ledger differs from archived SW0143 T")
    pre_state = prepared["preflight"]["source"]["source_state_sha256_before"]
    for arm in ARMS:
        directory = _output_dir(arm, prediction_root)
        if directory.exists():
            validate_prediction(arm, prediction_root=prediction_root, work_root=work_root,
                                prepared=prepared)
        else:
            _write_prediction_directory(directory, arm,
                candidate if arm == "connected_signed" else control,
                bindings, bind_sha, work=work, source_state_before=pre_state,
                source_state_after=pre_state, source_t=prepared["references"]["source_T"])
    return {"status": "prediction_complete", "candidate_output": str(_output_dir("connected_signed", prediction_root)),
        "control_output": str(_output_dir("source_T_control", prediction_root)),
        "resumed_from_committed_rows": True}


@torch.no_grad()
def predict_candidate(*, device="cuda:0", prediction_root=PREDICTION_ROOT,
                      work_root=WORK_ROOT, dataset=DATASET):
    """One native source T rollout per image; persist validated variable-K labels."""
    prepared = prepare_prediction_cpu(dataset=dataset)
    bindings = _prediction_bindings(prepared); bind_sha = _sha_json(bindings)
    output, work = _output_dir("connected_signed", prediction_root), _work_dir(work_root)
    control_output = _output_dir("source_T_control", prediction_root)
    work.mkdir(parents=True, exist_ok=True)
    binding_path = work / "bindings.json"
    if binding_path.exists():
        if json.loads(binding_path.read_text(encoding="utf-8")) != bindings:
            raise ValueError("existing SW0162 rows bind changed source/code inputs")
    else:
        _write_json_once(binding_path, bindings)
    if (output.exists() or control_output.exists()
            or all(_row_dir(work, i).is_dir() for i in range(COUNT))):
        return _finalize_rows_without_rollout(prepared=prepared, bindings=bindings, bind_sha=bind_sha,
            work=work, prediction_root=prediction_root, work_root=work_root)
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0162 fixed320 inference requires owner-held CUDA")
    torch.cuda.set_device(device)
    found = load_native32_foundation(SEED, device, verify_rgb_assets=False)
    found.encoder.eval(); found.wrapped.eval()
    graph = found.wrapped.core.graph_generator
    geodesic_operator.install_operator(graph, "T")
    coupling = found.wrapped.core.kuramoto
    if found.wrapped.core.kuramoto_backend != "factorized":
        raise ValueError("SW0162 requires the registered factorized backend")
    state_before = preflight._source_state_digest(found)
    if state_before != prepared["preflight"]["source"]["source_state_sha256_before"]:
        raise ValueError("loaded source model differs from validated eight-image preflight")
    gain_t = graph.log_coupling_gain
    if not torch.isfinite(gain_t).all():
        raise ValueError("SW0162 learned source coupling gain is nonfinite")
    gain = float(torch.exp(gain_t.detach()).cpu())
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("SW0162 actual source graph gain must be positive")
    for i, image_id in enumerate(IMAGE_IDS.tolist()):
        if _row_dir(work, i).exists():
            _load_row(work, i, image_id, prepared["rgb_hashes"][i], bind_sha)
            _load_control_row(work, i, image_id, prepared["rgb_hashes"][i], bind_sha)
            continue
        image = torch.from_numpy(np.asarray(prepared["rgb"][i:i + 1]).copy()).to(device)
        gamma = found.encode(image)
        if tuple(gamma.shape) != (1, 8, 1024) or not torch.isfinite(gamma).all():
            raise ValueError(f"SW0162 source gamma invalid at fixed index {i}")
        torch.cuda.reset_peak_memory_stats(device)
        rollout_started = time.monotonic()
        trace, actual_a = preflight.capture_source_coupling(
            coupling, lambda: late_rollout(found.wrapped, gamma, total_steps=1024, live_tail_steps=0))
        rollout_seconds = time.monotonic() - rollout_started
        _validate_trace(trace)
        q = preflight._actual_q(trace)
        source_qcc = sw136._qcc_labels(trace)
        if not np.array_equal(source_qcc[0], prepared["references"]["source_T"]["labels"][i]):
            raise ValueError(f"fresh source T QCC differs from archived source-T row {i}")
        preflight._validate_actual_A(actual_a)
        if tuple(q.shape) != (1, 1024, 1024) or not torch.isfinite(q).all():
            raise ValueError(f"SW0162 actual settled Q invalid at fixed index {i}")
        solver_started = time.monotonic()
        partition = connected_signed.connected_signed_merge(q[0].detach().cpu().double().numpy(),
            actual_a[0].detach().cpu().double().numpy(), gain, grid_shape=(32, 32))
        labels = connected_signed.canonical_labels(partition["labels"], min_size=8)
        solver_seconds = time.monotonic() - solver_started
        raw_sizes = [int(item["size"]) for item in partition["clusters"]]
        discarded_mass = sum(size for size in raw_sizes if size < 8)
        if labels.shape != (32, 32) or np.unique(labels).size > 129:
            raise ValueError(f"SW0162 connected labels invalid at fixed index {i}")
        trace_shas = {key: preflight._tensor_sha(trace[key]) for key in preflight.TRACE_FIELDS}
        _write_row(work, i, image_id, prepared["rgb_hashes"][i], bind_sha, labels, source_qcc[0],
            preflight._tensor_sha(q), preflight._tensor_sha(actual_a), gain, partition["merge_count"], trace_shas,
            {"rollout_seconds": rollout_seconds, "solver_seconds": solver_seconds,
             "peak_rss_bytes": _peak_rss_bytes(), "raw_group_sizes": raw_sizes,
             "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
             "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
             "retained_group_count": int(np.unique(labels).size), "discarded_mass": int(discarded_mass),
             "candidate_foreground_fraction": float(np.mean(labels > 0))})
        if i % 40 == 39:
            print(json.dumps({"stage": "SW0162_prediction", "predicted": i + 1, "count": COUNT}), flush=True)
        del image, gamma, trace, actual_a, q, source_qcc, partition, labels
    labels = np.stack([_load_row(work, i, int(image_id), prepared["rgb_hashes"][i], bind_sha)
                       for i, image_id in enumerate(IMAGE_IDS)], axis=0)
    control_labels = np.stack([_load_control_row(work, i, int(image_id), prepared["rgb_hashes"][i], bind_sha)
                               for i, image_id in enumerate(IMAGE_IDS)], axis=0)
    if not np.array_equal(control_labels, prepared["references"]["source_T"]["labels"]):
        raise AssertionError("fresh single-rollout source-T QCC rows do not exactly replay archive")
    state_after = preflight._source_state_digest(found)
    if state_before != state_after:
        raise AssertionError("source encoder/graph/core changed during prediction")
    _write_prediction_directory(output, "connected_signed", labels, bindings, bind_sha,
        work=work, source_state_before=state_before, source_state_after=state_after,
        source_t=prepared["references"]["source_T"])
    _write_prediction_directory(control_output, "source_T_control", control_labels, bindings, bind_sha,
        work=work, source_t=prepared["references"]["source_T"])
    result = _finalize_rows_without_rollout(prepared=prepared, bindings=bindings, bind_sha=bind_sha,
        work=work, prediction_root=prediction_root, work_root=work_root)
    result.update(status="prediction_complete", output=str(output), control_output=str(control_output))
    return result


def publish_control(*, prediction_root=PREDICTION_ROOT, dataset=DATASET):
    """Validate the same-rollout QCC control emitted with the candidate."""
    prepared = prepare_prediction_cpu(dataset=dataset)
    output = _output_dir("source_T_control", prediction_root)
    result = validate_prediction("source_T_control", prediction_root=prediction_root,
                                 dataset=dataset, prepared=prepared)
    return {"status": "control_complete", "prediction_sha256": result["prediction_sha256"],
            "protocol_sha256": result["protocol_sha256"], "output": str(output)}


def validate_prediction(arm, *, prediction_root=PREDICTION_ROOT, work_root=WORK_ROOT,
                        dataset=DATASET, prepared=None):
    if arm not in ARMS:
        raise ValueError("unregistered SW0162 arm")
    prepared = prepared or prepare_prediction_cpu(dataset=dataset)
    directory = _output_dir(arm, prediction_root)
    npz_path, protocol_path = directory / "predictions.npz", directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    bindings = _prediction_bindings(prepared); bind_sha = _sha_json(bindings)
    expected = {"experiment": "SW0162_connected_signed_partition", "status": "complete",
        "arm": arm, "seed": SEED, "image_ids": [1320, 1639], "count": COUNT,
        "grid_size": [32, 32], "prediction_shape": [COUNT, 32, 32],
        "implementation_fingerprint": implementation_fingerprint(), "bindings": bindings,
        "bindings_sha256": bind_sha, "optimizer_updates": 0,
        "ground_truth_used_for_prediction": False}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError(f"SW0162 {arm} prediction provenance mismatch")
    if protocol.get("prediction_sha256") != sha256_file(npz_path):
        raise ValueError("SW0162 prediction NPZ digest mismatch")
    if arm == "connected_signed":
        if (protocol.get("source_state_sha256_before") != protocol.get("source_state_sha256_after")
                or protocol.get("row_commits_sha256") != _row_ledger_sha(_work_dir(work_root))):
            raise ValueError("SW0162 candidate source state or row ledger changed")
    else:
        ref = prepared["references"]["source_T"]
        if (protocol.get("fresh_same_rollout_qcc_replay") is not True
                or protocol.get("row_commits_sha256") != _row_ledger_sha(_work_dir(work_root))
                or protocol.get("source_T_prediction_sha256") != ref["prediction_sha256"]
                or protocol.get("source_T_protocol_sha256") != ref["protocol_sha256"]):
            raise ValueError("SW0162 control no longer binds fresh source-T QCC replay")
    with np.load(npz_path, allow_pickle=False) as saved:
        ids = contract.require_fixed_image_ids(saved["image_ids"])
        labels = np.asarray(saved["labels"])
    if (labels.shape != (COUNT, GRID, GRID) or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels > 128)):
        raise ValueError("SW0162 direct native32 variable-K label contract mismatch")
    if arm == "source_T_control" and not np.array_equal(labels, prepared["references"]["source_T"]["labels"]):
        raise ValueError("SW0162 control labels differ from archived SW0143 T")
    if arm == "connected_signed":
        work = _work_dir(work_root)
        expected_t = prepared["references"]["source_T"]["labels"]
        for i, image_id in enumerate(IMAGE_IDS.tolist()):
            qcc = _load_control_row(work, i, image_id, prepared["rgb_hashes"][i], bind_sha)
            if not np.array_equal(qcc, expected_t[i]):
                raise ValueError(f"SW0162 fresh QCC row {i} differs from archived source T")
    return {"arm": arm, "ids": ids, "labels": labels.astype(np.int64, copy=False),
        "prediction_sha256": protocol["prediction_sha256"],
        "protocol_sha256": sha256_file(protocol_path), "protocol": protocol,
        "directory": directory}


def _bootstrap_delta(candidate, baseline):
    a, b = np.asarray(candidate, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
    if a.shape != (COUNT,) or b.shape != (COUNT,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired bootstrap requires finite fixed320 scores")
    delta = a - b
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    indices = rng.randint(0, COUNT, size=(BOOTSTRAP_REPLICATES, COUNT))
    draws = delta[indices].mean(axis=1)
    return {"mean": float(delta.mean()), "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
        "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}


def _registered_gate(means, paired):
    gate = {"fg_mean_above_source_T": means["connected_signed"]["fg_ari"] > means["source_T"]["fg_ari"],
        "fg_paired_ci_lower_above_zero": paired["connected_signed_minus_source_T_fg_ari"]["ci95"][0] > 0,
        "foreground_iou_not_below_source_T": means["connected_signed"]["foreground_iou"] >= means["source_T"]["foreground_iou"],
        "object_iou_not_below_source_T": means["connected_signed"]["matched_object_iou"] >= means["source_T"]["matched_object_iou"]}
    gate["seed1_expansion_gate_passed"] = all(gate.values())
    return gate


def score_all(*, output=SCORE_PATH, prediction_root=PREDICTION_ROOT, work_root=WORK_ROOT,
              dataset=DATASET, prepared=None):
    """Validate and freeze all predictions before opening ground-truth masks."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserving existing SW0162 score {output}")
    prepared = prepared or prepare_prediction_cpu(dataset=dataset)
    frozen = {arm: validate_prediction(arm, prediction_root=prediction_root, work_root=work_root,
        dataset=dataset, prepared=prepared) for arm in ARMS}
    refs = prepared["references"]
    frozen.update({"source_T": {"ids": refs["source_T"]["ids"], "labels": refs["source_T"]["labels"],
            "prediction_sha256": refs["source_T"]["prediction_sha256"], "protocol_sha256": refs["source_T"]["protocol_sha256"]},
        "source_qcc": {"ids": refs["source_qcc"]["ids"], "labels": refs["source_qcc"]["labels"],
            "prediction_sha256": refs["source_qcc"]["prediction_sha256"], "protocol_sha256": refs["source_qcc"]["protocol_sha256"]},
        "Slot70k": {"ids": refs["slot"]["ids"], "labels": refs["slot"]["labels"],
            "prediction_sha256": refs["slot"]["prediction_sha256"], "protocol_sha256": refs["slot"]["protocol_sha256"]}})
    before = {name: {"prediction_sha256": x["prediction_sha256"],
        "protocol_sha256": x.get("protocol_sha256")} for name, x in frozen.items()}
    for item in frozen.values():
        if not np.array_equal(item["ids"], IMAGE_IDS):
            raise ValueError("SW0162 prediction/reference image IDs differ")
    for arm in ARMS:
        if (sha256_file(frozen[arm]["directory"] / "predictions.npz") != before[arm]["prediction_sha256"]
                or sha256_file(frozen[arm]["directory"] / "protocol.json") != before[arm]["protocol_sha256"]):
            raise ValueError("SW0162 prediction changed before GT access")
    if (sha256_file(refs["source_T"]["protocol_path"]) != refs["source_T"]["protocol_sha256"]
            or sha256_file(refs["source_qcc"]["protocol_path"]) != refs["source_qcc"]["protocol_sha256"]
            or sha256_file(refs["slot"]["protocol_path"]) != refs["slot"]["protocol_sha256"]
            or sha256_file(refs["source_T"]["archive_path"]) != refs["source_T"]["prediction_sha256"]
            or sha256_file(refs["source_qcc"]["archive_path"]) != refs["source_qcc"]["prediction_sha256"]
            or sha256_file(refs["slot"]["prediction_path"]) != refs["slot"]["prediction_sha256"]):
        raise ValueError("SW0162 prediction/reference archive changed before GT access")

    # Only this point may open the canonical HDF5 mask dataset.
    with h5py.File(prepared["dataset"], "r") as hdf:
        if "mask" not in hdf or hdf["mask"].shape[0] <= int(IMAGE_IDS[-1]):
            raise ValueError("canonical HDF5 lacks fixed320 masks")
        gt = contract.modal_native32(np.asarray(hdf["mask"][int(IMAGE_IDS[0]):int(IMAGE_IDS[-1]) + 1]))
    scores = {name: contract.production_metrics32(item["labels"], gt) for name, item in frozen.items()}
    means = {name: {metric: float(scores[name][metric]["mean"]) for metric in contract.METRICS}
             for name in scores}
    paired = {}
    for model in ARMS:
        for baseline in ("source_T", "source_qcc", "Slot70k"):
            for metric in contract.METRICS:
                paired[f"{model}_minus_{baseline}_{metric}"] = _bootstrap_delta(
                    scores[model][metric]["per_image"], scores[baseline][metric]["per_image"])
    gate = _registered_gate(means, paired)
    for arm in ARMS:
        if (sha256_file(frozen[arm]["directory"] / "predictions.npz") != before[arm]["prediction_sha256"]
                or sha256_file(frozen[arm]["directory"] / "protocol.json") != before[arm]["protocol_sha256"]):
            raise ValueError("prediction changed during GT scoring")
    for reference in (refs["source_T"], refs["source_qcc"], refs["slot"]):
        if (sha256_file(reference["protocol_path"]) != reference["protocol_sha256"]
                or sha256_file(reference["archive_path"] if "archive_path" in reference else reference["prediction_path"])
                    != reference["prediction_sha256"]):
            raise ValueError("reference prediction changed during GT scoring")
    stat = prepared["dataset"].stat()
    if stat.st_size != prepared["dataset_size_bytes"] or stat.st_mtime_ns != prepared["dataset_mtime_ns"]:
        raise ValueError("canonical HDF5 changed during scoring")
    report = {"experiment": "SW0162_connected_signed_partition", "status": "complete",
        "seed_scope": [SEED], "implementation_fingerprint": implementation_fingerprint(),
        "scores": scores, "means": means, "paired_bootstrap": paired,
        "seed1_expansion_gate": gate,
        "all_metric_means_vs_slot": {metric: {"delta": means["connected_signed"][metric] - means["Slot70k"][metric],
            "strictly_above": means["connected_signed"][metric] > means["Slot70k"][metric]}
            for metric in contract.METRICS},
        "fg_ari_at_least_0_87": means["connected_signed"]["fg_ari"] >= .87,
        "prediction_diagnostics": {"candidate": frozen["connected_signed"]["protocol"].get("diagnostics"),
            "predicted_foreground_fraction": {name: float(np.mean(item["labels"] > 0)) for name, item in frozen.items()},
            "diagnostic_only": True},
        "bootstrap": {"rng": "numpy RandomState", "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES, "common_paired_image_indices": True},
        "prediction_hashes_before_gt": before,
        "ground_truth_opened_after_all_predictions_validated": True,
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
        "dataset": {"path": str(prepared["dataset"].resolve()), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns},
        "interpretation": "Fixed seed1 source-only inference; no automatic seed expansion."}
    _write_json_once(output, report)
    report["score_sha256"] = sha256_file(output)
    return report


def validate_score(path=SCORE_PATH, *, prediction_root=PREDICTION_ROOT, work_root=WORK_ROOT,
                   dataset=DATASET, prepared=None):
    """Strictly validate a completed score report against frozen predictions and GT-last contract."""
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    prepared = prepared or prepare_prediction_cpu(dataset=dataset)
    predictions = {arm: validate_prediction(arm, prediction_root=prediction_root,
        work_root=work_root, dataset=dataset, prepared=prepared) for arm in ARMS}
    if (report.get("experiment") != "SW0162_connected_signed_partition" or report.get("status") != "complete"
            or report.get("implementation_fingerprint") != implementation_fingerprint()
            or report.get("ground_truth_opened_after_all_predictions_validated") is not True
            or report.get("ground_truth_used_for_prediction") is not False
            or report.get("ground_truth_used_for_scoring") is not True
            or report.get("seed_scope") != [SEED]):
        raise ValueError("SW0162 score report contract/provenance mismatch")
    expected_hashes = {arm: {"prediction_sha256": predictions[arm]["prediction_sha256"],
        "protocol_sha256": predictions[arm]["protocol_sha256"]} for arm in ARMS}
    for name in ("source_T", "source_qcc", "Slot70k"):
        ref = prepared["references"][{"source_T": "source_T", "source_qcc": "source_qcc", "Slot70k": "slot"}[name]]
        expected_hashes[name] = {"prediction_sha256": ref["prediction_sha256"],
            "protocol_sha256": ref["protocol_sha256"]}
    if report.get("prediction_hashes_before_gt") != expected_hashes:
        raise ValueError("SW0162 score references do not match validated immutable predictions")
    names = (*ARMS, "source_T", "source_qcc", "Slot70k")
    scores = report.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(names):
        raise ValueError("SW0162 score metric table names mismatch")
    means = {}
    for name in names:
        if set(scores[name]) != set(contract.METRICS):
            raise ValueError(f"SW0162 metrics missing for {name}")
        means[name] = {}
        for metric in contract.METRICS:
            row = scores[name][metric]
            values = np.asarray(row.get("per_image"), dtype=np.float64)
            mean = row.get("mean")
            if (values.shape != (COUNT,) or not np.isfinite(values).all()
                    or row.get("valid_count") != COUNT or not np.isfinite(mean)
                    or abs(float(values.mean()) - float(mean)) > 1e-12):
                raise ValueError(f"SW0162 invalid score values for {name}/{metric}")
            means[name][metric] = float(mean)
    if report.get("means") != means:
        raise ValueError("SW0162 score means do not equal per-image metrics")
    paired = report.get("paired_bootstrap")
    expected_pairs = {}
    for model in ARMS:
        for baseline in ("source_T", "source_qcc", "Slot70k"):
            for metric in contract.METRICS:
                key = f"{model}_minus_{baseline}_{metric}"
                expected_pairs[key] = _bootstrap_delta(scores[model][metric]["per_image"],
                                                        scores[baseline][metric]["per_image"])
    if paired != expected_pairs or report.get("seed1_expansion_gate") != _registered_gate(means, expected_pairs):
        raise ValueError("SW0162 paired bootstrap or registered gate mismatch")
    if report.get("bootstrap") != {"rng": "numpy RandomState", "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES, "common_paired_image_indices": True}:
        raise ValueError("SW0162 bootstrap registration mismatch")
    dataset = Path(dataset)
    stat = dataset.stat()
    if report.get("dataset") != {"path": str(Path(dataset).resolve()), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}:
        raise ValueError("SW0162 score HDF5 identity changed")
    refs = prepared["references"]
    immutable_paths = {"connected_signed_npz": predictions["connected_signed"]["directory"] / "predictions.npz",
        "connected_signed_protocol": predictions["connected_signed"]["directory"] / "protocol.json",
        "source_T_control_npz": predictions["source_T_control"]["directory"] / "predictions.npz",
        "source_T_control_protocol": predictions["source_T_control"]["directory"] / "protocol.json",
        "source_T_npz": refs["source_T"]["archive_path"], "source_T_protocol": refs["source_T"]["protocol_path"],
        "source_qcc_npz": refs["source_qcc"]["archive_path"], "source_qcc_protocol": refs["source_qcc"]["protocol_path"],
        "slot_npz": refs["slot"]["prediction_path"], "slot_protocol": refs["slot"]["protocol_path"]}
    hashes_before_recompute = {name: sha256_file(file_path) for name, file_path in immutable_paths.items()}
    if (hashes_before_recompute["connected_signed_npz"] != expected_hashes["connected_signed"]["prediction_sha256"]
            or hashes_before_recompute["connected_signed_protocol"] != expected_hashes["connected_signed"]["protocol_sha256"]
            or hashes_before_recompute["source_T_control_npz"] != expected_hashes["source_T_control"]["prediction_sha256"]
            or hashes_before_recompute["source_T_control_protocol"] != expected_hashes["source_T_control"]["protocol_sha256"]
            or hashes_before_recompute["source_T_npz"] != expected_hashes["source_T"]["prediction_sha256"]
            or hashes_before_recompute["source_T_protocol"] != expected_hashes["source_T"]["protocol_sha256"]
            or hashes_before_recompute["source_qcc_npz"] != expected_hashes["source_qcc"]["prediction_sha256"]
            or hashes_before_recompute["source_qcc_protocol"] != expected_hashes["source_qcc"]["protocol_sha256"]
            or hashes_before_recompute["slot_npz"] != expected_hashes["Slot70k"]["prediction_sha256"]
            or hashes_before_recompute["slot_protocol"] != expected_hashes["Slot70k"]["protocol_sha256"]):
        raise ValueError("SW0162 immutable predictions changed before independent score recomputation")
    # Recompute the production metrics from validated prediction arrays and canonical GT.
    # This catches a coherent edit to per-image scores, means, bootstrap, and gates.
    with h5py.File(dataset, "r") as hdf:
        if "mask" not in hdf or hdf["mask"].shape[0] <= int(IMAGE_IDS[-1]):
            raise ValueError("canonical HDF5 lacks fixed320 masks for independent score validation")
        gt = contract.modal_native32(np.asarray(hdf["mask"][int(IMAGE_IDS[0]):int(IMAGE_IDS[-1]) + 1]))
    metric_inputs = {"connected_signed": predictions["connected_signed"]["labels"],
        "source_T_control": predictions["source_T_control"]["labels"],
        "source_T": refs["source_T"]["labels"], "source_qcc": refs["source_qcc"]["labels"],
        "Slot70k": refs["slot"]["labels"]}
    recomputed_scores = {name: contract.production_metrics32(labels, gt)
        for name, labels in metric_inputs.items()}
    recomputed_means = {name: {metric: float(recomputed_scores[name][metric]["mean"])
        for metric in contract.METRICS} for name in recomputed_scores}
    if scores != recomputed_scores or means != recomputed_means:
        raise ValueError("SW0162 score values differ from independently recomputed production metrics")
    stat_after = dataset.stat()
    hashes_after_recompute = {name: sha256_file(file_path) for name, file_path in immutable_paths.items()}
    if ((stat.st_size, stat.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns)
            or hashes_before_recompute != hashes_after_recompute):
        raise ValueError("SW0162 dataset or immutable prediction archives changed during score validation")
    return {"status": "valid", "score_sha256": sha256_file(path), "seed1_expansion_gate": report["seed1_expansion_gate"]}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("predict", "control", "score"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "predict": result = predict_candidate(device=args.device)
    elif args.stage == "control": result = publish_control()
    else: result = score_all(output=args.output or SCORE_PATH)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
