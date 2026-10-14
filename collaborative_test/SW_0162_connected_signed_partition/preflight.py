"""GT-free source-T parity and eight-image connected partition preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
               str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding.foundation import (
    implementation_fingerprint as source_implementation_fingerprint,
    load_native32_foundation,
)
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_operator, run as sw143
from collaborative_test.SW_0162_connected_signed_partition import connected_signed as solver
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity

SEED = 1
POSITIONS = (0, 1, 2, 3, 2048, 2049, 2050, 2051)
STEPS, SETTLE = 1024, 512
OUTPUT = HERE / "results_archive" / "preflight_20261014" / "source8_seed1.json"
TRACE_FIELDS = tuple(sw143.TRACE_FIELDS)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    fingerprint = {}
    fingerprint.update(sw143.implementation_fingerprint())
    fingerprint.update(source_implementation_fingerprint())
    paths = (HERE / "preflight.py", HERE / "connected_signed.py", HERE / "protocol.json",
             HERE / "DESIGN_KO.md", Path(sw136.__file__))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"SW0162 implementation dependency missing: {path}")
        fingerprint[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return fingerprint


def capture_source_coupling(graph, invoke_rollout):
    """Capture original prepare_coupling tuple[0] and return its tuple unchanged.

    The instance method is restored exactly even if the rollout raises. This is
    instance-local instrumentation; it does not alter the graph class or tuple.
    """
    original = graph.prepare_coupling
    had_instance_method = "prepare_coupling" in graph.__dict__
    instance_method = graph.__dict__.get("prepare_coupling")
    captured = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(captured):
            raise RuntimeError("SW0162 expected one source coupling preparation per static rollout")
        if not isinstance(result, tuple) or len(result) != 3 or not torch.is_tensor(result[0]):
            raise TypeError("native prepare_coupling must return its registered three-tensor tuple")
        captured.append(result[0].detach().clone())
        return result

    graph.prepare_coupling = capture
    try:
        rollout_output = invoke_rollout()
    finally:
        if had_instance_method:
            graph.prepare_coupling = instance_method
        else:
            delattr(graph, "prepare_coupling")
    if len(captured) != 1:
        raise RuntimeError("SW0162 source rollout did not prepare exactly one coupling tuple")
    return rollout_output, captured[0]


def state_sha256(modules, tensors=()):
    digest = hashlib.sha256()
    for module_name, module in modules:
        for key, value in sorted(module.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(module_name.encode() + b"\0" + key.encode() + b"\0")
            digest.update(str(tensor.dtype).encode() + b"\0")
            digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    for name, value in tensors:
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _source_state_digest(foundation):
    return state_sha256((("wrapped", foundation.wrapped), ("encoder", foundation.encoder),
                         ("patcher", foundation.patcher)),
                        (("feature_mean", foundation.feature_mean),
                         ("feature_std", foundation.feature_std)))


def select_train_rows(cache, pool_rows, ordered_ids, positions=POSITIONS):
    """Select the registered ordered TRAIN positions from source RGB cache rows."""
    rows = np.asarray(pool_rows)
    ids = list(map(int, ordered_ids))
    if (rows.ndim != 1 or rows.size != len(ids) or not np.issubdtype(rows.dtype, np.integer)
            or len(ids) != 4096 or len(set(ids)) != 4096):
        raise ValueError("SW0162 requires the exact unique 4096-row source TRAIN order")
    if tuple(positions) != POSITIONS:
        raise ValueError("SW0162 preflight source positions are registered and fixed")
    if cache.ndim != 4 or tuple(cache.shape[1:]) != (128, 128, 3) or cache.dtype != np.uint8:
        raise ValueError("registered TRAIN RGB cache must be uint8 NHWC 128x128")
    selected_rows = rows[np.asarray(POSITIONS, dtype=np.int64)].astype(np.int64, copy=False)
    if np.any(selected_rows < 0) or np.any(selected_rows >= cache.shape[0]):
        raise ValueError("source TRAIN pool rows are outside the registered RGB cache")
    selected = np.asarray(cache[selected_rows]).copy()
    if selected.shape != (8, 128, 128, 3) or selected.dtype != np.uint8:
        raise ValueError("SW0162 selected eight-image RGB cache has invalid schema")
    return selected, selected_rows.tolist(), [ids[i] for i in POSITIONS]


def _rng_state(device):
    state = {"cpu": torch.random.get_rng_state().clone()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device).clone()
    return state


def _set_rng_state(state, device):
    torch.random.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def prepare_preflight_cpu():
    """Validate the registered source and select only the eight TRAIN RGB rows.

    This runs before CUDA availability/device initialization. It reads no masks,
    validation labels, or ground-truth metrics.
    """
    checkpoint, manifest_path, manifest, pool_rows, ordered_ids, source_sha = \
        sw130.source_contract(SEED)
    cache_path = Path(sw130.TRAIN_RGB)
    complete_path = cache_path.with_suffix(cache_path.suffix + ".complete.json")
    if not complete_path.is_file():
        raise FileNotFoundError(f"registered TRAIN RGB completion metadata missing: {complete_path}")
    complete = json.loads(complete_path.read_text(encoding="utf-8-sig"))
    expected_rows = np.concatenate((np.arange(1000, dtype="<i8"),
                                    np.arange(1640, 70640, dtype="<i8")))
    expected_ids_sha = hashlib.sha256(expected_rows.tobytes()).hexdigest()
    if (complete.get("status") != "complete"
            or complete.get("cache_shape") != [70000, 128, 128, 3]
            or complete.get("ids_mapping_sha256") != expected_ids_sha):
        raise AssertionError("registered TRAIN RGB completion metadata/ID mapping changed")
    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    if tuple(cache.shape) != (70000, 128, 128, 3) or cache.dtype != np.uint8:
        raise ValueError("SW0162 registered RGB source cache schema changed")
    rgb8, selected_rows, selected_ids = select_train_rows(cache, pool_rows, ordered_ids)
    cache_stat = cache_path.stat()
    complete_sha = hashlib.sha256(complete_path.read_bytes()).hexdigest()
    del cache
    encoder_path = Path(sw130.ENCODER_SOURCE)
    stats_path = Path(sw130.FEATURE_STATS)
    if not encoder_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError("registered source encoder or feature statistics are missing")
    feature_stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    if feature_stats.get("mode") != "standardize":
        raise ValueError("registered source feature preprocessing mode changed")
    mean = torch.as_tensor(feature_stats["mean"])
    std = torch.as_tensor(feature_stats["std"])
    clip = float(feature_stats.get("clip", 3.0))
    if (not torch.isfinite(mean).all() or not torch.isfinite(std).all()
            or not torch.all(std > 0) or not np.isfinite(clip) or clip <= 0):
        raise ValueError("registered source feature statistics are not finite and usable")
    assets = {"train_manifest_sha256": complete_sha,
        "train_cache_shape": [70000, 128, 128, 3],
        "train_ids_mapping_sha256": expected_ids_sha,
        "selected_rgb_sha256": hashlib.sha256(rgb8.tobytes()).hexdigest(),
        "encoder_source_path": str(encoder_path),
        "encoder_source_sha256": sha256_file(encoder_path),
        "feature_stats_path": str(stats_path),
        "feature_stats_sha256": sha256_file(stats_path),
        "feature_stats_mode": "standardize", "feature_clip": clip,
        "scope": "registered TRAIN RGB only; no validation images or masks opened"}
    return {"checkpoint": checkpoint, "manifest_path": manifest_path,
        "manifest": manifest, "pool_rows": pool_rows, "ordered_ids": ordered_ids,
        "source_sha": source_sha, "assets": assets, "rgb8": rgb8,
        "selected_rows": selected_rows, "selected_ids": selected_ids,
        "rgb_sha": hashlib.sha256(rgb8.tobytes()).hexdigest(),
        "cache_size_bytes": cache_stat.st_size, "cache_mtime_ns": cache_stat.st_mtime_ns}


def source_report_fields(prepared, state_before, state_after):
    """Build the source block shared by the public producer and queue validator."""
    assets = prepared["assets"]
    return {"checkpoint_path": str(prepared["checkpoint"]),
        "checkpoint_sha256": prepared["source_sha"],
        "manifest_path": str(prepared["manifest_path"]),
        "manifest_sha256": sha256_file(prepared["manifest_path"]),
        "ordered_train_ids_sha256": hashlib.sha256(
            np.asarray(prepared["ordered_ids"], dtype="<i8").tobytes()).hexdigest(),
        "source_positions": list(POSITIONS), "selected_pool_rows": prepared["selected_rows"],
        "selected_image_ids": prepared["selected_ids"], "rgb_batch_sha256": prepared["rgb_sha"],
        "rgb_cache_path": str(sw130.TRAIN_RGB),
        "rgb_cache_size_bytes": prepared["cache_size_bytes"],
        "rgb_cache_mtime_ns": prepared["cache_mtime_ns"],
        "rgb_manifest_sha256": assets["train_manifest_sha256"],
        "encoder_source_sha256": assets["encoder_source_sha256"],
        "feature_stats_sha256": assets["feature_stats_sha256"],
        "feature_stats_mode": assets["feature_stats_mode"],
        "feature_clip": assets["feature_clip"],
        "rgb_asset_validation": assets,
        "source_state_sha256_before": state_before,
        "source_state_sha256_after": state_after}


def _tensor_sha(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _trace_check(trace, expected_batch=1):
    shapes = {"theta": (expected_batch, 1024, 1024, 4),
              "membrane": (expected_batch, 1024, 1024),
              "spikes": (expected_batch, 1024, 1024),
              "component_membrane": (expected_batch, 4, 1024, 1024),
              "component_spikes": (expected_batch, 4, 1024, 1024),
              "component_gates": (expected_batch, 4, 1024, 1024)}
    for name, shape in shapes.items():
        value = trace.get(name)
        if not torch.is_tensor(value) or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"SW0162 source trace {name} violates registered shape/finite contract")


def _actual_q(trace):
    q = spike_synchrony_affinity(trace["spikes"], trace["component_spikes"],
                                  settle=SETTLE, affinity_mode="spike")
    if tuple(q.shape) != (1, 1024, 1024) or not torch.isfinite(q).all():
        raise ValueError("SW0162 production spike Q has invalid shape or nonfinite values")
    return q


def _validate_actual_A(a):
    if tuple(a.shape) != (1, 1024, 1024) or not torch.isfinite(a).all():
        raise ValueError("captured actual source A must be finite [1,1024,1024]")
    if not torch.equal(a, a.transpose(1, 2)) or torch.any(a <= 0):
        raise ValueError("captured native A must retain source symmetric ReLU plus epsilon floor")


def _assert_connected_partition(clusters):
    covered = set()
    for cluster in clusters:
        patches = set(cluster["patches"])
        if not patches or len(patches) != cluster["size"] or covered.intersection(patches):
            raise ValueError("SW0162 component partition is empty or overlapping")
        covered.update(patches)
        reached = {min(patches)}
        todo = [next(iter(reached))]
        while todo:
            index = todo.pop()
            row, col = divmod(index, 32)
            candidates = []
            if row: candidates.append(index - 32)
            if row < 31: candidates.append(index + 32)
            if col: candidates.append(index - 1)
            if col < 31: candidates.append(index + 1)
            for other in candidates:
                if other in patches and other not in reached:
                    reached.add(other); todo.append(other)
        if reached != patches:
            raise ValueError("SW0162 agglomeration emitted a disconnected component")
    if covered != set(range(1024)):
        raise ValueError("SW0162 partition does not cover all 1024 patches exactly once")


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


@torch.no_grad()
def run_preflight(*, device="cuda:0", output=OUTPUT):
    """Run source-T parity and deterministic solver on eight TRAIN RGB images only."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserving existing SW0162 source preflight: {output}")
    prepared = prepare_preflight_cpu()
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0162 actual source preflight requires the registered owner-held CUDA device")
    torch.cuda.set_device(device)
    checkpoint = prepared["checkpoint"]
    manifest_path = prepared["manifest_path"]
    manifest = prepared["manifest"]
    pool_rows = prepared["pool_rows"]
    ordered_ids = prepared["ordered_ids"]
    source_sha = prepared["source_sha"]
    assets = prepared["assets"]
    found = load_native32_foundation(SEED, device, verify_rgb_assets=False)
    if not isinstance(assets, dict) or not assets:
        raise ValueError("SW0162 CPU preflight did not validate registered RGB assets")
    found.encoder.eval(); found.wrapped.eval()
    if found.wrapped.core.kuramoto_backend != "factorized":
        raise ValueError("SW0162 registered source T requires factorized coupling backend")
    if (list(map(int, found.image_ids)) != ordered_ids
            or not np.array_equal(found.pool_indices, np.asarray(pool_rows, dtype=np.int64))):
        raise ValueError("loaded SW0135 foundation source order differs from source contract")

    rgb8 = prepared["rgb8"]
    selected_rows = prepared["selected_rows"]
    selected_ids = prepared["selected_ids"]
    rgb_sha = prepared["rgb_sha"]

    graph = found.wrapped.core.graph_generator
    geodesic_operator.install_operator(graph, "T")
    gain_t = getattr(graph, "log_coupling_gain", None)
    if not torch.is_tensor(gain_t) or not torch.isfinite(gain_t).all():
        raise ValueError("actual source graph has no finite log coupling gain")
    gain = float(torch.exp(gain_t.detach()).cpu())
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("actual source coupling gain must be finite and positive")

    state_before = _source_state_digest(found)
    image_records = []
    peak_allocated = 0
    for row, image_id, rgb_array in zip(selected_rows, selected_ids, rgb8):
        image = torch.from_numpy(rgb_array[None].copy()).to(device)
        gamma = found.encode(image)
        if tuple(gamma.shape) != (1, 8, 1024) or not torch.isfinite(gamma).all():
            raise ValueError("SW0162 source encoder produced invalid native32 gamma")
        rng = _rng_state(device)
        original_trace = late_rollout(found.wrapped, gamma, total_steps=STEPS, live_tail_steps=0)
        _trace_check(original_trace)
        original_q = _actual_q(original_trace)
        original_qcc = sw136._qcc_labels(original_trace)
        _set_rng_state(rng, device)
        captured_trace, actual_a = capture_source_coupling(
            found.wrapped.core.kuramoto,
            lambda: late_rollout(found.wrapped, gamma, total_steps=STEPS, live_tail_steps=0))
        _trace_check(captured_trace)
        actual_q = _actual_q(captured_trace)
        actual_qcc = sw136._qcc_labels(captured_trace)
        _validate_actual_A(actual_a)
        if not torch.equal(actual_q, original_q):
            raise AssertionError("capturing source A changed actual source positive-product Q")
        if not np.array_equal(actual_qcc, original_qcc):
            raise AssertionError("capturing source A changed production source QCC labels")
        trace_hashes = {}
        for name in TRACE_FIELDS:
            if not torch.equal(captured_trace[name], original_trace[name]):
                raise AssertionError(f"source A capture changed registered trace field {name}")
            trace_hashes[name] = _tensor_sha(captured_trace[name])

        q_cpu = actual_q[0].detach().cpu().numpy().astype(np.float64, copy=True)
        a_cpu = actual_a[0].detach().cpu().numpy().astype(np.float64, copy=True)
        started = time.perf_counter()
        partition = solver.connected_signed_merge(q_cpu, a_cpu, gain, grid_shape=(32, 32))
        solver_seconds = time.perf_counter() - started
        repeat = solver.connected_signed_merge(q_cpu, a_cpu, gain, grid_shape=(32, 32))
        if (partition["merge_count"] != repeat["merge_count"]
                or not np.array_equal(partition["labels"], repeat["labels"])
                or not np.array_equal(partition["pair_energy"], repeat["pair_energy"])):
            raise AssertionError("SW0162 CPU connected solver is not deterministic on identical inputs")
        labels = solver.canonical_labels(partition["labels"], min_size=8)
        _assert_connected_partition(partition["clusters"])
        if labels.shape != (32, 32) or labels.dtype != np.int32 or not np.isfinite(labels).all():
            raise ValueError("SW0162 connected readout is malformed")
        sizes = sorted((int(cluster["size"]) for cluster in partition["clusters"]), reverse=True)
        counts = np.bincount(labels.reshape(-1))
        fg_fraction = float(np.count_nonzero(labels) / labels.size)
        image_records.append({
            "source_position": int(POSITIONS[len(image_records)]), "pool_row": int(row),
            "image_id": int(image_id), "rgb_sha256": hashlib.sha256(rgb_array.tobytes()).hexdigest(),
            "gamma_sha256": _tensor_sha(gamma), "trace_sha256": trace_hashes,
            "Q_sha256": _tensor_sha(actual_q), "actual_A_sha256": _tensor_sha(actual_a),
            "qcc_labels_sha256": hashlib.sha256(np.asarray(actual_qcc, dtype="<i8").tobytes()).hexdigest(),
            "trace_parity": True, "Q_parity": True, "QCC_parity": True,
            "actual_A_symmetric_positive_finite": True, "gain": gain,
            "merge_count": int(partition["merge_count"]), "raw_group_sizes_descending": sizes,
            "deterministic_solver_replay": True,
            "retained_group_count": int(np.unique(labels).size - (1 if np.any(labels == 0) else 0)),
            "background_size": int(counts[0]), "foreground_coverage": fg_fraction,
            "solver_seconds": float(solver_seconds),
        })
        peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(device)))
        del image, gamma, original_trace, captured_trace, original_q, actual_q, actual_a

    state_after = _source_state_digest(found)
    if state_before != state_after:
        raise AssertionError("SW0162 source model parameters/buffers changed during inference")
    record = {
        "experiment": "SW0162_connected_signed_partition", "status": "source8_preflight_complete",
        "seed": SEED, "implementation": implementation_fingerprint(),
        "source": source_report_fields(prepared, state_before, state_after),
        "source_operator": {"arm": "T", "steps": STEPS, "settle": SETTLE, "live_tail": 0,
            "optimizer_updates": 0, "backend": "factorized", "actual_gain_exp_log_coupling_gain": gain,
            "A_origin": "untouched GraphVectorKuramoto.prepare_coupling tuple[0]"},
        "images": image_records, "cuda_peak_allocated_bytes": peak_allocated,
        "ground_truth_used": False, "optimizer_updates": 0,
        "interpretation": "Eight TRAIN source parity and solver-feasibility check only; no accuracy or learning claim",
    }
    if len(image_records) != 8 or not all(item["trace_parity"] and item["Q_parity"] and item["QCC_parity"]
                                             for item in image_records):
        raise AssertionError("SW0162 source preflight did not validate all eight selected TRAIN images")
    _atomic_json(output, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    result = run_preflight(device=args.device, output=args.output)
    print(json.dumps({"status": result["status"], "images": len(result["images"]),
        "ground_truth_used": result["ground_truth_used"], "optimizer_updates": result["optimizer_updates"],
        "output": str(args.output)}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
