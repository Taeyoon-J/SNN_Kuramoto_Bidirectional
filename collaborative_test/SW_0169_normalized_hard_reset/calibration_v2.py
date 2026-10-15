"""Source-bound, mask-free SW0169 calibration/admission helpers."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from collaborative_test.SW_0169_normalized_hard_reset import hard_reset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
POSITIONS = tuple(range(8))
STEPS, SETTLE, TAIL = 1024, 512, 64
TRACE_FIELDS = ("theta", "membrane", "spikes", "component_membrane",
                "component_spikes", "component_gates")


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _import_keep_cuda(name):
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    module = importlib.import_module(name)
    if inherited is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = inherited
    return module


def _ordered_calibration_rows(ordered_ids, pool_rows):
    ids = list(map(int, ordered_ids))
    rows = np.asarray(pool_rows, dtype=np.int64)
    if len(ids) != 4096 or rows.shape != (4096,):
        raise ValueError("source contract must provide the registered ordered 4096 TRAIN rows")
    selected_ids = [ids[i] for i in POSITIONS]
    selected_rows = rows[np.asarray(POSITIONS, dtype=np.int64)]
    if len(set(selected_ids)) != 8 or np.any(selected_rows < 0):
        raise ValueError("calibration selection has duplicate IDs or invalid RGB rows")
    return selected_ids, selected_rows


def implementation_fingerprint():
    files = (
        "collaborative_test/SW_0169_normalized_hard_reset/DESIGN_KO.md",
        "collaborative_test/SW_0169_normalized_hard_reset/hard_reset.py",
        "collaborative_test/SW_0169_normalized_hard_reset/calibration_v2.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0130_phase_state_integration/model.py",
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0161_supervised_spike_partition/cache.py",
        "collaborative_test/SW_0162_connected_signed_partition/preflight.py",
        "collaborative_test/SW_0163_relation_context_assignment/context_cache.py",
        "collaborative_test/SW_0167_structured_relation_energy/audit_train_cache.py",
        "collaborative_test/SW_0167_structured_relation_energy/qcc_readout.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
        "snn_kuramoto_bidirectional/membrane_layer.py",
        "snn_kuramoto_bidirectional/dendric_layer.py",
        "snn_kuramoto_bidirectional/sinusoidal_gating.py",
    )
    result = {}
    for rel in files:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"required SW0169 dependency missing: {rel}")
        result[rel] = _sha_file(path)
    return result


def prepare_calibration_cpu():
    """Deep-check the registered first eight TRAIN cache/RGB rows, without GT."""
    audit = _import_keep_cuda(
        "collaborative_test.SW_0167_structured_relation_energy.audit_train_cache")
    sw130 = _import_keep_cuda("collaborative_test.SW_0130_phase_state_integration.run")
    sw155_train = _import_keep_cuda("collaborative_test.SW_0155_supervised_spike_relations.train")
    cached = audit._validate_upstream_and_rows()
    if len(cached["selected_ids"]) != 8 or tuple(POSITIONS) != tuple(range(8)):
        raise ValueError("SW0169 calibration requires the fixed eight TRAIN positions")
    checkpoint, manifest_path, manifest, pool_rows, ordered_ids, checkpoint_sha = sw130.source_contract(1)
    expected_ids, selected_rows = _ordered_calibration_rows(ordered_ids, pool_rows)
    actual_ids = list(map(int, cached["selected_ids"]))
    if actual_ids != expected_ids:
        raise ValueError("eight calibration rows do not match the source-cache TRAIN prefix")
    rgb_meta_path = sw130.TRAIN_RGB.with_suffix(sw130.TRAIN_RGB.suffix + ".complete.json")
    rgb_meta = json.loads(rgb_meta_path.read_text(encoding="utf-8-sig"))
    rgb_stat = sw130.TRAIN_RGB.stat()
    rgb_map = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if rgb_map.dtype != np.uint8 or tuple(rgb_map.shape) != (70000, 128, 128, 3):
        raise ValueError("registered TRAIN RGB mapping has invalid schema")
    rgb8 = np.asarray(rgb_map[selected_rows]).copy()
    del rgb_map
    if rgb8.dtype != np.uint8 or tuple(rgb8.shape) != (8, 128, 128, 3):
        raise ValueError("selected calibration RGB rows have invalid schema")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU calibration preparation unexpectedly initialized CUDA")
    return {
        "selected_ids": actual_ids,
        "positions": list(POSITIONS),
        "rgb_rows": selected_rows.tolist(),
        "rgb8": rgb8,
        "rgb_sha256": [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
                       for row in rgb8],
        "source_checkpoint_path": str(Path(checkpoint).resolve()),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_manifest_path": str(Path(manifest_path).resolve()),
        "source_manifest_sha256": _sha_file(manifest_path),
        "source_manifest_status": manifest.get("status"),
        "source_cache_manifest_sha256": cached["source_cache_manifest_sha256"],
        "source_cache_bindings_sha256": cached["source_cache_bindings_sha256"],
        "source_cache_state_sha256": cached["source_state_sha256"],
        "source_cache_trace_sha256": [row["trace_sha256"] for row in cached["rows"]],
        "context_cache_manifest_sha256": cached["context_cache_manifest_sha256"],
        "context_cache_bindings_sha256": cached["context_cache_bindings_sha256"],
        "context_cache_state_sha256": cached["context_state_sha256"],
        "context_rows": [{key: row[key] for key in
                           ("q_sha256", "adjacency_sha256", "context_file_sha256")}
                         for row in cached["rows"]],
        "cache_component_spikes": cached["component_spikes"],
        "cache_contexts": cached["contexts"],
        "gain": float(cached["gain"]),
        "gain_receipt_sha256": cached["gain_receipt_sha256"],
        "train_hdf5_path": str(Path(sw155_train.HDF5_PATH).resolve()),
        "train_hdf5_size_bytes": int(Path(sw155_train.HDF5_PATH).stat().st_size),
        "train_hdf5_mtime_ns": int(Path(sw155_train.HDF5_PATH).stat().st_mtime_ns),
        "train_rgb_path": str(Path(sw130.TRAIN_RGB).resolve()),
        "train_rgb_size_bytes": int(rgb_stat.st_size),
        "train_rgb_mtime_ns": int(rgb_stat.st_mtime_ns),
        "train_rgb_manifest_sha256": _sha_file(rgb_meta_path),
        "train_rgb_mapping_sha256": rgb_meta["ids_mapping_sha256"],
        "implementation_fingerprint": implementation_fingerprint(),
        "validation_data_read": False,
        "ground_truth_read": False,
    }


def _load_source(device):
    foundation = _import_keep_cuda("collaborative_test.SW_0135_native32_spike_binding.foundation")
    dev = torch.device(device)
    with torch.random.fork_rng(devices=[dev.index] if dev.type == "cuda" else []):
        torch.manual_seed(149001)
        if dev.type == "cuda":
            torch.cuda.manual_seed(149001)
        found = foundation.load_native32_foundation(1, dev, verify_rgb_assets=False)
    operator = _import_keep_cuda(
        "collaborative_test.SW_0143_native32_selective_geodesic.geodesic_operator")
    operator.install_operator(found.wrapped.core.graph_generator, "T")
    return found


def _source_digest(found):
    cache = _import_keep_cuda("collaborative_test.SW_0161_supervised_spike_partition.cache")
    return cache._state_digest(found)


def _q(trace):
    classifier = _import_keep_cuda("snn_kuramoto_bidirectional.spike_classifier")
    return classifier.spike_synchrony_affinity(
        trace["spikes"], trace["component_spikes"], settle=SETTLE,
        affinity_mode="spike")


def _cache_component_input(trace_components, saved_rows, batch_start, *, device):
    """Recreate SW0163's registered historical B4 cache input layout.

    Its first B4 was evaluated from the strided live trace view; later B4s
    were evaluated from contiguous saved source-cache rows. This is a fixed
    producer contract, not a fallback chosen by numerical agreement.
    """
    if batch_start == 0:
        return trace_components.detach()[..., SETTLE:], "live_trace_strided_b4"
    if batch_start == 4:
        rows = np.stack([saved_rows[i] for i in range(batch_start, batch_start + 4)])
        return torch.from_numpy(rows).to(device), "saved_contiguous_b4"
    raise ValueError("SW0169 cache replay only admits the registered B4 starts 0 and 4")


def _q_from_components(components):
    classifier = _import_keep_cuda("snn_kuramoto_bidirectional.spike_classifier")
    return classifier.spike_synchrony_affinity(
        components.mean(dim=1), components, settle=0, affinity_mode="spike")


def _q_difference(live, replay):
    live_cpu = live.detach().cpu()
    replay_cpu = replay.detach().cpu()
    return {
        "live_q_sha256": _sha_tensor(live_cpu),
        "cache_replay_q_sha256": _sha_tensor(replay_cpu),
        "live_vs_cache_replay_max_abs_diff": float((live_cpu - replay_cpu).abs().max()),
        "live_vs_cache_replay_threshold_0_5_edge_differences": int(
            ((live_cpu >= 0.5) != (replay_cpu >= 0.5)).sum()),
    }


def run_source_calibration_gpu(prepared, *, device="cuda:0"):
    """Capture eight original source B4 rollouts; assert exact parity then fix r."""
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("source calibration admission requires CUDA")
    if torch.cuda.is_initialized():
        torch.cuda.set_device(torch.device(device))
    else:
        torch.cuda.set_device(torch.device(device))
    from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
    from collaborative_test.SW_0162_connected_signed_partition.preflight import capture_source_coupling

    found = _load_source(device)
    found.encoder.eval(); found.patcher.eval(); found.wrapped.eval()
    if _source_digest(found) != prepared["source_cache_state_sha256"]:
        raise ValueError("loaded source differs from immutable SW0161 cache state")
    all_h, all_g, rows = [], [], []
    rgb = prepared["rgb8"]
    for start in (0, 4):
        batch = torch.from_numpy(np.ascontiguousarray(rgb[start:start + 4])).to(device)
        gamma = found.encode(batch)
        trace_native, a_native = capture_source_coupling(
            found.wrapped.core.kuramoto,
            lambda: late_rollout(found.wrapped, gamma, total_steps=STEPS,
                                 live_tail_steps=TAIL))
        trace_copy, a_copy = capture_source_coupling(
            found.wrapped.core.kuramoto,
            lambda: hard_reset.rollout(found.wrapped, gamma,
                                       torch.ones(4, device=device),
                                       total_steps=STEPS, live_tail_steps=TAIL,
                                       source_mode=True))
        for field in TRACE_FIELDS:
            if not torch.equal(trace_native[field], trace_copy[field]):
                raise ValueError(f"SW0169 source-mode parity failed in {field}, batch {start // 4}")
        q_native, q_copy = _q(trace_native), _q(trace_copy)
        if not torch.equal(q_native, q_copy):
            raise ValueError("SW0169 source-mode production Q differs from SW0134 native source")
        if not torch.equal(a_native, a_copy):
            raise ValueError("SW0169 source-mode coupling differs from native source coupling")
        cache_components, cache_q_mode = _cache_component_input(
            trace_native["component_spikes"], prepared["cache_component_spikes"],
            start, device=device)
        q_cache_replay = _q_from_components(cache_components)
        if q_cache_replay.shape != q_native.shape:
            raise ValueError("historical context-cache replay produced the wrong Q shape")
        for offset in range(4):
            index = start + offset
            cached_s = torch.from_numpy(prepared["cache_component_spikes"][index]).to(device)
            if not torch.equal(trace_native["component_spikes"][offset, :, :, SETTLE:], cached_s):
                raise ValueError(f"source replay differs from immutable B4 cache spikes at row {index}")
            cached_q, cached_a = prepared["cache_contexts"][index]
            cached_q = torch.from_numpy(cached_q)
            if not torch.equal(q_cache_replay[offset].cpu(), cached_q):
                raise ValueError(
                    f"historical {cache_q_mode} Q replay differs from immutable context cache "
                    f"at row {index}")
            if not torch.equal(a_native[offset].cpu(), torch.from_numpy(cached_a)):
                raise ValueError(f"source A differs from immutable context cache at row {index}")
            q_diagnostics = _q_difference(q_native[offset], q_cache_replay[offset])
            rows.append({"index": index, "image_id": prepared["selected_ids"][index],
                         "trace_sha256": {name: _sha_tensor(trace_native[name][offset])
                                          for name in TRACE_FIELDS},
                         "q_sha256": _sha_tensor(q_native[offset]),
                         "a_sha256": _sha_tensor(a_native[offset]),
                         "cache_q_producer_mode": cache_q_mode,
                         "live_component_spike_strides": list(
                             trace_native["component_spikes"][offset].stride()),
                         "cache_q_input_strides": list(cache_components[offset].stride()),
                         "cache_q_sha256": _sha_tensor(cached_q),
                         **q_diagnostics,
                         "cache_context_verified": True})
        all_h.append(trace_copy["h_wave"].detach().cpu())
        all_g.append(trace_copy["component_gates"].detach().cpu())
        del batch, gamma, trace_native, trace_copy, q_native, q_copy, a_native, a_copy
        del cache_components, q_cache_replay
    if _source_digest(found) != prepared["source_cache_state_sha256"]:
        raise ValueError("source state changed during calibration replay")
    calibration = hard_reset.source_calibration(torch.cat(all_h), torch.cat(all_g), settle=SETTLE)
    return {"source_six_trace_parity": True, "source_q_parity": True,
            "source_a_parity": True, "source_cache_replay_exact": True,
            "source_rows": rows,
            "source_model_state_sha256_before": prepared["source_cache_state_sha256"],
            "source_model_state_sha256_after": _source_digest(found),
            "r": calibration["r"].cpu().tolist(),
            "r_sha256": _sha_tensor(calibration["r"]),
            "active_counts": calibration["active_counts"],
            "r_float64": calibration["r_float64"],
            "calibration_mode": calibration["source"],
            "settle": SETTLE, "batch_size": 4,
            "implementation_fingerprint": prepared["implementation_fingerprint"]}
