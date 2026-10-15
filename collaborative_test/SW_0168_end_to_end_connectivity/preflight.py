"""Bounded TRAIN-only, two-update admission for SW0168."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from collaborative_test.SW_0168_end_to_end_connectivity import connectivity_objective as objective

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ("constrained_maximin", "matched_local_edge")
POSITIONS = (0, 1, 2, 3, 2048, 2049, 2050, 2051)
HEAD_SEED = 168001
STEPS, SETTLE, TAIL = 1024, 512, 64


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _import_keep_visibility(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    module = importlib.import_module(name)
    if visible is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return module


def implementation_fingerprint():
    rels = (
        "collaborative_test/SW_0168_end_to_end_connectivity/DESIGN_KO.md",
        "collaborative_test/SW_0168_end_to_end_connectivity/connectivity_objective.py",
        "collaborative_test/SW_0168_end_to_end_connectivity/preflight.py",
        "collaborative_test/SW_0167_structured_relation_energy/audit_train_cache.py",
        "collaborative_test/SW_0167_structured_relation_energy/qcc_readout.py",
        "collaborative_test/SW_0161_supervised_spike_partition/cache.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0155_supervised_spike_relations/objective.py",
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0162_connected_signed_partition/preflight.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/foreground_signed.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
    )
    out = {}
    for rel in rels:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"SW0168 dependency missing: {rel}")
        out[rel] = _sha(path)
    return out


def prepare_preflight_cpu():
    """Validate exactly eight cache/RGB/HDF TRAIN rows, without CUDA or VAL data."""
    audit = _import_keep_visibility(
        "collaborative_test.SW_0167_structured_relation_energy.audit_train_cache")
    sw130 = _import_keep_visibility("collaborative_test.SW_0130_phase_state_integration.run")
    sw155_train = _import_keep_visibility(
        "collaborative_test.SW_0155_supervised_spike_relations.train")
    sw155_objective = _import_keep_visibility(
        "collaborative_test.SW_0155_supervised_spike_relations.objective")
    qcc = _import_keep_visibility("collaborative_test.SW_0167_structured_relation_energy.qcc_readout")
    structured = _import_keep_visibility(
        "collaborative_test.SW_0167_structured_relation_energy.structured_energy")

    cached = audit._validate_upstream_and_rows()
    if len(cached["selected_ids"]) != 8:
        raise ValueError("SW0168 admission requires exactly eight registered TRAIN cache rows")
    contract = sw130.source_contract(1)
    checkpoint, manifest_path, manifest, pool_rows, ordered_ids, source_sha = contract
    if list(map(int, cached["selected_ids"])) != list(map(int, ordered_ids[:8])):
        raise ValueError("cache rows do not match seed-1 ordered TRAIN prefix")
    rgb_rows = np.asarray(pool_rows[:8], dtype=np.int64)
    if rgb_rows.shape != (8,) or np.any(rgb_rows < 0) or np.any(rgb_rows >= 70000):
        raise ValueError("source contract returned invalid TRAIN RGB row indices")
    train_rgb = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if train_rgb.dtype != np.uint8 or train_rgb.shape != (70000, 128, 128, 3):
        raise ValueError("registered source TRAIN RGB cache schema changed")
    rgb8 = np.asarray(train_rgb[rgb_rows]).copy()
    if rgb8.shape != (8, 128, 128, 3) or rgb8.dtype != np.uint8:
        raise ValueError("selected source RGB rows have invalid schema")
    hdf_stat = Path(sw155_train.HDF5_PATH).stat()
    aligned_rgb, masks = sw155_train.load_aligned_train_rows(
        sw155_train.HDF5_PATH, list(map(int, cached["selected_ids"])), rgb8, 0, 8)
    if aligned_rgb.dtype != torch.uint8 or not np.array_equal(aligned_rgb.cpu().numpy(), rgb8):
        raise ValueError("HDF5 TRAIN RGB rows differ from registered RGB cache")
    if masks.shape[0] != 8:
        raise ValueError("HDF5 TRAIN mask loader returned wrong row count")
    targets_modal32 = sw155_objective.modal_train_targets32(masks).cpu().numpy()

    # Source F is the production CPU-QCC over the original contiguous B4 groups.
    supports = []
    feasible = []
    for start in (0, 4):
        q_b4 = audit._recompute_cpu_q(np.stack(cached["component_spikes"][start:start + 4]))
        for offset in range(4):
            i = start + offset
            f = qcc.qcc_labels_from_affinity(q_b4[offset:offset + 1])[0].numpy().astype(np.int32)
            target = structured.feasible_target(f, targets_modal32[i].reshape(32, 32))
            supports.append(f)
            feasible.append(target)
    if len(supports) != 8 or any(s.shape != (32, 32) for s in supports + feasible):
        raise ValueError("CPU source F/feasible TRAIN targets have invalid native32 geometry")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation unexpectedly initialized CUDA")
    return {
        "cache": cached,
        "rgb8": rgb8,
        "supports": supports,
        "targets": feasible,
        "selected_ids": list(map(int, cached["selected_ids"])),
        "source_checkpoint_sha256": source_sha,
        "source_manifest_sha256": _sha(manifest_path),
        "source_cache_manifest_sha256": _sha(audit.source_cache.CACHE_ROOT / "manifest.json"),
        "context_cache_manifest_sha256": _sha(audit.context_cache.CACHE_ROOT / "manifest.json"),
        "source_state_sha256": cached["source_state_sha256"],
        "rng_cpu_state": torch.random.get_rng_state().tolist(),
        "rng_cuda_state": None,
        "hdf5_path": str(Path(sw155_train.HDF5_PATH).resolve()),
        "hdf5_size_bytes": hdf_stat.st_size,
        "hdf5_mtime_ns": hdf_stat.st_mtime_ns,
        "source_checkpoint_path": str(checkpoint),
        "source_manifest_path": str(manifest_path),
        "implementation_fingerprint": implementation_fingerprint(),
        "validation_data_read": False,
    }


def _cache_file_receipt():
    """Hash only manifests, ledgers, and the eight rows actually admitted."""
    audit = _import_keep_visibility(
        "collaborative_test.SW_0167_structured_relation_energy.audit_train_cache")
    records = {}
    for name, root in (("source", audit.source_cache.CACHE_ROOT),
                       ("context", audit.context_cache.CACHE_ROOT)):
        for leaf in ("manifest.json", "ledger.json"):
            path = Path(root) / leaf
            records[f"{name}/{leaf}"] = _sha(path)
        for index in range(8):
            path = Path(root) / "rows" / f"{index:04d}.npz"
            records[f"{name}/rows/{index:04d}.npz"] = _sha(path)
    return records


def _state_digest(module):
    cache = _import_keep_visibility(
        "collaborative_test.SW_0161_supervised_spike_partition.cache")
    return cache._state_digest(module)


def _load_source(device):
    foundation_mod = _import_keep_visibility(
        "collaborative_test.SW_0135_native32_spike_binding.foundation")
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        torch.manual_seed(149001)
        if device.type == "cuda":
            torch.cuda.manual_seed(149001)
        found = foundation_mod.load_native32_foundation(1, device, verify_rgb_assets=False)
    operator = _import_keep_visibility(
        "collaborative_test.SW_0143_native32_selective_geodesic.geodesic_operator")
    operator.install_operator(found.wrapped.core.graph_generator, "T")
    return found


def _capture_trace(found, rgb):
    from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
    from collaborative_test.SW_0162_connected_signed_partition.preflight import capture_source_coupling
    gamma = found.encode(rgb)
    trace, captured = capture_source_coupling(
        found.wrapped.core.kuramoto,
        lambda: late_rollout(found.wrapped, gamma, total_steps=STEPS, live_tail_steps=TAIL))
    trace_fields = ("theta", "membrane", "spikes", "component_membrane",
                    "component_spikes", "component_gates")
    if any(name not in trace for name in trace_fields):
        raise ValueError("native rollout did not return all registered trace fields")
    if any(not torch.isfinite(trace[name]).all() for name in trace_fields):
        raise FloatingPointError("native source rollout has nonfinite trace")
    return gamma, trace, captured


def _tensor_sha(value):
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _json_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _q(trace):
    from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
    return spike_synchrony_affinity(trace["spikes"], trace["component_spikes"],
                                   settle=SETTLE, affinity_mode="spike")


def _b1_source_readout(found, rgb_row):
    """Production B1 native-QCC and SW164 zero-residual baseline replay."""
    sw136 = _import_keep_visibility("collaborative_test.SW_0136_native32_transfer.evaluate")
    solver = _import_keep_visibility(
        "collaborative_test.SW_0164_foreground_preserving_partition.foreground_signed")
    image = torch.from_numpy(np.asarray(rgb_row[None]).copy()).to(next(found.encoder.parameters()).device)
    with torch.no_grad():
        gamma, trace, actual_a = _capture_trace(found, image)
        q = _q(trace)
        labels = sw136._qcc_labels(trace)[0].detach().cpu().numpy().astype(np.int32)
        gain = float(torch.exp(found.wrapped.core.graph_generator.log_coupling_gain.detach()).cpu())
    result = solver.foreground_induced_merge(
        q[0].detach().cpu().numpy().astype(np.float64, copy=True),
        actual_a[0].detach().cpu().numpy().astype(np.float64, copy=True),
        gain, labels, grid_shape=(32, 32))
    return {
        "qcc_labels": labels,
        "qcc_sha256": hashlib.sha256(labels.astype("<i4").tobytes()).hexdigest(),
        "e0_labels_sha256": hashlib.sha256(
            np.asarray(result["labels"], dtype="<i4").tobytes()).hexdigest(),
        "merge_log_sha256": _json_sha(result["merge_log"]),
        "merge_count": int(result["merge_count"]),
        "q_sha256": _tensor_sha(q[0]),
        "actual_a_sha256": _tensor_sha(actual_a[0]),
    }


def _params(found):
    wrapped = found.wrapped
    graph = wrapped.core.graph_generator
    if getattr(graph, "_sw0143_geodesic_arm", None) != "T":
        raise ValueError("SW0168 requires registered SW0143-T graph operator")
    if wrapped.core.kuramoto_backend != "factorized":
        raise ValueError("SW0168 requires registered factorized Kuramoto backend")
    graph.geodesic_contrast.requires_grad_(False)
    families = {
        "encoder": list(found.encoder.parameters()),
        "graph": [p for name, p in graph.named_parameters() if name != "geodesic_contrast"],
        "native_core": [p for name, p in wrapped.core.named_parameters()
                        if not name.startswith("graph_generator.")],
        "integration": [wrapped.a_d, wrapped.a_m, wrapped.b],
    }
    flattened = [p for group in families.values() for p in group]
    if not flattened or len({id(p) for p in flattened}) != len(flattened):
        raise ValueError("SW0168 trainable parameter family set is empty or duplicated")
    for p in flattened:
        p.requires_grad_(True)
    return families


def _norm(grads):
    values = [g for g in grads if g is not None]
    if not values:
        return 0.0
    if any(not bool(torch.isfinite(g).all().detach().cpu()) for g in values):
        raise FloatingPointError("nonfinite SW0168 gradient")
    return math.sqrt(sum(float(g.detach().double().square().sum().cpu()) for g in values))


def _family_norm(families):
    return {name: _norm([p.grad for p in params]) for name, params in families.items()}


def _snapshot(families):
    return {name: [p.detach().clone() for p in values] for name, values in families.items()}


def _deltas(before, families):
    return {name: math.sqrt(sum(float((old.detach().cpu().double() - new.detach().cpu().double()).square().sum())
                                for old, new in zip(before[name], values)))
            for name, values in families.items()}


def _objective(arm, q, target, support):
    if arm == "constrained_maximin":
        return objective.constrained_maximin_loss(q, target, support)
    return objective.matched_local_edge_loss(q, target, support)


def run_two_update_preflight(prepared, *, device):
    """Execute two disposable eight-image updates for both loss arms."""
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0168 source admission requires the exclusive CUDA device")
    if not torch.cuda.is_initialized():
        torch.cuda.set_device(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    cache = prepared["cache"]
    supports, targets, rgb8 = prepared["supports"], prepared["targets"], prepared["rgb8"]
    cache_before = _cache_file_receipt()

    # Capture a frozen source reference in the original B4 grouping. Q is
    # diagnostic only here: the training objective uses each fresh live Q.
    found = _load_source(dev)
    found.encoder.eval(); found.patcher.eval(); found.wrapped.eval()
    if _state_digest(found) != prepared["source_state_sha256"]:
        raise ValueError("loaded native foundation differs from immutable source cache state")
    torch.manual_seed(168000)
    torch.cuda.manual_seed(168000)
    baseline = []
    trace_fields = ("theta", "membrane", "spikes", "component_membrane",
                    "component_spikes", "component_gates")
    for start in (0, 4):
        batch_rgb = torch.from_numpy(rgb8[start:start+4]).to(dev)
        with torch.no_grad():
            gamma_ref, trace_ref, captured = _capture_trace(found, batch_rgb)
            q_ref = _q(trace_ref)
        if tuple(captured.shape) != (4, 1024, 1024) or not torch.isfinite(captured).all():
            raise ValueError("captured native prepare_coupling A has invalid B4 shape/values")
        if not torch.equal(captured, captured.transpose(-1, -2)) or torch.any(captured <= 0):
            raise ValueError("captured native A violates source symmetric-positive contract")
        for offset in range(4):
            i = start + offset
            _stored_q, stored_a = cache["contexts"][i]
            if not torch.equal(captured[offset].cpu(), torch.from_numpy(stored_a)):
                raise ValueError(f"actual native A differs from immutable B4 cache row {i}")
            if not torch.equal(trace_ref["component_spikes"][offset, :, :, SETTLE:].cpu(),
                               torch.from_numpy(cache["component_spikes"][i])):
                raise ValueError(f"same-mode live source spikes differ from cache row {i}")
            baseline.append({"gamma": _tensor_sha(gamma_ref[offset]),
                "traces": {name: _tensor_sha(trace_ref[name][offset]) for name in trace_fields},
                "q": _tensor_sha(q_ref[offset]), "a": _tensor_sha(captured[offset])})
        del batch_rgb, gamma_ref, trace_ref, q_ref, captured
    baseline_state_after = _state_digest(found)
    if baseline_state_after != prepared["source_state_sha256"]:
        raise ValueError("no-update source B4 replay mutated registered source model state")
    b1_baseline = []
    for i in range(8):
        value = _b1_source_readout(found, rgb8[i])
        b1_baseline.append({key: value[key] for key in
                            ("qcc_sha256", "e0_labels_sha256", "merge_log_sha256",
                             "merge_count", "q_sha256", "actual_a_sha256")})
    if _state_digest(found) != prepared["source_state_sha256"]:
        raise ValueError("B1 Source-T/E0 replay mutated frozen source state")
    del found
    trace_parity = {name: True for name in trace_fields}

    reports = {}
    for arm in ARMS:
        found = _load_source(dev)
        torch.manual_seed(168000)
        torch.cuda.manual_seed(168000)
        found.encoder.eval(); found.patcher.eval(); found.wrapped.eval()
        families = _params(found)
        if _state_digest(found) != prepared["source_state_sha256"]:
            raise ValueError("fresh paired arm does not load registered seed-1 foundation")
        arm_parity = []
        for start in (0, 4):
            batch_rgb = torch.from_numpy(rgb8[start:start+4]).to(dev)
            with torch.no_grad():
                gamma0, trace0, captured0 = _capture_trace(found, batch_rgb)
                q0 = _q(trace0)
            for offset in range(4):
                i = start + offset
                observed = {"gamma": _tensor_sha(gamma0[offset]),
                    "traces": {name: _tensor_sha(trace0[name][offset]) for name in trace_fields},
                    "q": _tensor_sha(q0[offset]), "a": _tensor_sha(captured0[offset])}
                arm_parity.append(observed == baseline[i])
            del batch_rgb, gamma0, trace0, captured0, q0
        if not all(arm_parity):
            raise ValueError(f"{arm} fresh source init failed exact B4 six-trace/Q/A parity")
        b1_parity = []
        for i in range(8):
            current = _b1_source_readout(found, rgb8[i])
            b1_parity.append(all(current[key] == b1_baseline[i][key] for key in b1_baseline[i]))
        if not all(b1_parity):
            raise ValueError(f"{arm} fresh source init failed exact B1 QCC/E0/merge replay")
        found.encoder.train(); found.patcher.train(); found.wrapped.train()
        groups = [dict(params=params, lr=3e-5, name=name) for name, params in families.items()
                  if name != "integration"]
        groups.append(dict(params=families["integration"], lr=1e-3, name="integration"))
        optimizer = torch.optim.Adam(groups)
        arm_steps = []
        for step in (1, 2):
            optimizer.zero_grad(set_to_none=True)
            initial = _snapshot(families)
            q_norms, s_live_norms, losses = [], [], []
            started = time.perf_counter()
            for start in (0, 4):
                rgb = torch.from_numpy(rgb8[start:start+4]).to(dev)
                gamma, trace, _ = _capture_trace(found, rgb)
                q = _q(trace)
                q.retain_grad()
                spikes = trace["component_spikes"]
                spikes.retain_grad()
                micro_losses = []
                for offset in range(4):
                    i = start + offset
                    loss = _objective(arm, q[offset], targets[i], supports[i])
                    if loss.ndim or not torch.isfinite(loss):
                        raise FloatingPointError("SW0168 objective is not a finite scalar")
                    micro_losses.append(loss)
                    losses.append(float(loss.detach().cpu()))
                (torch.stack(micro_losses).sum() / 8.0).backward()
                if q.grad is None or spikes.grad is None:
                    raise ValueError("SW0168 objective supplies no actual Q/component-spike gradient")
                q_norms.extend(float(q.grad[offset].detach().double().norm().cpu())
                                for offset in range(4))
                # Q uses all 512 settled steps, but only the final 64 settled
                # values retain autograd credit through late_rollout.
                s_live_norms.extend(_norm([spikes.grad[offset, ..., STEPS-TAIL:STEPS]])
                                    for offset in range(4))
                del rgb, gamma, trace, q, spikes, micro_losses, loss
            family_grads = _family_norm(families)
            critical = {name: family_grads[name] for name in ("encoder", "graph", "native_core")}
            if any(value <= 0 for value in critical.values()):
                raise ValueError(f"critical rollout family has no gradient at {arm} update {step}: {critical}")
            if max(q_norms) <= 0 or max(s_live_norms) <= 0:
                raise ValueError(f"native Q/live-tail spike credit is dead at {arm} update {step}")
            active = [p for group in optimizer.param_groups for p in group["params"]]
            preclip = math.sqrt(sum(float(p.grad.detach().double().square().sum().cpu())
                                     for p in active if p.grad is not None))
            torch.nn.utils.clip_grad_norm_(active, 1.0)
            clip_scale = min(1.0, 1.0 / (preclip + 1e-12))
            optimizer.step()
            found.wrapped.project_integrations_()
            deltas = _deltas(initial, families)
            if any(deltas[name] <= 0 for name in critical):
                raise ValueError(f"critical model family did not change at {arm} update {step}")
            arm_steps.append({"update": step, "mean_loss": float(np.mean(losses)),
                              "family_gradient_norms": family_grads,
                              "family_parameter_delta_l2": deltas,
                              "q_gradient_norm_mean": float(np.mean(q_norms)),
                              "live_tail_component_spike_gradient_norm_mean": float(np.mean(s_live_norms)),
                              "preclip_global_gradient_norm": preclip,
                              "clip_scale": clip_scale,
                              "elapsed_seconds": time.perf_counter() - started})
        reports[arm] = {"steps": arm_steps, "updates": 2,
                        "initial_source_parity_exact": all(arm_parity),
                        "initial_b1_readout_exact": all(b1_parity),
                        "initial_model_state_sha256": prepared["source_state_sha256"]}
        del optimizer, found
        torch.cuda.empty_cache()
    cache_after = _cache_file_receipt()
    if cache_before != cache_after:
        raise RuntimeError("immutable source/context cache files changed during disposable updates")
    return {"experiment": "SW0168_actual_spike_maximin_preflight",
            "status": "complete_admitted",
            "arms": reports, "trace_replay_exact": trace_parity,
            "initial_b4_trace_q_a_parity": {arm: reports[arm]["initial_source_parity_exact"]
                                             for arm in ARMS},
            "source_b4_reference_rows": [
                {"index": i, "image_id": int(prepared["selected_ids"][i]),
                 "gamma_sha256": row["gamma"], "trace_sha256": row["traces"],
                 "q_sha256": row["q"], "actual_a_sha256": row["a"],
                 "cached_component_spikes_exact": True, "cached_actual_a_exact": True}
                for i, row in enumerate(baseline)],
            "source_b1_readout_reference_rows": [
                {"index": i, "image_id": int(prepared["selected_ids"][i]),
                 **row} for i, row in enumerate(b1_baseline)],
            "source_model_state_sha256": prepared["source_state_sha256"],
            "source_cache_manifest_sha256": prepared["source_cache_manifest_sha256"],
            "context_cache_manifest_sha256": prepared["context_cache_manifest_sha256"],
            "selected_image_ids": prepared["selected_ids"],
            "validation_data_read": False, "training_ground_truth_scope": "exact eight registered TRAIN masks",
            "batch_scope": "8 unique registered TRAIN images per disposable update; not the full-run logical B16",
            "optimizer_updates": 2, "source_cache_file_receipt_before": cache_before,
            "source_cache_file_receipt_after": cache_after,
            "source_cache_unchanged": True,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev))}


def validate_report(report, *, expected_source, expected_fingerprint):
    """Strict receipt check shared by parent queue and downstream trainer."""
    if (not isinstance(report, dict)
            or report.get("experiment") != "SW0168_actual_spike_maximin_preflight"
            or report.get("status") != "complete_admitted"
            or report.get("implementation_fingerprint") != expected_fingerprint
            or report.get("source") != expected_source
            or report.get("selected_image_ids") != expected_source.get("selected_ids")
            or report.get("validation_data_read") is not False
            or report.get("source_model_state_sha256") != expected_source.get("source_state_sha256")
            or report.get("source_cache_unchanged") is not True
            or report.get("source_cache_file_receipt_before") != report.get("source_cache_file_receipt_after")
            or report.get("optimizer_updates") != 2
            or report.get("trace_replay_exact") != {name: True for name in
                ("theta", "membrane", "spikes", "component_membrane", "component_spikes", "component_gates")}
            or report.get("initial_b4_trace_q_a_parity") != {arm: True for arm in ARMS}
            or not isinstance(report.get("source_b1_readout_reference_rows"), list)
            or len(report["source_b1_readout_reference_rows"]) != 8
            or not isinstance(report.get("source_b4_reference_rows"), list)
            or len(report["source_b4_reference_rows"]) != 8):
        raise ValueError("SW0168 preflight provenance/scope/replay receipt is invalid")
    for i, row in enumerate(report["source_b4_reference_rows"]):
        if (row.get("index") != i or row.get("image_id") != expected_source["selected_ids"][i]
                or row.get("cached_component_spikes_exact") is not True
                or row.get("cached_actual_a_exact") is not True
                or not isinstance(row.get("trace_sha256"), dict)
                or set(row["trace_sha256"]) != {"theta", "membrane", "spikes", "component_membrane", "component_spikes", "component_gates"}
                or any(not isinstance(digest, str) or len(digest) != 64
                       for digest in [row.get("gamma_sha256"), row.get("q_sha256"), row.get("actual_a_sha256"), *row["trace_sha256"].values()])):
            raise ValueError(f"SW0168 source B4 trace receipt malformed at row {i}")
    for i, row in enumerate(report["source_b1_readout_reference_rows"]):
        hashes = (row.get("qcc_sha256"), row.get("e0_labels_sha256"),
                  row.get("merge_log_sha256"), row.get("q_sha256"),
                  row.get("actual_a_sha256"))
        if (row.get("index") != i or row.get("image_id") != expected_source["selected_ids"][i]
                or type(row.get("merge_count")) is not int or row["merge_count"] < 0
                or any(not isinstance(d, str) or len(d) != 64
                       or any(c not in "0123456789abcdef" for c in d) for d in hashes)):
            raise ValueError(f"SW0168 B1 Source-T/E0 replay receipt malformed at row {i}")
    arms = report.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("SW0168 preflight must contain both registered loss arms")
    for arm in ARMS:
        value = arms[arm]
        steps = value.get("steps") if isinstance(value, dict) else None
        if (value.get("updates") != 2 or value.get("initial_source_parity_exact") is not True
                or value.get("initial_b1_readout_exact") is not True
                or value.get("initial_model_state_sha256") != expected_source.get("source_state_sha256")
                or not isinstance(steps, list) or len(steps) != 2):
            raise ValueError(f"SW0168 {arm} update record is malformed")
        for i, row in enumerate(steps, 1):
            if row.get("update") != i:
                raise ValueError(f"SW0168 {arm} update cursor is invalid")
            for key in ("mean_loss", "q_gradient_norm_mean", "live_tail_component_spike_gradient_norm_mean",
                        "preclip_global_gradient_norm", "clip_scale", "elapsed_seconds"):
                number = row.get(key)
                if type(number) not in (int, float) or not math.isfinite(number):
                    raise ValueError(f"SW0168 {arm} update has invalid {key}")
            if (row["q_gradient_norm_mean"] <= 0
                    or row["live_tail_component_spike_gradient_norm_mean"] <= 0
                    or row["clip_scale"] <= 0 or row["clip_scale"] > 1):
                raise ValueError(f"SW0168 {arm} lacks live Q/spike credit or valid clipping")
            family = row.get("family_gradient_norms")
            deltas = row.get("family_parameter_delta_l2")
            required = {"encoder", "graph", "native_core", "integration"}
            if (not isinstance(family, dict) or set(family) != required
                    or not isinstance(deltas, dict) or set(deltas) != required
                    or any(not math.isfinite(float(family[n])) or family[n] < 0 for n in required)
                    or any(not math.isfinite(float(deltas[n])) or deltas[n] < 0 for n in required)
                    or any(family[n] <= 0 for n in ("encoder", "graph", "native_core"))
                    or any(deltas[n] <= 0 for n in ("encoder", "graph", "native_core"))):
                raise ValueError(f"SW0168 {arm} critical family credit/change is invalid")
    return True


def _source_identity(prepared):
    return {key: prepared[key] for key in (
        "selected_ids", "source_checkpoint_sha256", "source_manifest_sha256",
        "source_cache_manifest_sha256", "context_cache_manifest_sha256",
        "source_state_sha256", "hdf5_path", "hdf5_size_bytes", "hdf5_mtime_ns")}


def run_registered_preflight(*, physical_gpu):
    """Queue-child entrypoint: CPU admission first, then the leased GPU probe."""
    if type(physical_gpu) is not int or physical_gpu not in (0, 1, 2, 3):
        raise ValueError("physical GPU must be one of the four registered devices")
    assigned = os.environ.get("CUDA_VISIBLE_DEVICES")
    if assigned != str(physical_gpu):
        raise RuntimeError("preflight child visibility differs from leased GPU")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before registered TRAIN source admission")
    prepared = prepare_preflight_cpu()
    expected_source = _source_identity(prepared)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != assigned or torch.cuda.is_initialized():
        raise RuntimeError("CPU admission altered GPU visibility or initialized CUDA")
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    report = run_two_update_preflight(prepared, device=device)
    current_hdf = Path(prepared["hdf5_path"]).stat()
    if (current_hdf.st_size != prepared["hdf5_size_bytes"]
            or current_hdf.st_mtime_ns != prepared["hdf5_mtime_ns"]):
        raise RuntimeError("canonical TRAIN HDF5 changed during SW0168 preflight")
    report.update({"source": expected_source,
                   "implementation_fingerprint": implementation_fingerprint(),
                   "physical_gpu": physical_gpu,
                   "cuda_visible_devices": assigned})
    validate_report(report, expected_source=expected_source,
                    expected_fingerprint=report["implementation_fingerprint"])
    return report


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.worker or args.output is None or args.physical_gpu is None:
        raise SystemExit("SW0168 preflight runs only as an owner-queue child")
    _atomic_json(args.output, run_registered_preflight(physical_gpu=args.physical_gpu))

