"""Non-gating fixed TRAIN8 diagnostics for completed SW0175 training arms.

This module audits positions 256..263 after both registered 64-update arms
complete. It never reads validation/reserve data and never changes a model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np  # Initialize NumPy/OpenMP before torch on Windows.
import torch

from collaborative_test.SW_0175_critical_component_binding import train_v2 as train
from collaborative_test.SW_0175_critical_component_binding import predict_evaluate_v2 as evaluate
from collaborative_test.SW_0168_end_to_end_connectivity import connectivity_objective as route_api
from collaborative_test.SW_0164_foreground_preserving_partition import foreground_signed as solver
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract

HERE = Path(__file__).resolve().parent
POSITIONS = tuple(range(256, 264))
ENDPOINTS = ("conditional", "standalone")
METRICS = tuple(contract.METRICS)


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def implementation_fingerprint() -> dict:
    paths = (
        "collaborative_test/SW_0175_critical_component_binding/DESIGN_KO.md",
        "collaborative_test/SW_0175_critical_component_binding/critical_component_objective.py",
        "collaborative_test/SW_0175_critical_component_binding/train_v2.py",
        "collaborative_test/SW_0175_critical_component_binding/predict_evaluate_v2.py",
        "collaborative_test/SW_0175_critical_component_binding/train8_audit.py",
        "collaborative_test/SW_0175_critical_component_binding/train8_queue.py",
        "collaborative_test/SW_0171_contextual_instance_graph/contextual_encoder.py",
        "collaborative_test/SW_0169_normalized_hard_reset/calibration_v2.py",
        "collaborative_test/SW_0168_end_to_end_connectivity/connectivity_objective.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/foreground_signed.py",
        "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
        "snn_kuramoto_bidirectional/evaluation.py",
    )
    result = {}
    root = train.preflight.ROOT
    for rel in paths:
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"SW0175 TRAIN8 audit dependency missing: {rel}")
        result[rel] = _sha_file(path)
    return {"files": result, "positions": list(POSITIONS), "scope": "TRAIN only",
            "optimizer_updates": 0, "selection_or_gate_use": False}


def prepare_inputs_cpu() -> dict:
    """Validate the completed arms and load only the registered TRAIN cache."""
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0175 TRAIN8 audit inputs must be prepared before CUDA initialization")
    admitted = train._load_admitted_inputs()
    prepared = admitted["prepared"]
    arrays = prepared.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("SW0175 admitted TRAIN cache has no arrays mapping")
    required = {"image_ids", "rgb", "source_support", "feasible_targets", "modal_gt"}
    if set(arrays) != required:
        raise ValueError("SW0175 admitted TRAIN cache schema changed")
    expected = {
        "image_ids": (512,), "rgb": (512, 128, 128, 3),
        "source_support": (512, 32, 32), "feasible_targets": (512, 32, 32),
        "modal_gt": (512, 32, 32),
    }
    for name, shape in expected.items():
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise ValueError(f"SW0175 TRAIN8 cache {name} has shape {value.shape}, expected {shape}")
    if arrays["rgb"].dtype != np.uint8:
        raise ValueError("SW0175 TRAIN8 RGB cache is not uint8")
    for name in ("source_support", "feasible_targets", "modal_gt"):
        value = np.asarray(arrays[name])
        if not np.issubdtype(value.dtype, np.integer) or np.any(value < 0):
            raise ValueError(f"SW0175 TRAIN8 {name} must be nonnegative integer labels")
    ids = np.asarray(arrays["image_ids"], dtype=np.int64)
    if len(np.unique(ids)) != 512:
        raise ValueError("SW0175 TRAIN cache IDs are not unique")
    manifests = {}
    for arm in train.ARMS:
        root = train.OUTPUT_ROOT / f"{arm}_seed1"
        manifest = train.validate_training_artifact(arm, root)
        if manifest.get("completed_updates") != train.UPDATES:
            raise ValueError(f"SW0175 {arm} is not complete at update 64")
        manifests[arm] = manifest
    first, second = (manifests[name]["bindings"] for name in train.ARMS)
    if {k: v for k, v in first.items() if k != "arm"} != {
            k: v for k, v in second.items() if k != "arm"}:
        raise ValueError("SW0175 matched arms no longer share training/source bindings")
    for arm in train.ARMS:
        ordered_ids = manifests[arm]["bindings"].get("ordered_train_ids")
        if (not isinstance(ordered_ids, list) or len(ordered_ids) != 512
                or [int(ids[i]) for i in POSITIONS]
                   != [int(ordered_ids[i]) for i in POSITIONS]):
            raise ValueError("SW0175 TRAIN8 audit positions differ from the bound canonical TRAIN order")
    input_sha = prepared.get("arrays_sha256")
    if not isinstance(input_sha, str) or len(input_sha) != 64:
        raise ValueError("SW0175 TRAIN cache archive hash is missing")
    for arm in train.ARMS:
        if manifests[arm]["bindings"].get("input_cache_arrays_sha256") != input_sha:
            raise ValueError(f"SW0175 {arm} is not bound to the admitted TRAIN arrays")
    return {"admitted": admitted, "prepared": prepared, "arrays": arrays,
            "manifests": manifests, "positions": list(POSITIONS),
            "image_ids": [int(ids[i]) for i in POSITIONS],
            "arrays_sha256": input_sha,
            "cache_manifest_sha256": prepared.get("cache_manifest_sha256"),
            "fingerprint": implementation_fingerprint()}


def _adjusted_rand(left, right) -> float:
    """ARI over a fixed grid, invariant to arbitrary partition label names."""
    a, b = np.asarray(left).reshape(-1), np.asarray(right).reshape(-1)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("partition comparison requires aligned grids with at least two cells")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    contingency = np.zeros((int(ai.max()) + 1, int(bi.max()) + 1), dtype=np.int64)
    np.add.at(contingency, (ai, bi), 1)
    choose2 = lambda x: x * (x - 1) / 2.0
    sum_cells = float(choose2(contingency).sum())
    sum_rows = float(choose2(contingency.sum(axis=1)).sum())
    sum_cols = float(choose2(contingency.sum(axis=0)).sum())
    total = float(choose2(a.size))
    expected = sum_rows * sum_cols / total
    maximum = 0.5 * (sum_rows + sum_cols)
    denom = maximum - expected
    if denom == 0:
        return 1.0 if np.array_equal(ai, bi) else 0.0
    return float((sum_cells - expected) / denom)


def _quantiles(values) -> dict:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not x.size:
        return {"count": 0, "q00": None, "q25": None, "q50": None,
                "q75": None, "q100": None}
    if not np.isfinite(x).all():
        raise FloatingPointError("SW0175 TRAIN8 rho diagnostic contains nonfinite values")
    q = np.quantile(x, [0, .25, .5, .75, 1])
    return {"count": int(x.size), "q00": float(q[0]), "q25": float(q[1]),
            "q50": float(q[2]), "q75": float(q[3]), "q100": float(q[4])}


def _support_mask(support):
    """Use the exact V2 caller adapter for cached integer label grids and masks."""
    return train._support_mask(support)


def _routes_and_clamp(q, rho, target, support):
    if q.ndim != 2 or rho.shape != (4, q.shape[0], q.shape[1]):
        raise ValueError("SW0175 TRAIN8 route diagnostics need Q[N,N] and rho[4,N,N]")
    plan = route_api.forest_routes(q.detach(), target, _support_mask(support))
    edge_components = [[] for _ in range(4)]
    q_np = q.detach().cpu().numpy()
    rho_np = rho.detach().cpu().numpy()
    edges = plan["positive_edges"]
    blocked = 0
    zero_q_mass = 0
    if edges:
        ii = torch.as_tensor([edge[0] for edge in edges], dtype=torch.long, device=q.device)
        jj = torch.as_tensor([edge[1] for edge in edges], dtype=torch.long, device=q.device)
        node_count = int(q.shape[-1])
        edge_rho = rho.reshape(4, -1)[:, ii * node_count + jj].detach().clone()
        edge_rho.requires_grad_(True)
        with torch.enable_grad():
            edge_products = edge_rho.clamp_min(0.0).prod(dim=0)
            actual_edge_q = q[ii, jj].detach()
            if not torch.equal(edge_products.detach(), actual_edge_q):
                raise ValueError("SW0175 TRAIN8 edge rho product differs from production Q")
            partials = torch.autograd.grad(edge_products.sum(), edge_rho)[0]
        blocked_flags = (~torch.any(partials != 0, dim=0)).detach().cpu().tolist()
    else:
        actual_edge_q = q.new_empty((0,))
        blocked_flags = []
    for edge_index, (i, j, represented) in enumerate(edges):
        values = rho_np[:, i, j]
        for component in range(4):
            edge_components[component].append(float(values[component]))
        if blocked_flags[edge_index]:
            blocked += int(represented)
        if float(q_np[i, j]) == 0.0:
            zero_q_mass += int(represented)
    return {"positive_route_edges": int(len(plan["positive_edges"])),
            "positive_route_pair_mass": int(plan["positive_pair_mass"]),
            "positive_clamp_blocked_pair_mass": int(blocked),
            "positive_route_zero_q_pair_mass": int(zero_q_mass),
            "component_rho_quantiles_on_positive_routes": [
                _quantiles(values) for values in edge_components],
            "negative_route_pair_mass": int(plan["negative_pair_mass"]),
            "excluded_unreachable_negative_pair_mass": int(
                plan["excluded_unreachable_negative_pair_mass"])}


def _event_stats(trace) -> dict:
    gates = trace["component_gates"].detach().float()
    spikes = trace["component_spikes"].detach().float()
    if gates.ndim != 4 or spikes.shape != gates.shape or gates.shape[1] != 4:
        raise ValueError("SW0175 TRAIN8 event traces must have shape [1,4,N,T]")
    if not bool(torch.isfinite(gates).all()) or not bool(torch.isfinite(spikes).all()):
        raise FloatingPointError("SW0175 TRAIN8 event traces are nonfinite")
    gate_on = gates > 0
    spike_event = spikes > 0
    gate_count = gate_on.sum(dim=(0, 2, 3)).clamp_min(1)
    return {"component_gate_mean_amplitude": gates.mean(dim=(0, 2, 3)).cpu().tolist(),
            "component_spike_mean_amplitude": spikes.mean(dim=(0, 2, 3)).cpu().tolist(),
            "component_gate_active_fraction": gate_on.float().mean(dim=(0, 2, 3)).cpu().tolist(),
            "component_spike_event_fraction": spike_event.float().mean(dim=(0, 2, 3)).cpu().tolist(),
            "component_events_given_active_gate":
                (spike_event.sum(dim=(0, 2, 3)).float() / gate_count.float()).cpu().tolist()}


def _score_partition(labels, gt) -> dict:
    from snn_kuramoto_bidirectional.evaluation import evaluate_patch_masks
    prediction = torch.from_numpy(np.asarray(labels, dtype=np.int64)[None])
    target = torch.from_numpy(np.asarray(gt, dtype=np.int64)[None])
    result = evaluate_patch_masks(prediction, target)
    scores = {}
    for name in METRICS:
        value = float(result["per_image"][name][0])
        scores[name] = value if np.isfinite(value) else None
    return scores


def _seal_partition_labels(rows: dict[str, list[dict]]) -> None:
    """Replace in-memory ndarray labels with JSON-safe immutable digests."""
    for model_rows in rows.values():
        for row in model_rows:
            endpoints = row.get("endpoints")
            if not isinstance(endpoints, dict) or set(endpoints) != set(ENDPOINTS):
                raise ValueError("SW0175 TRAIN8 endpoint row schema changed")
            for endpoint in ENDPOINTS:
                item = endpoints[endpoint]
                labels = np.asarray(item.pop("labels"), dtype=np.int32)
                if labels.shape != (32, 32):
                    raise ValueError("SW0175 TRAIN8 partition labels have invalid shape")
                item["partition_sha256"] = hashlib.sha256(
                    labels.astype("<i4", copy=False).tobytes()).hexdigest()


def _run_model_rows(found, arrays, *, model_name: str) -> list[dict]:
    from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
    from collaborative_test.SW_0175_critical_component_binding import predict_evaluate_v2 as evaluator
    import time

    rows = []
    for position in POSITIONS:
        image_id = int(arrays["image_ids"][position])
        image = torch.from_numpy(np.ascontiguousarray(arrays["rgb"][position:position + 1]))
        image = image.to(next(found.wrapped.parameters()).device)
        started = time.perf_counter()
        with torch.no_grad():
            gamma, trace, adjacency, _projection, q, rho = train._native_forward(
                found, image, live_tail_steps=0)
            qcc = np.asarray(sw136._qcc_labels(trace)[0], dtype=np.int32)
            support = np.asarray(arrays["source_support"][position], dtype=np.int32)
            target = np.asarray(arrays["feasible_targets"][position], dtype=np.int32)
            gt = np.asarray(arrays["modal_gt"][position], dtype=np.int32)
            if q.shape[0] != 1 or rho.shape[:2] != (1, 4) or adjacency.shape[0] != 1:
                raise ValueError("SW0175 TRAIN8 model forward returned invalid batch geometry")
            if not np.array_equal(support > 0, np.asarray(arrays["source_support"][position]) > 0):
                raise ValueError("SW0175 source support changed during diagnostics")
            q_loss = train._native_q_loss(q[0], target, support)
            candidate_loss = train._loss_for_arm("critical_component", q[0], rho[0], target, support)
            routes = _routes_and_clamp(q[0], rho[0], target, support)
            gain = float(torch.exp(found.wrapped.core.graph_generator.log_coupling_gain.detach()).cpu())
            if not np.isfinite(gain) or gain <= 0:
                raise FloatingPointError("SW0175 TRAIN8 actual learned graph gain is invalid")
            q_np = q[0].detach().cpu().numpy().astype(np.float64, copy=True)
            a_np = adjacency[0].detach().cpu().numpy().astype(np.float64, copy=True)
            outputs = {}
            for endpoint, endpoint_support in (("conditional", support), ("standalone", qcc)):
                if endpoint_support.shape != (32, 32):
                    raise ValueError(f"SW0175 TRAIN8 {endpoint} support geometry changed")
                partition = solver.foreground_induced_merge(
                    q_np, a_np, gain, endpoint_support, grid_shape=(32, 32))
                labels = np.asarray(partition["labels"], dtype=np.int32)
                if (labels.shape != (32, 32) or not np.array_equal(labels > 0, endpoint_support > 0)
                        or labels.max(initial=0) > 1024):
                    raise ValueError(f"SW0175 TRAIN8 {endpoint} readout violated foreground/capacity contract")
                outputs[endpoint] = {"labels": labels, "scores": _score_partition(labels, gt),
                                     "merge_count": int(partition["merge_count"]),
                                     "foreground_patches": int(np.count_nonzero(endpoint_support))}
            rows.append({"position": int(position), "image_id": image_id,
                "model": model_name, "gain": gain,
                "native_q_loss": float(q_loss.detach().cpu()),
                "critical_component_loss": float(candidate_loss.detach().cpu()),
                "routes": routes, "events": _event_stats(trace),
                "endpoints": outputs,
                "q_sha256": hashlib.sha256(q_np.astype("<f4", copy=False).tobytes()).hexdigest(),
                "actual_a_sha256": hashlib.sha256(a_np.astype("<f4", copy=False).tobytes()).hexdigest(),
                "standalone_qcc_sha256": hashlib.sha256(qcc.astype("<i4", copy=False).tobytes()).hexdigest(),
                "rollout_seconds": float(time.perf_counter() - started)})
        del image, gamma, trace, adjacency, q, rho
    return rows


def run_train8_audit(*, device="cuda:0", prepared=None) -> dict:
    """Run source-zero-context plus both frozen trained heads on TRAIN 256..263."""
    p = prepared or prepare_inputs_cpu()
    arrays, manifests = p["arrays"], p["manifests"]
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0175 TRAIN8 audit requires an assigned CUDA device")
    torch.cuda.set_device(device)
    from collaborative_test.SW_0169_normalized_hard_reset import calibration_v2 as source_cal
    from collaborative_test.SW_0171_contextual_instance_graph.contextual_encoder import ContextualFeatureEncoder
    from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_operator

    source = source_cal._load_source(device)
    geodesic_operator.install_operator(source.wrapped.core.graph_generator, "T")
    source_sha = train._source_sha(source, source.encoder)
    train._validate_source_proof_digest(p["admitted"]["proof"], source_sha)
    base_encoder = source.encoder
    base_sha = train.preflight._module_sha(base_encoder)
    frozen = train._frozen_source_components(source, base_encoder)
    for arm in train.ARMS:
        binding = manifests[arm]["bindings"]
        if (binding.get("source_state_sha256") != source_sha
                or binding.get("base_encoder_sha256") != base_sha
                or manifests[arm].get("frozen_source_components_before") != frozen):
            raise ValueError(f"SW0175 source-zero model no longer matches {arm} training lineage")
    source.encoder = ContextualFeatureEncoder(base_encoder, seed=train.CONTEXT_SEED).to(device)
    source.encoder.eval(); source.patcher.eval(); source.wrapped.eval()
    for module in (source.encoder, source.patcher, source.wrapped):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    source_zero_context_state_before = train._state_sha(source.wrapped, source.encoder)
    source_rows = _run_model_rows(source, arrays, model_name="source_T_zero_context")
    source_zero_context_state_after = train._state_sha(source.wrapped, source.encoder)
    if source_zero_context_state_after != source_zero_context_state_before:
        raise RuntimeError("SW0175 source-zero-context model changed during diagnostic inference")
    source_partition = {endpoint: [row["endpoints"][endpoint]["labels"] for row in source_rows]
                        for endpoint in ENDPOINTS}
    del source
    if device.type == "cuda":
        torch.cuda.empty_cache()

    by_model = {"source_T_zero_context": source_rows}
    for arm in train.ARMS:
        found, _state_before = evaluate._load_model(arm, {"manifests": manifests}, device)
        rows = _run_model_rows(found, arrays, model_name=arm)
        state_after = train._state_sha(found.wrapped, found.encoder)
        if state_after != manifests[arm]["final_model_state_sha256"]:
            raise RuntimeError(f"SW0175 {arm} model state changed during TRAIN8 audit")
        for index, row in enumerate(rows):
            row["partition_ari_vs_source_zero_context"] = {
                endpoint: _adjusted_rand(row["endpoints"][endpoint]["labels"],
                    source_partition[endpoint][index]) for endpoint in ENDPOINTS}
        by_model[arm] = rows
        del found
        torch.cuda.empty_cache()

    aggregate = {}
    for model_name, rows in by_model.items():
        aggregate[model_name] = {}
        for endpoint in ENDPOINTS:
            aggregate[model_name][endpoint] = {}
            for metric in METRICS:
                values = [row["endpoints"][endpoint]["scores"][metric] for row in rows]
                defined = [value for value in values if value is not None]
                aggregate[model_name][endpoint][metric] = {
                    "mean_over_defined": float(np.mean(defined)) if defined else None,
                    "defined_count": len(defined), "per_image": values}
        if model_name != "source_T_zero_context":
            aggregate[model_name]["partition_ari_vs_source_zero_context"] = {
                endpoint: [row["partition_ari_vs_source_zero_context"][endpoint] for row in rows]
                for endpoint in ENDPOINTS}
    _seal_partition_labels(by_model)
    return {"experiment": "SW0175_fixed_train8_posttraining_diagnostic",
        "status": "complete_non_gating_diagnostic", "positions": list(POSITIONS),
        "image_ids": p["image_ids"], "arms": list(train.ARMS),
        "source_zero_context_model_state_sha256": source_sha,
        "source_zero_context_wrapped_encoder_sha256_before": source_zero_context_state_before,
        "source_zero_context_wrapped_encoder_sha256_after": source_zero_context_state_after,
        "training_manifest_sha256": {arm: _sha_file(train.OUTPUT_ROOT / f"{arm}_seed1" / "manifest.json")
                                      for arm in train.ARMS},
        "train_cache_manifest_sha256": p["cache_manifest_sha256"],
        "train_cache_arrays_sha256": p["arrays_sha256"],
        "validation_or_reserve_data_read": False, "ground_truth_scope": "TRAIN512 only",
        "optimizer_updates": 0, "used_for_checkpoint_or_arm_selection": False,
        "fingerprint": p["fingerprint"], "aggregates": aggregate, "rows": by_model}


__all__ = ["POSITIONS", "prepare_inputs_cpu", "implementation_fingerprint",
           "_support_mask", "_adjusted_rand", "_routes_and_clamp",
           "_seal_partition_labels", "run_train8_audit"]
