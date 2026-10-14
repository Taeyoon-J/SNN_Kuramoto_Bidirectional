"""Two-update, source-bound feasibility check for SW0158 (GPU entrypoint only).

This is not a training runner. It evaluates exact source-T parity at eta=0,
then two disposable supervised TRAIN-mask updates and a matched live/detached
temporal-credit comparison. A report with failed admission is still preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

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
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_operator, run as sw143
from collaborative_test.SW_0155_supervised_spike_relations import objective, train as sw155
from collaborative_test.SW_0158_membrane_spike_credit import spike_credit
from snn_kuramoto_bidirectional.membrane_layer import R_m

EXPERIMENT = "SW0158_membrane_spike_credit_preflight"
SEED = 1
POSITIONS = (0, 1, 2, 3)
STEPS, SETTLE, TAIL = 1024, 512, 64
OUTPUT = HERE / "results_archive" / "preflight_seed1.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    paths = [HERE / name for name in ("protocol.json", "README.md", "spike_credit.py", "preflight.py")]
    paths.extend(ROOT / rel for rel in (
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0155_supervised_spike_relations/objective.py",
        "collaborative_test/SW_0157_train_gate_credit/results_archive/completed_20261014/0_train_gate_credit_seed1.json",
        "snn_kuramoto_bidirectional/spike_classifier.py",
        "snn_kuramoto_bidirectional/kuramoto_layer.py",
        "snn_kuramoto_bidirectional/dendric_layer.py",
        "snn_kuramoto_bidirectional/membrane_layer.py",
        "snn_kuramoto_bidirectional/sinusoidal_gating.py"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0158 preflight dependency missing: {missing[0]}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _state_sha(modules):
    digest = hashlib.sha256()
    for prefix, module in modules:
        for name, value in sorted(module.state_dict().items()):
            tensor = value.detach().cpu().contiguous().reshape(-1)
            digest.update(f"{prefix}.{name}:{tuple(value.shape)}:{value.dtype}".encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _flat_grads(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    rows = []
    for parameter, grad in zip(parameters, grads):
        if grad is None:
            rows.append(torch.zeros_like(parameter).reshape(-1))
        else:
            if not torch.isfinite(grad).all():
                raise FloatingPointError("nonfinite temporal-credit gradient")
            rows.append(grad.detach().reshape(-1))
    return torch.cat(rows) if rows else torch.empty(0)


def _norm(value):
    return float(value.detach().double().norm().cpu())


def temporal_credit_summary(live, repeated_live, detached, forward_parity):
    """Compare matched membrane-parameter gradients without confusing siblings."""
    if not all(torch.is_tensor(value) and value.ndim == 1 for value in
               (live, repeated_live, detached)) or not (live.shape == repeated_live.shape == detached.shape):
        raise ValueError("matched temporal gradients must be equal-length vectors")
    finite = all(bool(torch.isfinite(value).all()) for value in (live, repeated_live, detached))
    if not isinstance(forward_parity, dict) or not forward_parity or not all(
            value is True for value in forward_parity.values()):
        raise ValueError("live and detached forwards must have exact matching parity")
    residual = _norm(live - repeated_live)
    extra = _norm(live - detached)
    return {"live_gradient_finite": finite,
            "repeated_live_gradient_residual_l2": residual,
            "live_detached_membrane_gradient_delta_l2": extra,
            "additional_membrane_credit_detected": bool(finite and extra > max(1e-12, 10 * residual))}


def validate_report_record(record, *, expected_fingerprint, upstream):
    """CPU admission guard for the actual-source preflight receipt."""
    try:
        if (record.get("experiment") != EXPERIMENT or record.get("status") != "complete"
                or record.get("seed") != SEED
                or record.get("implementation_fingerprint") != expected_fingerprint
                or record.get("upstream") != upstream
                or record.get("optimizer_updates") != 2
                or record.get("source_positions") != list(POSITIONS)
                or record.get("image_ids") != upstream.get("source_training_ids", [])[:4]
                or record.get("ground_truth_used") is not True
                or record.get("ground_truth_scope") != "the four registered source TRAIN masks only"
                or record.get("validation_or_test_ground_truth_used") is not False
                or not isinstance(record.get("rgb_batch_sha256"), str)
                or len(record["rgb_batch_sha256"]) != 64
                or record.get("source_encoder_patcher_and_graph_frozen_state_sha256_before") !=
                    record.get("source_encoder_patcher_and_graph_frozen_state_sha256_after")
                or record.get("training_admission") is not True):
            return False
        parity = record["eta_zero_source_T_parity"]
        if (parity.get("six_trace_exact") is not True or parity.get("production_Q_exact") is not True
                or parity.get("production_QCC_exact") is not True
                or set(parity.get("trace_parity", {})) != set(sw143.TRACE_FIELDS)
                or not all(value is True for value in parity["trace_parity"].values())):
            return False
        credit = record["matched_temporal_credit"]
        if (credit.get("same_state_same_rng_forward_exact") is not True
                or not credit.get("live_gradient_finite") or not credit.get("detached_gradient_finite")
                or credit.get("additional_membrane_credit_detected") is not True
                or not math.isfinite(credit.get("live_detached_membrane_gradient_delta_l2", float("nan")))
                or not math.isfinite(credit.get("repeated_live_gradient_residual_l2", float("nan")))):
            return False
        recurrence = record.get("post_update1_recurrence_reset", {})
        amplitude = record.get("post_update1_amplitude_diagnostics", {})
        if (recurrence.get("finite") is not True
                or recurrence.get("same_spike_used_by_dendrite_and_reset") is not True
                or recurrence.get("recurrent_assignment_exact") is not True
                or recurrence.get("native_reset_equation_exact") is not True
                or not isinstance(amplitude.get("active_native_event_gate_count"), int)
                or amplitude.get("active_native_event_gate_count", 0) <= 0
                or not isinstance(amplitude.get("eligible_node_count"), int)
                or amplitude.get("eligible_node_count", 0) <= 0
                or not math.isfinite(amplitude.get("per_node_amplitude_factor_temporal_variance_mean", float("nan")))
                or amplitude.get("per_node_amplitude_factor_temporal_variance_mean", -1) < 0
                or not math.isfinite(amplitude.get("fraction_eligible_nodes_with_nonzero_temporal_variance", float("nan")))
                or not math.isfinite(amplitude.get("aggregate_amplitude_factor_mean", float("nan")))
                or not math.isfinite(amplitude.get("aggregate_amplitude_factor_std", float("nan")))
                or not math.isfinite(record.get("post_update1_zero_eta_Q_mean_abs_delta", float("nan")))
                or record.get("post_update1_zero_eta_Q_mean_abs_delta", -1) < 0):
            return False
        updates = record.get("updates")
        if not isinstance(updates, list) or [row.get("update") for row in updates] != [1, 2]:
            return False
        for row in updates:
            gradients = row.get("family_gradient_norms", {})
            changes = row.get("family_parameter_change_l2", {})
            if set(gradients) != {"native_core", "integration_tensors", "eta"} or set(changes) != set(gradients):
                return False
            if any(not math.isfinite(float(v)) or float(v) <= 0 for v in gradients.values()):
                return False
            if any(not math.isfinite(float(v)) or float(v) <= 0 for v in changes.values()):
                return False
        return True
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return False


def validate_report(path=OUTPUT, *, expected_fingerprint=None, upstream=None):
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        return validate_report_record(record,
            expected_fingerprint=expected_fingerprint or implementation_fingerprint(),
            upstream=upstream if upstream is not None else validate_upstream())
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return False


def _capture_rng(device):
    dev = torch.device(device)
    return (torch.get_rng_state().clone(), torch.cuda.get_rng_state(dev).clone())


def _restore_rng(state, device):
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1], torch.device(device))


def _trace_finite(trace):
    fields = sw143.TRACE_FIELDS
    result = {}
    for key in fields:
        tensor = trace.get(key)
        if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
            raise FloatingPointError(f"SW0158 trace {key} is absent or nonfinite")
        result[key] = tuple(tensor.shape)
    return result


def validate_recurrence_reset(trace):
    """Verify emitted S is the next-step dendrite/reset input in the trace."""
    state = trace.get("credit_state")
    if not isinstance(state, dict):
        raise ValueError("captured rollout state is required for recurrence validation")
    required = ("native_membrane", "previous_membrane", "previous_spike",
                "dendrite_previous_spike", "h_wave", "alpha", "threshold")
    if any(not torch.is_tensor(state.get(key)) for key in required):
        raise ValueError("captured recurrence/reset fields are incomplete")
    if not all(torch.isfinite(state[key]).all() for key in required):
        raise FloatingPointError("captured recurrence/reset trace is nonfinite")
    emitted = trace["component_spikes"].reshape(-1, trace["component_spikes"].shape[-2],
                                                  trace["component_spikes"].shape[-1])
    previous = state["previous_spike"]
    dendrite_previous = state["dendrite_previous_spike"]
    if not torch.equal(previous, dendrite_previous):
        raise AssertionError("dendrite and membrane did not receive the same prior emitted spike")
    if not torch.equal(previous[..., 1:], emitted[..., :-1]):
        raise AssertionError("adaptive emitted spike was not assigned to recurrent membrane state")
    expected_native = (state["previous_membrane"] * state["alpha"]
        + (1.0 - state["alpha"]) * R_m * state["h_wave"]
        - state["threshold"] * state["previous_spike"])
    if not torch.equal(state["native_membrane"], expected_native):
        raise AssertionError("native membrane reset equation/input changed")
    return {"finite": True, "same_spike_used_by_dendrite_and_reset": True,
            "recurrent_assignment_exact": True, "native_reset_equation_exact": True}


def amplitude_diagnostics(trace, *, settle=SETTLE):
    events = trace["component_events"].detach()[..., settle:]
    gates = trace["component_gates"].detach()[..., settle:]
    spikes = trace["component_spikes"].detach()[..., settle:]
    base = events * gates
    active = base != 0
    ratios = spikes[active] / base[active]
    counts = active.sum(dim=-1)
    # Compute per-node temporal variance only across timesteps with a nonzero
    # native E*g support; spatial variation between nodes is not temporal credit.
    safe_ratio = torch.where(active, spikes / base.clamp_min(torch.finfo(base.dtype).tiny),
                             torch.zeros_like(base))
    mean_by_node = safe_ratio.sum(dim=-1) / counts.clamp_min(1)
    variance_by_node = (((safe_ratio - mean_by_node.unsqueeze(-1)) ** 2)
                        * active.to(safe_ratio.dtype)).sum(dim=-1) / counts.clamp_min(1)
    eligible = counts >= 2
    node_variances = variance_by_node[eligible]
    return {"active_native_event_gate_count": int(active.sum().item()),
            "eligible_node_count": int(eligible.sum().item()),
            "per_node_amplitude_factor_temporal_variance_mean": (
                float(node_variances.mean().cpu()) if node_variances.numel() else None),
            "fraction_eligible_nodes_with_nonzero_temporal_variance": (
                float((node_variances > 1e-12).float().mean().cpu()) if node_variances.numel() else None),
            "aggregate_amplitude_factor_mean": float(ratios.float().mean().cpu()) if ratios.numel() else None,
            "aggregate_amplitude_factor_std": float(ratios.float().std(unbiased=False).cpu()) if ratios.numel() else None}


def _loss(foundation, gamma, masks, *, detached=False, live_tail=TAIL, capture_state=False):
    trace = spike_credit.late_rollout_credit(foundation.wrapped, gamma,
        total_steps=STEPS, live_tail_steps=live_tail, detached_control=detached,
        capture_state=capture_state)
    loss, details = objective.supervised_relation_loss(trace, masks, settle=SETTLE)
    return trace, loss, details


def _configure(foundation):
    wrapped, core = foundation.wrapped, foundation.wrapped.core
    foundation.encoder.requires_grad_(False)
    foundation.patcher.requires_grad_(False)
    if core.graph_generator is not None:
        core.graph_generator.requires_grad_(False)
    native = [p for name, p in core.named_parameters()
              if not name.startswith("graph_generator.")]
    integrations = [wrapped.a_d, wrapped.a_m, wrapped.b]
    for parameter in native + integrations:
        parameter.requires_grad_(True)
    wrapped.sw0158_eta.requires_grad_(True)
    if any(p.requires_grad for p in foundation.encoder.parameters()):
        raise AssertionError("source encoder must remain frozen in preflight")
    if any(p.requires_grad for p in foundation.patcher.parameters()):
        raise AssertionError("native patcher must remain frozen in preflight")
    if core.graph_generator is not None and any(p.requires_grad for p in core.graph_generator.parameters()):
        raise AssertionError("registered SW0143-T graph must remain frozen")
    families = {"native_core": native, "integration_tensors": integrations,
                "eta": [wrapped.sw0158_eta]}
    flattened = [p for values in families.values() for p in values]
    if len({id(p) for p in flattened}) != len(flattened):
        raise AssertionError("duplicate SW0158 selected parameters")
    optimizer = torch.optim.Adam([
        {"name": "native_core", "params": native, "lr": 3e-5},
        {"name": "integration_tensors", "params": integrations, "lr": 1e-3},
        {"name": "eta", "params": [wrapped.sw0158_eta], "lr": 1e-3}])
    return families, optimizer


def validate_upstream():
    source = sw155.source_contract()
    assets = sw130.validate_rgb_assets()
    t_ref = sw143.validate_arm_prediction(sw143._arm_dir("T", SEED), seed=SEED)
    if t_ref.get("arm") != "T" or not np.array_equal(
            t_ref["ids"], np.asarray(sw143.contract.IMAGE_IDS, dtype=np.int64)):
        raise ValueError("SW0158 requires the exact validated SW0143-T seed1 fixed320 result")
    report = spike_credit.registered_margin_rms()
    return {"source_sha256": source["source_sha256"],
            "source_manifest_sha256": sha256_file(source["manifest_path"]),
            "source_training_ids_sha256": sw155.ids_sha(source["image_ids"]),
            "source_training_ids": source["image_ids"],
            "rgb_asset_validation": assets,
            "sw0143_T_prediction_sha256": t_ref["prediction_sha256"],
            "sw0143_T_protocol_sha256": t_ref["protocol_sha256"],
            "sw0157_calibration_sha256": spike_credit.sha256_file(spike_credit.SW157_REPORT),
            "component_margin_rms": list(report)}


def run_preflight(*, device="cuda:0", output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0158 preflight: {output}")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0158 actual-source preflight requires its assigned CUDA device")
    torch.cuda.set_device(dev); torch.cuda.init()
    upstream = validate_upstream()
    src = sw155.source_contract()
    foundation = sw155._load_foundation(dev)
    if (foundation.provenance.get("source_core_sha256") != upstream["source_sha256"]
            or foundation.provenance.get("source_manifest_sha256") != upstream["source_manifest_sha256"]
            or foundation.provenance.get("rgb_asset_validation") != upstream["rgb_asset_validation"]
            or list(map(int, foundation.image_ids)) != src["image_ids"]
            or not np.array_equal(foundation.pool_indices, src["rows"])):
        raise ValueError("fresh native32 foundation differs from the pinned source contract")
    foundation.encoder.eval(); foundation.patcher.eval(); foundation.wrapped.eval()
    graph = foundation.wrapped.core.graph_generator
    if graph is None:
        raise ValueError("SW0158 requires the registered learned source graph")
    geodesic_operator.install_operator(graph, "T")
    spike_credit.install_spike_credit(foundation.wrapped, upstream["component_margin_rms"])

    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered source TRAIN RGB cache schema changed")
    rgb = torch.from_numpy(np.asarray(cache[src["rows"][:4]], dtype=np.uint8).copy())
    rgb_hash = hashlib.sha256(rgb.numpy().tobytes()).hexdigest()
    rgb_gpu = rgb.to(dev)
    with torch.no_grad():
        gamma = foundation.encode(rgb_gpu)
    if gamma.ndim != 3 or gamma.shape[1:] != (8, 1024) or not torch.isfinite(gamma).all():
        raise ValueError("source native32 gamma must be finite [B,8,1024]")
    masks_rgb, masks = sw155.load_aligned_train_rows(
        sw155.HDF5_PATH, src["image_ids"], np.asarray(cache[src["rows"][:4]]).copy(), 0, 4)
    if not torch.equal(masks_rgb, torch.from_numpy(np.asarray(cache[src["rows"][:4]]).copy())):
        raise AssertionError("canonical TRAIN HDF5 images differ from the source RGB cache")
    masks = masks.to(dev)
    labels = objective.modal_train_targets32(masks)

    with torch.no_grad():
        native = late_rollout(foundation.wrapped, gamma, total_steps=STEPS, live_tail_steps=0)
        zero = spike_credit.late_rollout_credit(
            foundation.wrapped, gamma, total_steps=STEPS, live_tail_steps=0)
    parity = sw143.assert_same_batch_trace(native, zero, "SW0158 eta-zero")
    from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
    native_q = spike_synchrony_affinity(native["spikes"], native["component_spikes"],
                                         settle=SETTLE, affinity_mode="spike")
    zero_q = spike_synchrony_affinity(zero["spikes"], zero["component_spikes"],
                                       settle=SETTLE, affinity_mode="spike")
    native_qcc = sw136._qcc_labels(native)
    zero_qcc = sw136._qcc_labels(zero)
    initial = {"six_trace_exact": all(parity.values()),
               "production_Q_exact": torch.equal(native_q, zero_q),
               "production_QCC_exact": np.array_equal(native_qcc, zero_qcc),
               "native_Q_sha256": hashlib.sha256(native_q.detach().cpu().numpy().tobytes()).hexdigest(),
               "native_QCC_sha256": hashlib.sha256(native_qcc.tobytes()).hexdigest(),
               "trace_parity": parity}
    if not all((initial["six_trace_exact"], initial["production_Q_exact"], initial["production_QCC_exact"])):
        raise AssertionError("eta-zero SW0158 path failed exact source-T parity")

    families, optimizer = _configure(foundation)
    frozen_modules = (("encoder", foundation.encoder), ("patcher", foundation.patcher),
                      ("graph", foundation.wrapped.core.graph_generator))
    frozen_before = _state_sha(frozen_modules)
    trainable_before = {name: [p.detach().clone() for p in params]
                        for name, params in families.items()}
    update_rows = []
    def do_update(update):
        nonlocal trainable_before
        optimizer.zero_grad(set_to_none=True)
        trace, loss, details = _loss(foundation, gamma, masks, live_tail=TAIL)
        _trace_finite(trace)
        if not torch.isfinite(loss):
            raise FloatingPointError("SW0158 preflight relation loss is nonfinite")
        loss.backward()
        grad_norms = {}
        for name, params in families.items():
            squares = 0.0
            for parameter in params:
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"nonfinite {name} gradient")
                    squares += float(parameter.grad.detach().double().square().sum().cpu())
            grad_norms[name] = math.sqrt(squares)
        if grad_norms["eta"] <= 0:
            raise FloatingPointError("SW0158 eta did not receive positive initial credit")
        torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups
            for p in group["params"] if p.grad is not None], 1.0)
        optimizer.step()
        with torch.no_grad():
            foundation.wrapped.project_integrations_()
        changes = {name: math.sqrt(sum(float((p.detach() - before).double().square().sum().cpu())
            for p, before in zip(families[name], trainable_before[name])))
            for name in families}
        trainable_before = {name: [p.detach().clone() for p in params]
                            for name, params in families.items()}
        update_rows.append({"update": update, "loss": float(loss.detach().cpu()),
                            "family_gradient_norms": grad_norms,
                            "family_parameter_change_l2": changes,
                            "pair_counts": details["pair_counts"].detach().cpu().tolist()})
        del trace, loss, details

    do_update(1)

    with torch.no_grad():
        recurrence_trace = spike_credit.late_rollout_credit(foundation.wrapped, gamma,
            total_steps=STEPS, live_tail_steps=0, capture_state=True)
    recurrence = validate_recurrence_reset(recurrence_trace)
    amplitude = amplitude_diagnostics(recurrence_trace)
    from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
    post_update_q = spike_synchrony_affinity(recurrence_trace["spikes"],
        recurrence_trace["component_spikes"], settle=SETTLE, affinity_mode="spike")
    del recurrence_trace
    eta_after_update1 = foundation.wrapped.sw0158_eta.detach().clone()
    qchange_rng = _capture_rng(dev)
    try:
        with torch.no_grad():
            foundation.wrapped.sw0158_eta.zero_()
            zero_trace_post_update1 = spike_credit.late_rollout_credit(
                foundation.wrapped, gamma, total_steps=STEPS, live_tail_steps=0)
            zero_q_post_update1 = spike_synchrony_affinity(zero_trace_post_update1["spikes"],
                zero_trace_post_update1["component_spikes"], settle=SETTLE,
                affinity_mode="spike")
            post_update1_q_delta = float((post_update_q - zero_q_post_update1).abs().mean().cpu())
            del zero_trace_post_update1, zero_q_post_update1
    finally:
        with torch.no_grad():
            foundation.wrapped.sw0158_eta.copy_(eta_after_update1)
        _restore_rng(qchange_rng, dev)

    # Attribute the second-step temporal pathway at identical post-step-1 state,
    # before the actual second optimizer update. Restore identical RNG each run.
    membrane_parameters = list(foundation.wrapped.core.membrane_layer.parameters()) + [foundation.wrapped.b]
    def matched(mode):
        rng = _capture_rng(dev)
        trace, loss, _ = _loss(foundation, gamma, masks, detached=mode, live_tail=TAIL)
        _trace_finite(trace)
        vector = _flat_grads(loss, membrane_parameters)
        snapshot = {key: value.detach().clone() for key, value in trace.items()
                    if key in sw143.TRACE_FIELDS}
        q = spike_synchrony_affinity(trace["spikes"], trace["component_spikes"],
                                     settle=SETTLE, affinity_mode="spike").detach()
        loss_value = float(loss.detach().cpu())
        del trace, loss
        _restore_rng(rng, dev)
        return vector, snapshot, q, loss_value

    initial_rng = _capture_rng(dev)
    live1 = matched(False)
    _restore_rng(initial_rng, dev)
    live2 = matched(False)
    _restore_rng(initial_rng, dev)
    detached = matched(True)
    _restore_rng(initial_rng, dev)
    equality = {key: torch.equal(live1[1][key], detached[1][key]) for key in sw143.TRACE_FIELDS}
    equality["production_Q"] = torch.equal(live1[2], detached[2])
    equality["loss"] = live1[3] == detached[3]
    residual = _norm(live1[0] - live2[0])
    extra_credit = _norm(live1[0] - detached[0])
    summary = temporal_credit_summary(live1[0], live2[0], detached[0], equality)
    credit = {"same_state_same_rng_forward_exact": all(equality.values()),
              "forward_parity": equality,
              "live_gradient_finite": summary["live_gradient_finite"],
              "detached_gradient_finite": bool(torch.isfinite(detached[0]).all()),
              "live_detached_membrane_gradient_delta_l2": summary["live_detached_membrane_gradient_delta_l2"],
              "repeated_live_gradient_residual_l2": summary["repeated_live_gradient_residual_l2"],
              "additional_membrane_credit_detected": summary["additional_membrane_credit_detected"]}

    do_update(2)

    frozen_after = _state_sha(frozen_modules)
    report = {"experiment": EXPERIMENT, "status": "complete", "seed": SEED,
        "implementation_fingerprint": implementation_fingerprint(), "upstream": upstream,
        "source_positions": list(POSITIONS), "image_ids": src["image_ids"][:4],
        "rgb_batch_sha256": rgb_hash, "steps": STEPS, "settle": SETTLE,
        "live_tail_steps": TAIL, "optimizer_updates": 2,
        "ground_truth_used": True, "ground_truth_scope": "the four registered source TRAIN masks only",
        "validation_or_test_ground_truth_used": False,
        "eta_zero_source_T_parity": initial, "updates": update_rows,
        "matched_temporal_credit": credit,
        "post_update1_recurrence_reset": recurrence,
        "post_update1_amplitude_diagnostics": amplitude,
        "post_update1_zero_eta_Q_mean_abs_delta": post_update1_q_delta,
        "source_encoder_patcher_and_graph_frozen_state_sha256_before": frozen_before,
        "source_encoder_patcher_and_graph_frozen_state_sha256_after": frozen_after,
        "training_admission": bool(initial["six_trace_exact"] and initial["production_Q_exact"]
            and initial["production_QCC_exact"] and credit["same_state_same_rng_forward_exact"]
            and credit["live_gradient_finite"] and credit["detached_gradient_finite"]
            and credit["additional_membrane_credit_detected"]
            and frozen_before == frozen_after
            and all(all(value > 0 for value in row["family_gradient_norms"].values())
                    for row in update_rows)
            and all(all(value > 0 for value in row["family_parameter_change_l2"].values())
                    for row in update_rows)
            and recurrence["native_reset_equation_exact"]
            and amplitude["active_native_event_gate_count"] > 0),
        "limitations": ["two disposable updates on the first four TRAIN images only",
            "not a full training result or performance claim", "E binary occupancy is not modified"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    if not validate_report(output, expected_fingerprint=report["implementation_fingerprint"],
                           upstream=upstream):
        raise ValueError("written SW0158 preflight receipt failed its own strict validator")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(run_preflight(device=args.device, output=args.output), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
