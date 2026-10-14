"""Bounded SW0159 source-T parity and two-update feasibility preflight."""
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
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"), str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_operator, run as sw143
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
from collaborative_test.SW_0155_supervised_spike_relations import objective, train as sw155
from collaborative_test.SW_0159_one_shot_event_regime.calibration import (
    REPORT_SHA256, install_bias, one_shot_bias, sha256_file, source_margin_rms,
)
from snn_kuramoto_bidirectional.membrane_layer import act_fun_adp
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity

SEED = 1
POSITIONS = (0, 1, 2, 3, 2048, 2049, 2050, 2051)
STEPS, SETTLE, TAIL = 1024, 512, 64
OUTPUT = HERE / "results_archive" / "preflight_seed1.json"
EXPERIMENT = "SW0159_one_shot_event_regime_preflight"


def fingerprint():
    rels = [
        "collaborative_test/SW_0159_one_shot_event_regime/protocol.json",
        "collaborative_test/SW_0159_one_shot_event_regime/README.md",
        "collaborative_test/SW_0159_one_shot_event_regime/calibration.py",
        "collaborative_test/SW_0159_one_shot_event_regime/preflight.py",
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0130_phase_state_integration/model.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0155_supervised_spike_relations/objective.py",
        "collaborative_test/SW_0157_train_gate_credit/results_archive/completed_20261014/0_train_gate_credit_seed1.json",
        "snn_kuramoto_bidirectional/spike_classifier.py",
        "snn_kuramoto_bidirectional/membrane_layer.py",
    ]
    paths = [ROOT / rel for rel in rels]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("a frozen SW0159 source dependency is missing")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def initial_bias(margin_rms=None):
    values = source_margin_rms() if margin_rms is None else tuple(margin_rms)
    return one_shot_bias(values)


def _trace_q(trace):
    q = spike_synchrony_affinity(trace["spikes"], trace["component_spikes"],
                                  settle=SETTLE, affinity_mode="spike")
    qcc = sw136._qcc_labels(trace)
    return q, qcc


def _six_trace_equal(left, right):
    names = tuple(sw143.TRACE_FIELDS)
    return {name: bool(torch.equal(left[name], right[name])) for name in names}


def _family_norm(params):
    acc = 0.0
    for p in params:
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise FloatingPointError("nonfinite selected gradient")
            acc += float(p.grad.detach().double().square().sum().cpu())
    return math.sqrt(acc)


def validate_train_rows_api():
    """Assert the actual SW0155 loader accepts ordered uint8 cache rows."""
    src = sw155.source_contract()
    validate_source_contract_types(src)
    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    rgb_chunks, mask_chunks = [], []
    for start in (0, 2048):
        rgb = np.asarray(cache[src["rows"][start:start+4]], dtype=np.uint8).copy()
        if rgb.shape != (4, 128, 128, 3):
            raise ValueError("source cache RGB batch schema changed")
        local_ids = src["image_ids"][start:start+4]
        aligned, masks = sw155.load_aligned_train_rows(
            sw155.HDF5_PATH, local_ids, rgb, 0, 4)
        if aligned.dtype != torch.uint8 or tuple(aligned.shape) != (4, 128, 128, 3):
            raise AssertionError("SW0155 loader did not return registered uint8 RGB")
        if masks.shape[0] != 4:
            raise AssertionError("actual source contract/loader row alignment failed")
        rgb_chunks.append(rgb); mask_chunks.append(masks)
    rgb = np.concatenate(rgb_chunks)
    return src, cache, rgb, torch.from_numpy(rgb.copy()), torch.cat(mask_chunks)


def validate_upstream():
    src = sw155.source_contract()
    validate_source_contract_types(src)
    assets = sw130.validate_rgb_assets()
    ref = sw143.validate_arm_prediction(sw143._arm_dir("T", SEED), seed=SEED)
    if ref.get("arm") != "T" or not np.array_equal(ref["ids"], np.asarray(sw143.contract.IMAGE_IDS)):
        raise ValueError("registered SW0143-T seed1 fixed320 reference failed validation")
    rms = source_margin_rms()
    init = one_shot_bias(rms)
    init = {key: (list(value) if isinstance(value, tuple) else value) for key, value in init.items()}
    return {"source_sha256": src["source_sha256"],
            "source_manifest_sha256": sha256_file(src["manifest_path"]),
            "source_training_ids_sha256": sw155.ids_sha(src["image_ids"]),
            "source_training_ids": src["image_ids"],
            "rgb_asset_validation": assets,
            "sw0143_T_prediction_sha256": ref["prediction_sha256"],
            "sw0143_T_protocol_sha256": ref["protocol_sha256"],
            "sw0157_calibration_sha256": REPORT_SHA256,
            "margin_rms": list(rms), "initialization": init}


def validate_source_contract_types(source):
    """Check the concrete SW0155 producer contract before indexing its RGB cache."""
    ids, rows = source.get("image_ids"), source.get("rows")
    if (not isinstance(ids, list) or len(ids) != 4096
            or any(type(value) is not int for value in ids)
            or not isinstance(rows, np.ndarray) or rows.shape != (4096,)
            or rows.dtype.kind not in "iu" or len(np.unique(rows)) != 4096):
        raise ValueError("SW0155 source_contract ID/list or pool-row/array schema changed")
    return True


def _rollout(foundation, gamma):
    trace = late_rollout(foundation.wrapped, gamma, total_steps=STEPS, live_tail_steps=TAIL)
    for key in sw143.TRACE_FIELDS:
        if key not in trace or not torch.isfinite(trace[key]).all():
            raise FloatingPointError(f"nonfinite/missing trace field {key}")
    return trace


def _event_regime(trace, bias, reference_events=None):
    """Summarize only settled native events; never infer E where gate is closed."""
    threshold = (0.06 * torch.exp(bias.detach())).view(1, 4, 1, 1)
    events = act_fun_adp(trace["component_membrane"] - threshold)
    gates = trace["component_gates"]
    spikes = trace["component_spikes"]
    if not torch.equal(spikes, events * gates):
        raise AssertionError("native actual spikes are not exactly E*g")
    settled_e, settled_g = events[..., SETTLE:], gates[..., SETTLE:]
    support = settled_g > 0
    transitions = (settled_e[..., 1:] != settled_e[..., :-1]) & support[..., 1:]
    result = {"settled_event_on_fraction": float(settled_e.float().mean().cpu()),
        "settled_gate_positive_count": int(support.sum().item()),
        "settled_event_on_fraction_when_gate_positive": (
            float(settled_e[support].float().mean().cpu()) if bool(support.any()) else None),
        "settled_on_off_transitions_gate_positive": int(transitions.sum().item()),
        "per_component_settled_on_fraction": [float(settled_e[:, c].float().mean().cpu()) for c in range(4)]}
    if reference_events is not None:
        flips = ((settled_e != reference_events[..., SETTLE:]) & support)
        result["settled_event_flips_vs_source_gate_positive"] = int(flips.sum().item())
        result["settled_event_differs_from_source"] = bool(flips.any())
    return result, events


def run_preflight(*, device="cuda:0", output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserving existing SW0159 preflight: {output}")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0159 source feasibility requires the assigned CUDA device")
    torch.cuda.set_device(dev)
    upstream = validate_upstream()
    src, cache, rgb_rows, aligned_rgb, masks = validate_train_rows_api()
    foundation = sw155._load_foundation(dev)
    if (foundation.provenance.get("source_core_sha256") != upstream["source_sha256"]
            or foundation.provenance.get("source_manifest_sha256") != upstream["source_manifest_sha256"]
            or foundation.provenance.get("rgb_asset_validation") != upstream["rgb_asset_validation"]
            or list(map(int, foundation.image_ids)) != src["image_ids"]
            or not np.array_equal(foundation.pool_indices, src["rows"])):
        raise ValueError("fresh source97 seed1 foundation differs from registered source")
    foundation.encoder.eval(); foundation.patcher.eval(); foundation.wrapped.eval()
    graph = foundation.wrapped.core.graph_generator
    if graph is None:
        raise ValueError("source-T learned graph is required")
    geodesic_operator.install_operator(graph, "T")
    ids = src["image_ids"]
    if not np.array_equal(aligned_rgb.numpy(), rgb_rows):
        raise AssertionError("canonical TRAIN RGB differs from the registered source cache")
    source_digest = hashlib.sha256(rgb_rows.tobytes()).hexdigest()
    gamma_chunks = []
    for start in (0, 4):
        rgb = torch.from_numpy(rgb_rows[start:start+4]).to(dev)
        with torch.no_grad():
            gamma_chunks.append(foundation.encode(rgb).detach())
    gamma = torch.cat(gamma_chunks)
    if tuple(gamma.shape) != (8, 8, 1024) or not torch.isfinite(gamma).all():
        raise ValueError("registered source gamma must be finite [8,8,1024]")

    control = foundation.wrapped
    with torch.no_grad():
        control.b.zero_()
    with torch.no_grad():
        control_trace = _rollout(foundation, gamma[:4])
        control_q, control_qcc = _trace_q(control_trace)
        source_stats, control_events = _event_regime(control_trace, control.b)
        source_trace = late_rollout(control, gamma[:4], total_steps=STEPS, live_tail_steps=TAIL)
        source_q, source_qcc = _trace_q(source_trace)
    trace_parity = _six_trace_equal(control_trace, source_trace)
    if (not all(trace_parity.values()) or not torch.equal(control_q, source_q)
            or not np.array_equal(control_qcc, source_qcc)):
        raise AssertionError("control b=0 failed exact SW0143-T source parity")
    del source_trace, source_q

    control_stats = [source_stats]
    control_events_by_batch = [control_events]
    with torch.no_grad():
        second_control = _rollout(foundation, gamma[4:8])
        second_stats, second_events = _event_regime(second_control, control.b)
    control_stats.append(second_stats); control_events_by_batch.append(second_events)
    del second_control

    candidate = sw155._load_foundation(dev)
    candidate.encoder.eval(); candidate.patcher.eval(); candidate.wrapped.eval()
    geodesic_operator.install_operator(candidate.wrapped.core.graph_generator, "T")
    if _module_sha(candidate.wrapped.core.graph_generator) != _module_sha(graph):
        raise AssertionError("independent candidate graph initialization differs from frozen source")
    initial = initial_bias(upstream["margin_rms"])
    install_bias(candidate.wrapped, initial["bias"])
    # Reuse identical gamma and mask batches; this is an 8-image feasibility set,
    # not the full registered training run.
    masks_t = masks.to(dev)
    threshold = 0.06 * torch.exp(candidate.wrapped.b.detach())
    if not torch.allclose(threshold.cpu(), torch.tensor(initial["thresholds"]), rtol=2e-6, atol=1e-6):
        raise AssertionError("installed initial threshold does not match the frozen source calibration")
    event_rows, q_means, q_hash = [], [], hashlib.sha256()
    qcc_count, event_changed, transition_count = 0, False, 0
    for batch_index, start in enumerate((0, 4)):
        with torch.no_grad():
            candidate_trace = _rollout(candidate, gamma[start:start+4])
            event_row, _ = _event_regime(candidate_trace, candidate.wrapped.b,
                                         reference_events=control_events_by_batch[batch_index])
            event_rows.append(event_row)
            event_changed |= event_row["settled_event_differs_from_source"]
            transition_count += event_row["settled_on_off_transitions_gate_positive"]
            candidate_q, candidate_qcc = _trace_q(candidate_trace)
            if not torch.isfinite(candidate_q).all() or np.any(candidate_qcc < 0):
                raise FloatingPointError("candidate production Q/QCC is nonfinite or invalid")
            q_means.append(float(candidate_q.detach().mean().cpu()))
            q_hash.update(candidate_q.detach().cpu().contiguous().numpy().tobytes())
            qcc_count += int(sum(len(np.unique(row[row >= 0])) for row in candidate_qcc))
            if not torch.isfinite(candidate_trace["component_membrane"]).all():
                raise FloatingPointError("candidate membrane trace is nonfinite")
            del candidate_trace, candidate_q
    candidate_q_mean = float(np.mean(q_means))
    candidate_q_sha = q_hash.hexdigest()
    del control_trace

    core = candidate.wrapped.core
    core.graph_generator.requires_grad_(False)
    candidate.encoder.requires_grad_(False); candidate.patcher.requires_grad_(False)
    native = [p for n, p in core.named_parameters() if not n.startswith("graph_generator.")]
    integrations = [candidate.wrapped.a_d, candidate.wrapped.a_m, candidate.wrapped.b]
    for p in native + integrations:
        p.requires_grad_(True)
    families = {"native_core": native, "integration_tensors": integrations}
    membrane_parameters = list(core.membrane_layer.parameters())
    optimizer = torch.optim.Adam([
        {"name": "native_core", "params": native, "lr": 3e-5},
        {"name": "integration_tensors", "params": integrations, "lr": 1e-3}])
    update_rows = []
    for index in range(2):
        start = index * 4
        local_gamma, local_masks = gamma[start:start+4], masks_t[start:start+4]
        optimizer.zero_grad(set_to_none=True)
        totals = []
        bias_grad_norm = 0.0
        for micro in range(4):
            take = micro
            trace = _rollout(candidate, local_gamma[take:take+1])
            loss, details = objective.supervised_relation_loss(trace, local_masks[take:take+1], settle=SETTLE)
            if not torch.isfinite(loss):
                raise FloatingPointError("candidate supervised loss is nonfinite")
            (loss / 4).backward(); totals.append(float(loss.detach().cpu()))
            if candidate.wrapped.b.grad is not None:
                if not torch.isfinite(candidate.wrapped.b.grad).all():
                    raise FloatingPointError("nonfinite existing b gradient")
                bias_grad_norm = float(candidate.wrapped.b.grad.detach().double().norm().cpu())
            del trace, loss, details
        norms = {name: _family_norm(params) for name, params in families.items()}
        membrane_gradient_norm = _family_norm(membrane_parameters)
        torch.nn.utils.clip_grad_norm_(native + integrations, 1.0)
        before_b = candidate.wrapped.b.detach().clone()
        before_native = [p.detach().clone() for p in native]
        before_integration = [p.detach().clone() for p in integrations]
        before_membrane = [p.detach().clone() for p in membrane_parameters]
        optimizer.step()
        with torch.no_grad(): candidate.wrapped.project_integrations_()
        b_change = float((candidate.wrapped.b.detach() - before_b).norm().cpu())
        native_change = math.sqrt(sum(float((p.detach() - old).double().square().sum().cpu())
                                      for p, old in zip(native, before_native)))
        integration_change = math.sqrt(sum(float((p.detach() - old).double().square().sum().cpu())
                                           for p, old in zip(integrations, before_integration)))
        membrane_change = math.sqrt(sum(float((p.detach() - old).double().square().sum().cpu())
                                         for p, old in zip(membrane_parameters, before_membrane)))
        update_rows.append({"update": index + 1, "mean_loss": float(np.mean(totals)),
                            "family_gradient_norms": norms, "b_gradient_norm": bias_grad_norm,
                            "membrane_gradient_norm": membrane_gradient_norm,
                            "membrane_parameter_change_l2": membrane_change,
                            "b_parameter_change_l2": b_change,
                            "family_parameter_change_l2": {"native_core": native_change,
                                "integration_tensors": integration_change}})
    rejection_reasons = []
    if not event_changed:
        rejection_reasons.append("initial_candidate_events_match_source")
    if transition_count <= 0:
        rejection_reasons.append("no_gate_positive_settled_event_transitions")
    for row in update_rows:
        if row["b_gradient_norm"] <= 0:
            rejection_reasons.append(f"update_{row['update']}_existing_b_gradient_zero")
        if row["membrane_gradient_norm"] <= 0:
            rejection_reasons.append(f"update_{row['update']}_membrane_gradient_zero")
        if row["b_parameter_change_l2"] <= 0:
            rejection_reasons.append(f"update_{row['update']}_existing_b_unchanged")
        if row["membrane_parameter_change_l2"] <= 0:
            rejection_reasons.append(f"update_{row['update']}_membrane_parameters_unchanged")
        if any(value <= 0 for value in row["family_parameter_change_l2"].values()):
            rejection_reasons.append(f"update_{row['update']}_selected_family_unchanged")
    report = {"experiment": EXPERIMENT, "status": "complete", "seed": SEED,
        "implementation_fingerprint": fingerprint(), "upstream": upstream,
        "source_positions": list(POSITIONS), "image_ids": [ids[i] for i in POSITIONS],
        "rgb_batch_sha256": source_digest, "steps": STEPS, "settle": SETTLE,
        "live_tail_steps": TAIL, "optimizer_updates": 2,
        "preflight_logical_batch": 4, "preflight_microbatch": 1,
        "control_source_T_parity": {"trace_fields": trace_parity,
            "six_trace_exact": all(trace_parity.values()), "production_Q_exact": True,
            "production_QCC_exact": True},
        "initialization": initial, "thresholds_after_init": threshold.cpu().tolist(),
        "candidate_event_changed_from_source": bool(event_changed),
        "candidate_temporal_event_transitions_gate_positive": transition_count,
        "source_event_regime_by_batch": control_stats,
        "candidate_event_regime_by_batch": event_rows,
        "candidate_initial_production_Q_mean": candidate_q_mean,
        "candidate_initial_production_Q_sha256": candidate_q_sha,
        "candidate_initial_QCC_label_count_sum": qcc_count,
        "candidate_updates": update_rows, "ground_truth_used": True,
        "ground_truth_scope": "the eight registered source TRAIN masks only",
        "validation_or_test_ground_truth_used": False,
        "training_admission": not rejection_reasons,
        "admission_reasons": rejection_reasons,
        "interpretation_limits": ["two disposable updates on eight TRAIN images only",
            "threshold initialization jointly changes event cutoff and reset; no separate causal attribution",
            "no full-training or validation performance claim"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    if not validate_report(output, expected_fingerprint=report["implementation_fingerprint"], upstream=upstream):
        raise ValueError("written SW0159 preflight receipt failed strict validation")
    return report


def _module_sha(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def validate_report_record(record, *, expected_fingerprint=None, upstream=None):
    """Strict, CPU-only preflight receipt contract for the owner queue."""
    try:
        if (record.get("experiment") != EXPERIMENT or record.get("status") != "complete"
                or record.get("seed") != SEED or record.get("optimizer_updates") != 2
                or record.get("source_positions") != list(POSITIONS)
                or record.get("preflight_logical_batch") != 4
                or record.get("preflight_microbatch") != 1
                or len(record.get("image_ids", [])) != 8
                or record.get("ground_truth_used") is not True
                or record.get("ground_truth_scope") != "the eight registered source TRAIN masks only"
                or record.get("validation_or_test_ground_truth_used") is not False
                or type(record.get("training_admission")) is not bool
                or not isinstance(record.get("admission_reasons"), list)):
            return False
        if expected_fingerprint is not None and record.get("implementation_fingerprint") != expected_fingerprint:
            return False
        if upstream is not None and record.get("upstream") != upstream:
            return False
        if upstream is not None and record["image_ids"] != [
                upstream["source_training_ids"][i] for i in POSITIONS]:
            return False
        parity = record["control_source_T_parity"]
        if (parity.get("six_trace_exact") is not True or parity.get("production_Q_exact") is not True
                or parity.get("production_QCC_exact") is not True
                or set(parity.get("trace_fields", {})) != set(sw143.TRACE_FIELDS)
                or not all(value is True for value in parity["trace_fields"].values())):
            return False
        init = record["initialization"]
        if (init.get("fraction") != 0.5 or len(init.get("margin_rms", [])) != 4
                or len(init.get("thresholds", [])) != 4 or len(init.get("bias", [])) != 4):
            return False
        for rms, threshold, bias in zip(init["margin_rms"], init["thresholds"], init["bias"]):
            if not all(math.isfinite(float(v)) for v in (rms, threshold, bias)) or rms <= 0 or threshold <= 0:
                return False
            if abs(0.06 * math.exp(float(bias)) - float(threshold)) > 2e-5:
                return False
            if abs(float(threshold) - 0.5 * float(rms)) > 1e-8:
                return False
        actual_thresholds = record.get("thresholds_after_init", [])
        if len(actual_thresholds) != 4 or any(
                not math.isfinite(float(actual)) or not math.isclose(float(actual), float(wanted), rel_tol=2e-6, abs_tol=1e-6)
                for actual, wanted in zip(actual_thresholds, init["thresholds"])):
            return False
        if (type(record.get("candidate_event_changed_from_source")) is not bool
                or not isinstance(record.get("candidate_temporal_event_transitions_gate_positive"), int)
                or record.get("candidate_temporal_event_transitions_gate_positive", -1) < 0):
            return False
        if (not math.isfinite(float(record.get("candidate_initial_production_Q_mean", float("nan"))))
                or not isinstance(record.get("candidate_initial_QCC_label_count_sum"), int)
                or record.get("candidate_initial_QCC_label_count_sum", 0) < 0
                or len(str(record.get("candidate_initial_production_Q_sha256", ""))) != 64):
            return False
        if len(record.get("source_event_regime_by_batch", [])) != 2 or len(record.get("candidate_event_regime_by_batch", [])) != 2:
            return False
        for row in record["candidate_event_regime_by_batch"]:
            fraction = float(row["settled_event_on_fraction"])
            if (not math.isfinite(fraction) or not 0 <= fraction <= 1
                    or not isinstance(row.get("settled_gate_positive_count"), int)
                    or row["settled_gate_positive_count"] < 0):
                return False
            if type(row.get("settled_event_differs_from_source")) is not bool:
                return False
            for component_fraction in row.get("per_component_settled_on_fraction", []):
                if not math.isfinite(float(component_fraction)) or not 0 <= float(component_fraction) <= 1:
                    return False
        if (record["candidate_event_changed_from_source"] != any(
                row["settled_event_differs_from_source"] for row in record["candidate_event_regime_by_batch"])
                or record["candidate_temporal_event_transitions_gate_positive"] != sum(
                    int(row["settled_on_off_transitions_gate_positive"])
                    for row in record["candidate_event_regime_by_batch"])):
            return False
        updates = record.get("candidate_updates")
        if not isinstance(updates, list) or [row.get("update") for row in updates] != [1, 2]:
            return False
        for row in updates:
            norms = row.get("family_gradient_norms", {})
            if set(norms) != {"native_core", "integration_tensors"}:
                return False
            if any(not math.isfinite(float(value)) or float(value) < 0 for value in norms.values()):
                return False
            if any(not math.isfinite(float(row.get(key, float("nan")))) or row[key] < 0
                   for key in ("mean_loss", "b_gradient_norm", "membrane_gradient_norm",
                               "membrane_parameter_change_l2", "b_parameter_change_l2")):
                return False
            changes = row.get("family_parameter_change_l2", {})
            if set(changes) != {"native_core", "integration_tensors"} or any(
                    not math.isfinite(float(value)) or float(value) < 0 for value in changes.values()):
                return False
        should_admit = (record["candidate_event_changed_from_source"]
            and record["candidate_temporal_event_transitions_gate_positive"] > 0
            and all(row["b_gradient_norm"] > 0 and row["membrane_gradient_norm"] > 0
                and row["membrane_parameter_change_l2"] > 0 and row["b_parameter_change_l2"] > 0
                and all(value > 0 for value in row["family_parameter_change_l2"].values())
                for row in updates))
        reasons = []
        if not record["candidate_event_changed_from_source"]:
            reasons.append("initial_candidate_events_match_source")
        if record["candidate_temporal_event_transitions_gate_positive"] <= 0:
            reasons.append("no_gate_positive_settled_event_transitions")
        for row in updates:
            if row["b_gradient_norm"] <= 0:
                reasons.append(f"update_{row['update']}_existing_b_gradient_zero")
            if row["membrane_gradient_norm"] <= 0:
                reasons.append(f"update_{row['update']}_membrane_gradient_zero")
            if row["b_parameter_change_l2"] <= 0:
                reasons.append(f"update_{row['update']}_existing_b_unchanged")
            if row["membrane_parameter_change_l2"] <= 0:
                reasons.append(f"update_{row['update']}_membrane_parameters_unchanged")
            if any(value <= 0 for value in row["family_parameter_change_l2"].values()):
                reasons.append(f"update_{row['update']}_selected_family_unchanged")
        if record["training_admission"] != should_admit or record["admission_reasons"] != reasons:
            return False
        return True
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return False


def validate_report(path=OUTPUT, *, expected_fingerprint=None, upstream=None):
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        return validate_report_record(record,
            expected_fingerprint=expected_fingerprint or fingerprint(),
            upstream=upstream)
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    report = run_preflight(device=args.device, output=args.output)
    print(json.dumps({"status": report["status"], "training_admission": report["training_admission"],
                      "image_count": len(report["image_ids"]), "optimizer_updates": 2}, allow_nan=False), flush=True)
    return 0 if report["training_admission"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
