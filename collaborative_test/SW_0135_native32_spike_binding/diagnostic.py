"""Read-only B1/B4 batching diagnostic for the registered native32 source."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for item in (str(ROOT), str(ROOT / "collaborative_test"), str(ROOT / "snn_kuramoto_bidirectional")):
    if item not in sys.path:
        sys.path.insert(0, item)

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding.foundation import (
    load_native32_foundation, make_criterion32, sha256_file,
)
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding.foundation import implementation_fingerprint as foundation_fingerprint
from snn_kuramoto_bidirectional.loss_function import phase_locking_value
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from snn_kuramoto_bidirectional.membrane_layer import act_fun_adp

STEPS, SETTLE, TAIL, BATCH = 1024, 512, 64, 4
B1_REPORT_SHA256 = "dcccd773cc303eb0bbb843b686e1bd61d9d9f998bc8f752a85f0b0b57a15445c"
B4_REPORT_SHA256 = "f604293605c880fb07a2d400914d744f6201ea06c4494bd605bbe5b4d1273214"


def _loss(theta, q, criterion):
    band = theta[:, SETTLE:]
    phase, phase_parts = criterion(plv=phase_locking_value(band, combine="mean"), theta=band)
    positive_q, q_parts = criterion(plv=q)
    return phase + 5.0 * positive_q, {
        "phase": phase, "positive_q": positive_q,
        "phase_parts": phase_parts, "q_parts": q_parts,
    }


def _roll(wrapped, gamma, criterion):
    trace = late_rollout(wrapped, gamma, total_steps=STEPS, live_tail_steps=TAIL)
    q = spike_synchrony_affinity(trace["spikes"], trace["component_spikes"],
                                 settle=SETTLE, affinity_mode="spike")
    old, parts = _loss(trace["theta"], q, criterion)
    return trace, q, old, parts


def _roll_with_graph(wrapped, gamma, criterion):
    captured = {}
    graph_module = wrapped.core.graph_generator
    handle = graph_module.register_forward_hook(
        lambda _module, _inputs, output: captured.__setitem__("graph", output.detach().clone()))
    try:
        trace, q, old, parts = _roll(wrapped, gamma, criterion)
    finally:
        handle.remove()
    if "graph" not in captured:
        raise RuntimeError("native graph generator did not execute during rollout")
    graph = captured["graph"]
    coupling = wrapped.core.kuramoto.prepare_coupling(
        graph, batch_size=gamma.shape[0], num_units=wrapped.core.in_dim,
        device=gamma.device)
    return trace, q, old, parts, graph, tuple(value.detach().clone() for value in coupling)


def implementation_fingerprint():
    values = foundation_fingerprint()
    values["collaborative_test/SW_0135_native32_spike_binding/diagnostic.py"] = sha256_file(
        HERE / "diagnostic.py")
    values["collaborative_test/SW_0135_native32_spike_binding/parity.py"] = sha256_file(
        HERE / "parity.py")
    return values


def validate_resource_references(foundation):
    """Bind this diagnostic to the completed B1 and B4 resource-only records."""
    b1_path = HERE / "results_archive/resource_probe_seed0.json"
    b4_path = HERE / "results_archive/batched_resource_probe_seed0.json"
    if sha256_file(b1_path) != B1_REPORT_SHA256 or sha256_file(b4_path) != B4_REPORT_SHA256:
        raise ValueError("registered B1/B4 resource reference bytes changed")
    b1 = json.loads(b1_path.read_text(encoding="utf-8"))
    b4 = json.loads(b4_path.read_text(encoding="utf-8"))
    expected_ids = [int(x) for x in foundation.image_ids[:16]]
    common = {"source_core_sha256": foundation.provenance["source_core_sha256"],
              "source_manifest_sha256": foundation.provenance["source_manifest_sha256"]}
    if (b1.get("status") != "resource_probe_complete" or b1.get("seed") != 0
            or b1.get("logical_batch_size") != 16 or b1.get("microbatch_size") != 1
            or b1.get("image_ids") != expected_ids
            or any(b1.get(k) != v for k, v in common.items())
            or b1.get("optimizer_updates") != 0
            or b1.get("ground_truth_used") is not False):
        raise ValueError("B1 resource reference does not match registered seed0 source/data")
    b1_sha = sha256_file(b1_path)
    if (b4.get("status") != "batched_resource_probe_complete"
            or b4.get("seed") != 0 or b4.get("microbatch_images") != 4
            or b4.get("image_ids") != expected_ids
            or any(b4.get(k) != v for k, v in common.items())
            or b4.get("training_admission") is not False
            or b4.get("optimizer_updates") != 0
            or b4.get("ground_truth_used") is not False
            or b4.get("b1_comparison", {}).get("sha256") != b1_sha):
        raise ValueError("B4 resource reference is not bound to the registered B1 record")
    return {"b1_path": str(b1_path), "b1_sha256": b1_sha,
            "b4_path": str(b4_path), "b4_sha256": B4_REPORT_SHA256,
            "image_ids": expected_ids}


def _tensor_stats(a, b):
    """Finite, descriptive CPU64 difference statistics; never a pass threshold."""
    a = torch.as_tensor(a).detach().to(device="cpu", dtype=torch.float64)
    b = torch.as_tensor(b).detach().to(device="cpu", dtype=torch.float64)
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("diagnostic comparison requires equal finite tensors")
    d = b - a
    na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    return {"shape": list(a.shape), "max_abs": float(d.abs().max()) if d.numel() else 0.0,
            "rms": float(d.square().mean().sqrt()) if d.numel() else 0.0,
            "relative_l2": float(torch.linalg.vector_norm(d) / na.clamp_min(1e-300)),
            "cosine": float(torch.dot(a.reshape(-1), b.reshape(-1)) /
                            (na * nb).clamp_min(1e-300))}


def _flatten_grads(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    pieces = []
    for p, grad in zip(parameters, grads):
        if grad is None:
            pieces.append(torch.zeros(p.numel(), dtype=torch.float64))
        else:
            if not torch.isfinite(grad).all():
                raise FloatingPointError("nonfinite diagnostic gradient")
            pieces.append(grad.detach().to(device="cpu", dtype=torch.float64).reshape(-1))
    return torch.cat(pieces)


def _trace_summary(trace):
    result = {}
    for key in ("theta", "membrane", "spikes", "component_membrane", "component_spikes",
                "component_gates"):
        value = trace[key].detach()
        result[key] = {"shape": list(value.shape), "mean": float(value.float().mean()),
                       "std": float(value.float().std(unbiased=False)),
                       "finite": bool(torch.isfinite(value).all())}
    return result


def run_diagnostic(seed=0, device="cuda:0", output=None):
    if int(seed) != 0:
        raise ValueError("registered B1/B4 comparison is seed0 only")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("registered batching diagnostic requires exclusive CUDA")
    if output is None:
        raise ValueError("--output is required; diagnostic records are create-once")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing diagnostic: {output}")
    torch.cuda.set_device(device); torch.cuda.init()
    foundation = load_native32_foundation(0, device, verify_rgb_assets=True)
    resource_references = validate_resource_references(foundation)
    if len(foundation.pool_indices) != 4096 or len(foundation.image_ids) != 4096:
        raise AssertionError("registered source order must contain 4096 TRAIN examples")
    rows = np.asarray(foundation.pool_indices[:BATCH], dtype=np.int64)
    ids = [int(x) for x in foundation.image_ids[:BATCH]]
    if len(set(ids)) != BATCH:
        raise AssertionError("first four registered global IDs must be unique")
    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r")
    images = torch.from_numpy(np.asarray(cache[rows]).copy()).to(device)
    del cache
    if images.dtype != torch.uint8 or tuple(images.shape) != (BATCH, 128, 128, 3):
        raise ValueError("registered TRAIN cache returned unexpected RGB batch")
    gamma = foundation.encode(images)
    if tuple(gamma.shape) != (BATCH, 8, 1024) or not torch.isfinite(gamma).all():
        raise ValueError("native32 encoder returned invalid gamma")
    criterion = make_criterion32()
    wrapped = foundation.wrapped
    wrapped.eval(); foundation.encoder.eval()
    joint_named, _groups = sw130.joint_parameters(wrapped, foundation.encoder)
    params = [p for _, p in joint_named]
    before = [p.detach().clone() for p in params]

    started = time.perf_counter()
    trace4, q4, loss4, parts4 = _roll(wrapped, gamma, criterion)
    # Isolate reduction semantics with fixed traces/Q; each row contributes equally.
    theta_leaf = trace4["theta"].detach().requires_grad_(True)
    q_leaf = q4.detach().requires_grad_(True)
    batch_loss, batch_parts = _loss(theta_leaf, q_leaf, criterion)
    batch_trace_grad = torch.autograd.grad(batch_loss, (theta_leaf, q_leaf), retain_graph=False)
    per_image = []
    per_image_theta, per_image_q = [], []
    for i in range(BATCH):
        ti = trace4["theta"][i:i + 1].detach().requires_grad_(True)
        qi = q4[i:i + 1].detach().requires_grad_(True)
        li, pi = _loss(ti, qi, criterion)
        gt, gq = torch.autograd.grad(li / BATCH, (ti, qi))
        per_image.append({"phase": float(pi["phase"].detach()),
                          "positive_q": float(pi["positive_q"].detach()),
                          "total": float(li.detach())})
        per_image_theta.append(gt.detach())
        per_image_q.append(gq.detach())
    theta_mean_grad = torch.cat(per_image_theta, dim=0)
    q_mean_grad = torch.cat(per_image_q, dim=0)
    reduction = {
        "batch_loss": float(batch_loss.detach()),
        "mean_per_image_loss": float(np.mean([x["total"] for x in per_image])),
        "batch_vs_mean_loss_abs": abs(float(batch_loss.detach()) -
                                       float(np.mean([x["total"] for x in per_image]))),
        "phase_batch": float(batch_parts["phase"].detach()),
        "phase_mean_per_image": float(np.mean([x["phase"] for x in per_image])),
        "positive_q_batch": float(batch_parts["positive_q"].detach()),
        "positive_q_mean_per_image": float(np.mean([x["positive_q"] for x in per_image])),
        "dtheta_gradient_batch_vs_mean_images": _tensor_stats(batch_trace_grad[0], theta_mean_grad),
        "dq_gradient_batch_vs_mean_images": _tensor_stats(batch_trace_grad[1], q_mean_grad),
        "per_image": per_image,
    }
    del theta_leaf, q_leaf, batch_trace_grad, theta_mean_grad, q_mean_grad
    del trace4, q4, loss4, parts4
    torch.cuda.empty_cache()

    # Compare true B4 and four B1 executions, accumulating the same mean objective gradient.
    trace4, q4, old4, _, graph4, coupling4 = _roll_with_graph(wrapped, gamma, criterion)
    grad4 = _flatten_grads(old4, params)
    trace_keys = ("theta", "membrane", "spikes", "component_membrane",
                  "component_spikes", "component_gates")
    trace4_cpu = {k: trace4[k].detach().cpu() for k in trace_keys}
    q4_cpu = q4.detach().cpu()
    graph4_cpu = graph4.detach().cpu()
    coupling4_cpu = tuple(value.cpu() for value in coupling4)
    old4_value = float(old4.detach())
    b4_summary = _trace_summary(trace4)
    threshold = 0.06 * torch.exp(wrapped.b.detach().view(1, 4, 1, 1))
    event4 = act_fun_adp(trace4["component_membrane"].detach() - threshold).cpu()
    event_rates = event4.float().mean(dim=(0, 2, 3)).tolist()
    del trace4, q4, old4, graph4, coupling4
    torch.cuda.empty_cache()
    b1_traces, b1_q, b1_losses, b1_grads, b1_gamma = [], [], [], [], []
    b1_graph, b1_coupling = [], []
    for i in range(BATCH):
        gi = foundation.encode(images[i:i + 1])
        ti, qi, li, _, graph_i, coupling_i = _roll_with_graph(wrapped, gi, criterion)
        b1_gamma.append(gi.detach().cpu())
        b1_traces.append({k: ti[k].detach().cpu() for k in trace_keys})
        b1_q.append(qi.detach().cpu())
        b1_graph.append(graph_i.detach().cpu())
        b1_coupling.append(tuple(value.cpu() for value in coupling_i))
        b1_losses.append(float(li.detach()))
        b1_grads.append(_flatten_grads(li / BATCH, params))
        del gi, ti, qi, li
    grad1 = torch.stack(b1_grads).sum(dim=0)
    keys = ("theta", "membrane", "spikes", "component_membrane", "component_spikes",
            "component_gates")
    trace_diffs = {}
    for key in keys:
        one = torch.cat([record[key] for record in b1_traces], dim=0)
        trace_diffs[key] = _tensor_stats(trace4_cpu[key], one)
    q1 = torch.cat(b1_q, dim=0)
    old1 = float(np.mean(b1_losses))
    gamma1 = torch.cat(b1_gamma, dim=0)
    graph1 = torch.cat(b1_graph, dim=0)
    coupling_diffs = [_tensor_stats(a, torch.cat([row[j] for row in b1_coupling], dim=0))
                       for j, a in enumerate(coupling4_cpu)]
    graph_diff = _tensor_stats(graph4_cpu, graph1)
    comparison_order = [("gamma", _tensor_stats(gamma, gamma1)),
                        ("prepared_graph", graph_diff),
                        ("prepared_coupling", coupling_diffs[0]),
                        ("theta", trace_diffs["theta"]),
                        ("membrane", trace_diffs["membrane"]),
                        ("emitted_spikes", trace_diffs["spikes"]),
                        ("component_spikes", trace_diffs["component_spikes"]),
                        ("q", _tensor_stats(q4_cpu, q1))]
    earliest_difference = next((name for name, stats in comparison_order
                                if stats["max_abs"] != 0.0), None)
    event1 = act_fun_adp(torch.cat([row["component_membrane"] for row in b1_traces], dim=0)
                         - threshold.cpu())
    event_diff = _tensor_stats(event4, event1)
    output_record = {
        "experiment": "SW0135_native32_spike_binding",
        "stage": "batching_diagnostic",
        "status": "diagnostic_complete",
        "seed": 0,
        "image_ids": ids,
        "pool_indices": rows.tolist(),
        "source_core_sha256": foundation.provenance["source_core_sha256"],
        "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
        "implementation_fingerprint": implementation_fingerprint(),
        "diagnostic_sha256": sha256_file(HERE / "diagnostic.py"),
        "resource_reference_records": resource_references,
        "prior_resource_norm_comparison": json.loads(
            Path(resource_references["b4_path"]).read_text(encoding="utf-8")
        )["b1_comparison"]["norm_comparison"],
        "batch_size": BATCH, "steps": STEPS, "settle": SETTLE,
        "live_tail_steps": TAIL, "ground_truth_used": False,
        "masks_read": False, "optimizer_updates": 0, "training_admission": False,
        "registered_reduction": reduction,
        "true_rollout_comparison": {
            "gamma_b4_vs_independent_b1": _tensor_stats(gamma, gamma1),
            "prepared_graph_b4_vs_b1": graph_diff,
            "prepared_coupling_b4_vs_b1": coupling_diffs,
            "trace_b4_vs_concatenated_b1": trace_diffs,
            "q_b4_vs_concatenated_b1": _tensor_stats(q4_cpu, q1),
            "old_loss_b4": old4_value, "old_loss_mean_b1": old1,
            "old_loss_abs_difference": abs(old4_value - old1),
            "old_gradient_b4_vs_accumulated_b1": _tensor_stats(grad4, grad1),
            "b1_old_losses": b1_losses,
            "b4_trace_summary": b4_summary,
            "event_occupancy_by_component": event_rates,
            "events_b4_vs_b1": event_diff,
            "topk_indices": None,
            "topk_unavailable_reason": "The registered graph-generator API returns adjacency but does not expose selection indices.",
            "earliest_nonzero_difference_stage": earliest_difference,
            "interpretation": "Descriptive batch-size differences; no equivalence gate or causal attribution.",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    if any(not math.isfinite(v) for v in [output_record["elapsed_seconds"], old1,
                                            output_record["true_rollout_comparison"]["old_loss_b4"]]):
        raise FloatingPointError("diagnostic report contains nonfinite scalar")
    if any(not torch.equal(p.detach(), original) for p, original in zip(params, before)):
        raise AssertionError("diagnostic modified source parameters")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(output_record, f, indent=2, allow_nan=False); f.write("\n")
    return output_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_diagnostic(args.seed, args.device, args.output)
    print(json.dumps({"status": result["status"], "seed": result["seed"],
                      "image_ids": result["image_ids"],
                      "training_admission": False}, allow_nan=False))


if __name__ == "__main__":
    main()
