"""No-training causal credit audit for the unchanged SW0097 SNN core."""
import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"), str(ROOT / "collaborative_test")]
from SW_0094_aligned_joint_pilot.run import GAMMA, VAL_GAMMA, DATASET, hparams, aligned_affinity
from SW_0102_cannot_link_draft.calibration_preflight import make_criterion, expected_training_ids
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks, spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity, spike_synchrony_components

CONTROL = ROOT / "trained_models/SW0097_graph_adaptation"
SOURCE = ROOT / "trained_models/SW0095_full70k_aligned_loss"
OUTDIR = ROOT / "collaborative_test/SW_0104_causal_credit/results_archive"
IDS = list(range(1320, 1336))
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_core(seed, device, steps, training=True):
    path = CONTROL / f"seed{seed}_positive_frozen/core.pt"
    hp = hparams("raw")
    hp.num_time_steps = steps
    core = S2NetCore(hp, device=device).to(device)
    core.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    core.graph_generator.requires_grad_(False)
    core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
    core.train(training)
    if core.graph_generator.uses_feedback:
        raise AssertionError("expected registered frozen graph without feedback");
    if core.kuramoto.spike_pulse_gain is not None:
        raise AssertionError("pulse coupling must be disabled for this diagnosis")
    return core, path


def install_capture(core):
    records = {k: [] for k in ("x", "event", "gate", "h", "mem", "theta")}
    module = sys.modules[core.membrane_layer.__class__.__module__]
    original = module.act_fun_adp
    def capture_act(x):
        e = original(x)
        records["x"].append(x)
        records["event"].append(e)
        return e
    module.act_fun_adp = capture_act
    def mem_pre(_module, inputs):
        h_wave, g_wave = inputs
        records["h"].append(h_wave)
        records["gate"].append(g_wave)
    def mem_post(_module, _inputs, output):
        records["mem"].append(output[0])
    def theta_post(_module, _inputs, output):
        records["theta"].append(output)
    handles = [core.membrane_layer.register_forward_pre_hook(mem_pre),
               core.membrane_layer.register_forward_hook(mem_post),
               core.kuramoto.register_forward_hook(theta_post)]
    return records, module, original, handles


def cleanup(module, original, handles):
    module.act_fun_adp = original
    for handle in handles:
        handle.remove()


def fold(records, key, bsz, dim, n, steps):
    rows = records[key]
    if len(rows) != steps:
        raise AssertionError(f"{key} hook count {len(rows)} != {steps}")
    return torch.stack(rows).reshape(steps, bsz, dim, n).permute(1, 2, 3, 0)


def q_from_components(z, settle):
    return spike_synchrony_affinity(z.mean(dim=1), z, settle=settle)


def safe_quantiles(value):
    flat = value.detach().float().reshape(-1)
    quantiles = torch.quantile(flat, torch.tensor([0., .01, .1, .5, .9, .99, 1.], device=flat.device))
    return {k: float(v) for k, v in zip(("min", "p01", "p10", "p50", "p90", "p99", "max"), quantiles)}


def norm(values):
    sq = 0.
    for g in values:
        if g is not None:
            if not torch.isfinite(g).all():
                raise FloatingPointError("nonfinite gradient")
            sq += float(g.detach().double().square().sum())
    return math.sqrt(sq)


def cosine(left, right):
    dot = sum(float((a.detach().double() * b.detach().double()).sum())
              for a, b in zip(left, right) if a is not None and b is not None)
    nl, nr = norm(left), norm(right)
    return dot / (nl * nr) if nl and nr else None


def capture_forward(core, gamma, steps):
    bsz = gamma.shape[0]
    records, module, original, handles = install_capture(core)
    try:
        outputs = core(gamma, return_core_out=True, num_time_steps=steps, return_theta=True)
    finally:
        cleanup(module, original, handles)
    _, spikes, core_out, theta_stack = outputs
    if len(records["x"]) != steps or len(records["event"]) != steps or len(records["gate"]) != steps \
            or len(records["h"]) != steps or len(records["mem"]) != steps or len(records["theta"]) != steps:
        raise AssertionError("one or more actual module hooks missed a time step")
    D, N = core.osc_dim, core.in_dim
    event = fold(records, "event", bsz, D, N, steps)
    x = fold(records, "x", bsz, D, N, steps)
    h = fold(records, "h", bsz, D, N, steps)
    mem = fold(records, "mem", bsz, D, N, steps)
    gate = fold(records, "gate", bsz, D, N, steps)
    theta = torch.stack(records["theta"], dim=1)
    if theta.shape != (bsz, steps, N, D):
        raise AssertionError(f"theta shape mismatch: {tuple(theta.shape)}")
    if any(not torch.equal(gate[:, 0], gate[:, d]) for d in range(1, D)):
        raise AssertionError("folded raw gate differs across repeated components")
    reconstructed = event * gate
    actual = core.last_component_spikes
    if actual is None or not torch.equal(reconstructed, actual):
        delta = float((reconstructed - actual).abs().max()) if actual is not None else None
        raise AssertionError(f"event*gate did not exactly reconstruct historical component spike; maxdiff={delta}")
    if tuple(actual.shape) != (bsz, D, N, steps):
        raise AssertionError(f"component history shape mismatch: {tuple(actual.shape)}")
    return {"spikes": spikes, "core_out": core_out, "theta": theta,
            "event": event, "gate": gate, "x": x, "h": h, "mem": mem,
            "z": reconstructed, "records": records}


def losses(capture, settle, criterion):
    e, g = capture["event"], capture["gate"]
    z = capture["z"]
    z_gate = e.detach() * g
    z_event = e * g.detach()
    q_full = q_from_components(z, settle)
    q_gate = q_from_components(z_gate, settle)
    q_event = q_from_components(z_event, settle)
    if not torch.equal(q_full, q_gate) or not torch.equal(q_full, q_event):
        raise AssertionError("detach readouts changed q values")
    lfull = 5. * criterion(plv=q_full)[0]
    lgate = 5. * criterion(plv=q_gate)[0]
    levent = 5. * criterion(plv=q_event)[0]
    if not (torch.equal(lfull, lgate) and torch.equal(lfull, levent)):
        raise AssertionError("detach readouts changed old 5x spike loss values")
    return lfull, lgate, levent


def param_family(name):
    for p in ("kuramoto.", "dendric_layer.", "membrane_layer.", "gamma_to_drive.", "graph_generator."):
        if name.startswith(p):
            return p[:-1]
    return "other"


def gradients(core, capture, lfull, lgate, levent):
    named = [(n, p) for n, p in core.named_parameters() if p.requires_grad]
    if any(n.startswith("graph_generator.") for n, _ in named):
        raise AssertionError("graph parameters must be frozen")
    params = [p for _, p in named]
    node_categories = {k: capture["records"][k] for k in ("theta", "h", "mem", "x", "gate")}
    node_tensors = [t for items in node_categories.values() for t in items]
    inputs = params + node_tensors
    outputs = []
    for loss, retain in ((lfull, True), (lgate, True), (levent, False)):
        outputs.append(torch.autograd.grad(loss, inputs, retain_graph=retain, allow_unused=True))
    gs = outputs
    pcount = len(params)
    gfull, ggate, gevent = [tuple(out[:pcount]) for out in gs]
    family = {}
    for fam in sorted({param_family(n) for n, _ in named}):
        ix = [i for i, (name, _) in enumerate(named) if param_family(name) == fam]
        a, b, c = [tuple(out[i] for i in ix) for out in (gfull, ggate, gevent)]
        difference = [None if x is None and y is None and z is None else
                      (torch.zeros_like(p) if x is None else x) -
                      (torch.zeros_like(p) if y is None else y) -
                      (torch.zeros_like(p) if z is None else z)
                      for (name, p), x, y, z in zip([named[i] for i in ix], a, b, c)]
        full_norm, gate_norm, event_norm = norm(a), norm(b), norm(c)
        rel = norm(difference) / max(full_norm, 1e-30)
        maxabs = max((float(x.abs().max()) for x in difference if x is not None), default=0.)
        full_max = max((float(x.abs().max()) for x in a if x is not None), default=0.)
        family[fam] = {"full_norm": full_norm, "gate_credit_norm": gate_norm,
                       "event_credit_norm": event_norm, "gate_event_cosine": cosine(b, c),
                       "sum_relative_residual": rel, "sum_max_abs_residual": maxabs,
                       "sum_tolerance": 1e-5 * max(1., full_max),
                       "sum_identity_pass": rel <= 1e-5 and maxabs <= 1e-5 * max(1., full_max)}
    node_norms = {}
    start = pcount
    for key, tensors in node_categories.items():
        stop = start + len(tensors)
        node_norms[key] = {}
        for name, out in zip(("full", "gate_credit", "event_credit"), gs):
            node_norms[key][name] = norm(out[start:stop])
        start = stop
    return {"parameters_by_family": family, "captured_node_gradient_norms": node_norms,
            "all_parameter_families_sum_identity_pass": all(x["sum_identity_pass"] for x in family.values())}


def event_stats(capture):
    e, x, g = capture["event"], capture["x"], capture["gate"]
    unit_time = e.permute(0, 1, 2, 3).reshape(-1, e.shape[-1])
    return {"event_logit_quantiles": safe_quantiles(x),
            "event_binary_rate": float(e.detach().mean()),
            "always_on_unit_fraction": float((unit_time.detach().mean(-1) == 1).float().mean()),
            "always_off_unit_fraction": float((unit_time.detach().mean(-1) == 0).float().mean()),
            "constant_unit_fraction": float((unit_time.detach().var(-1, unbiased=False) == 0).float().mean()),
            "gate_mean": float(g.detach().mean()), "gate_variance": float(g.detach().var(unbiased=False)),
            "spike_activity_occupancy": float(capture["z"].detach().mean())}


def run_train(device, smoke_only=False):
    gamma_path = ROOT / GAMMA.relative_to(ROOT)
    gamma = torch.load(gamma_path, map_location="cpu", weights_only=True, mmap=True)
    results = {"status": "running", "purpose": "SW0104 explicit output event/gate credit decomposition; no GT, no optimizer",
               "gamma_cache_sha256": sha(gamma_path), "device": str(device),
               "contract": {"batches_per_seed": 1 if smoke_only else 4, "batch": 16,
                            "steps": 64, "settle": 32, "shuffle": [117, 118, 119]}, "seeds": {}}
    criterion = make_criterion()
    for seed in (range(1) if smoke_only else range(3)):
        ids = expected_training_ids(seed)
        cdir = CONTROL / f"seed{seed}_positive_frozen"
        manifest = json.loads((cdir / "manifest.json").read_text())
        if manifest.get("training_ids") != ids or manifest.get("train_steps") != 64 or manifest.get("train_settle") != 32:
            raise AssertionError(f"seed{seed}: registered SW0097 training contract mismatch")
        core, checkpoint = load_core(seed, device, 64)
        source_sha = sha(SOURCE / f"seed{seed}/core.pt")
        checkpoint_sha = sha(checkpoint)
        if source_sha != manifest.get("source_sha256"):
            raise AssertionError(f"seed{seed}: source hash mismatch")
        gen = torch.Generator().manual_seed(117 + seed)
        indices = torch.randperm(70000, generator=gen)[:4096]
        rows = []
        for b in range(1 if smoke_only else 4):
            ix = indices[b * 16:(b + 1) * 16]
            cap = capture_forward(core, gamma[ix].to(device), 64)
            lfull, lgate, levent = losses(cap, 32, criterion)
            grad = gradients(core, cap, lfull, lgate, levent)
            row = {"batch": b, "training_ids": ids[b * 16:(b + 1) * 16],
                   "source_sha256": source_sha, "control_checkpoint_sha256": checkpoint_sha,
                   "full_weighted_spike_loss": float(lfull.detach()),
                   "gate_detached_readout_loss": float(lgate.detach()), "event_detached_readout_loss": float(levent.detach()),
                   "loss_values_exactly_equal": torch.equal(lfull, lgate) and torch.equal(lfull, levent),
                   "gradient_credit_decomposition": grad, "event_statistics": event_stats(cap),
                   "captured_steps": {k: len(v) if isinstance(v, list) else 1 for k, v in cap["records"].items()},
                   "gate_folded_shape": list(cap["gate"].shape), "event_folded_shape": list(cap["event"].shape),
                   "actual_event_times_gate_exactly_reconstructs_spikes": True,
                   "feedback_disabled": not core.graph_generator.uses_feedback,
                   "pulse_disabled": core.kuramoto.spike_pulse_gain is None}
            rows.append(row)
            del cap, lfull, lgate, levent
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results["seeds"][str(seed)] = {"source_sha256": source_sha, "no_optimizer_or_gt": True, "batches": rows}
    results["status"] = "smoke_passed" if smoke_only else "complete"
    return results


def prediction(groups):
    return spatial_components_to_patch_labels(groups, 16).long().cpu()


def run_validation(device):
    gamma = torch.load(VAL_GAMMA, map_location="cpu", weights_only=True)
    if tuple(gamma.shape) != (320, 8, 256):
        raise AssertionError("validation cache shape mismatch")
    records = {}
    # All actual/gate/event predictions are formed for every seed before labels.
    for seed in range(3):
        core, ckpt = load_core(seed, device, 1024, training=False)
        all_predictions = {"actual": [], "gate": [], "event": []}
        q_rows = {"actual": [], "gate": [], "event": []}
        gates, events = [], []
        for start in (0, 8):
            with torch.inference_mode():
                cap = capture_forward(core, gamma[start:start + 8].to(device), 1024)
            z, e, g = cap["z"], cap["event"], cap["gate"]
            q_a, q_g, q_e = (q_from_components(z, 512), q_from_components(g, 512), q_from_components(e, 512))
            if not torch.equal(z, core.last_component_spikes):
                raise AssertionError("validation reconstructed actual spikes mismatch")
            for name, q, comp, activity in (
                ("actual", q_a, z, cap["spikes"]),
                ("gate", q_g, g, g.mean(dim=1)),
                ("event", q_e, e, e.mean(dim=1))):
                groups = spike_synchrony_components(activity.detach().cpu(), synchrony_threshold=.50,
                    min_group_size=2, settle=512, components=comp.detach().cpu(), background="largest_component")
                all_predictions[name].append(prediction(groups))
                q_rows[name].append(q.detach().cpu())
            gates.append(g.detach().cpu()); events.append(e.detach().cpu())
            del cap, z, e, g, q_a, q_g, q_e
        for key in all_predictions:
            all_predictions[key] = torch.cat(all_predictions[key])
            q_rows[key] = torch.cat(q_rows[key])
        records[str(seed)] = {"checkpoint_sha256": sha(ckpt), "predictions": all_predictions,
                              "q": q_rows, "gate": torch.cat(gates), "event": torch.cat(events)}
        del core
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with h5py.File(DATASET, "r") as f:
        masks = torch.from_numpy(f["mask"][IDS])
        gt = clevr_mask_patch(masks, 8)["patch_labels"].reshape(16, 256).cpu()
    output = {"status": "complete", "purpose": "fixed validation causal-credit readouts; labels accessed only after all predictions",
              "ids": IDS, "batch": 8, "steps": 1024, "settle": 512, "threshold": .50,
              "ground_truth_used_for_training": False, "predictions_constructed_before_gt": True, "seeds": {}}
    for seed in range(3):
        r = records[str(seed)]
        per_pred = {}
        for key, pred in r["predictions"].items():
            score = evaluate_patch_masks(pred.reshape(16, 16, 16), gt.reshape(16, 16, 16))["per_image"]
            per_pred[key] = {m: [float(v) for v in score[m]] for m in METRICS}
        control_eval = json.loads((CONTROL / f"seed{seed}_positive_frozen/evaluation.json").read_text())
        archived = control_eval["sweep"][0]["scored_targets"]["our_hdf5"]["per_image"]
        q_actual = r["q"]["actual"]
        q_g, q_e = r["q"]["gate"], r["q"]["event"]
        off = ~torch.eye(256, dtype=torch.bool)
        gate_scalar = r["gate"][:, 0]
        centered = gate_scalar[..., 512:] - gate_scalar[..., 512:].mean(-1, keepdim=True)
        normalized = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        corr = torch.matmul(normalized, normalized.transpose(1, 2)).clamp(-1., 1.)
        q_gate_formula = torch.stack([corr.clamp_min(0)] * 4).prod(dim=0)
        max_gate_formula_diff = float((q_gate_formula - q_g).abs().max())
        output["seeds"][str(seed)] = {"checkpoint_sha256": r["checkpoint_sha256"],
            "mean_metrics": {k: {m: float(np.mean(v[m])) for m in METRICS} for k, v in per_pred.items()},
            "per_image_metrics": per_pred,
            "actual_matches_archived_97": {m: max(abs(a-b) for a,b in zip(per_pred["actual"][m], archived[m][:16])) <= 1e-9 for m in METRICS},
            "q_gate_is_positive_scalar_pearson_fourth_power_max_abs_diff": max_gate_formula_diff,
            "q_gate_scalar_formula_within_1e_6": max_gate_formula_diff <= 1e-6,
            "actual_vs_gate_edge_agreement_at_050": float(((q_actual >= .50) == (q_g >= .50))[off.expand(16, -1, -1)].float().mean()),
            "actual_vs_event_edge_agreement_at_050": float(((q_actual >= .50) == (q_e >= .50))[off.expand(16, -1, -1)].float().mean()),
            "actual_minus_gate_q_mean": float((q_actual - q_g).mean()),
            "actual_minus_event_q_mean": float((q_actual - q_e).mean()),
            "event_logit_proxy_binary_event_rate": float(r["event"].mean()),
            "event_always_on_fraction": float((r["event"].mean(-1) == 1).float().mean()),
            "gate_mean": float(r["gate"].mean()), "gate_variance": float(r["gate"].var(unbiased=False)),
            "group_counts": {k: [int((p > 0).unique().numel() - 1) for p in v] for k, v in r["predictions"].items()}}
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--stage", choices=("train-smoke", "train", "validation"), required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.stage.startswith("train"):
        result = run_train(device, smoke_only=args.stage == "train-smoke")
        name = "train_smoke.json" if args.stage == "train-smoke" else "train_credit.json"
    else:
        result = run_validation(device)
        name = "validation16.json"
    result["created_unix"] = time.time()
    path = OUTDIR / name
    if path.exists():
        raise FileExistsError(f"preserve existing result {path}")
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "path": str(path),
                      "seeds": list(result.get("seeds", {}))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
