"""Read-only, GT-free first-four-batch diagnostic for the completed SW0121 seed-0 pilot.

Loads immutable SW0097 source and completed SW0121/SW0117 artifacts, then forwards
the same four registered B16 TRAIN batches once through each core. Prints compact
JSON to stdout and never writes checkpoints, optimizer state, or result files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
from collaborative_test.SW_0121_competitive_assignment import run as pilot
from collaborative_test.SW_0117_joint_analytic_rgb import run as joint
from collaborative_test.SW_0117_joint_analytic_rgb import coordinator as joint_coordinator
from collaborative_test.SW_0121_competitive_assignment.assignment import (
    CompetitiveAssignmentHead, analytic_rgb_terms, patch_features,
)
from snn_kuramoto_bidirectional.spike_classifier import (
    spike_synchrony_affinity, spike_synchrony_components,
)

EPS = 1e-8
EXPECTED = {
    "manifest.json": "722a44d1762f77de42c77abadc84552b055ff98a19b6c9ce4cc88db876d85449",
    "core.pt": "8fde18e89d750c75b0a193247dd524159b59ca9ca519ed84c5613b1df52f06e9",
    "encoder.pt": "03bdcfc618c7ac1223f06e0684122040dc56735fa7ff42fa1b538b8e4c3f5642",
    "head.pt": "85e2dcfcfdcad8229de35b2f07bae5f3f8bd0688b3bbe02f69507cf3a92c1b39",
    "evaluation.json": "2ecfec351b01716d09d1d79149291885cb6dfaec366e30deb778ae38774c69e6",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def quantiles(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"count": 0, "min": None, "p01": None, "p50": None, "p99": None, "max": None}
    return {"count": int(a.size), "min": float(a.min()),
            "p01": float(np.quantile(a, .01)), "p50": float(np.quantile(a, .50)),
            "p99": float(np.quantile(a, .99)), "max": float(a.max())}


def _component_rhos(components):
    if components.ndim != 4 or components.shape[1] != 4:
        raise ValueError("expected actual component traces [B,4,N,T]")
    x = components[..., pilot.SETTLE:]
    factors = []
    for d in range(4):
        xd = x[:, d].float()
        centered = xd - xd.mean(dim=-1, keepdim=True)
        normalized = centered / centered.norm(dim=-1, keepdim=True).clamp_min(EPS)
        factors.append((normalized @ normalized.transpose(-1, -2)).clamp(-1.0, 1.0))
    return torch.stack(factors, dim=1)


def product_path_summary(rho, dcdq):
    """Count exact Q Jacobian dead paths and the associated |dC/dQ| signal mass."""
    if rho.ndim != 4 or rho.shape[1] != 4 or dcdq.shape != (rho.shape[0], rho.shape[2], rho.shape[3]):
        raise ValueError("rho must be [B,4,N,N] and dC/dQ must be [B,N,N]")
    positive = rho.clamp_min(0)
    dq_drho = []
    for d in range(4):
        other_product = torch.stack([positive[:, k] for k in range(4) if k != d]).prod(dim=0)
        dq_drho.append(torch.where(rho[:, d] < 0, torch.zeros_like(other_product), other_product))
    dq_drho = torch.stack(dq_drho, dim=1)
    all_blocked = (dq_drho == 0).all(dim=1)
    strict_negative = (rho < 0).any(dim=1)
    offdiag = ~torch.eye(rho.shape[-1], device=rho.device, dtype=torch.bool)
    magnitude = dcdq.detach().abs()[..., offdiag]
    total = magnitude.sum().clamp_min(1e-30)
    return {
        "all_four_dQ_drho_jacobians_zero_fraction_offdiag": float(all_blocked[..., offdiag].float().mean()),
        "strictly_negative_rho_blocks_all_four_fraction_offdiag": float(strict_negative[..., offdiag].float().mean()),
        "abs_dC_dQ_mass_fraction_on_all_four_blocked_pairs": float(magnitude[all_blocked[..., offdiag]].sum() / total),
        "abs_dC_dQ_mass_fraction_on_strictly_negative_pairs": float(magnitude[strict_negative[..., offdiag]].sum() / total),
        "zero_dQ_drho_fraction_by_component_offdiag": [
            float((dq_drho[:, d][..., offdiag] == 0).float().mean()) for d in range(4)],
        "dq_drho": dq_drho,
        "all_blocked_mask": all_blocked,
        "strict_negative_mask": strict_negative,
    }


def pearson_path_summary(components):
    """Summarize exact per-component Pearson factors and product dead paths."""
    if components.ndim != 4 or components.shape[1] != 4:
        raise ValueError("expected actual component traces [B,4,N,T]")
    x = components[..., pilot.SETTLE:]
    norms = torch.stack([
        (x[:, d].float() - x[:, d].float().mean(dim=-1, keepdim=True)).norm(dim=-1)
        for d in range(4)], dim=1)
    rho = _component_rhos(components)
    positive = rho.clamp_min(0)
    eye = torch.eye(components.shape[2], device=components.device, dtype=torch.bool)
    offdiag = ~eye
    q_rebuilt = positive.prod(dim=1)
    q_actual = spike_synchrony_affinity(components.mean(dim=1), components=components,
                                        settle=pilot.SETTLE, affinity_mode="spike")
    if not torch.equal(q_rebuilt, q_actual):
        raise AssertionError("reconstructed component product is not bitwise production Q")
    path = product_path_summary(rho, torch.ones_like(q_actual))
    dq_drho = path.pop("dq_drho")
    all_jacobians_blocked = path.pop("all_blocked_mask")
    strict_negative_block = path.pop("strict_negative_mask")
    dead_component = norms <= EPS
    return {
        "centered_trace_norm_by_component": [quantiles(norms[:, d].detach().cpu().numpy()) for d in range(4)],
        "degenerate_trace_fraction_by_component": [float(dead_component[:, d].float().mean()) for d in range(4)],
        "strictly_negative_rho_fraction_by_component_offdiag": [
            float((rho[:, d][..., offdiag] < 0).float().mean()) for d in range(4)],
        "strictly_negative_rho_fraction_any_component_offdiag": float(strict_negative_block[..., offdiag].float().mean()),
        "any_strictly_negative_blocks_all_component_jacobians_fraction_offdiag": float(
            strict_negative_block[..., offdiag].float().mean()),
        "zero_dq_drho_fraction_by_component_offdiag": [
            float((dq_drho[:, d][..., offdiag] == 0).float().mean()) for d in range(4)],
        "zero_dq_drho_all_components_fraction_offdiag": float(all_jacobians_blocked[..., offdiag].float().mean()),
        "rho_exact_zero_fraction_by_component_offdiag": [
            float((rho[:, d][..., offdiag] == 0).float().mean()) for d in range(4)],
        "q_exact_zero_fraction_offdiag": float((q_actual[..., offdiag] == 0).float().mean()),
        "q_offdiag_quantiles": quantiles(q_actual[..., offdiag].detach().cpu().numpy()),
        "q_threshold_margin_abs_quantiles": quantiles(
            (q_actual[..., offdiag] - .5).abs().detach().cpu().numpy()),
        "production_q_reconstruction_max_abs": float((q_rebuilt - q_actual).abs().max()),
    }


def discover_completed(root, experiment, arm):
    matches = []
    for manifest_path in Path(root).rglob("manifest.json"):
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (m.get("experiment") == experiment and m.get("seed") == 0
                and m.get("status") == "training_complete" and m.get("arm") == arm
                and (manifest_path.parent / "TRAINING_COMPLETED").is_file()):
            matches.append((manifest_path.parent, m))
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one completed {experiment}/{arm} seed0 run; found {len(matches)}")
    return matches[0]


def history_summary(folder, manifest):
    path = folder / "history.json"
    if manifest.get("history_sha256") != sha(path):
        raise AssertionError(f"history SHA mismatch: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 256:
        raise AssertionError(f"expected 256 history rows: {path}")
    result = {"directory_name_discovered": folder.name, "manifest_sha256": sha(folder / "manifest.json"),
              "history_sha256": sha(path), "updates": len(rows), "fields": {}}
    for key in ("gradient_norm_preclip", "core_gradient_norm_preclip", "head_gradient_norm_preclip",
                "R", "C", "old", "total"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            result["fields"][key] = quantiles(vals)
    return result


def production_groups(spikes, components):
    return spike_synchrony_components(
        spikes.detach().cpu(), synchrony_threshold=.5, min_group_size=2,
        settle=pilot.SETTLE, components=components.detach().cpu(),
        background="largest_component")


def groups_summary(groups):
    out = []
    for per_image in groups:
        foreground_patches = sum(len(g) for g in per_image)
        out.append({"foreground_groups": len(per_image),
                    "foreground_patches": foreground_patches,
                    "background_patches": 256 - foreground_patches,
                    "group_sizes_descending": sorted((len(g) for g in per_image), reverse=True)})
    return out


def coassignment_agreement(left, right):
    eqs = []
    for gl, gr in zip(left, right):
        # Production foreground groups omit BG; restore the production label-0
        # background before comparing the full hard partition.
        ll = np.zeros(256, dtype=np.int16)
        rr = np.zeros(256, dtype=np.int16)
        for k, group in enumerate(gl):
            ll[list(group)] = k + 1
        for k, group in enumerate(gr):
            rr[list(group)] = k + 1
        same_l = ll[:, None] == ll[None, :]
        same_r = rr[:, None] == rr[None, :]
        eqs.append(float(np.mean(same_l == same_r)))
    return eqs


def family_gradient_norms(loss, core, encoder, head, retain_graph=True):
    families = pilot._family_params(core, encoder)
    result = {}
    for name in ("encoder", "graph", "core"):
        grad = torch.autograd.grad(loss, families[name], retain_graph=retain_graph,
                                   allow_unused=True)
        result[name] = pilot._norm(grad)
    head_grads = torch.autograd.grad(loss, list(head.parameters()), retain_graph=retain_graph,
                                     allow_unused=True)
    result["head"] = pilot._norm(head_grads)
    return result


def _head(state, device):
    rms = state["channel_rms"].detach().cpu().reshape(-1)
    model = CompetitiveAssignmentHead(rms).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def validate_warm_head_binding(candidate_manifest, warm_head_path):
    if candidate_manifest.get("head_warmup_sha256") != sha(warm_head_path):
        raise AssertionError("warmup head checkpoint differs from the candidate's recorded source")
    return True


def _slot_summary(p):
    hard = p.argmax(dim=-1)
    counts = torch.bincount(hard.reshape(-1), minlength=11).detach().cpu().tolist()
    mass = p.detach().mean(dim=(0, 1)).cpu().tolist()
    entropy = -(p.detach().clamp_min(1e-12) * p.detach().clamp_min(1e-12).log()).sum(-1).mean()
    return {"argmax_patch_count_by_slot": [int(x) for x in counts],
            "probability_mass_by_slot": [float(x) for x in mass],
            "mean_patch_entropy_nats": float(entropy)}


def run_diagnostic(device):
    device = torch.device(device)
    torch.set_num_threads(2)
    # Discover trained runs by their immutable manifests rather than assuming arm folder names.
    candidate_dir, candidate_manifest = discover_completed(
        pilot.OUT, "SW0121", "analytic_candidate")
    source_root = joint.OUT
    control_dir, control_manifest = discover_completed(source_root, "SW0117", "control")
    joined_dir, joined_manifest = discover_completed(source_root, "SW0117", "analytic_candidate")
    expected_eval_sha = EXPECTED["evaluation.json"]
    eval_path = candidate_dir / "evaluation.json"
    if sha(eval_path) != expected_eval_sha:
        raise AssertionError("SW0121 evaluation SHA differs from the registered frozen report")
    eval_report = json.loads(eval_path.read_text(encoding="utf-8"))
    if (eval_report.get("experiment") != "SW0121" or eval_report.get("seed") != 0
            or eval_report.get("checkpoint_sha256") != EXPECTED["core.pt"]
            or eval_report.get("encoder_checkpoint_sha256") != EXPECTED["encoder.pt"]
            or eval_report.get("head_checkpoint_sha256") != EXPECTED["head.pt"]
            or eval_report.get("ids") != [1320, 1639] or eval_report.get("images") != 320):
        raise AssertionError("SW0121 endpoint report does not bind the completed registered candidate")
    for name, digest in EXPECTED.items():
        path = candidate_dir / name if name != "evaluation.json" else eval_path
        if sha(path) != digest:
            raise AssertionError(f"SW0121 frozen artifact SHA mismatch for {path}")
    if (candidate_manifest.get("core_sha256") != EXPECTED["core.pt"]
            or candidate_manifest.get("encoder_sha256") != EXPECTED["encoder.pt"]
            or candidate_manifest.get("head_sha256") != EXPECTED["head.pt"]
            or candidate_manifest.get("source_core_sha256") != joint.base.EXPECTED_SOURCE_SHAS[0]
            or candidate_manifest.get("encoder_source_sha256") != joint.base.EXPECTED_ENCODER_SHA256
            or candidate_manifest.get("preprocessing_sha256") != joint.base.EXPECTED_PREPROCESSING_SHA256
            or sha(candidate_dir / "manifest.json") != EXPECTED["manifest.json"]):
        raise AssertionError("SW0121 manifest does not bind the reported frozen candidate")
    if (not joint_coordinator.valid_result({"stage": "train", "seed": 0, "arm": "control"})
            or not joint_coordinator.valid_result({"stage": "evaluate", "seed": 0, "arm": "control"})
            or not joint_coordinator.valid_result({"stage": "train", "seed": 0, "arm": "analytic_candidate"})
            or not joint_coordinator.valid_result({"stage": "evaluate", "seed": 0, "arm": "analytic_candidate"})):
        raise AssertionError("SW0117 matched-control run artifacts failed registered validation")
    if control_manifest.get("updates") != 256 or joined_manifest.get("updates") != 256:
        raise AssertionError("SW0117 history source runs are not full registered pilots")

    ids, rows = joint.base.train_indices(0)
    expected_ids = candidate_manifest.get("training_ids")
    if (len(ids) != 4096 or not np.array_equal(ids, np.asarray(expected_ids, dtype=np.int64))
            or candidate_manifest.get("training_ids_sha256") != pilot._train_ids_hash(ids)):
        raise AssertionError("SW0121 training order/IDs do not match the registered source order")
    src_values = joint.load_models(0, device)
    src_core, src_encoder = src_values[0], src_values[1]
    cand_core_values = joint.load_models(0, device)
    cand_core, cand_encoder = cand_core_values[0], cand_core_values[1]
    cand_core.load_state_dict(torch.load(candidate_dir / "core.pt", map_location=device, weights_only=True), strict=True)
    cand_encoder.load_state_dict(torch.load(candidate_dir / "encoder.pt", map_location=device, weights_only=True), strict=True)
    candidate_head_state = torch.load(candidate_dir / "head.pt", map_location=device, weights_only=True)
    trained_head = _head(candidate_head_state, device)
    validate_warm_head_binding(candidate_manifest, pilot.WARMUP_HEAD)
    warm_head_state, _, _, _ = pilot._load_head_artifacts(device)
    warm_head = _head(warm_head_state, device)
    if not torch.equal(trained_head.channel_rms, warm_head.channel_rms):
        raise AssertionError("final and warmup head RMS buffers differ")
    for model in (src_core, src_encoder, cand_core, cand_encoder):
        model.eval()
    rgb_cache, rgb_meta, rgb_sha = joint.load_rgb_training_cache()
    if candidate_manifest.get("rgb_cache_sha256") != rgb_sha:
        raise AssertionError("SW0121 candidate manifest does not bind this exact TRAIN RGB cache")
    lossfn = joint.criterion()
    patcher = cand_core_values[2]
    mean, std, clip = cand_core_values[3:6]

    batches = []
    for bi in range(4):
        lo = bi * 16
        batch_rows = rows[lo:lo+16]
        image_ids = ids[lo:lo+16]
        images = joint.read_rgb(rgb_cache, batch_rows, device)
        rgb_targets = joint.rgb_base.rgb_patch_means(images / 255.0).detach()
        with torch.no_grad():
            src_parts = pilot._forward(src_core, src_encoder, patcher, mean, std, clip, images, lossfn)
        with torch.enable_grad():
            cand_parts = pilot._forward(cand_core, cand_encoder, patcher, mean, std, clip, images, lossfn)
            sources = {"source97": src_parts, "candidate121": cand_parts}
            groups = {key: production_groups(value["spikes"], value["components"])
                      for key, value in sources.items()}

            # Apply both the exact warmup head and final learned head across both
            # models' traces; each assignment prediction is paired with both Q's
            # so the target is held fixed for the affinity-error comparison.
            psets = {}
            for model_name, parts in sources.items():
                for head_name, head in (("warmup_head", warm_head), ("final_head", trained_head)):
                    psets[f"{model_name}_{head_name}"] = head(parts["components"][..., -pilot.SETTLE:])

            p_stats = {}
            for pname, p in psets.items():
                with torch.no_grad():
                    r, c_candidate, _ = analytic_rgb_terms(cand_parts["q"].detach(), p.detach(), rgb_targets)
                    _, c_source, _ = analytic_rgb_terms(src_parts["q"], p.detach(), rgb_targets)
                p_stats[pname] = {"head_occupancy": _slot_summary(p.detach()),
                                  "R": float(r.detach()),
                                  "C_fixedP_sourceQ": float(c_source.detach()),
                                  "C_fixedP_candidateQ": float(c_candidate.detach())}

            native_p = psets["candidate121_final_head"]
            r_native, c_native, _ = analytic_rgb_terms(cand_parts["q"], native_p, rgb_targets)
            r_grads = family_gradient_norms(r_native, cand_core, cand_encoder, trained_head, True)
            c_grads = family_gradient_norms(c_native, cand_core, cand_encoder, trained_head, True)
            q_grad = torch.autograd.grad(c_native, cand_parts["q"], retain_graph=True)[0]
            rho_diag = pearson_path_summary(cand_parts["components"])
            rho_live = _component_rhos(cand_parts["components"])
            path_stats = product_path_summary(rho_live.detach(), q_grad)
            offdiag = ~torch.eye(256, dtype=torch.bool, device=device)
            q_from_rho = rho_live.clamp_min(0).prod(dim=1)
            if not torch.equal(q_from_rho, cand_parts["q"]):
                raise AssertionError("candidate Q differs bitwise from reconstructed signed Pearson product")
            _, c_from_rho, _ = analytic_rgb_terms(q_from_rho, native_p, rgb_targets)
            if not torch.allclose(c_from_rho, c_native, rtol=0, atol=2e-7):
                raise AssertionError("C readout differs between actual Q and reconstructed product")
            dcd_rho = torch.autograd.grad(c_from_rho, rho_live, retain_graph=True)[0]
            dc_drho_norms = [float(dcd_rho[:, d][..., offdiag].norm()) for d in range(4)]
            dc_drho_zero_fracs = [float((dcd_rho[:, d][..., offdiag] == 0).float().mean())
                                  for d in range(4)]

            edge = {}
            q0, q1 = src_parts["q"].detach(), cand_parts["q"].detach()
            e0, e1 = q0 >= .5, q1 >= .5
            edge_agreement = (e0[:, offdiag] == e1[:, offdiag]).float().mean()
            edge.update({"source_edges_ge_0_5_directed_per_image": e0[:, offdiag].sum(-1).cpu().tolist(),
                         "candidate_edges_ge_0_5_directed_per_image": e1[:, offdiag].sum(-1).cpu().tolist(),
                         "edge_agreement_fraction": float(edge_agreement),
                         "q_mae": float((q0 - q1).abs().mean()),
                         "q_max_abs_difference": float((q0 - q1).abs().max()),
                         "q_threshold_margin_lt_0_01_fraction_candidate": float(
                             ((q1[..., offdiag] - .5).abs() < .01).float().mean()),
                         "q_gradient_absolute_quantiles_candidate_C": quantiles(
                             q_grad.detach().abs()[..., offdiag].cpu().numpy()),
                         "dC_dQ_abs_gradient_mass_on_all_four_blocked_pairs": path_stats["abs_dC_dQ_mass_fraction_on_all_four_blocked_pairs"],
                         "dC_dQ_abs_gradient_mass_on_strictly_negative_pairs": path_stats["abs_dC_dQ_mass_fraction_on_strictly_negative_pairs"],
                         "all_four_dQ_drho_jacobians_zero_fraction_offdiag": path_stats["all_four_dQ_drho_jacobians_zero_fraction_offdiag"],
                         "dC_drho_norm_by_component_from_reconstructed_signed_rho": dc_drho_norms,
                         "dC_drho_exact_zero_fraction_by_component": dc_drho_zero_fracs})
            src_groups, cand_groups = groups["source97"], groups["candidate121"]
            batches.append({"batch_index": bi,
                "dataset_image_ids": [int(x) for x in image_ids],
                "gamma_cache_rows": [int(x) for x in batch_rows],
                "source_primary_phase_loss": float(src_parts["primary"].detach()),
                "candidate_primary_phase_loss": float(cand_parts["primary"].detach()),
                "candidate_positive_actual_Q_loss": float(cand_parts["positive"].detach()),
                "assignment_readouts": p_stats,
                "native_final_head_R_grad_norms": r_grads,
                "native_final_head_C_grad_norms": c_grads,
                "learned_head_native_R": float(r_native.detach()),
                "learned_head_native_C": float(c_native.detach()),
                "head_gradient_norm_R": r_grads["head"],
                "head_gradient_norm_C": c_grads["head"],
                "Q_threshold_edges": edge,
                "pearson_product_paths": rho_diag,
                "production_H_source": groups_summary(src_groups),
                "production_H_candidate": groups_summary(cand_groups),
                "H_pairwise_coassignment_agreement_per_image": coassignment_agreement(src_groups, cand_groups)})

    history_paths = {
        "SW0117_control": (control_dir, control_manifest),
        "SW0117_analytic_candidate": (joined_dir, joined_manifest),
        "SW0121_analytic_candidate": (candidate_dir, candidate_manifest),
    }
    hist = {key: history_summary(folder, manifest) for key, (folder, manifest) in history_paths.items()}
    return {
        "status": "complete_readonly_diagnostic",
        "experiment": "SW0121_training_credit_diagnostic",
        "training_rows_only": True, "ground_truth_read": False,
        "optimizer_or_checkpoint_modified": False,
        "scope": "first four matched ordered TRAIN B16 batches; each RGB batch loaded once and forwarded once per model",
        "device": str(device), "training_image_ids": [int(x) for x in ids[:64]],
        "training_order_sha256": pilot._train_ids_hash(ids),
        "rgb_cache_sha256": rgb_sha,
        "models": {
            "SW0097_seed0_source_core_sha256": joint.base.EXPECTED_SOURCE_SHAS[0],
            "SW0097_registered_encoder_sha256": joint.base.EXPECTED_ENCODER_SHA256,
            "SW0121_manifest_sha256": sha(candidate_dir / "manifest.json"),
            "SW0121_core_sha256": sha(candidate_dir / "core.pt"),
            "SW0121_encoder_sha256": sha(candidate_dir / "encoder.pt"),
            "SW0121_head_sha256": sha(candidate_dir / "head.pt"),
            "SW0121_evaluation_sha256": sha(eval_path),
            "SW0121_evaluation_path_discovered": str(eval_path),
        },
        "history_gradient_quantiles": hist,
        "batches": batches,
        "interpretation_limits": [
            "Q=prod_d max(rho_d,0); a strictly negative rho factor zeroes Q Jacobians for the other component correlations on that patch pair.",
            "Degenerate centered-trace rows are reported separately; the zero-product-Jacobian fraction is an analytic path census, not a causal performance attribution.",
            "Fixed-P cross-Q C values isolate affinity changes conditional on that P; they do not replace the trained joint trajectory.",
            "No validation labels or object-mask correlations were read.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run_diagnostic(args.device)
    print(json.dumps(result, allow_nan=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
