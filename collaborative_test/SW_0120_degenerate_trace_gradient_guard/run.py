"""SW0120 seed-0 diagnostic pilot: detach degenerate centered-trace gradients in Q only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run.py"
ARCHIVE = HERE / "results_archive"
OUT = ROOT / "trained_models/SW0120_degenerate_trace_gradient_guard"
PROOF_PATH = HERE / "update3_guard_proof_20261009.json"
PROOF_HELPER_SHA256 = "f430f41146d37bab1e9c6e09f19b661f7d3193a6fa4e33d85c374e65056745df"
LAMBDA = 7.865416617457706
BATCH, UPDATES, TIME_STEPS, SETTLE = 16, 256, 64, 32
CORE_GRAPH_LR, ENCODER_LR, CLIP_NORM = 3e-5, 3e-6, 1.0
ARM = "analytic_candidate"
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0115_analytic_partition_rgb import run as rgb_base
from collaborative_test.SW_0115_analytic_partition_rgb.loss import (
    batch_reconstruction_loss, production_partition,
)
from collaborative_test.SW_0117_joint_analytic_rgb import run as sw117
from collaborative_test.SW_0120_degenerate_trace_gradient_guard.affinity import guarded_spike_affinity
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from snn_kuramoto_bidirectional.loss_function import phase_locking_value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def require_vacant(path, label):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"preserve existing {label}; refusing overwrite: {path}")
    return path


def implementation_fingerprint():
    files = [RUNNER, HERE / "affinity.py", HERE / "protocol.json", HERE / "coordinator.py",
             HERE / "test_run.py", PROOF_PATH,
             ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
             ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/run.py",
             ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/loss.py",
             sw117.RUNNER, ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
             ROOT / "collaborative_test/SW_0094_aligned_joint_pilot/run.py"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files}


def validate_proof_record(proof, expected_source_sha, expected_source_manifest_sha,
                          expected_candidate_history_sha, expected_lambda_sha):
    diag = proof.get("update_3_guarded_diagnostic", {})
    forward = diag.get("guarded_forward_identity", {})
    expected_forward = ("Q_torch_equal", "primary_loss_torch_equal", "primary_tensor_is_original",
                        "old_loss_torch_equal", "positive_Q_loss_torch_equal", "RGB_loss_torch_equal",
                        "RGB_per_image_torch_equal", "production_labels_torch_equal", "production_H_torch_equal")
    candidate_history_path = sw117.OUT / "seed0_analytic_candidate/history.json"
    replay = proof.get("replay_comparisons", [])
    if (proof.get("status") != "guarded_update3_probe_complete"
            or proof.get("source_core_sha256") != expected_source_sha
            or proof.get("source_manifest_sha256") != expected_source_manifest_sha
            or proof.get("candidate_history_sha256") != expected_candidate_history_sha
            or proof.get("lambda_joint") != LAMBDA
            or proof.get("lambda_sha256") != expected_lambda_sha
            or diag.get("q_guard_helper_sha256") != PROOF_HELPER_SHA256
            or proof.get("writes_performed") is not False
            or proof.get("update_3_optimizer_step_applied_to_registered_state") is not False
            or proof.get("optimizer_updates_replayed") != 2
            or proof.get("forward_backward_batches") != 3
            or proof.get("shuffle_seed") != 117
            or proof.get("training_ids_count") != 4096
            or len(replay) != 3
            or [row.get("update") for row in replay] != [1, 2, 3]
            or not all(row.get("passed") is True for row in replay)
            or diag.get("optimizer_step_3_applied_only_to_throwaway_in_memory_state") is not True
            or not all(forward.get(k) is True for k in expected_forward)
            or diag.get("theta_finite") is not True):
        raise AssertionError("update-3 proof provenance or exact-forward identity failed")
    guards = diag.get("component_gradient_guard", {})
    for key in ("5_positive_Q", "lambda_RGB"):
        item = guards.get(key, {})
        if (item.get("degenerate_trace_count", 0) < 1
                or item.get("valid_trace_count", 0) < 1
                or item.get("guarded_degenerate_gradient_max_abs") != 0
                or item.get("valid_trace_gradient_exactly_matches_legacy") is not True):
            raise AssertionError(f"update-3 {key} component-gradient proof failed")
    family_stats = diag.get("guarded_old_rgb_gradient_stats", {})
    deltas = diag.get("throwaway_optimizer_step_parameter_deltas", {})
    for family in ("encoder", "graph", "core"):
        if (not math.isfinite(float(family_stats.get(family, {}).get("rgb_gradient_norm", 0)))
                or float(family_stats[family]["rgb_gradient_norm"]) <= 0
                or not math.isfinite(float(deltas.get(family, {}).get("l1_parameter_delta", 0)))
                or float(deltas[family]["l1_parameter_delta"]) <= 0):
            raise AssertionError(f"update-3 proof lacks RGB gradient/update credit for {family}")
    if (not math.isfinite(float(diag.get("guarded_combined_preclip_gradient_norm", float("nan"))))
            or float(diag["guarded_combined_preclip_gradient_norm"])
            >= float(diag.get("legacy_combined_preclip_gradient_norm", float("inf")))):
        raise AssertionError("update-3 proof does not show a finite combined-gradient drop")
    return diag


def validate_update3_proof():
    if not PROOF_PATH.is_file():
        raise FileNotFoundError(f"guarded update-3 proof required before training: {PROOF_PATH}")
    if sha(HERE / "affinity.py") != PROOF_HELPER_SHA256:
        raise AssertionError("embedded guard helper differs from the reviewed SW0120 helper")
    proof = json.loads(PROOF_PATH.read_text(encoding="utf-8"))
    source, source_manifest, source_record, ids, rows = sw117.source_contract(0)
    candidate_history_path = sw117.OUT / "seed0_analytic_candidate/history.json"
    if not candidate_history_path.is_file():
        raise FileNotFoundError(candidate_history_path)
    validate_proof_record(proof, sha(source), sha(source_manifest), sha(candidate_history_path),
                         sha(sw117.LAMBDA_PATH))
    return proof, sha(PROOF_PATH), source, source_manifest, source_record, ids, rows


def find_completed_control():
    matches = []
    for manifest_path in sw117.OUT.rglob("manifest.json"):
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (m.get("experiment") == "SW0117" and m.get("seed") == 0
                and m.get("arm") == "analytic_candidate"):
            matches.append((manifest_path, m))
    if len(matches) != 1:
        raise AssertionError(f"expected unique existing SW0117 candidate control, found {len(matches)}")
    manifest_path, m = matches[0]
    core_path = manifest_path.parent / "core.pt"
    history_path = manifest_path.parent / "history.json"
    optimizer_path = manifest_path.parent / "optimizer.pt"
    proof, _, source, source_manifest, _, ids, _ = validate_update3_proof()
    endpoint = sw117.validate_source_endpoint_review()
    if (m.get("status") != "training_complete" or m.get("source_core_sha256") != sha(source)
            or m.get("source_manifest_sha256") != sha(source_manifest)
            or m.get("implementation_fingerprint") != sw117.implementation_fingerprint()
            or m.get("training_ids") != ids.tolist() or m.get("matched_shuffle_seed") != 117
            or m.get("updates") != 256 or m.get("batch_size") != 16
            or m.get("time_steps") != 64 or m.get("settle") != 32
            or m.get("core_graph_lr") != sw117.CORE_GRAPH_LR
            or m.get("encoder_lr") != sw117.ENCODER_LR
            or m.get("clip_norm") != sw117.CLIP_NORM
            or m.get("lambda_joint") != LAMBDA
            or m.get("lambda_artifact_sha256") != sha(sw117.LAMBDA_PATH)
            or m.get("encoder_source_sha256") != sha(sw117.ENCODER_PATH)
            or m.get("preprocessing_sha256") != sha(sw117.STATS_PATH)
            or m.get("source_endpoint_review_sha256") != sha(sw117.SOURCE_ENDPOINT_REVIEW)
            or m.get("source_endpoint_report_sha256") != endpoint.get("report_sha256")
            or not core_path.is_file() or not history_path.is_file() or not optimizer_path.is_file()
            or m.get("core_sha256") != sha(core_path)
            or m.get("history_sha256") != sha(history_path)
            or m.get("optimizer_sha256") != sha(optimizer_path)
            or not (manifest_path.parent / "encoder.pt").is_file()
            or m.get("encoder_sha256") != sha(manifest_path.parent / "encoder.pt")
            or not (manifest_path.parent / "TRAINING_COMPLETED").is_file()):
        raise AssertionError("existing SW0117 analytic candidate is not an exact registered control")
    proof = json.loads(PROOF_PATH.read_text(encoding="utf-8"))
    if (sha(manifest_path) != proof.get("candidate_manifest_sha256")
            or sha(history_path) != proof.get("candidate_history_sha256")):
        raise AssertionError("SW0117 control manifest/history do not match update-3 proof")
    eval_path = manifest_path.parent / "evaluation.json"
    sidecar_path = manifest_path.parent / "evaluation_manifest.json"
    if not eval_path.is_file() or not sidecar_path.is_file():
        raise AssertionError("completed SW0117 control evaluation and sidecar are required")
    report = json.loads(eval_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    score = report["sweep"][0]["scored_targets"]["our_hdf5"]
    metric_names = {"fg_ari", "foreground_iou", "matched_object_iou"}
    if (report.get("experiment") != "SW0117" or report.get("arm") != "analytic_candidate"
            or report.get("seed") != 0 or report.get("ground_truth_used_for_prediction") is not False
            or report.get("training_manifest_sha256") != sha(manifest_path)
            or report.get("checkpoint_sha256") != sha(core_path)
            or report.get("encoder_checkpoint_sha256") != sha(manifest_path.parent / "encoder.pt")
            or report.get("evaluation_runner_sha256") != sw117.sha(sw117.RUNNER)
            or report.get("shared_evaluator_sha256") != sw117.sha(
                sw117.ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py")
            or sidecar.get("evaluation_sha256") != sha(eval_path)
            or sidecar.get("training_manifest_sha256") != sha(manifest_path)
            or sidecar.get("checkpoint_sha256") != sha(core_path)
            or sidecar.get("evaluation_contract", {}).get("ids") != [1320, 1639]
            or sidecar.get("evaluation_contract", {}).get("images") != 320
            or sidecar.get("evaluation_contract", {}).get("batch_size") != 8
            or sidecar.get("evaluation_contract", {}).get("time_steps") != 1024
            or sidecar.get("evaluation_contract", {}).get("settle") != 512
            or sidecar.get("evaluation_contract", {}).get("ground_truth_used_for_prediction") is not False
            or set(score.get("metrics", {})) != metric_names
            or set(score.get("valid_count", {})) != metric_names
            or set(score.get("per_image", {})) != metric_names
            or any(score["valid_count"][k] != 320
                   or not math.isfinite(float(score["metrics"][k]))
                   or len(score["per_image"][k]) != 320
                   or not all(math.isfinite(float(v)) for v in score["per_image"][k])
                   or abs(sum(float(v) for v in score["per_image"][k]) / 320.0
                            - float(score["metrics"][k])) > 1e-12
                   for k in metric_names)):
        raise AssertionError("SW0117 control evaluation does not match the registered frozen 320-image contract")
    return manifest_path.parent, m


def validate_control(output):
    proof, proof_sha, source, source_manifest, _, _, _ = validate_update3_proof()
    control_dir, manifest = find_completed_control()
    sw117.validate_source_endpoint_review()
    output = require_vacant(output, "SW0120 matched-control validation")
    record = {
        "status": "passed", "experiment": "SW0120", "control_experiment": "SW0117",
        "seed": 0, "arm": "analytic_candidate", "source_core_sha256": sha(source),
        "source_manifest_sha256": sha(source_manifest), "proof_sha256": proof_sha,
        "control_manifest_sha256": sha(control_dir / "manifest.json"),
        "control_core_sha256": sha(control_dir / "core.pt"),
        "control_history_sha256": sha(control_dir / "history.json"),
        "control_optimizer_sha256": sha(control_dir / "optimizer.pt"),
        "control_evaluation_sha256": sha(control_dir / "evaluation.json"),
        "lambda_joint": LAMBDA, "lambda_sha256": sha(sw117.LAMBDA_PATH),
        "updates": UPDATES, "batch_size": BATCH, "time_steps": TIME_STEPS,
        "settle": SETTLE, "matched_shuffle_seed": 117,
        "ground_truth_used_for_training": False,
    }
    write(output, record)
    return record


def objective_parts_guarded(core, gamma, rgb_patches, lossfn):
    _, spikes, core_out, plv, theta = sw117._forward_with_plv(
        core, gamma, lossfn, SETTLE, "phase", "mean")
    components = core.last_component_spikes
    q = guarded_spike_affinity(components.mean(dim=1), components=components,
                               settle=SETTLE, eps=1e-8)
    primary, _ = lossfn(plv=plv, theta=theta)
    positive, _ = lossfn(plv=q)
    old = primary + 5.0 * positive
    labels, hard = production_partition(spikes, components, settle=SETTLE)
    rgb, per_image, predictions, details = batch_reconstruction_loss(q, hard, rgb_patches)
    return old, primary, positive, rgb, q, labels, hard, per_image, details, spikes, core_out, theta, components


def guarded_parts_from_legacy(parts, rgb_patches, lossfn):
    """Recompute only Q-dependent values on the legacy rollout graph."""
    old, primary, positive, rgb, q, labels, hard, per_image, details, spikes, core_out, theta, components = parts
    guarded_q = guarded_spike_affinity(components.mean(dim=1), components=components,
                                       settle=SETTLE, eps=1e-8)
    guarded_positive, _ = lossfn(plv=guarded_q)
    guarded_old = primary + 5.0 * guarded_positive
    guarded_rgb, guarded_per_image, predictions, guarded_details = batch_reconstruction_loss(
        guarded_q, hard, rgb_patches)
    return (guarded_old, primary, guarded_positive, guarded_rgb, guarded_q, labels, hard,
            guarded_per_image, guarded_details, spikes, core_out, theta, components)


def require_exact_guard_parity(legacy, guarded):
    names = ("old", "primary", "positive Q", "RGB", "Q", "labels", "spikes", "core_out", "theta", "components")
    indices = (0, 1, 2, 3, 4, 5, 9, 10, 11, 12)
    for name, idx in zip(names, indices):
        if isinstance(legacy[idx], torch.Tensor):
            if not torch.equal(legacy[idx], guarded[idx]):
                raise AssertionError(f"guarded forward changed {name}")
        elif legacy[idx] != guarded[idx]:
            raise AssertionError(f"guarded forward changed {name}")
    if len(legacy[6]) != len(guarded[6]) or any(
            not torch.equal(a, b) for a, b in zip(legacy[6], guarded[6])):
        raise AssertionError("guarded forward changed production H")
    if not torch.equal(legacy[7], guarded[7]):
        raise AssertionError("guarded forward changed per-image RGB loss")


def direct_guard_checks(legacy_parts, guarded_parts):
    components = legacy_parts[12]
    old_pos = 5.0 * legacy_parts[2]
    new_pos = 5.0 * guarded_parts[2]
    old_rgb = LAMBDA * legacy_parts[3]
    new_rgb = LAMBDA * guarded_parts[3]
    report = {}
    centered = components[..., SETTLE:].detach().float()
    centered = centered - centered.mean(dim=-1, keepdim=True)
    degenerate = centered.norm(dim=-1) <= 1e-8
    valid = ~degenerate
    for name, old_loss, new_loss in (("positive_Q", old_pos, new_pos), ("lambda_RGB", old_rgb, new_rgb)):
        g_old, = torch.autograd.grad(old_loss, components, retain_graph=True)
        g_new, = torch.autograd.grad(new_loss, components, retain_graph=True)
        dmask = degenerate.unsqueeze(-1).expand_as(g_new)
        vmask = valid.unsqueeze(-1).expand_as(g_new)
        if not torch.equal(g_new[dmask], torch.zeros_like(g_new[dmask])):
            raise AssertionError(f"guarded {name} gradient is nonzero on degenerate rows")
        if not torch.equal(g_new[vmask], g_old[vmask]):
            raise AssertionError(f"guarded {name} valid-row gradient differs from legacy")
        report[name] = {"degenerate_rows": int(degenerate.sum()), "valid_rows": int(valid.sum()),
                        "guarded_degenerate_max_abs": float(g_new[dmask].abs().max()) if dmask.any() else 0.0,
                        "valid_gradients_exact": True}
    return report


def preflight(output, device="cuda"):
    # Validate all immutable proof/control/source inputs before creating output.
    proof, proof_sha, source, source_manifest, source_record, ids, rows = validate_update3_proof()
    control_dir, control_manifest = find_completed_control()
    control_validation_path = ARCHIVE / "sw0117_control_validation.json"
    control_validation = json.loads(control_validation_path.read_text(encoding="utf-8"))
    if (control_validation.get("status") != "passed"
            or control_validation.get("proof_sha256") != proof_sha
            or control_validation.get("control_manifest_sha256") != sha(control_dir / "manifest.json")
            or control_validation.get("control_core_sha256") != sha(control_dir / "core.pt")):
        raise AssertionError("SW0117 matched control must pass the separate provenance task")
    endpoint = sw117.validate_source_endpoint_review()
    require_vacant(output, "SW0120 preflight")
    (core, encoder, patcher, mean, std, clip, loaded_source, loaded_source_manifest,
     source_record, loaded_ids, loaded_rows) = sw117.load_models(0, device)
    if loaded_source != source or loaded_source_manifest != source_manifest:
        raise AssertionError("SW117 source changed during preflight")
    cached_gamma, _ = base.validate_gamma_cache(base.GAMMA_TRAIN, base.GAMMA_TRAIN_MANIFEST)
    rgb_cache, rgb_manifest, rgb_cache_sha = sw117.load_rgb_training_cache()
    families = sw117.eligible_params(core, encoder)
    reference = base.make_core(device, TIME_STEPS)
    reference.load_state_dict(torch.load(source, map_location=device, weights_only=True), strict=True)
    reference._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
    reference.eval()
    torch.manual_seed(117)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(117)
    core.train(); core.graph_generator.train(); encoder.train(); reference.train()
    lossfn = sw117.criterion()
    parity, gradients, real_rgb, shuffled_rgb = [], [], [], []
    fixed_perm = torch.as_tensor(np.random.default_rng(11501).permutation(256), device=device)
    for b in range(4):
        start = b * BATCH
        batch_rows = loaded_rows[start:start + BATCH]
        images = sw117.read_rgb(rgb_cache, batch_rows, device)
        patches = rgb_base.rgb_patch_means(images / 255.0)
        gamma = sw117.encode_rgb(encoder, patcher, mean, std, clip, images)
        cached = cached_gamma[torch.as_tensor(batch_rows, dtype=torch.long)].to(device)
        # Independent source-reference parity and exact same-input implementation identity.
        legacy, p = sw117.compare_initial_rollout(core, reference, gamma, cached, patches, lossfn)
        guarded = guarded_parts_from_legacy(legacy, patches, lossfn)
        require_exact_guard_parity(legacy, guarded)
        old, rgb = guarded[0], guarded[3]
        stats, _, _ = sw117.collect_joint_gradients(old, rgb, families, candidate=True)
        qgrad, = torch.autograd.grad(rgb, guarded[4], retain_graph=True)
        qnorm = sw117.grad_norm((qgrad,))
        sw117.validate_gradient_families(stats, "analytic_candidate", qnorm)
        guard_report = direct_guard_checks(legacy, guarded)
        shuffled_h = [h.index_select(0, fixed_perm) for h in guarded[6]]
        _, shuffled_per, _, _ = batch_reconstruction_loss(guarded[4], shuffled_h, patches)
        real_rgb.extend(float(x.detach()) for x in guarded[7])
        shuffled_rgb.extend(float(x.detach()) for x in shuffled_per)
        parity.append({"batch": b, "input_reference": p, "guard_q_losses_labels_H_exact": True,
                       "direct_component_gradient_guard": guard_report,
                       "rgb_family_norms": {name: stats[name]["rgb_gradient_norm"]
                                            for name in ("encoder", "graph", "core")},
                       "rgb_to_q_norm": qnorm})
        core.zero_grad(set_to_none=True); encoder.zero_grad(set_to_none=True)
    scramble_excess = rgb_base.mean_scramble_excess(real_rgb, shuffled_rgb)
    if not math.isfinite(scramble_excess) or scramble_excess <= 0:
        raise AssertionError("fixed count-preserving H row scramble did not increase RGB loss")

    # Disposable fresh source copy: prove every family receives an update.
    (throw_core, throw_enc, throw_patcher, tmean, tstd, tclip, *_rest, throw_ids, throw_rows) = \
        sw117.load_models(0, device)
    throw_families = sw117.eligible_params(throw_core, throw_enc)
    optimizer = sw117.optimizer_for(throw_core, throw_enc)
    torch.manual_seed(117)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(117)
    image = sw117.read_rgb(rgb_cache, throw_rows[:BATCH], device)
    patches = rgb_base.rgb_patch_means(image / 255.0)
    gamma = sw117.encode_rgb(throw_enc, throw_patcher, tmean, tstd, tclip, image)
    guarded = objective_parts_guarded(throw_core, gamma, patches, lossfn)
    before_core, before_enc = sw117.clone_state(throw_core), sw117.clone_state(throw_enc)
    _, old_by, rgb_by = sw117.collect_joint_gradients(guarded[0], guarded[3], throw_families, True)
    sw117.apply_family_gradients(throw_families, old_by, rgb_by, LAMBDA)
    norm = torch.nn.utils.clip_grad_norm_(throw_families["all"], CLIP_NORM)
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise FloatingPointError("invalid SW0120 throwaway update gradient")
    optimizer.step()
    changed = sw117.changed_trainable_groups(before_core, throw_core.state_dict(),
                                              before_enc, throw_enc.state_dict())
    if not all(changed.values()):
        raise AssertionError(f"SW0120 throwaway update missed parameter family: {changed}")

    record = {
        "status": "passed", "experiment": "SW0120", "seed": 0, "arm": ARM,
        "source_core_sha256": sha(source), "source_manifest_sha256": sha(source_manifest),
        "source_status": source_record.get("status"),
        "implementation_fingerprint": implementation_fingerprint(),
        "update3_proof_sha256": proof_sha, "guard_helper_sha256": PROOF_HELPER_SHA256,
        "control_validation_sha256": sha(control_validation_path),
        "lambda_joint": LAMBDA, "lambda_sha256": sha(sw117.LAMBDA_PATH),
        "sw0117_control_manifest_sha256": sha(control_dir / "manifest.json"),
        "sw0117_control_core_sha256": sha(control_dir / "core.pt"),
        "encoder_sha256": sha(sw117.ENCODER_PATH), "preprocessing_sha256": sha(sw117.STATS_PATH),
        "gamma_train_sha256": sha(base.GAMMA_TRAIN),
        "gamma_train_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
        "rgb_cache_sha256": rgb_cache_sha, "rgb_cache_manifest_sha256": sha(sw117.RGB_CACHE_MANIFEST),
        "training_ids": ids.tolist(),
        "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "matched_shuffle_seed": 117, "batch_size": BATCH, "updates": UPDATES,
        "time_steps": TIME_STEPS, "settle": SETTLE, "ground_truth_used": False,
        "same_input_reference_parity": parity,
        "fixed_h_rows_scramble_rgb_loss_excess": float(scramble_excess),
        "throwaway_preclip_gradient_norm": float(norm),
        "throwaway_parameter_groups_changed": changed,
        "objective": "phase_primary + 5*positive_actual_spike_product + frozen_lambda*partition_RGB; guarded Q backward only",
    }
    write(output, record)
    return record


def train(output, device="cuda"):
    # All proof, source, cache, control and preflight gates precede output mkdir.
    proof, proof_sha, source, source_manifest, source_record, ids, rows = validate_update3_proof()
    control_dir, control_manifest = find_completed_control()
    endpoint_review = sw117.validate_source_endpoint_review()
    pf_path = ARCHIVE / "preflight_seed0_guarded_candidate.json"
    if not pf_path.is_file():
        raise FileNotFoundError(pf_path)
    pf = json.loads(pf_path.read_text(encoding="utf-8"))
    fingerprint = implementation_fingerprint()
    if (pf.get("status") != "passed" or pf.get("experiment") != "SW0120"
            or pf.get("implementation_fingerprint") != fingerprint
            or pf.get("update3_proof_sha256") != proof_sha or pf.get("lambda_joint") != LAMBDA
            or pf.get("source_core_sha256") != sha(source)
            or pf.get("sw0117_control_core_sha256") != sha(control_dir / "core.pt")
            or pf.get("control_validation_sha256") != sha(ARCHIVE / "sw0117_control_validation.json")):
        raise AssertionError("fresh SW0120 preflight/source/proof binding required")
    output = Path(output)
    require_vacant(output, "SW0120 training output")

    (core, encoder, patcher, mean, std, clip, loaded_source, loaded_manifest,
     loaded_record, train_ids, train_rows) = sw117.load_models(0, device)
    if loaded_source != source or loaded_manifest != source_manifest:
        raise AssertionError("SW0117 source changed before SW0120 training")
    cache, rgb_manifest, cache_sha = sw117.load_rgb_training_cache()
    gamma_cache, gamma_manifest = base.validate_gamma_cache(base.GAMMA_TRAIN, base.GAMMA_TRAIN_MANIFEST)
    if (sha(sw117.LAMBDA_PATH) != pf.get("lambda_sha256")
            or float(json.loads(sw117.LAMBDA_PATH.read_text())["lambda_joint"]) != LAMBDA
            or cache_sha != pf.get("rgb_cache_sha256")
            or sha(base.GAMMA_TRAIN) != pf.get("gamma_train_sha256")):
        raise AssertionError("SW0120 training cache/lambda changed after preflight")
    families = sw117.eligible_params(core, encoder)
    optimizer = sw117.optimizer_for(core, encoder)
    torch.manual_seed(117)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(117)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "training", "experiment": "SW0120", "seed": 0, "arm": ARM,
        "source_core": str(source), "source_core_sha256": sha(source),
        "source_manifest_sha256": sha(source_manifest), "source_status": source_record.get("status"),
        "update3_proof_sha256": proof_sha, "preflight_sha256": sha(pf_path),
        "implementation_fingerprint": fingerprint,
        "source_endpoint_review_sha256": sha(sw117.SOURCE_ENDPOINT_REVIEW),
        "source_endpoint_report_sha256": endpoint_review["report_sha256"],
        "encoder_source_sha256": sha(sw117.ENCODER_PATH), "preprocessing_sha256": sha(sw117.STATS_PATH),
        "gamma_train_sha256": sha(base.GAMMA_TRAIN),
        "gamma_train_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
        "rgb_cache_sha256": cache_sha, "rgb_cache_manifest_sha256": sha(sw117.RGB_CACHE_MANIFEST),
        "sw0117_control_manifest_sha256": sha(control_dir / "manifest.json"),
        "sw0117_control_core_sha256": sha(control_dir / "core.pt"),
        "training_ids": train_ids.tolist(),
        "training_ids_sha256": hashlib.sha256(np.asarray(train_ids, dtype="<i8").tobytes()).hexdigest(),
        "matched_shuffle_seed": 117, "updates": UPDATES, "batch_size": BATCH,
        "core_graph_lr": CORE_GRAPH_LR, "encoder_lr": ENCODER_LR, "clip_norm": CLIP_NORM,
        "time_steps": TIME_STEPS, "settle": SETTLE, "lambda_joint": LAMBDA,
        "lambda_artifact_sha256": sha(sw117.LAMBDA_PATH),
        "objective": "phase_primary + 5*positive_actual_spike_product + frozen_lambda*partition_RGB; guarded Q backward only",
        "ground_truth_used_for_training": False,
    }
    write(output / "manifest.json", manifest)
    history = []
    lossfn = sw117.criterion()
    core.train(); core.graph_generator.train(); encoder.train()
    for update in range(UPDATES):
        start = update * BATCH
        image = sw117.read_rgb(cache, train_rows[start:start+BATCH], device)
        patches = rgb_base.rgb_patch_means(image / 255.0)
        gamma = sw117.encode_rgb(encoder, patcher, mean, std, clip, image)
        parts = objective_parts_guarded(core, gamma, patches, lossfn)
        old, primary, positive, rgb, q, labels, hard, per_image = parts[:8]
        total = old + LAMBDA * rgb
        if not torch.isfinite(total):
            raise FloatingPointError(f"nonfinite SW0120 total loss at update {update+1}")
        optimizer.zero_grad(set_to_none=True)
        stats, old_by, rgb_by = sw117.collect_joint_gradients(old, rgb, families, candidate=True)
        sw117.apply_family_gradients(families, old_by, rgb_by, LAMBDA)
        norm = torch.nn.utils.clip_grad_norm_(families["all"], CLIP_NORM)
        if not torch.isfinite(norm) or float(norm) <= 0:
            raise FloatingPointError(f"invalid SW0120 gradient at update {update+1}")
        optimizer.step()
        history.append({"update": update+1, "total": float(total.detach()),
            "old": float(old.detach()), "primary": float(primary.detach()),
            "positive_actual_spike_product": float(positive.detach()), "rgb": float(rgb.detach()),
            "lambda_joint": LAMBDA, "gradient_norm_preclip": float(norm),
            "old_rgb_gradient_stats": stats, "mean_groups": float(np.mean([h.shape[1] for h in hard])),
            "mean_rgb_per_image": float(per_image.detach().mean())})
        if update == 0 or (update + 1) % 32 == 0:
            write(output / "progress.json", {"status": "training", "update": update+1,
                                               "total_updates": UPDATES})
    if any(not torch.isfinite(p).all() for p in list(core.parameters()) + list(encoder.parameters())):
        raise FloatingPointError("nonfinite SW0120 final parameters")
    torch.save(core.state_dict(), output / "core.pt")
    torch.save(encoder.state_dict(), output / "encoder.pt")
    torch.save(optimizer.state_dict(), output / "optimizer.pt")
    write(output / "history.json", history)
    manifest.update(status="training_complete", completed=time.time(), core_sha256=sha(output / "core.pt"),
                    encoder_sha256=sha(output / "encoder.pt"), optimizer_sha256=sha(output / "optimizer.pt"),
                    history_sha256=sha(output / "history.json"))
    write(output / "manifest.json", manifest)
    (output / "TRAINING_COMPLETED").write_text("complete\n", encoding="utf-8")


def evaluate(checkpoint, output, device="cuda"):
    checkpoint, output = Path(checkpoint), Path(output)
    train_manifest_path = checkpoint.parent / "manifest.json"
    if not train_manifest_path.is_file():
        raise FileNotFoundError(train_manifest_path)
    train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    if (train_manifest.get("experiment") != "SW0120" or train_manifest.get("status") != "training_complete"
            or train_manifest.get("core_sha256") != sha(checkpoint)
            or train_manifest.get("update3_proof_sha256") != sha(PROOF_PATH)):
        raise AssertionError("evaluation requires the completed proof-bound SW0120 checkpoint")
    original_runner = sw117.RUNNER
    sw117.RUNNER = RUNNER
    try:
        # Evaluator uses the unchanged actual spike/Q classifier. Guard is a
        # training-backward intervention only.
        sw117.evaluate(0, "analytic_candidate", checkpoint, output, device=device)
    finally:
        sw117.RUNNER = original_runner
    report = json.loads(output.read_text(encoding="utf-8"))
    report.update(experiment="SW0120", arm=ARM, seed=0,
                  training_manifest_sha256=sha(train_manifest_path),
                  checkpoint_sha256=sha(checkpoint), evaluation_runner_sha256=sha(RUNNER),
                  guarded_affinity_used_for_prediction=False)
    write(output, report)
    sidecar_path = output.parent / "evaluation_manifest.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar.update(experiment="SW0120", arm=ARM, evaluation_sha256=sha(output),
                   evaluation_runner_sha256=sha(RUNNER), guarded_affinity_used_for_prediction=False)
    write(sidecar_path, sidecar)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("validate-control", "preflight", "train", "evaluate"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "validate-control":
        result = validate_control(Path(args.output))
    elif args.stage == "preflight":
        result = preflight(Path(args.output), args.device)
    elif args.stage == "train":
        train(Path(args.output), args.device)
        result = {"status": "training_complete", "output": args.output}
    else:
        if not args.checkpoint:
            parser.error("evaluate requires --checkpoint")
        result = evaluate(Path(args.checkpoint), Path(args.output), args.device)
    # Keep full arrays in the durable artifact; stdout is for the dispatcher.
    summary = {"status": result.get("status"), "stage": args.stage,
               "experiment": result.get("experiment", "SW0120"),
               "seed": result.get("seed", 0), "arm": result.get("arm", ARM),
               "output": args.output}
    if args.stage == "evaluate":
        score = result["sweep"][0]["scored_targets"]["our_hdf5"]
        summary["metrics"] = {key: float(score["metrics"][key])
                              for key in sorted(score["metrics"])}
    print(json.dumps(summary, separators=(",", ":"), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
