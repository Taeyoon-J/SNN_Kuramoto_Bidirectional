"""SW0125 full-horizon forward, late-tail gradient, and paired RGB pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run.py"
ARCHIVE = HERE / "results_archive"
OUT = ROOT / "trained_models/SW0125_late_rollout_credit"
SEEDS = (0, 1, 2)
ARM = "late_candidate"
BATCH, UPDATES = 16, 256
FULL_STEPS, SETTLE, TAIL_STEPS = 1024, 512, 64
TRAIN_CORE_LR, ENCODER_LR, CLIP_NORM = 3e-5, 3e-6, 1.0
LAMBDA = 7.865416617457706
LAMBDA_PATH = ROOT / "collaborative_test/SW_0117_joint_analytic_rgb/results_archive/lambda_joint_seed0.json"
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0115_analytic_partition_rgb import run as rgb_base
from collaborative_test.SW_0115_analytic_partition_rgb.loss import (
    batch_reconstruction_loss, production_partition,
)
from collaborative_test.SW_0117_joint_analytic_rgb import run as sw117
from collaborative_test.SW_0117_joint_analytic_rgb import coordinator as sw117_coordinator
from collaborative_test.SW_0122_joint_rgb_seed_replication import run as sw122
from collaborative_test.SW_0122_joint_rgb_seed_replication import coordinator as sw122_coordinator
from collaborative_test.SW_0125_late_rollout_credit.late_rollout import late_rollout
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_input_encoder
from snn_kuramoto_bidirectional.training.train_s2net_core import phase_locking_value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def require_new(path, what):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"preserving existing {what}: {path}")
    return path


def implementation_fingerprint():
    paths = [RUNNER, HERE / "evaluate.py", HERE / "late_rollout.py", HERE / "verify_foundation.py",
             HERE / "coordinator.py", HERE / "protocol.json", LAMBDA_PATH,
             sw117.RUNNER, sw117.HERE / "protocol.json", sw117.HERE / "coordinator.py",
             sw122.RUNNER, sw122.HERE / "protocol.json", sw122.HERE / "coordinator.py",
             sw122.HISTORICAL_SUMMARY,
             ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
             ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/run.py",
             ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/loss.py",
             ROOT / "collaborative_test/SW_0094_aligned_joint_pilot/run.py",
             ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
             ROOT / "snn_kuramoto_bidirectional/kuramoto_layer.py",
             ROOT / "snn_kuramoto_bidirectional/dendric_layer.py",
             ROOT / "snn_kuramoto_bidirectional/membrane_layer.py",
             ROOT / "snn_kuramoto_bidirectional/sinusoidal_gating.py",
             ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
             ROOT / "snn_kuramoto_bidirectional/evaluation.py",
             ROOT / "snn_kuramoto_bidirectional/loss_function.py",
             ROOT / "snn_kuramoto_bidirectional/training/train_gamma_initializer.py",
             ROOT / "snn_kuramoto_bidirectional/training/train_s2net_core.py",
             ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in paths}


def source_contract(seed):
    source, manifest_path, manifest = base.source_paths(seed)
    ids, rows = base.train_indices(seed)
    if (len(ids) != 4096 or len(rows) != 4096 or len(np.unique(ids)) != 4096
            or manifest.get("status") != "complete"
            or manifest.get("source_model_seed") != seed
            or manifest.get("unique_images_seen") != 4096
            or manifest.get("steps") != 256 or manifest.get("batch") != 16
            or manifest.get("seed") != 117 + seed
            or manifest.get("training_ids") != ids.tolist()
            or manifest.get("ground_truth_used_for_training") is not False
            or sha(source) != base.EXPECTED_SOURCE_SHAS[seed]):
        raise AssertionError(f"SW0097 source/order contract failed for seed {seed}")
    return source, manifest_path, manifest, ids, rows


def validate_foundation_report(seed, source, source_manifest, ids, gamma_manifest_sha,
                               rgb_cache_sha, rgb_cache_manifest_sha):
    """Require the fresh corrected full-horizon proof before any pilot stage."""
    proof_path = ARCHIVE / f"foundation_layoutfix_seed{seed}_source_20261009.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    required_trace = {"theta", "spikes", "membrane", "component_spikes", "component_membrane"}
    required_loss = {"primary_phase", "positive_actual_q", "analytic_actual_h_rgb"}
    expected_impl = {
        "foundation": sha(HERE / "late_rollout.py"),
        "foundation_verifier": sha(HERE / "verify_foundation.py"),
        "source_core": sha(ROOT / "snn_kuramoto_bidirectional/s2net_cls.py"),
        "synchrony": sha(ROOT / "snn_kuramoto_bidirectional/spike_classifier.py"),
        "kuramoto": sha(ROOT / "snn_kuramoto_bidirectional/kuramoto_layer.py"),
        "dendrite": sha(ROOT / "snn_kuramoto_bidirectional/dendric_layer.py"),
        "membrane": sha(ROOT / "snn_kuramoto_bidirectional/membrane_layer.py"),
        "gating": sha(ROOT / "snn_kuramoto_bidirectional/sinusoidal_gating.py"),
        "phase_loss": sha(ROOT / "snn_kuramoto_bidirectional/loss_function.py"),
        "rgb_loss": sha(ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/loss.py"),
    }
    if (proof.get("status") != "complete" or proof.get("experiment") != "SW0125"
            or proof.get("seed") != seed or proof.get("optimizer_updates") != 0
            or proof.get("ground_truth_used") is not False
            or proof.get("source_core_sha256") != sha(source)
            or proof.get("source_manifest_sha256") != sha(source_manifest)
            or proof.get("encoder_sha256") != sha(sw117.ENCODER_PATH)
            or proof.get("preprocessing_sha256") != sha(sw117.STATS_PATH)
            or proof.get("gamma_train_sha256") != sha(base.GAMMA_TRAIN)
            or proof.get("gamma_train_manifest_sha256") != gamma_manifest_sha
            or proof.get("rgb_train_cache_manifest_sha256") != rgb_cache_manifest_sha
            or proof.get("rgb_train_cache_sha256") != rgb_cache_sha
            or proof.get("total_time_steps") != FULL_STEPS
            or proof.get("loss_settle_steps") != SETTLE
            or proof.get("gradient_tail_steps") != TAIL_STEPS
            or proof.get("detached_prefix_steps") != FULL_STEPS - TAIL_STEPS
            or proof.get("global_image_ids") != ids[:BATCH].tolist()
            or proof.get("trace_parity_exact", {}) != {key: True for key in required_trace}
            or proof.get("loss_parity_exact", {}) != {key: True for key in required_loss}
            or proof.get("production_hard_labels_and_H_exact") is not True
            or proof.get("implementation_sha256") != expected_impl):
        raise AssertionError(f"corrected SW0125 source foundation proof is stale or invalid: {proof_path}")
    gamma_delta = float(proof.get("gamma_cache_max_abs_diff", float("nan")))
    if not math.isfinite(gamma_delta) or gamma_delta > 2e-5:
        raise AssertionError("foundation proof's live/cache gamma parity is invalid")
    gradients = proof.get("late_tail_rgb_gradient_norm_by_family", {})
    for family in ("encoder", "graph", "oscillator_drive", "kuramoto", "dendritic", "membrane"):
        value = float(gradients.get(family, float("nan")))
        if not math.isfinite(value) or value <= 0:
            raise AssertionError(f"foundation proof lacks finite nonzero {family} RGB credit")
    return {"path": str(proof_path), "sha256": sha(proof_path),
            "implementation_sha256": expected_impl}


def _completed_task(coordinator, stage, seed, arm):
    task = next((row for row in coordinator.task_plan()
                 if row.get("stage") == stage and row.get("seed") == seed
                 and row.get("arm") == arm), None)
    if task is None or not coordinator.valid_result(task):
        raise AssertionError(f"registered matched reference is incomplete: {stage} seed{seed} {arm}")
    return task


def validate_short_reference(seed):
    if seed == 0:
        coordinator, runner = sw117_coordinator, sw117
        source_metrics = None
    else:
        coordinator, runner = sw122_coordinator, sw122
        source_ref = sw122.validate_source_reference(seed)
        source_metrics = source_ref["source_metrics"]
    for arm in ("control", "analytic_candidate"):
        for stage in ("preflight", "train", "evaluate"):
            _completed_task(coordinator, stage, seed, arm)
    if seed == 0:
        endpoint = runner.validate_source_endpoint_review()
        report_path = runner.SOURCE_ENDPOINT_REPORT
        report = json.loads(report_path.read_text(encoding="utf-8"))
        score = report["scores"]["source16_vs_modal8"]
        source_metrics = score["mean"]
        source_eval_sha = endpoint["report_sha256"]
    else:
        source_eval_sha = source_ref["source_evaluation_sha256"]
    candidate_eval = coordinator.artifact_path(
        next(row for row in coordinator.task_plan()
             if row.get("stage") == "evaluate" and row.get("seed") == seed
             and row.get("arm") == "analytic_candidate"))
    candidate_report = json.loads(candidate_eval.read_text(encoding="utf-8"))
    candidate_score = candidate_report["sweep"][0]["scored_targets"]["our_hdf5"]
    metrics = ("fg_ari", "foreground_iou", "matched_object_iou")
    for name in metrics:
        if (candidate_score.get("valid_count", {}).get(name) != 320
                or len(candidate_score.get("per_image", {}).get(name, [])) != 320
                or not math.isfinite(float(candidate_score["metrics"][name]))):
            raise AssertionError(f"matched SW seed-{seed} candidate metric is invalid: {name}")
    return {"source_metrics": source_metrics, "source_evaluation_sha256": source_eval_sha,
            "short_candidate_metrics": candidate_score["metrics"],
            "short_candidate_evaluation_sha256": sha(candidate_eval),
            "short_candidate_manifest_sha256": sha(candidate_eval.parent / "manifest.json")}


def validate_immutable_lambda():
    evidence = sw122.validate_historical_seed0()
    summary = json.loads(sw122.HISTORICAL_SUMMARY.read_text(encoding="utf-8"))
    if (summary.get("lambda_joint") != LAMBDA
            or evidence.get("lambda_sha256") != sha(LAMBDA_PATH)):
        raise AssertionError("SW0125 requires the exact immutable SW0117 seed-0 lambda")
    return {"lambda_joint": LAMBDA, "lambda_artifact_sha256": evidence["lambda_sha256"],
            "historical_seed0_summary_sha256": evidence["summary_sha256"]}


def load_models(seed, device):
    source, source_manifest, manifest, ids, rows = source_contract(seed)
    if (sha(sw117.ENCODER_PATH) != base.EXPECTED_ENCODER_SHA256
            or sha(sw117.STATS_PATH) != base.EXPECTED_PREPROCESSING_SHA256):
        raise AssertionError("registered encoder/preprocessing asset SHA mismatch")
    core = base.make_core(device, steps=64)
    core.load_state_dict(torch.load(source, map_location=device, weights_only=True), strict=True)
    core.requires_grad_(True)
    core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.shape[0])]
    if (core.graph_generator is None or core.graph_generator.uses_feedback
            or core.kuramoto.spike_pulse_gain is not None or core.osc_dim != 4):
        raise AssertionError("SW0097 source dynamics are outside the registered static D4 contract")
    encoder = load_input_encoder(str(sw117.ENCODER_PATH), num_kernels=8,
                                 kernel_size=3, channels=3, device=device)
    encoder.requires_grad_(True)
    patcher = sw117.FeaturePatchGammaInitializer(grid_size=16).to(device)
    stats = torch.load(sw117.STATS_PATH, map_location="cpu", weights_only=True)
    mean, std, clip = sw117.preprocessing_tensors(stats, device)
    return core, encoder, patcher, mean, std, clip, source, source_manifest, manifest, ids, rows


def _families(core, encoder):
    return sw117.eligible_params(core, encoder)


def _named_credit_groups(core, encoder):
    core_named = dict(core.named_parameters())
    groups = {
        "encoder": list(encoder.parameters()),
        "graph": [p for n, p in core_named.items() if n.startswith("graph_generator.")],
        "oscillator_drive": [p for n, p in core_named.items()
                             if n.startswith(("gamma_channel_proj.", "gamma_phase_gain"))],
        "kuramoto": [p for n, p in core_named.items() if n.startswith("kuramoto.")],
        "dendritic": [p for n, p in core_named.items() if n.startswith("dendric_layer.")],
        "membrane": [p for n, p in core_named.items() if n.startswith("membrane_layer.")],
    }
    if any(not params for params in groups.values()):
        raise AssertionError("SW0125 source is missing a required trainable credit family")
    return groups


def _credit_norms(loss, groups):
    result = {}
    for name, params in groups.items():
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        value = sw117.grad_norm(grads)
        if not math.isfinite(value) or value <= 0:
            raise AssertionError(f"SW0125 objective has no finite nonzero {name} credit")
        result[name] = value
    return result


def _changed_parameter_groups(before, module, groups):
    after = dict(module.named_parameters())
    before_named = dict(before)
    result = {}
    name_for_id = {id(parameter): name for name, parameter in module.named_parameters()}
    for family, params in groups.items():
        names = [name_for_id[id(parameter)] for parameter in params]
        result[family] = any(not torch.equal(before_named[name], after[name]) for name in names)
    return result


def _objective(core, gamma, patches, lossfn, live_tail=True):
    rollout = late_rollout(core, gamma, total_steps=FULL_STEPS,
                           live_tail_steps=TAIL_STEPS if live_tail else 0)
    components, spikes, theta = (rollout["component_spikes"], rollout["spikes"], rollout["theta"])
    if tuple(components.shape[1:]) != (4, 256, FULL_STEPS):
        raise AssertionError("SW0125 requires all four actual component traces for 1024 frames")
    q = spike_synchrony_affinity(spikes, components=components, settle=SETTLE,
                                 affinity_mode="spike")
    plv = phase_locking_value(theta, settle=SETTLE, combine="mean")
    primary, _ = lossfn(plv=plv, theta=theta[:, SETTLE:])
    positive, _ = lossfn(plv=q)
    old = primary + 5.0 * positive
    labels, hard = production_partition(spikes, components, settle=SETTLE)
    rgb_loss, per_image, _, _ = batch_reconstruction_loss(q, hard, patches)
    return {"rollout": rollout, "q": q, "plv": plv, "primary": primary,
            "positive": positive, "old": old, "labels": labels, "hard": hard,
            "rgb_loss": rgb_loss, "rgb_per_image": per_image}


def _source_reference_core(seed, device, source_path):
    reference = base.make_core(device, steps=64)
    reference.load_state_dict(torch.load(source_path, map_location=device, weights_only=True), strict=True)
    reference.requires_grad_(False)
    reference._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.shape[0])]
    return reference.eval()


def _read_batch(seed, rows, rgb_cache, gamma_cache, device):
    images = sw117.read_rgb(rgb_cache, rows, device)
    patches = rgb_base.rgb_patch_means(images / 255.0)
    cached = gamma_cache[torch.as_tensor(rows, dtype=torch.long)].to(device)
    return images, patches, cached


def _compare_no_grad(reference, candidate, gamma, cached, patches, lossfn):
    gamma_delta = float((gamma.detach() - cached).abs().max())
    if not math.isfinite(gamma_delta) or gamma_delta > 2e-5:
        raise AssertionError(f"live encoder gamma exceeds registered cache parity: {gamma_delta}")
    with torch.no_grad():
        _, ref_spikes, ref_mem, ref_theta = reference(
            gamma, return_core_out=True, return_theta=True, num_time_steps=FULL_STEPS)
        ref_components = reference.last_component_spikes.clone()
        ref_component_mem = reference.last_component_out.clone()
    parts = _objective(candidate, gamma, patches, lossfn, live_tail=True)
    traces = parts["rollout"]
    exact = {
        "spikes": torch.equal(ref_spikes, traces["spikes"]),
        "core_out": torch.equal(ref_mem, traces["membrane"]),
        "theta": torch.equal(ref_theta, traces["theta"]),
        "component_spikes": torch.equal(ref_components, traces["component_spikes"]),
        "component_membrane": torch.equal(ref_component_mem, traces["component_membrane"]),
    }
    if not all(exact.values()):
        raise AssertionError(f"same-input full/late rollout mismatch: {exact}")
    ref_q = spike_synchrony_affinity(ref_spikes, components=ref_components,
                                    settle=SETTLE, affinity_mode="spike")
    if not torch.equal(ref_q, parts["q"]):
        raise AssertionError("same-input actual-Q forward values differ")
    ref_labels, ref_hard = production_partition(ref_spikes, ref_components, settle=SETTLE)
    if not torch.equal(ref_labels, parts["labels"]) or any(
            not torch.equal(left, right) for left, right in zip(ref_hard, parts["hard"])):
        raise AssertionError("same-input production labels/H differ")
    ref_plv = phase_locking_value(ref_theta, settle=SETTLE, combine="mean")
    ref_primary, _ = lossfn(plv=ref_plv, theta=ref_theta[:, SETTLE:])
    ref_positive, _ = lossfn(plv=ref_q)
    ref_rgb, _, _, _ = batch_reconstruction_loss(ref_q, ref_hard, patches)
    values_exact = {"primary": torch.equal(ref_primary, parts["primary"]),
                    "positive_q": torch.equal(ref_positive, parts["positive"]),
                    "rgb": torch.equal(ref_rgb, parts["rgb_loss"])}
    if not all(values_exact.values()):
        raise AssertionError(f"same-input objective value mismatch: {values_exact}")
    return parts, {"gamma_cache_max_abs_diff": gamma_delta,
                   "full_trace_exact": exact, "q_exact": True,
                   "production_labels_h_exact": True, "objective_values_exact": values_exact}


def _gradient_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return sw117.grad_norm(grads), grads


def preflight(seed, output, device="cuda:0"):
    if seed not in SEEDS:
        raise ValueError("SW0125 only supports registered source seeds 0, 1, and 2")
    output = require_new(output, "SW0125 preflight")
    references = validate_short_reference(seed)
    lambda_reference = validate_immutable_lambda()
    endpoint = (sw117.validate_source_endpoint_review() if seed == 0
                else sw122.validate_source_reference(seed))
    source, source_manifest, source_meta, ids, rows = source_contract(seed)
    (core, encoder, patcher, mean, std, clip, *_rest) = load_models(seed, device)
    gamma_cache, gamma_manifest = base.validate_gamma_cache(base.GAMMA_TRAIN, base.GAMMA_TRAIN_MANIFEST)
    rgb_cache, rgb_manifest, rgb_cache_sha = sw117.load_rgb_training_cache()
    foundation = validate_foundation_report(
        seed, source, source_manifest, ids, sha(base.GAMMA_TRAIN_MANIFEST),
        rgb_cache_sha, sha(sw117.RGB_CACHE_MANIFEST))
    reference_core = _source_reference_core(seed, device, source)
    core.train(); core.graph_generator.train(); encoder.train()
    families = _families(core, encoder)
    detailed_groups = _named_credit_groups(core, encoder)
    lossfn = sw117.criterion()
    before_core, before_encoder = sw117.clone_state(core), sw117.clone_state(encoder)
    rows_report, real_losses, scrambled_losses = [], [], []
    permutation = torch.as_tensor(np.random.default_rng(11501).permutation(256),
                                  dtype=torch.long, device=device)
    for batch_idx in range(4):
        start = batch_idx * BATCH
        batch_rows = rows[start:start+BATCH]
        images, patches, cached = _read_batch(seed, batch_rows, rgb_cache, gamma_cache, device)
        gamma = sw117.encode_rgb(encoder, patcher, mean, std, clip, images)
        parts, parity = _compare_no_grad(reference_core, core, gamma, cached, patches, lossfn)
        stats, old_by, rgb_by = sw117.collect_joint_gradients(
            parts["old"], parts["rgb_loss"], families, candidate=True)
        q_grad = torch.autograd.grad(parts["rgb_loss"], parts["q"], retain_graph=True)[0]
        q_norm = sw117.grad_norm((q_grad,))
        sw117.validate_gradient_families(stats, "analytic_candidate", q_norm)
        detailed_rgb_norms = _credit_norms(parts["rgb_loss"], detailed_groups)
        shuffled = [h.index_select(0, permutation) for h in parts["hard"]]
        _, scramble_image_losses, _, _ = batch_reconstruction_loss(parts["q"], shuffled, patches)
        if not torch.isfinite(scramble_image_losses).all():
            raise FloatingPointError("nonfinite fixed H-row scramble loss")
        real_losses.extend(float(value.detach()) for value in parts["rgb_per_image"])
        scrambled_losses.extend(float(value.detach()) for value in scramble_image_losses)
        rows_report.append({
            "batch": batch_idx, "training_ids": ids[start:start+BATCH].tolist(),
            "same_input_parity": parity,
            "primary_phase_last512": float(parts["primary"].detach()),
            "positive_actual_q_last512": float(parts["positive"].detach()),
            "rgb_loss_last512": float(parts["rgb_loss"].detach()),
            "old_gradient_norms": {name: stats[name]["old_gradient_norm"]
                                   for name in ("encoder", "graph", "core", "joint")},
            "rgb_gradient_norms": {name: stats[name]["rgb_gradient_norm"]
                                   for name in ("encoder", "graph", "core", "joint")},
            "rgb_to_q_gradient_norm": q_norm,
            "rgb_gradient_norms_by_named_family": detailed_rgb_norms,
            "hard_group_counts": [int(h.shape[1]) for h in parts["hard"]],
        })
        core.zero_grad(set_to_none=True); encoder.zero_grad(set_to_none=True)
    if any(not torch.equal(before_core[k], core.state_dict()[k]) for k in before_core):
        raise AssertionError("four-batch SW0125 preflight changed source core before throwaway update")
    if any(not torch.equal(before_encoder[k], encoder.state_dict()[k]) for k in before_encoder):
        raise AssertionError("four-batch SW0125 preflight changed encoder before throwaway update")
    scramble_excess = rgb_base.mean_scramble_excess(real_losses, scrambled_losses)
    if not math.isfinite(scramble_excess) or scramble_excess <= 0:
        raise AssertionError("fixed count-preserving H scramble did not increase RGB loss")

    # Disposable update: same full objective and two-gradient recipe as training.
    (test_core, test_encoder, test_patcher, test_mean, test_std, test_clip,
     *_rest) = load_models(seed, device)
    test_core.train(); test_core.graph_generator.train(); test_encoder.train()
    test_families = _families(test_core, test_encoder)
    test_detailed_groups = _named_credit_groups(test_core, test_encoder)
    optimizer = sw117.optimizer_for(test_core, test_encoder)
    test_images, test_patches, test_cached = _read_batch(seed, rows[:BATCH], rgb_cache,
                                                         gamma_cache, device)
    test_gamma = sw117.encode_rgb(test_encoder, test_patcher, test_mean, test_std,
                                  test_clip, test_images)
    test_reference = _source_reference_core(seed, device, source)
    test_parts, _ = _compare_no_grad(test_reference, test_core, test_gamma,
                                     test_cached, test_patches, lossfn)
    before_test_core, before_test_encoder = sw117.clone_state(test_core), sw117.clone_state(test_encoder)
    before_test_core_named = {name: value.detach().clone()
                              for name, value in test_core.named_parameters()}
    before_test_encoder_named = {f"encoder.{name}": value.detach().clone()
                                 for name, value in test_encoder.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    _, old_grads, rgb_grads = sw117.collect_joint_gradients(
        test_parts["old"], test_parts["rgb_loss"], test_families, candidate=True)
    sw117.apply_family_gradients(test_families, old_grads, rgb_grads, LAMBDA)
    gradnorm = torch.nn.utils.clip_grad_norm_(test_families["all"], CLIP_NORM)
    detailed_combined_norms = {}
    for name, params in test_detailed_groups.items():
        value = sw117.grad_norm([parameter.grad for parameter in params])
        if not math.isfinite(value) or value <= 0:
            raise AssertionError(f"SW0125 disposable update lacks {name} gradient")
        detailed_combined_norms[name] = value
    total = test_parts["old"] + LAMBDA * test_parts["rgb_loss"]
    if not torch.isfinite(total) or not torch.isfinite(gradnorm) or float(gradnorm) <= 0:
        raise FloatingPointError("SW0125 disposable full-horizon update has invalid loss/gradient")
    optimizer.step()
    changed = sw117.changed_trainable_groups(before_test_core, test_core.state_dict(),
                                              before_test_encoder, test_encoder.state_dict())
    if not all(changed.values()):
        raise AssertionError(f"SW0125 disposable update failed parameter families: {changed}")
    changed_core_families = _changed_parameter_groups(
        before_test_core_named, test_core, {name: params for name, params in test_detailed_groups.items()
                                            if name != "encoder"})
    changed_encoder = any(not torch.equal(before_test_encoder_named[f"encoder.{name}"], value)
                          for name, value in test_encoder.named_parameters())
    if not all(changed_core_families.values()) or not changed_encoder:
        raise AssertionError(f"SW0125 disposable update failed detailed parameter groups: "
                             f"{changed_core_families}, encoder={changed_encoder}")

    record = {
        "status": "passed", "experiment": "SW0125", "seed": seed, "arm": ARM,
        "implementation_fingerprint": implementation_fingerprint(),
        "source_core_sha256": sha(source), "source_manifest_sha256": sha(source_manifest),
        "source_evaluation_sha256": endpoint.get("source_evaluation_sha256", endpoint.get("report_sha256")),
        "short_reference": references,
        "lambda_reference": lambda_reference,
        "foundation_report": foundation,
        "foundation_report_sha256": foundation["sha256"],
        "encoder_sha256": sha(sw117.ENCODER_PATH),
        "preprocessing_sha256": sha(sw117.STATS_PATH),
        "gamma_train_sha256": sha(base.GAMMA_TRAIN),
        "gamma_train_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
        "rgb_cache_sha256": rgb_cache_sha,
        "rgb_cache_manifest_sha256": sha(sw117.RGB_CACHE_MANIFEST),
        "rgb_cache_source_size_bytes": rgb_manifest.get("source_size_bytes"),
        "training_ids": ids.tolist(),
        "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "shuffle_seed": 117 + seed, "batch_size": BATCH, "updates": UPDATES,
        "total_time_steps": FULL_STEPS, "settle": SETTLE, "tail_steps": TAIL_STEPS,
        "lambda_joint": LAMBDA, "lambda_artifact_sha256": sha(LAMBDA_PATH),
        "ground_truth_used": False, "batches": rows_report,
        "fixed_h_scramble_rgb_excess": scramble_excess,
        "throwaway_b16_adam_update": True,
        "throwaway_gradient_norm_preclip": float(gradnorm),
        "throwaway_gradient_norms_by_named_family": detailed_combined_norms,
        "throwaway_parameter_groups_changed": changed,
        "throwaway_named_core_groups_changed": changed_core_families,
        "throwaway_encoder_changed": changed_encoder,
    }
    write_once(output, record)
    return record


def train(seed, output, device="cuda:0", steps=UPDATES):
    if seed not in SEEDS or steps != UPDATES:
        raise ValueError("SW0125 only supports registered three seeds and 256 updates")
    validate_short_reference(seed)
    lambda_reference = validate_immutable_lambda()
    output = require_new(output, "SW0125 training output")
    pf_path = ARCHIVE / f"preflight_seed{seed}_{ARM}.json"
    if not pf_path.is_file():
        raise FileNotFoundError(pf_path)
    pf = json.loads(pf_path.read_text(encoding="utf-8"))
    if (pf.get("status") != "passed" or pf.get("experiment") != "SW0125"
            or pf.get("seed") != seed or pf.get("arm") != ARM
            or pf.get("implementation_fingerprint") != implementation_fingerprint()
            or pf.get("lambda_joint") != LAMBDA):
        raise AssertionError("current SW0125 preflight and immutable lambda are required")
    source, source_manifest, source_meta, ids, rows = source_contract(seed)
    (core, encoder, patcher, mean, std, clip, *_rest) = load_models(seed, device)
    gamma_cache, _ = base.validate_gamma_cache(base.GAMMA_TRAIN, base.GAMMA_TRAIN_MANIFEST)
    rgb_cache, rgb_meta, rgb_cache_sha = sw117.load_rgb_training_cache()
    foundation = validate_foundation_report(
        seed, source, source_manifest, ids, sha(base.GAMMA_TRAIN_MANIFEST),
        rgb_cache_sha, sha(sw117.RGB_CACHE_MANIFEST))
    bindings = {
        "source_core_sha256": sha(source), "source_manifest_sha256": sha(source_manifest),
        "encoder_sha256": sha(sw117.ENCODER_PATH), "preprocessing_sha256": sha(sw117.STATS_PATH),
        "gamma_train_sha256": sha(base.GAMMA_TRAIN),
        "gamma_train_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
        "rgb_cache_sha256": rgb_cache_sha, "rgb_cache_manifest_sha256": sha(sw117.RGB_CACHE_MANIFEST),
        "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "foundation_report_sha256": foundation["sha256"],
        "lambda_artifact_sha256": lambda_reference["lambda_artifact_sha256"],
        "lambda_joint": LAMBDA,
        "implementation_fingerprint": implementation_fingerprint(),
    }
    if any(pf.get(key) != value for key, value in bindings.items()):
        raise AssertionError("SW0125 preflight source/cache/order/code bindings changed")
    optimizer = sw117.optimizer_for(core, encoder)
    families = _families(core, encoder)
    lossfn = sw117.criterion()
    torch.manual_seed(117 + seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(117 + seed)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "training", "experiment": "SW0125", "seed": seed, "arm": ARM,
        **bindings, "preflight_sha256": sha(pf_path),
        "source_evaluation_sha256": pf.get("source_evaluation_sha256"),
        "short_reference": pf["short_reference"],
        "training_ids": ids.tolist(), "shuffle_seed": 117 + seed,
        "updates": steps, "batch_size": BATCH, "time_steps": FULL_STEPS,
        "settle": SETTLE, "tail_steps": TAIL_STEPS,
        "core_graph_lr": TRAIN_CORE_LR, "encoder_lr": ENCODER_LR,
        "clip_norm": CLIP_NORM, "lambda_joint": LAMBDA,
        "lambda_artifact_sha256": sha(LAMBDA_PATH),
        "objective": "phase_primary_last512 + 5*positive_actual_four_component_Q_last512 + lambda*unchanged_SW0115_actual_H_RGB_last512",
        "gradient_contract": "full 1024-step forward; prefix960 no-grad detached-state boundary; last64 live tail; truncated BPTT, not full-horizon BPTT",
        "ground_truth_used_for_training": False,
    }
    write_once(output / "manifest.json", manifest)
    core.train(); core.graph_generator.train(); encoder.train()
    history = []
    for update in range(steps):
        start = update * BATCH
        batch_rows = rows[start:start+BATCH]
        images, patches, _cached = _read_batch(seed, batch_rows, rgb_cache, gamma_cache, device)
        gamma = sw117.encode_rgb(encoder, patcher, mean, std, clip, images)
        parts = _objective(core, gamma, patches, lossfn, live_tail=True)
        total = parts["old"] + LAMBDA * parts["rgb_loss"]
        if not torch.isfinite(total):
            raise FloatingPointError(f"nonfinite SW0125 total at update {update+1}")
        optimizer.zero_grad(set_to_none=True)
        stats, old_by, rgb_by = sw117.collect_joint_gradients(
            parts["old"], parts["rgb_loss"], families, candidate=True)
        sw117.apply_family_gradients(families, old_by, rgb_by, LAMBDA)
        gradnorm = torch.nn.utils.clip_grad_norm_(families["all"], CLIP_NORM)
        if not torch.isfinite(gradnorm) or float(gradnorm) <= 0:
            raise FloatingPointError(f"invalid SW0125 joint gradient at update {update+1}")
        optimizer.step()
        history.append({
            "update": update+1, "total": float(total.detach()),
            "primary_phase_last512": float(parts["primary"].detach()),
            "positive_actual_q_last512": float(parts["positive"].detach()),
            "rgb_last512": float(parts["rgb_loss"].detach()),
            "lambda_joint": LAMBDA, "gradient_norm_preclip": float(gradnorm),
            "family_gradient_stats": stats,
            "mean_hard_groups": float(np.mean([h.shape[1] for h in parts["hard"]])),
        })
        if update == 0 or (update + 1) % 32 == 0:
            write_once(output / f"progress_{update+1:04d}.json",
                       {"status": "training", "seed": seed, "update": update+1,
                        "total_updates": steps})
    if any(not torch.isfinite(p).all() for p in list(core.parameters()) + list(encoder.parameters())):
        raise FloatingPointError("nonfinite final SW0125 parameter")
    torch.save(core.state_dict(), output / "core.pt")
    torch.save(encoder.state_dict(), output / "encoder.pt")
    torch.save(optimizer.state_dict(), output / "optimizer.pt")
    write_once(output / "history.json", history)
    manifest.update(status="training_complete", completed=time.time(),
                    core_sha256=sha(output / "core.pt"), encoder_sha256=sha(output / "encoder.pt"),
                    optimizer_sha256=sha(output / "optimizer.pt"), history_sha256=sha(output / "history.json"))
    # Replace only this attempt's own in-progress manifest atomically.
    temp = output / "manifest.complete.tmp"
    temp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(output / "manifest.json")
    (output / "TRAINING_COMPLETED").write_text("complete\n", encoding="utf-8")
    return manifest


def evaluate(seed, checkpoint, output, device="cuda:0"):
    if seed not in SEEDS:
        raise ValueError("unsupported SW0125 seed")
    checkpoint, output = Path(checkpoint), Path(output)
    sidecar = output.parent / "evaluation_manifest.json"
    require_new(output, "SW0125 evaluation report"); require_new(sidecar, "SW0125 evaluation sidecar")
    train_manifest_path = checkpoint.parent / "manifest.json"
    encoder_path = checkpoint.parent / "encoder.pt"
    manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "training_complete" or manifest.get("experiment") != "SW0125"
            or manifest.get("seed") != seed or manifest.get("arm") != ARM
            or manifest.get("core_sha256") != sha(checkpoint)
            or manifest.get("encoder_sha256") != sha(encoder_path)
            or manifest.get("history_sha256") != sha(checkpoint.parent / "history.json")
            or manifest.get("optimizer_sha256") != sha(checkpoint.parent / "optimizer.pt")
            or not (checkpoint.parent / "TRAINING_COMPLETED").is_file()):
        raise AssertionError("evaluation requires hash-validated completed SW0125 training")
    rgb_val, _, rgb_val_sha = sw117.validate_rgb_validation_cache()
    (eval_core, encoder, patcher, mean, std, clip, *_rest) = load_models(seed, device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True), strict=True)
    encoder.eval(); patcher.eval()
    gammas = []
    with torch.no_grad():
        for start in range(0, 320, 8):
            images = sw117.read_rgb(rgb_val, np.arange(start, start+8), device)
            gammas.append(sw117.encode_rgb(encoder, patcher, mean, std, clip, images).cpu())
    gamma = torch.cat(gammas, dim=0)
    gamma_path = output.parent / "trained_encoder_gamma_validation.pt"
    gamma_manifest_path = output.parent / "trained_encoder_gamma_validation_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    require_new(gamma_path, "trained encoder validation gamma")
    require_new(gamma_manifest_path, "trained encoder validation gamma manifest")
    torch.save(gamma, gamma_path)
    gamma_meta = {"status": "complete", "kind": "validation", "ids": [1320, 1639],
                  "image_ids": [1320, 1639],
                  "images": 320, "gamma_sha256": sha(gamma_path),
                  "encoder_sha256": sha(encoder_path), "preprocessing_sha256": sha(sw117.STATS_PATH),
                  "rgb_cache_sha256": rgb_val_sha,
                  "rgb_cache_manifest_sha256": sha(sw117.VAL_RGB_MANIFEST),
                  "ground_truth_used_for_gamma": False}
    write_once(gamma_manifest_path, gamma_meta)
    old_path, old_manifest, old_validator = base.GAMMA_VAL, base.GAMMA_VAL_MANIFEST, base.validate_gamma_cache

    def validate_trained_gamma(path, manifest_path, validation=False):
        path, manifest_path = Path(path), Path(manifest_path)
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = torch.load(path, map_location="cpu", weights_only=True)
        if (not validation or path != gamma_path or manifest_path != gamma_manifest_path
                or tuple(value.shape) != (320, 8, 256) or record.get("gamma_sha256") != sha(path)
                or record.get("status") != "complete" or record.get("kind") != "validation"
                or record.get("image_ids") != [1320, 1639]
                or record.get("encoder_sha256") != sha(encoder_path)
                or record.get("preprocessing_sha256") != sha(sw117.STATS_PATH)
                or record.get("rgb_cache_sha256") != rgb_val_sha
                or record.get("rgb_cache_manifest_sha256") != sha(sw117.VAL_RGB_MANIFEST)
                or record.get("ground_truth_used_for_gamma") is not False
                or not torch.isfinite(value).all()):
            raise AssertionError("SW0125 trained gamma validation binding mismatch")
        return value, record

    try:
        base.GAMMA_VAL, base.GAMMA_VAL_MANIFEST = gamma_path, gamma_manifest_path
        base.validate_gamma_cache = validate_trained_gamma
        base.evaluate(seed, "control", checkpoint, output, device=device)
    finally:
        base.GAMMA_VAL, base.GAMMA_VAL_MANIFEST, base.validate_gamma_cache = old_path, old_manifest, old_validator
    report = json.loads(output.read_text(encoding="utf-8"))
    report.update(status="complete", experiment="SW0125", seed=seed, arm=ARM,
                  training_manifest_sha256=sha(train_manifest_path),
                  checkpoint_sha256=sha(checkpoint), encoder_checkpoint_sha256=sha(encoder_path),
                  trained_encoder_gamma_sha256=sha(gamma_path),
                  trained_encoder_gamma_manifest_sha256=sha(gamma_manifest_path),
                  training_runner_sha256=sha(RUNNER),
                  evaluation_runner_sha256=sha(HERE / "evaluate.py"),
                  shared_evaluator_sha256=sha(ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py"),
                  ground_truth_used_for_prediction=False,
                  evaluation_contract={"ids": [1320, 1639], "images": 320,
                      "batch_size": 8, "time_steps": FULL_STEPS, "settle": SETTLE,
                      "membrane_vth": 0.06, "readout_threshold": 0.50,
                      "min_group_size": 2, "background": "largest_component",
                      "ground_truth_used_for_prediction": False})
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(output)
    write_once(sidecar, {"experiment": "SW0125", "seed": seed, "arm": ARM,
        "evaluation_sha256": sha(output), "training_manifest_sha256": sha(train_manifest_path),
        "checkpoint_sha256": sha(checkpoint), "encoder_checkpoint_sha256": sha(encoder_path),
        "trained_encoder_gamma_sha256": sha(gamma_path),
        "trained_encoder_gamma_manifest_sha256": sha(gamma_manifest_path),
        "training_runner_sha256": sha(RUNNER),
        "evaluation_runner_sha256": sha(HERE / "evaluate.py"),
        "shared_evaluator_sha256": sha(ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py"),
        "evaluation_contract": report["evaluation_contract"]})
    return report


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("preflight", "train", "evaluate"):
        sp = sub.add_parser(stage)
        sp.add_argument("--seed", type=int, choices=SEEDS, required=True)
        sp.add_argument("--device", default="cuda:0")
        if stage in ("preflight", "train"):
            sp.add_argument("--output", type=Path, required=True)
        else:
            sp.add_argument("--checkpoint", type=Path, required=True)
            sp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight(args.seed, args.output, args.device)
    elif args.stage == "train":
        result = train(args.seed, args.output, args.device)
    else:
        result = evaluate(args.seed, args.checkpoint, args.output, args.device)
    print(json.dumps({"status": result.get("status"), "experiment": "SW0125",
                      "stage": args.stage, "seed": args.seed,
                      "output": str(args.output)}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
