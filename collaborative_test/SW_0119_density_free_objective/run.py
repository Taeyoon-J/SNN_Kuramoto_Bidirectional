"""SW0119 seed-0 pilot removing only PLV density-balance penalties."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run.py"
ARCHIVE = HERE / "results_archive"
OUT = ROOT / "trained_models/SW0119_density_free_objective"
BATCH, UPDATES, TRAIN_STEPS, TRAIN_SETTLE = 16, 256, 64, 32
LAMBDA = 7.865416617457706
ARM = "analytic_candidate"  # Preserve SW0117 evaluator/task-arm compatibility.

import sys
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0115_analytic_partition_rgb import run as rgb_base
from collaborative_test.SW_0115_analytic_partition_rgb.loss import (
    batch_reconstruction_loss, production_partition,
)
from collaborative_test.SW_0117_joint_analytic_rgb import run as sw117
from collaborative_test.SW_0117_joint_analytic_rgb import coordinator as sw117_coordinator
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from snn_kuramoto_bidirectional.training.train_s2net_core import _forward_with_plv


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    files = [RUNNER, HERE / "coordinator.py", HERE / "protocol.json",
             sw117.SOURCE_ENDPOINT_REVIEW, sw117.RUNNER,
             ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
             ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/run.py",
             ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/loss.py",
             ROOT / "collaborative_test/SW_0094_aligned_joint_pilot/run.py",
             ROOT / "snn_kuramoto_bidirectional/loss_function.py",
             ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
             ROOT / "snn_kuramoto_bidirectional/training/train_s2net_core.py",
             ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in files}


def loss_without_density(lossfn, plv, theta=None):
    full, full_parts = lossfn(plv=plv, theta=theta)
    density_free_fn = copy.deepcopy(lossfn)
    density_free_fn.plv_balance_weight = 0.0
    retained, retained_parts = density_free_fn(plv=plv, theta=theta)
    removed = float(lossfn.plv_balance_weight) * full_parts.get(
        "plv_balance", torch.zeros_like(full))
    if not torch.allclose(full - retained, removed, rtol=1e-6, atol=2e-6):
        raise AssertionError("density-free criterion changed more than plv_balance")
    return retained, removed, full, full_parts, retained_parts


def objective_parts_density_free(core, gamma, rgb_patches, lossfn):
    _, spikes, core_out, plv, theta = _forward_with_plv(
        core, gamma, lossfn, TRAIN_SETTLE, "phase", "mean")
    components = core.last_component_spikes
    if tuple(components.shape[1:]) != (4, 256, TRAIN_STEPS):
        raise AssertionError("SW0119 component spike trace shape changed")
    q = spike_synchrony_affinity(components.mean(dim=1), components=components,
                                 settle=TRAIN_SETTLE, affinity_mode="spike")
    if q.shape != (gamma.shape[0], 256, 256) or not torch.isfinite(q).all():
        raise AssertionError("SW0119 Q must be the finite actual four-component product")
    primary, removed_primary, full_primary, pparts, retained_pparts = loss_without_density(lossfn, plv, theta)
    positive, removed_q, full_positive, qparts, retained_qparts = loss_without_density(lossfn, q)
    old = primary + 5.0 * positive
    full_old = full_primary + 5.0 * full_positive
    density_removed = removed_primary + 5.0 * removed_q
    labels, hard = production_partition(spikes, components, settle=TRAIN_SETTLE)
    rgb_loss, per_image, _, details = batch_reconstruction_loss(q, hard, rgb_patches)
    return (old, primary, positive, rgb_loss, q, labels, hard, per_image, details,
            spikes, core_out, theta, components, density_removed,
            removed_primary, removed_q, full_old, full_primary, full_positive)


def _control_task(stage):
    return next(task for task in sw117_coordinator.task_plan()
                if task["stage"] == stage and task["arm"] == "analytic_candidate")


def validate_registered_control():
    endpoint = sw117.validate_source_endpoint_review()
    for stage in ("preflight", "train", "evaluate"):
        task = _control_task(stage)
        if not sw117_coordinator.valid_result(task):
            raise AssertionError(f"completed SW0117 candidate {stage} is not valid")
    manifest_path = sw117.OUT / "seed0_analytic_candidate/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source, source_manifest_path, source_record, ids, _rows = sw117.source_contract(0)
    preflight_path = sw117.ARCHIVE / "preflight_seed0_analytic_candidate.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    expected_ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    if (manifest.get("lambda_joint") != LAMBDA
            or manifest.get("lambda_artifact_sha256") != sha(sw117.LAMBDA_PATH)
            or manifest.get("source_core_sha256") != sha(source)
            or manifest.get("source_manifest_sha256") != sha(source_manifest_path)
            or manifest.get("training_ids") != ids.tolist()
            or manifest.get("training_ids_sha256") != expected_ids_sha
            or manifest.get("matched_shuffle_seed") != 117
            or manifest.get("updates") != UPDATES or manifest.get("batch_size") != BATCH
            or manifest.get("time_steps") != TRAIN_STEPS or manifest.get("settle") != TRAIN_SETTLE
            or manifest.get("core_graph_lr") != sw117.CORE_GRAPH_LR
            or manifest.get("encoder_lr") != sw117.ENCODER_LR
            or manifest.get("clip_norm") != sw117.CLIP_NORM
            or manifest.get("preflight_sha256") != sha(preflight_path)
            or manifest.get("implementation_fingerprint") != sw117.implementation_fingerprint()
            or manifest.get("encoder_source_sha256") != sha(sw117.ENCODER_PATH)
            or manifest.get("preprocessing_sha256") != sha(sw117.STATS_PATH)
            or manifest.get("gamma_train_sha256") != sha(base.GAMMA_TRAIN)
            or manifest.get("gamma_train_manifest_sha256") != sha(base.GAMMA_TRAIN_MANIFEST)
            or manifest.get("rgb_cache_sha256") != sha(sw117.RGB_CACHE)
            or manifest.get("rgb_cache_manifest_sha256") != sha(sw117.RGB_CACHE_MANIFEST)
            or preflight.get("lambda_joint") != LAMBDA
            or preflight.get("implementation_fingerprint") != sw117.implementation_fingerprint()
            or preflight.get("source_core_sha256") != sha(source)
            or preflight.get("source_manifest_sha256") != sha(source_manifest_path)
            or preflight.get("training_ids_sha256") != expected_ids_sha
            or preflight.get("rgb_cache_sha256") != sha(sw117.RGB_CACHE)
            or preflight.get("gamma_train_sha256") != sha(base.GAMMA_TRAIN)
            or manifest.get("source_endpoint_report_sha256") != endpoint["report_sha256"]
            or manifest.get("implementation_fingerprint") != sw117.implementation_fingerprint()):
        raise AssertionError("matched SW0117 endpoint is not bound to the registered recipe")
    return manifest_path.parent, manifest, endpoint


def _finite_gradients(stats, source):
    for family in ("encoder", "graph", "core", "joint"):
        norm = float(stats[family][source + "_gradient_norm"])
        if not math.isfinite(norm) or norm <= 0:
            raise FloatingPointError(f"SW0119 {source} gradient is not finite/nonzero for {family}")


def verify_gradient_decomposition(full_loss, retained_loss, removed_loss, families):
    report = {}
    for family in ("encoder", "graph", "core"):
        params = families[family]
        full = torch.autograd.grad(full_loss, params, retain_graph=True, allow_unused=True)
        split = torch.autograd.grad(retained_loss + removed_loss, params,
                                    retain_graph=True, allow_unused=True)
        diffs, scales = [], []
        for full_grad, split_grad in zip(full, split):
            if full_grad is None and split_grad is None:
                continue
            if full_grad is None:
                full_grad = torch.zeros_like(split_grad)
            if split_grad is None:
                split_grad = torch.zeros_like(full_grad)
            if not torch.isfinite(full_grad).all() or not torch.isfinite(split_grad).all():
                raise FloatingPointError("nonfinite SW0119 loss-decomposition gradient")
            diffs.append(float((full_grad.detach().double() - split_grad.detach().double()).abs().max()))
            scales.append(float(full_grad.detach().double().abs().max()))
        max_abs = max(diffs, default=0.0)
        scale = max(scales, default=0.0)
        relative = max_abs / max(scale, 1e-30)
        if relative > 1e-5:
            raise AssertionError(f"SW0119 removed balance gradient decomposition failed for {family}: {relative}")
        report[family] = {"max_abs_difference": max_abs, "reference_max_abs": scale,
                          "relative_max_abs_difference": relative}
    return report


def preflight(output, device="cuda"):
    output = require_vacant(output, "SW0119 preflight")
    control_dir, control_manifest, endpoint = validate_registered_control()
    if not sw117.LAMBDA_PATH.is_file():
        raise FileNotFoundError(sw117.LAMBDA_PATH)
    lambda_record = json.loads(sw117.LAMBDA_PATH.read_text(encoding="utf-8"))
    if lambda_record.get("status") != "passed" or lambda_record.get("lambda_joint") != LAMBDA:
        raise AssertionError("immutable SW0117 RGB coefficient is not valid")

    (core, encoder, patcher, mean, std, clip, source, source_manifest, source_record,
     ids, rows) = sw117.load_models(0, device)
    gamma_cache, gamma_meta = base.validate_gamma_cache(base.GAMMA_TRAIN, base.GAMMA_TRAIN_MANIFEST)
    rgb_cache, rgb_meta, rgb_cache_sha = sw117.load_rgb_training_cache()
    reference = base.make_core(device, TRAIN_STEPS)
    reference.load_state_dict(torch.load(source, map_location=device, weights_only=True), strict=True)
    reference._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
    reference.train(); core.train(); core.graph_generator.train(); encoder.train()
    torch.manual_seed(117)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(117)
    lossfn = sw117.criterion()
    families = sw117.eligible_params(core, encoder)
    initial_core, initial_encoder = sw117.clone_state(core), sw117.clone_state(encoder)
    records, real, shuffled = [], [], []
    permutation = torch.as_tensor(np.random.default_rng(11501).permutation(256),
                                  dtype=torch.long, device=device)

    for batch_index in range(4):
        start = batch_index * BATCH
        batch_rows = rows[start:start + BATCH]
        image = sw117.read_rgb(rgb_cache, batch_rows, device)
        patches = rgb_base.rgb_patch_means(image / 255.0)
        gamma = sw117.encode_rgb(encoder, patcher, mean, std, clip, image)
        cached = gamma_cache[torch.as_tensor(batch_rows, dtype=torch.long)].to(device)
        # Preserve SW0117's independent cache-input and same-input identity guards.
        legacy_parts, parity = sw117.compare_initial_rollout(
            core, reference, gamma, cached, patches, lossfn)
        parts = objective_parts_density_free(core, gamma, patches, lossfn)
        with torch.no_grad():
            ref_parts = objective_parts_density_free(reference, gamma.detach(), patches, lossfn)
        density_identity = legacy_parts[0].detach() - parts[0].detach()
        removed = parts[13]
        if not torch.allclose(density_identity, removed, rtol=1e-6, atol=2e-6):
            raise AssertionError("SW0119 loss difference is not exactly the removed balance contribution")
        if not torch.equal(parts[0].detach(), ref_parts[0].detach()):
            raise AssertionError("density-free retained objective differs on same-input source reference")
        for index in (4, 5, 9, 10, 11, 12):
            if not torch.equal(parts[index].detach(), ref_parts[index].detach()):
                raise AssertionError(f"same-input reference mismatch in density-free field {index}")
        for index in (4, 5, 9, 10, 11, 12):
            if not torch.equal(legacy_parts[index].detach(), parts[index].detach()):
                raise AssertionError(f"density removal changed rollout/readout field {index}")
        for other in (legacy_parts, ref_parts):
            if len(parts[6]) != len(other[6]) or any(
                    not torch.equal(a.detach(), b.detach()) for a, b in zip(parts[6], other[6])):
                raise AssertionError("density removal changed production H partition")
        if not torch.equal(legacy_parts[7].detach(), parts[7].detach()):
            raise AssertionError("density removal changed RGB readout loss")
        del legacy_parts, ref_parts
        grad_decomposition = verify_gradient_decomposition(parts[16], parts[0], removed, families)

        old, rgb_loss = parts[0], parts[3]
        stats, old_by, rgb_by = sw117.collect_joint_gradients(old, rgb_loss, families, candidate=True)
        _finite_gradients(stats, "old")
        _finite_gradients(stats, "rgb")
        qgrad, = torch.autograd.grad(rgb_loss, parts[4], retain_graph=True)
        qnorm = sw117.grad_norm((qgrad,))
        if not math.isfinite(qnorm) or qnorm <= 0:
            raise FloatingPointError("SW0119 RGB loss has no finite Q gradient")
        sw117.validate_gradient_families(stats, ARM, qnorm)
        shuffled_h = [h.index_select(0, permutation) for h in parts[6]]
        _, shuffled_per, _, _ = rgb_base.batch_reconstruction_loss(parts[4], shuffled_h, patches)
        if not torch.isfinite(shuffled_per).all():
            raise FloatingPointError("nonfinite count-preserving scramble loss")
        real.extend(float(v.detach()) for v in parts[7])
        shuffled.extend(float(v.detach()) for v in shuffled_per)
        records.append({
            "batch": batch_index, "global_ids": ids[start:start+BATCH].tolist(),
            "cache_gamma_max_abs_diff": parity["max_gamma_cache_abs_diff"],
            "cache_live_trace_max_abs_diff": parity["cache_vs_live_trace_max_abs_diff"],
            "cached_baseline_old_loss_abs_diff": parity["cached_baseline_old_loss_abs_diff"],
            "cached_labels_H_exact": parity["cached_baseline_labels_H_exact"],
            "same_input_reference_identity_exact": True,
            "legacy_minus_density_free": float(density_identity.detach()),
            "removed_balance_terms": float(removed.detach()),
            "removed_primary_balance": float(parts[14].detach()),
            "removed_Q_balance": float(parts[15].detach()),
            "removed_balance_gradient_decomposition": grad_decomposition,
            "density_free_old": float(old.detach()), "primary": float(parts[1].detach()),
            "positive_actual_spike_product": float(parts[2].detach()),
            "rgb_loss": float(rgb_loss.detach()),
            "old_gradient_norms": {f: stats[f]["old_gradient_norm"] for f in ("encoder", "graph", "core", "joint")},
            "rgb_gradient_norms": {f: stats[f]["rgb_gradient_norm"] for f in ("encoder", "graph", "core", "joint")},
            "old_rgb_gradient_cosines": {f: stats[f]["gradient_cosine"] for f in ("encoder", "graph", "core", "joint")},
            "rgb_to_Q_gradient_norm": qnorm,
            "group_counts": [int(h.shape[1]) for h in parts[6]],
        })
        core.zero_grad(set_to_none=True); encoder.zero_grad(set_to_none=True)

    scramble_excess = rgb_base.mean_scramble_excess(real, shuffled)
    if not math.isfinite(scramble_excess) or scramble_excess <= 0:
        raise AssertionError("fixed count-preserving H row scramble did not increase RGB loss")
    if any(not torch.equal(initial_core[k], core.state_dict()[k]) for k in initial_core):
        raise AssertionError("preflight modified core before throwaway update")
    if any(not torch.equal(initial_encoder[k], encoder.state_dict()[k]) for k in initial_encoder):
        raise AssertionError("preflight modified encoder before throwaway update")

    # A separate disposable source copy verifies a real finite B16 Adam update.
    (throw_core, throw_encoder, throw_patcher, tmean, tstd, tclip, *_rest) = sw117.load_models(0, device)
    throw_core.train(); throw_core.graph_generator.train(); throw_encoder.train()
    throw_families = sw117.eligible_params(throw_core, throw_encoder)
    optimizer = sw117.optimizer_for(throw_core, throw_encoder)
    torch.manual_seed(117)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(117)
    first_image = sw117.read_rgb(rgb_cache, rows[:BATCH], device)
    first_patches = rgb_base.rgb_patch_means(first_image / 255.0)
    first_gamma = sw117.encode_rgb(throw_encoder, throw_patcher, tmean, tstd, tclip, first_image)
    throw_parts = objective_parts_density_free(throw_core, first_gamma, first_patches, lossfn)
    before_core, before_encoder = sw117.clone_state(throw_core), sw117.clone_state(throw_encoder)
    optimizer.zero_grad(set_to_none=True)
    _, old_by, rgb_by = sw117.collect_joint_gradients(
        throw_parts[0], throw_parts[3], throw_families, candidate=True)
    sw117.apply_family_gradients(throw_families, old_by, rgb_by, LAMBDA)
    norm = torch.nn.utils.clip_grad_norm_(throw_families["all"], sw117.CLIP_NORM)
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise FloatingPointError("invalid SW0119 throwaway B16 gradient")
    optimizer.step()
    changed = sw117.changed_trainable_groups(before_core, throw_core.state_dict(),
                                             before_encoder, throw_encoder.state_dict())
    if not all(changed.values()):
        raise AssertionError(f"SW0119 throwaway update missed trainable family: {changed}")

    record = {
        "status": "passed", "experiment": "SW0119", "seed": 0, "arm": ARM,
        "implementation_fingerprint": implementation_fingerprint(),
        "source_core_sha256": sha(source), "source_manifest_sha256": sha(source_manifest),
        "source_status": source_record.get("status"),
        "source_endpoint_review_sha256": endpoint["review_sha256"],
        "source_endpoint_report_sha256": endpoint["report_sha256"],
        "sw0117_control_manifest_sha256": sha(control_dir / "manifest.json"),
        "sw0117_control_evaluation_sha256": sha(control_dir / "evaluation.json"),
        "encoder_sha256": sha(sw117.ENCODER_PATH), "preprocessing_sha256": sha(sw117.STATS_PATH),
        "gamma_train_sha256": sha(base.GAMMA_TRAIN),
        "gamma_train_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
        "rgb_cache_sha256": sha(sw117.RGB_CACHE),
        "rgb_cache_manifest_sha256": sha(sw117.RGB_CACHE_MANIFEST),
        "training_ids": ids.tolist(),
        "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "matched_shuffle_seed": 117, "updates": UPDATES, "batch_size": BATCH,
        "time_steps": TRAIN_STEPS, "settle": TRAIN_SETTLE,
        "lambda_joint": LAMBDA, "lambda_sha256": sha(sw117.LAMBDA_PATH),
        "ground_truth_used": False, "batches": records,
        "scrambled_minus_real_rgb_loss": float(scramble_excess),
        "throwaway_parameter_groups_changed": changed,
        "throwaway_gradient_norm_preclip": float(norm),
        "throwaway_b16_adam_update": True,
        "removed_objective": "10 * PLV plv_balance + 5 * 10 * actual-Q plv_balance",
    }
    write(output, record)
    return record


def train(output, device="cuda"):
    endpoint = sw117.validate_source_endpoint_review()
    control_dir, control_manifest, _ = validate_registered_control()
    pf_path = ARCHIVE / "preflight_seed0_density_free.json"
    if not pf_path.is_file():
        raise FileNotFoundError(pf_path)
    pf = json.loads(pf_path.read_text(encoding="utf-8"))
    fingerprint = implementation_fingerprint()
    if (pf.get("status") != "passed" or pf.get("experiment") != "SW0119"
            or pf.get("implementation_fingerprint") != fingerprint
            or pf.get("lambda_joint") != LAMBDA
            or pf.get("source_core_sha256") != control_manifest.get("source_core_sha256")
            or pf.get("source_manifest_sha256") != control_manifest.get("source_manifest_sha256")
            or pf.get("sw0117_control_manifest_sha256") != sha(control_dir / "manifest.json")
            or pf.get("sw0117_control_evaluation_sha256") != sha(control_dir / "evaluation.json")
            or pf.get("lambda_sha256") != sha(sw117.LAMBDA_PATH)
            or pf.get("encoder_sha256") != sha(sw117.ENCODER_PATH)
            or pf.get("preprocessing_sha256") != sha(sw117.STATS_PATH)
            or pf.get("gamma_train_sha256") != sha(base.GAMMA_TRAIN)
            or pf.get("gamma_train_manifest_sha256") != sha(base.GAMMA_TRAIN_MANIFEST)
            or pf.get("rgb_cache_sha256") != sha(sw117.RGB_CACHE)
            or pf.get("rgb_cache_manifest_sha256") != sha(sw117.RGB_CACHE_MANIFEST)):
        raise AssertionError("current SW0119 preflight/control/source binding required")
    output = require_vacant(output, "SW0119 training output")
    (core, encoder, patcher, mean, std, clip, source, source_manifest, source_record,
     ids, rows) = sw117.load_models(0, device)
    if sha(source) != pf.get("source_core_sha256") or sha(source_manifest) != pf.get("source_manifest_sha256"):
        raise AssertionError("SW0119 source checkpoint changed after preflight")
    gamma_cache, _ = base.validate_gamma_cache(base.GAMMA_TRAIN, base.GAMMA_TRAIN_MANIFEST)
    rgb_cache, rgb_meta, rgb_cache_sha = sw117.load_rgb_training_cache()
    ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    lambda_record = json.loads(sw117.LAMBDA_PATH.read_text(encoding="utf-8"))
    if (ids_sha != pf.get("training_ids_sha256") or ids.tolist() != pf.get("training_ids")
            or rgb_cache_sha != pf.get("rgb_cache_sha256")
            or sha(base.GAMMA_TRAIN) != pf.get("gamma_train_sha256")
            or sha(base.GAMMA_TRAIN_MANIFEST) != pf.get("gamma_train_manifest_sha256")
            or sha(sw117.RGB_CACHE_MANIFEST) != pf.get("rgb_cache_manifest_sha256")
            or sha(sw117.LAMBDA_PATH) != pf.get("lambda_sha256")
            or lambda_record.get("status") != "passed"
            or lambda_record.get("lambda_joint") != LAMBDA):
        raise AssertionError("SW0119 ordered IDs or immutable cache/coefficient changed")
    lam_record = json.loads(sw117.LAMBDA_PATH.read_text(encoding="utf-8"))
    if lam_record.get("status") != "passed" or lam_record.get("lambda_joint") != LAMBDA:
        raise AssertionError("SW0119 immutable coefficient does not match registered SW0117")
    families = sw117.eligible_params(core, encoder)
    optimizer = sw117.optimizer_for(core, encoder)
    lossfn = sw117.criterion()
    torch.manual_seed(117)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(117)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "training", "experiment": "SW0119", "seed": 0, "arm": ARM,
        "source_core": str(source), "source_core_sha256": sha(source),
        "source_manifest_sha256": sha(source_manifest), "source_status": source_record.get("status"),
        "preflight_sha256": sha(pf_path), "implementation_fingerprint": fingerprint,
        "source_endpoint_review_sha256": endpoint["review_sha256"],
        "source_endpoint_report_sha256": endpoint["report_sha256"],
        "encoder_source_sha256": sha(sw117.ENCODER_PATH), "preprocessing_sha256": sha(sw117.STATS_PATH),
        "gamma_train_sha256": sha(base.GAMMA_TRAIN),
        "gamma_train_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
        "rgb_cache_sha256": rgb_cache_sha, "rgb_cache_manifest_sha256": sha(sw117.RGB_CACHE_MANIFEST),
        "sw0117_control_manifest_sha256": sha(control_dir / "manifest.json"),
        "sw0117_control_evaluation_sha256": sha(control_dir / "evaluation.json"),
        "training_ids": ids.tolist(), "training_ids_sha256": ids_sha,
        "matched_shuffle_seed": 117, "updates": UPDATES, "batch_size": BATCH,
        "core_graph_lr": sw117.CORE_GRAPH_LR, "encoder_lr": sw117.ENCODER_LR,
        "clip_norm": sw117.CLIP_NORM, "time_steps": TRAIN_STEPS, "settle": TRAIN_SETTLE,
        "lambda_joint": LAMBDA, "lambda_artifact_sha256": sha(sw117.LAMBDA_PATH),
        "objective": "SW0117 objective minus only 10*plv_balance(PLV) and 50*plv_balance(actual Q)",
        "ground_truth_used_for_training": False,
    }
    write(output / "manifest.json", manifest)
    history = []
    core.train(); core.graph_generator.train(); encoder.train()
    for update in range(UPDATES):
        start = update * BATCH
        image = sw117.read_rgb(rgb_cache, rows[start:start+BATCH], device)
        patches = rgb_base.rgb_patch_means(image / 255.0)
        gamma = sw117.encode_rgb(encoder, patcher, mean, std, clip, image)
        parts = objective_parts_density_free(core, gamma, patches, lossfn)
        old, primary, positive, rgb_loss, _, _, hard, per_image = parts[:8]
        total = old + LAMBDA * rgb_loss
        if not torch.isfinite(total):
            raise FloatingPointError(f"nonfinite SW0119 total at update {update+1}")
        optimizer.zero_grad(set_to_none=True)
        stats, old_by, rgb_by = sw117.collect_joint_gradients(old, rgb_loss, families, candidate=True)
        sw117.apply_family_gradients(families, old_by, rgb_by, LAMBDA)
        gradnorm = torch.nn.utils.clip_grad_norm_(families["all"], sw117.CLIP_NORM)
        if not torch.isfinite(gradnorm) or float(gradnorm) <= 0:
            raise FloatingPointError(f"invalid SW0119 gradient at update {update+1}")
        optimizer.step()
        history.append({"update": update+1, "total": float(total.detach()),
                        "density_free_old": float(old.detach()), "primary": float(primary.detach()),
                        "positive_actual_spike_product": float(positive.detach()),
                        "rgb": float(rgb_loss.detach()),
                        "removed_balance_contribution": float(parts[13].detach()),
                        "lambda_joint": LAMBDA, "gradient_norm_preclip": float(gradnorm),
                        "gradient_stats": stats, "mean_groups": float(np.mean([h.shape[1] for h in hard])),
                        "mean_rgb_per_image": float(per_image.detach().mean())})
        if update == 0 or (update + 1) % 32 == 0:
            write(output / "progress.json", {"status": "training", "update": update+1,
                                                "total_updates": UPDATES})
    if any(not torch.isfinite(p).all() for p in list(core.parameters()) + list(encoder.parameters())):
        raise FloatingPointError("nonfinite SW0119 final parameter")
    torch.save(core.state_dict(), output / "core.pt")
    torch.save(encoder.state_dict(), output / "encoder.pt")
    torch.save(optimizer.state_dict(), output / "optimizer.pt")
    write(output / "history.json", history)
    manifest.update(status="training_complete", completed=time.time(),
                    core_sha256=sha(output / "core.pt"), encoder_sha256=sha(output / "encoder.pt"),
                    optimizer_sha256=sha(output / "optimizer.pt"), history_sha256=sha(output / "history.json"))
    write(output / "manifest.json", manifest)
    (output / "TRAINING_COMPLETED").write_text("complete\n", encoding="utf-8")


def evaluate(checkpoint, output, device="cuda"):
    checkpoint, output = Path(checkpoint), Path(output)
    manifest_path = checkpoint.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("experiment") != "SW0119" or manifest.get("status") != "training_complete"
            or manifest.get("arm") != ARM or manifest.get("core_sha256") != sha(checkpoint)
            or manifest.get("encoder_sha256") != sha(checkpoint.parent / "encoder.pt")
            or manifest.get("implementation_fingerprint") != implementation_fingerprint()
            or manifest.get("preflight_sha256") != sha(ARCHIVE / "preflight_seed0_density_free.json")):
        raise AssertionError("evaluation checkpoint lacks a completed SW0119 provenance chain")
    original_runner = sw117.RUNNER
    sw117.RUNNER = RUNNER
    try:
        sw117.evaluate(0, ARM, checkpoint, output, device=device)
    finally:
        sw117.RUNNER = original_runner
    report = json.loads(output.read_text(encoding="utf-8"))
    report.update(experiment="SW0119", arm=ARM, seed=0,
                  training_manifest_sha256=sha(manifest_path),
                  checkpoint_sha256=sha(checkpoint), encoder_checkpoint_sha256=sha(checkpoint.parent / "encoder.pt"),
                  evaluation_runner_sha256=sha(RUNNER))
    write(output, report)
    sidecar_path = output.parent / "evaluation_manifest.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar.update(experiment="SW0119", arm=ARM, evaluation_sha256=sha(output),
                   training_manifest_sha256=sha(manifest_path), checkpoint_sha256=sha(checkpoint),
                   evaluation_runner_sha256=sha(RUNNER))
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
        result = validate_registered_control()
        out = require_vacant(args.output, "SW0119 control validation")
        control_dir, manifest, endpoint = result
        payload = {"status": "passed", "experiment": "SW0119", "seed": 0, "arm": ARM,
                   "control_manifest_sha256": sha(control_dir / "manifest.json"),
                   "control_evaluation_sha256": sha(control_dir / "evaluation.json"),
                   "source_core_sha256": manifest["source_core_sha256"],
                   "source_endpoint_report_sha256": endpoint["report_sha256"],
                   "lambda_joint": LAMBDA}
        write(out, payload)
        result = payload
    elif args.stage == "preflight":
        result = preflight(args.output, args.device)
    elif args.stage == "train":
        train(args.output, args.device)
        result = {"status": "training_complete"}
    else:
        if not args.checkpoint:
            parser.error("evaluate requires --checkpoint")
        result = evaluate(args.checkpoint, args.output, args.device)
    summary = {"status": result.get("status"), "stage": args.stage,
               "experiment": "SW0119", "seed": 0, "arm": ARM, "output": args.output}
    if args.stage == "evaluate":
        score = result["sweep"][0]["scored_targets"]["our_hdf5"]
        summary["metrics"] = score.get("metrics", {})
    print(json.dumps(summary, separators=(",", ":"), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
