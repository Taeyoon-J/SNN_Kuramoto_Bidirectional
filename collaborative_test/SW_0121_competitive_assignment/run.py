"""Offline-reviewed SW0121 seed-0 competitive assignment pilot runner."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run.py"
OUT = ROOT / "trained_models/SW0121_competitive_assignment"
ARCHIVE = HERE / "results_archive"
WARMUP_HEAD = ARCHIVE / "head_warmup_seed0.pt"
WARMUP_OPTIMIZER = ARCHIVE / "head_warmup_optimizer_seed0.pt"
WARMUP_FEATURES = ARCHIVE / "head_warmup_features_seed0.pt"
WARMUP_METADATA = ARCHIVE / "head_warmup_seed0.json"
LAMBDA_PATH = ARCHIVE / "lambda_seed0.json"
SEED = 0
ARM = "analytic_candidate"  # Preserve the SW0117 evaluator's registered arm contract.
BATCH = 16
WARMUP_BATCHES = 32
UPDATES = 256
TRAIN_STEPS = 64
SETTLE = 32
CORE_LR = 3e-5
ENCODER_LR = 3e-6
HEAD_LR = 3e-4
CLIP = 1.0
LAMBDA_SCALE = 0.25
LAMBDA_MIN = 1e-4
LAMBDA_MAX = 1e4

sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
from collaborative_test.SW_0117_joint_analytic_rgb import run as prior
from collaborative_test.SW_0117_joint_analytic_rgb import coordinator as prior_coordinator
from collaborative_test.SW_0115_analytic_partition_rgb import run as rgb_base
from collaborative_test.SW_0121_competitive_assignment.assignment import (
    CompetitiveAssignmentHead, analytic_rgb_terms, fit_channel_rms, patch_features,
)
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from snn_kuramoto_bidirectional.training.train_s2net_core import _forward_with_plv


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def implementation_fingerprint():
    files = [RUNNER, HERE / "coordinator.py", HERE / "assignment.py", HERE / "protocol.json",
             ROOT / "collaborative_test/SW_0117_joint_analytic_rgb/run.py",
             ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/run.py",
             ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/loss.py",
             ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
             ROOT / "collaborative_test/SW_0094_aligned_joint_pilot/run.py",
             ROOT / "collaborative_test/SW_0106_spike_partition_rgb/build_rgb_cache.py",
             ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
             ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
             ROOT / "snn_kuramoto_bidirectional/evaluation.py",
             ROOT / "snn_kuramoto_bidirectional/loss_function.py",
             ROOT / "snn_kuramoto_bidirectional/training/train_gamma_initializer.py",
             ROOT / "snn_kuramoto_bidirectional/training/train_s2net_core.py"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files}


def source(seed=SEED, device="cpu"):
    return prior.load_models(seed, device)


def load_training_data():
    ids, rows = prior.base.train_indices(SEED)
    gamma, gamma_meta = prior.base.validate_gamma_cache(
        prior.base.GAMMA_TRAIN, prior.base.GAMMA_TRAIN_MANIFEST)
    rgb, rgb_meta, rgb_digest = prior.load_rgb_training_cache()
    if len(ids) != 4096 or len(rows) != 4096:
        raise AssertionError("SW0121 requires the exact registered 4096-image source order")
    return ids, rows, gamma, gamma_meta, rgb, rgb_meta, rgb_digest


def validate_registered_control():
    try:
        task = {"seed": 0, "arm": "analytic_candidate"}
        if (not prior_coordinator.valid_result({"stage": "train", **task})
                or not prior_coordinator.valid_result({"stage": "evaluate", **task})):
            return False
        folder = prior.OUT / "seed0_analytic_candidate"
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        ids, _ = prior.base.train_indices(0)
        lambda_meta = json.loads(prior.LAMBDA_PATH.read_text(encoding="utf-8"))
        return (manifest.get("source_core_sha256") == prior.base.EXPECTED_SOURCE_SHAS[0]
                and manifest.get("encoder_source_sha256") == prior.base.EXPECTED_ENCODER_SHA256
                and manifest.get("preprocessing_sha256") == prior.base.EXPECTED_PREPROCESSING_SHA256
                and manifest.get("training_ids") == ids.tolist()
                and manifest.get("training_ids_sha256") == _train_ids_hash(ids)
                and manifest.get("matched_shuffle_seed") == 117
                and manifest.get("updates") == 256 and manifest.get("batch_size") == 16
                and manifest.get("time_steps") == 64 and manifest.get("settle") == 32
                and manifest.get("core_graph_lr") == CORE_LR
                and manifest.get("encoder_lr") == ENCODER_LR
                and manifest.get("clip_norm") == CLIP
                and lambda_meta.get("status") == "passed"
                and manifest.get("lambda_joint") == lambda_meta.get("lambda_joint")
                and manifest.get("lambda_artifact_sha256") == prior.sha(prior.LAMBDA_PATH))
    except (OSError, KeyError, TypeError, ValueError, AssertionError):
        return False


def _batch_rgb(rgb_cache, rows, device):
    return prior.read_rgb(rgb_cache, rows, device)


def _forward(core, encoder, patcher, mean, std, clip, images, lossfn):
    gamma = prior.encode_rgb(encoder, patcher, mean, std, clip, images)
    _, spikes, core_out, plv, theta = _forward_with_plv(
        core, gamma, lossfn, SETTLE, "phase", "mean")
    components = core.last_component_spikes
    if tuple(components.shape) != (images.shape[0], 4, 256, TRAIN_STEPS):
        raise AssertionError("SW0121 actual component trace shape changed")
    q = spike_synchrony_affinity(components.mean(dim=1), components=components,
                                 settle=SETTLE, affinity_mode="spike")
    primary, _ = lossfn(plv=plv, theta=theta)
    positive, _ = lossfn(plv=q)
    old = primary + 5.0 * positive
    return {"gamma": gamma, "spikes": spikes, "core_out": core_out, "plv": plv,
            "theta": theta, "components": components, "q": q,
            "primary": primary, "positive": positive, "old": old}


def _family_params(core, encoder):
    core_rows = list(core.named_parameters())
    encoder_rows = list(encoder.named_parameters())
    if (not core_rows or not encoder_rows or any(not p.requires_grad for _, p in core_rows + encoder_rows)):
        raise AssertionError("SW0121 requires trainable source core, graph, and encoder")
    families = {
        "encoder": [p for _, p in encoder_rows],
        "graph": [p for name, p in core_rows if name.startswith("graph_generator.")],
        "core": [p for name, p in core_rows if not name.startswith("graph_generator.")],
    }
    if any(not values for values in families.values()):
        raise AssertionError("one SW0121 trainable parameter family is empty")
    return families


def _norm(grads):
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            raise FloatingPointError("nonfinite SW0121 gradient")
        total += float(grad.detach().double().square().sum())
    return math.sqrt(total)


def _joint_grads(loss, families, retain=True):
    result = {}
    for name in ("encoder", "graph", "core"):
        params = families[name]
        grads = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
        result[name] = (params, grads, _norm(grads))
    return result


def _flatten_grads(families, old, aux, lam):
    all_params, all_grads = [], []
    for name in ("encoder", "graph", "core"):
        params = families[name]
        old_grads, aux_grads = old[name][1], aux[name][1]
        for param, og, ag in zip(params, old_grads, aux_grads):
            if og is None and ag is None:
                continue
            grad = torch.zeros_like(param) if og is None else og.detach().clone()
            if ag is not None:
                grad.add_(ag.detach(), alpha=float(lam))
            param.grad = grad
            all_params.append(param)
            all_grads.append(grad)
    return all_params, all_grads


def _trainable_snapshots(core, encoder, head):
    return {
        "encoder": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
        "core": {k: v.detach().clone() for k, v in core.state_dict().items()},
        "head": {k: v.detach().clone() for k, v in head.state_dict().items()},
    }


def _changed(before, core, encoder, head):
    return {
        "encoder": any(not torch.equal(before["encoder"][k], v) for k, v in encoder.state_dict().items()),
        "graph": any(not torch.equal(before["core"][k], v) for k, v in core.state_dict().items()
                     if k.startswith("graph_generator.")),
        "core": any(not torch.equal(before["core"][k], v) for k, v in core.state_dict().items()
                    if not k.startswith("graph_generator.")),
        "head": any(not torch.equal(before["head"][k], v) for k, v in head.state_dict().items()),
    }


def _criterion():
    return prior.base.criterion()


def _train_ids_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _load_head_artifacts(device):
    head_state = torch.load(WARMUP_HEAD, map_location=device, weights_only=True)
    opt_state = torch.load(WARMUP_OPTIMIZER, map_location=device, weights_only=True)
    meta = json.loads(WARMUP_METADATA.read_text(encoding="utf-8"))
    cache = torch.load(WARMUP_FEATURES, map_location="cpu", weights_only=True)
    if (meta.get("status") != "complete" or meta.get("experiment") != "SW0121"
            or meta.get("head_sha256") != sha(WARMUP_HEAD)
            or meta.get("optimizer_sha256") != sha(WARMUP_OPTIMIZER)
            or meta.get("features_sha256") != sha(WARMUP_FEATURES)
            or meta.get("warmup_ids_sha256") != hashlib.sha256(
                np.asarray(cache.get("image_ids", []), dtype="<i8").tobytes()).hexdigest()
            or tuple(cache["features"].shape) != (512, 256, 128)
            or tuple(cache["rgb_patch_means"].shape) != (512, 256, 3)
            or len(cache["image_ids"]) != 512):
        raise AssertionError("head warmup/RMS artifacts are incomplete or changed")
    return head_state, opt_state, meta, cache


def _collect_warmup_features(device, core, encoder, patcher, mean, std, clip,
                             rows, ids, gamma_cache, rgb_cache):
    reference = prior.base.make_core(device, TRAIN_STEPS)
    source_path, *_ = prior.source_contract(SEED)
    reference.load_state_dict(torch.load(source_path, map_location=device, weights_only=True), strict=True)
    reference._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
    core.train(); core.graph_generator.train(); encoder.train(); reference.train()
    lossfn = _criterion()
    feat_rows, rgb_rows, cache_diffs = [], [], []
    for batch in range(WARMUP_BATCHES):
        lo = batch * BATCH
        batch_rows = rows[lo:lo+BATCH]
        images = _batch_rgb(rgb_cache, batch_rows, device)
        patches = rgb_base.rgb_patch_means(images / 255.0)
        gamma = prior.encode_rgb(encoder, patcher, mean, std, clip, images)
        cached = gamma_cache[torch.as_tensor(batch_rows, dtype=torch.long)].to(device)
        parts, parity = prior.compare_initial_rollout(core, reference, gamma, cached, patches, lossfn)
        if parity["cached_baseline_old_loss_abs_diff"] > 2e-5:
            raise AssertionError("warmup source old-loss parity exceeded registered tolerance")
        components = parts[12]
        feature = patch_features(components[..., -SETTLE:])
        if not torch.isfinite(feature).all():
            raise FloatingPointError("nonfinite warmup spike features")
        feat_rows.append(feature.detach().cpu().float())
        rgb_rows.append(patches.detach().cpu().float())
        cache_diffs.append(parity["max_gamma_cache_abs_diff"])
    features = torch.cat(feat_rows, dim=0)
    rgb_patches = torch.cat(rgb_rows, dim=0)
    channel_rms = fit_channel_rms(features)
    cache = {"features": features, "rgb_patch_means": rgb_patches,
             "image_ids": np.asarray(ids[:WARMUP_BATCHES * BATCH], dtype=np.int64).tolist()}
    return channel_rms, cache, max(cache_diffs)


def _warm_head(channel_rms, feature_cache, device):
    torch.manual_seed(121)
    head = CompetitiveAssignmentHead(channel_rms).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
    records = []
    features = feature_cache["features"]
    rgb = feature_cache["rgb_patch_means"]
    # Exactly the same 32 ordered B16 batches used to fit channel RMS.
    for batch in range(WARMUP_BATCHES):
        lo = batch * BATCH
        trace = features[lo:lo+BATCH].to(device)
        target = rgb[lo:lo+BATCH].to(device)
        p = head(_features_to_spikes(trace))
        # Use the reconstruction term only; affinity is intentionally absent.
        dummy_q = torch.zeros((BATCH, 256, 256), device=device, dtype=p.dtype)
        r, _, detail = analytic_rgb_terms(dummy_q, p, target)
        if not torch.isfinite(r):
            raise FloatingPointError("nonfinite SW0121 head warmup reconstruction")
        opt.zero_grad(set_to_none=True)
        r.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), CLIP)
        if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
            raise FloatingPointError("invalid SW0121 head warmup gradient")
        opt.step()
        records.append({"batch": batch, "R": float(r.detach()),
                        "head_grad_norm_preclip": float(grad_norm)})
    return head, opt, records


def _global_mean_reconstruction(rgb_patches):
    mean = rgb_patches.mean(dim=1, keepdim=True)
    reconstruction = mean.expand_as(rgb_patches)
    variance = rgb_patches.var(dim=1, unbiased=False).mean(dim=-1).clamp_min(1e-6)
    return ((reconstruction - rgb_patches).square().mean(dim=(1, 2)) / variance).mean()


def _features_to_spikes(features):
    if features.ndim != 3 or tuple(features.shape[1:]) != (256, 128):
        raise ValueError("cached warmup feature rows must be [B,256,128]")
    return features.reshape(features.shape[0], 256, 4, 32).permute(0, 2, 1, 3).contiguous()


def _joint_components(core, encoder, patcher, mean, std, clip, image):
    values = _forward(core, encoder, patcher, mean, std, clip, image, _criterion())
    features = patch_features(values["components"][..., -SETTLE:])
    return values, features


def _lambda_calibration(device, seed, head, families, rows, gamma_cache, rgb_cache,
                        core, encoder, patcher, mean, std, clip):
    ratios, records = [], []
    lossfn = _criterion()
    for batch in range(4):
        lo = batch * BATCH
        batch_rows = rows[lo:lo+BATCH]
        images = _batch_rgb(rgb_cache, batch_rows, device)
        patches = rgb_base.rgb_patch_means(images / 255.0)
        parts = _forward(core, encoder, patcher, mean, std, clip, images, lossfn)
        features = patch_features(parts["components"][..., -SETTLE:])
        p = head(_features_to_spikes(features))
        recon, consistency, _ = analytic_rgb_terms(parts["q"], p, patches)
        aux = recon + consistency
        old_grads = _joint_grads(parts["old"], families)
        aux_grads = _joint_grads(aux, families)
        old_norm = math.sqrt(sum(old_grads[k][2] ** 2 for k in ("encoder", "graph", "core")))
        aux_norm = math.sqrt(sum(aux_grads[k][2] ** 2 for k in ("encoder", "graph", "core")))
        if not (math.isfinite(old_norm) and old_norm > 0 and math.isfinite(aux_norm) and aux_norm > 0):
            raise FloatingPointError("SW0121 lambda calibration requires finite nonzero old and R+C gradients")
        head_grads = torch.autograd.grad(recon, list(head.parameters()), retain_graph=True, allow_unused=True)
        q_grads = torch.autograd.grad(consistency, parts["q"], retain_graph=True, allow_unused=True)
        component_grads = torch.autograd.grad(
            recon, parts["components"], retain_graph=True, allow_unused=True)
        head_norm = _norm(head_grads)
        q_norm = _norm(q_grads)
        component_norm = _norm(component_grads)
        per_family = {}
        for family in ("encoder", "graph", "core"):
            og, ag = old_grads[family][2], aux_grads[family][2]
            if not (math.isfinite(og) and og > 0 and math.isfinite(ag) and ag > 0):
                raise FloatingPointError(f"inert SW0121 old/R+C gradient for {family}")
            per_family[family] = {"old_norm": og, "aux_norm": ag}
        assignment_patch_std = float(p.detach().std(dim=1, unbiased=False).mean())
        if not (math.isfinite(head_norm) and head_norm > 0 and math.isfinite(q_norm) and q_norm > 0
                and math.isfinite(component_norm) and component_norm > 0
                and math.isfinite(assignment_patch_std) and assignment_patch_std > 0):
            raise FloatingPointError("SW0121 requires nonzero R-to-head/component and C-to-Q gradients and spatial variation")
        ratios.append(old_norm / aux_norm)
        records.append({"batch": batch, "gamma_cache_rows": [int(x) for x in batch_rows],
                        "old_norm": old_norm, "aux_norm": aux_norm,
                        "family_norms": per_family,
                        "R_to_head_norm": head_norm,
                        "R_to_component_spikes_norm": component_norm,
                        "C_to_Q_norm": q_norm, "assignment_patch_std": assignment_patch_std,
                        "R": float(recon.detach()), "C": float(consistency.detach()),
                        "slot_probability_std": float(p.detach().std())})
    lam = LAMBDA_SCALE * float(np.median(np.asarray(ratios, dtype=np.float64)))
    if not math.isfinite(lam) or not LAMBDA_MIN <= lam <= LAMBDA_MAX:
        raise AssertionError(f"calibrated SW0121 lambda outside preregistered bounds: {lam}")
    return lam, records


def _apply_joint(core_optimizer, head_optimizer, core, encoder, head, families,
                 old_grads, aux_grads, head_loss, lam):
    core_optimizer.zero_grad(set_to_none=True)
    head_optimizer.zero_grad(set_to_none=True)
    params, grads = _flatten_grads(families, old_grads, aux_grads, lam)
    head_grads = torch.autograd.grad(head_loss, list(head.parameters()), allow_unused=True)
    for param, grad in zip(head.parameters(), head_grads):
        param.grad = None if grad is None else grad.detach().clone()
    core_norm = torch.nn.utils.clip_grad_norm_(params, CLIP)
    head_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), CLIP)
    if (not torch.isfinite(core_norm) or float(core_norm) <= 0
            or not torch.isfinite(head_norm) or float(head_norm) <= 0):
        raise FloatingPointError("invalid SW0121 split optimizer gradients")
    core_optimizer.step()
    head_optimizer.step()
    return float(core_norm), float(head_norm)


def preflight(seed, output, device="cuda"):
    if seed != 0:
        raise ValueError("SW0121 is seed0-only until its registered promotion gate passes")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0121 preflight: {output}")
    if any(p.exists() for p in (WARMUP_HEAD, WARMUP_OPTIMIZER, WARMUP_FEATURES,
                                WARMUP_METADATA, LAMBDA_PATH)):
        raise FileExistsError("preserve any existing SW0121 warmup/calibration artifacts")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    endpoint = prior.validate_source_endpoint_review()
    ids, rows, gamma_cache, gamma_meta, rgb_cache, rgb_meta, rgb_digest = load_training_data()
    (core, encoder, patcher, mean, std, clip, source_path, source_manifest, source_record,
     model_ids, model_rows) = source(seed, device)
    if not (np.array_equal(ids, model_ids) and np.array_equal(rows, model_rows)):
        raise AssertionError("source loader/training cache order mismatch")
    source_before = _trainable_snapshots(core, encoder, nn.Identity().to(device))
    rms, warm_cache, gamma_diff = _collect_warmup_features(
        device, core, encoder, patcher, mean, std, clip, rows, ids, gamma_cache, rgb_cache)
    if gamma_diff > 2e-5:
        raise AssertionError("first 512 live gamma rows diverge from registered cache")
    head, head_optimizer, warm_records = _warm_head(rms, warm_cache, device)
    if any(not torch.equal(source_before["core"][k], v) for k, v in core.state_dict().items()):
        raise AssertionError("head warmup modified frozen source core")
    if any(not torch.equal(source_before["encoder"][k], v) for k, v in encoder.state_dict().items()):
        raise AssertionError("head warmup modified frozen source encoder")
    # A fixed count-preserving row permutation must worsen the same reconstruction objective.
    fixed_perm = torch.as_tensor(np.random.default_rng(12101).permutation(256), device=device)
    real_values, shuffled_values, global_values = [], [], []
    for batch in range(WARMUP_BATCHES):
        lo = batch * BATCH
        trace = warm_cache["features"][lo:lo+BATCH].to(device)
        target = warm_cache["rgb_patch_means"][lo:lo+BATCH].to(device)
        p = head(_features_to_spikes(trace))
        p_shuffled = p.index_select(1, fixed_perm)
        dummy_q = torch.zeros((BATCH, 256, 256), device=device)
        r0, _, _ = analytic_rgb_terms(dummy_q, p, target)
        r1, _, _ = analytic_rgb_terms(dummy_q, p_shuffled, target)
        real_values.append(float(r0.detach())); shuffled_values.append(float(r1.detach()))
        global_values.append(float(_global_mean_reconstruction(target).detach()))
        assignment_patch_std = float(p.detach().std(dim=1, unbiased=False).mean())
        if not torch.isfinite(p).all() or not math.isfinite(assignment_patch_std) or assignment_patch_std <= 0:
            raise AssertionError("competitive assignment is constant or nonfinite")
    real_mean, global_mean = float(np.mean(real_values)), float(np.mean(global_values))
    if not real_mean < global_mean:
        raise AssertionError("warmed assignment reconstruction did not beat global-mean baseline")
    scramble_delta = float(np.mean(shuffled_values) - np.mean(real_values))
    if not math.isfinite(scramble_delta) or scramble_delta <= 0:
        raise AssertionError("fixed row scramble did not increase SW0121 reconstruction loss")
    families = _family_params(core, encoder)
    lam, calibration = _lambda_calibration(device, seed, head, families, rows, gamma_cache,
                                           rgb_cache, core, encoder, patcher, mean, std, clip)
    # Test a disposable, fresh source copy and both independent optimizer states.
    (test_core, test_encoder, test_patcher, test_mean, test_std, test_clip, *_rest) = source(seed, device)
    test_head = CompetitiveAssignmentHead(rms).to(device)
    # Use the calibrated/warmed head, not a newly initialized head.
    test_head.load_state_dict(head.state_dict(), strict=True)
    test_core_opt = prior.optimizer_for(test_core, test_encoder)
    test_head_opt = torch.optim.Adam(test_head.parameters(), lr=HEAD_LR)
    test_head_opt.load_state_dict(copy.deepcopy(head_optimizer.state_dict()))
    test_families = _family_params(test_core, test_encoder)
    first_rows = rows[:BATCH]
    images = _batch_rgb(rgb_cache, first_rows, device)
    patches = rgb_base.rgb_patch_means(images / 255.0)
    vals = _forward(test_core, test_encoder, test_patcher, test_mean, test_std, test_clip, images, _criterion())
    features = patch_features(vals["components"][..., -SETTLE:])
    p = test_head(_features_to_spikes(features))
    rloss, closs, _ = analytic_rgb_terms(vals["q"], p, patches)
    aux = rloss + closs
    old_g = _joint_grads(vals["old"], test_families)
    aux_g = _joint_grads(aux, test_families)
    before = _trainable_snapshots(test_core, test_encoder, test_head)
    _apply_joint(test_core_opt, test_head_opt, test_core, test_encoder, test_head,
                 test_families, old_g, aux_g, rloss, lam)
    changed = _changed(before, test_core, test_encoder, test_head)
    if not all(changed.values()):
        raise AssertionError(f"SW0121 two-optimizer throwaway update did not change all groups: {changed}")
    # Persist only after every scientific guard passed; all artifacts are create-once.
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), WARMUP_HEAD)
    torch.save(head_optimizer.state_dict(), WARMUP_OPTIMIZER)
    torch.save(warm_cache, WARMUP_FEATURES)
    warm_meta = {"status": "complete", "experiment": "SW0121", "seed": 0,
                 "source_core_sha256": sha(source_path), "source_manifest_sha256": sha(source_manifest),
                 "encoder_sha256": sha(prior.ENCODER_PATH), "preprocessing_sha256": sha(prior.STATS_PATH),
                 "training_ids_sha256": _train_ids_hash(ids), "training_ids": ids[:512].tolist(),
                  "warmup_ids_sha256": _train_ids_hash(ids[:512]),
                 "warmup_batches": 32, "batch_size": 16, "head_seed": 121,
                 "rms": rms.tolist(), "rms_sha256": hashlib.sha256(rms.numpy().tobytes()).hexdigest(),
                 "head_sha256": sha(WARMUP_HEAD), "optimizer_sha256": sha(WARMUP_OPTIMIZER),
                 "features_sha256": sha(WARMUP_FEATURES), "rgb_cache_sha256": rgb_digest,
                 "implementation_fingerprint": implementation_fingerprint(),
                 "ground_truth_used": False}
    write(WARMUP_METADATA, warm_meta)
    lambda_record = {"status": "passed", "experiment": "SW0121", "seed": 0,
                     "lambda": lam, "rule": "0.25*median(first4 ||grad(old)||/||grad(R+C)||), encoder+graph+core only",
                     "preflight_sha256_pending": True,
                     "implementation_fingerprint": implementation_fingerprint()}
    write(LAMBDA_PATH, lambda_record)
    record = {"status": "passed", "experiment": "SW0121", "seed": 0,
              "source_core_sha256": sha(source_path), "source_manifest_sha256": sha(source_manifest),
              "source_endpoint_review_sha256": endpoint["review_sha256"],
              "encoder_sha256": sha(prior.ENCODER_PATH), "preprocessing_sha256": sha(prior.STATS_PATH),
              "gamma_train_sha256": sha(prior.base.GAMMA_TRAIN),
              "gamma_train_manifest_sha256": sha(prior.base.GAMMA_TRAIN_MANIFEST),
              "rgb_cache_sha256": rgb_digest, "rgb_cache_manifest_sha256": sha(prior.RGB_CACHE_MANIFEST),
              "training_ids": ids.tolist(), "training_ids_sha256": _train_ids_hash(ids),
              "matched_shuffle_seed": 117, "batch_size": BATCH, "time_steps": TRAIN_STEPS,
              "settle": SETTLE, "ground_truth_used": False,
              "implementation_fingerprint": implementation_fingerprint(),
              "lambda": lam, "calibration_batches": calibration,
              "head_warmup": {"batches": warm_records, "features_sha256": sha(WARMUP_FEATURES),
                              "head_sha256": sha(WARMUP_HEAD), "optimizer_sha256": sha(WARMUP_OPTIMIZER)},
              "channel_rms": rms.tolist(), "channel_rms_sha256": warm_meta["rms_sha256"],
              "warmup_R_mean": real_mean, "global_mean_R_baseline": global_mean,
              "fixed_scramble_R_delta": scramble_delta,
              "throwaway_update": {"passed": True, "changed": changed},
              "source_gamma_max_diff_first512": gamma_diff,
              "artifacts": {"head_sha256": sha(WARMUP_HEAD),
                            "optimizer_sha256": sha(WARMUP_OPTIMIZER),
                            "features_sha256": sha(WARMUP_FEATURES),
                            "warmup_metadata_sha256": sha(WARMUP_METADATA)}}
    write(output, record)
    lambda_record["preflight_sha256"] = sha(output)
    lambda_record.pop("preflight_sha256_pending", None)
    write(LAMBDA_PATH, lambda_record)
    return record


def train(seed, output, device="cuda", steps=UPDATES):
    if seed != 0 or steps != UPDATES:
        raise ValueError("SW0121 seed0 only; exact 256-update schedule required")
    prior.validate_source_endpoint_review()
    if not validate_registered_control():
        raise AssertionError("the completed SW0117 analytic-candidate control must validate before SW0121 training")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0121 training output: {output}")
    pf_path = ARCHIVE / "preflight_seed0.json"
    if not pf_path.is_file():
        raise FileNotFoundError(pf_path)
    pf = json.loads(pf_path.read_text(encoding="utf-8"))
    if (pf.get("status") != "passed" or pf.get("experiment") != "SW0121"
            or pf.get("implementation_fingerprint") != implementation_fingerprint()
            or pf.get("lambda") is None):
        raise AssertionError("fresh successful SW0121 preflight with current code is required")
    head_state, head_opt_state, warm_meta, warm_cache = _load_head_artifacts(device)
    if warm_meta.get("implementation_fingerprint") != implementation_fingerprint():
        raise AssertionError("head warmup implementation fingerprint changed")
    lam_record = json.loads(LAMBDA_PATH.read_text(encoding="utf-8"))
    lam = float(lam_record.get("lambda", float("nan")))
    if (lam != float(pf["lambda"]) or lam_record.get("preflight_sha256") != sha(pf_path)
            or not LAMBDA_PATH.is_file()):
        raise AssertionError("immutable SW0121 lambda is not bound to successful preflight")
    ids, rows, gamma_cache, _, rgb_cache, _, rgb_digest = load_training_data()
    (core, encoder, patcher, mean, std, clip, source_path, source_manifest,
     source_record, source_ids, source_rows) = source(seed, device)
    expected = {"source_core_sha256": sha(source_path),
                "source_manifest_sha256": sha(source_manifest),
                "encoder_sha256": sha(prior.ENCODER_PATH),
                "preprocessing_sha256": sha(prior.STATS_PATH),
                "gamma_train_sha256": sha(prior.base.GAMMA_TRAIN),
                "gamma_train_manifest_sha256": sha(prior.base.GAMMA_TRAIN_MANIFEST),
                "rgb_cache_sha256": rgb_digest,
                "rgb_cache_manifest_sha256": sha(prior.RGB_CACHE_MANIFEST),
                "training_ids_sha256": _train_ids_hash(ids),
                "implementation_fingerprint": implementation_fingerprint()}
    if any(pf.get(k) != v for k, v in expected.items()):
        raise AssertionError("SW0121 source/cache/order/code changed since preflight")
    if any(warm_meta.get(k) != expected[k] for k in (
            "source_core_sha256", "source_manifest_sha256", "encoder_sha256",
            "preprocessing_sha256", "training_ids_sha256", "rgb_cache_sha256")):
        raise AssertionError("head warmup source, encoder, cache, or ordered IDs changed")
    if warm_meta.get("warmup_ids_sha256") != _train_ids_hash(ids[:512]):
        raise AssertionError("head warmup cache differs from the first 512 ordered TRAIN IDs")
    expected_warm_artifacts = {
        "head_sha256": sha(WARMUP_HEAD), "optimizer_sha256": sha(WARMUP_OPTIMIZER),
        "features_sha256": sha(WARMUP_FEATURES), "warmup_metadata_sha256": sha(WARMUP_METADATA)}
    if pf.get("artifacts") != expected_warm_artifacts:
        raise AssertionError("head warmup/RMS artifacts no longer match successful preflight")
    if (lam_record.get("status") != "passed"
            or lam_record.get("implementation_fingerprint") != implementation_fingerprint()
            or lam_record.get("preflight_sha256") != sha(pf_path)
            or lam_record.get("lambda") != float(pf.get("lambda", float("nan")))):
        raise AssertionError("immutable calibrated lambda artifact is not bound to current preflight")
    head = CompetitiveAssignmentHead(torch.tensor(warm_meta["rms"], dtype=torch.float32)).to(device)
    head.load_state_dict(head_state, strict=True)
    core_opt = prior.optimizer_for(core, encoder)
    head_opt = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
    head_opt.load_state_dict(head_opt_state)
    families = _family_params(core, encoder)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "training", "experiment": "SW0121", "seed": seed,
                "arm": ARM, "source_core_sha256": sha(source_path),
                "source_manifest_sha256": sha(source_manifest), "preflight_sha256": sha(pf_path),
                "implementation_fingerprint": implementation_fingerprint(),
                "source_endpoint_review_sha256": sha(prior.SOURCE_ENDPOINT_REVIEW),
                "encoder_source_sha256": sha(prior.ENCODER_PATH),
                "preprocessing_sha256": sha(prior.STATS_PATH),
                "gamma_train_sha256": sha(prior.base.GAMMA_TRAIN),
                "gamma_train_manifest_sha256": sha(prior.base.GAMMA_TRAIN_MANIFEST),
                "rgb_cache_sha256": rgb_digest,
                "rgb_cache_manifest_sha256": sha(prior.RGB_CACHE_MANIFEST),
                "training_ids": ids.tolist(), "training_ids_sha256": _train_ids_hash(ids),
                "matched_shuffle_seed": 117, "updates": UPDATES, "batch_size": BATCH,
                "time_steps": TRAIN_STEPS, "settle": SETTLE, "core_graph_lr": CORE_LR,
                "encoder_lr": ENCODER_LR, "head_lr": HEAD_LR, "clip_norm": CLIP,
                "lambda": lam, "lambda_sha256": sha(LAMBDA_PATH),
                "head_warmup_sha256": sha(WARMUP_HEAD),
                "head_warmup_optimizer_sha256": sha(WARMUP_OPTIMIZER),
                "head_warmup_features_sha256": sha(WARMUP_FEATURES),
                "ground_truth_used_for_training": False,
                "objective": "phase_primary+5*positive_actual_Q+lambda*(R+C); head Adam uses unweighted R only"}
    write(output / "manifest.json", manifest)
    torch.manual_seed(117)
    torch.cuda.manual_seed_all(117) if str(device).startswith("cuda") else None
    core.train(); core.graph_generator.train(); encoder.train(); head.train()
    history = []
    for update in range(UPDATES):
        lo = update * BATCH
        batch_rows = rows[lo:lo+BATCH]
        images = _batch_rgb(rgb_cache, batch_rows, device)
        patches = rgb_base.rgb_patch_means(images / 255.0)
        vals = _forward(core, encoder, patcher, mean, std, clip, images, _criterion())
        p = head(_features_to_spikes(patch_features(vals["components"][..., -SETTLE:])))
        recon, consistency, _ = analytic_rgb_terms(vals["q"], p, patches)
        aux = recon + consistency
        total = vals["old"] + lam * aux
        if not torch.isfinite(total):
            raise FloatingPointError(f"nonfinite SW0121 objective at update {update+1}")
        old_g = _joint_grads(vals["old"], families)
        aux_g = _joint_grads(aux, families)
        core_norm, head_norm = _apply_joint(core_opt, head_opt, core, encoder, head,
                                            families, old_g, aux_g, recon, lam)
        history.append({"update": update + 1, "old": float(vals["old"].detach()),
                        "primary": float(vals["primary"].detach()),
                        "positive_actual_Q": float(vals["positive"].detach()),
                        "R": float(recon.detach()), "C": float(consistency.detach()),
                        "lambda": lam, "total": float(total.detach()),
                        "core_gradient_norm_preclip": core_norm,
                        "head_gradient_norm_preclip": head_norm,
                        "slot_probability_std": float(p.detach().std()),
                        "slot_occupancy_mean": p.detach().mean(dim=(0, 1)).cpu().tolist()})
        if update == 0 or (update + 1) % 32 == 0:
            write(output / "progress.json", {"status": "training", "experiment": "SW0121",
                                               "seed": seed, "update": update + 1,
                                               "total_updates": UPDATES})
    if any(not torch.isfinite(x).all() for x in list(core.parameters()) + list(encoder.parameters())
           + list(head.parameters())):
        raise FloatingPointError("nonfinite final SW0121 trained parameter")
    torch.save(core.state_dict(), output / "core.pt")
    torch.save(encoder.state_dict(), output / "encoder.pt")
    torch.save(head.state_dict(), output / "head.pt")
    torch.save(core_opt.state_dict(), output / "core_optimizer.pt")
    torch.save(head_opt.state_dict(), output / "head_optimizer.pt")
    torch.save({"core": core_opt.state_dict(), "head": head_opt.state_dict()},
               output / "optimizer.pt")
    write(output / "history.json", history)
    manifest.update(status="training_complete", completed=time.time(),
                    core_sha256=sha(output / "core.pt"), encoder_sha256=sha(output / "encoder.pt"),
                    head_sha256=sha(output / "head.pt"),
                    core_optimizer_sha256=sha(output / "core_optimizer.pt"),
                    head_optimizer_sha256=sha(output / "head_optimizer.pt"),
                    optimizer_sha256=sha(output / "optimizer.pt"),
                    history_sha256=sha(output / "history.json"))
    write(output / "manifest.json", manifest)
    (output / "TRAINING_COMPLETED").write_text("complete\n", encoding="utf-8")


def evaluate(seed, checkpoint, output, device="cuda"):
    if seed != 0:
        raise ValueError("SW0121 evaluation is seed0-only pending promotion")
    checkpoint, output = Path(checkpoint), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "training_complete" or manifest.get("experiment") != "SW0121"
            or manifest.get("implementation_fingerprint") != implementation_fingerprint()
            or manifest.get("core_sha256") != sha(checkpoint)
            or manifest.get("encoder_sha256") != sha(checkpoint.parent / "encoder.pt")
            or manifest.get("head_sha256") != sha(checkpoint.parent / "head.pt")
            or manifest.get("history_sha256") != sha(checkpoint.parent / "history.json")
            or manifest.get("core_optimizer_sha256") != sha(checkpoint.parent / "core_optimizer.pt")
            or manifest.get("head_optimizer_sha256") != sha(checkpoint.parent / "head_optimizer.pt")):
        raise AssertionError("SW0121 evaluation source artifacts failed manifest binding")
    if output.exists() or (output.parent / "evaluation_manifest.json").exists():
        raise FileExistsError("preserve existing SW0121 evaluation outputs")
    prior.evaluate(seed, "analytic_candidate", checkpoint, output, device=device)
    report = json.loads(output.read_text(encoding="utf-8"))
    report.update(experiment="SW0121", arm=ARM, seed=seed,
                  training_manifest_sha256=sha(manifest_path),
                  checkpoint_sha256=sha(checkpoint),
                  encoder_checkpoint_sha256=sha(checkpoint.parent / "encoder.pt"),
                  head_checkpoint_sha256=sha(checkpoint.parent / "head.pt"),
                  assignment_head_used_for_prediction=False,
                  ground_truth_used_for_prediction=False,
                  evaluation_runner_sha256=sha(RUNNER))
    write(output, report)
    sidecar = output.parent / "evaluation_manifest.json"
    write(sidecar, {"experiment": "SW0121", "seed": seed, "arm": ARM,
                    "evaluation_file": output.name, "evaluation_sha256": sha(output),
                    "training_manifest_sha256": sha(manifest_path),
                    "checkpoint_sha256": sha(checkpoint),
                    "encoder_checkpoint_sha256": sha(checkpoint.parent / "encoder.pt"),
                    "head_checkpoint_sha256": sha(checkpoint.parent / "head.pt"),
                    "assignment_head_used_for_prediction": False,
                    "evaluation_runner_sha256": sha(RUNNER),
                    "shared_evaluator_sha256": report.get("shared_evaluator_sha256"),
                    "evaluation_contract": {"ids": [1320, 1639], "images": 320,
                        "batch_size": 8, "time_steps": 1024, "settle": 512,
                        "membrane_vth": 0.06, "readout_threshold": 0.50,
                        "min_group_size": 2, "background": "largest_component",
                        "ground_truth_used_for_prediction": False}})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("preflight", "train", "evaluate"):
        sp = sub.add_parser(stage)
        sp.add_argument("--seed", type=int, choices=(0,), required=True)
        sp.add_argument("--device", default="cuda")
        if stage in ("preflight", "train"):
            sp.add_argument("--output", type=Path, required=True)
        else:
            sp.add_argument("--checkpoint", type=Path, required=True)
            sp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "preflight":
        record = preflight(args.seed, args.output, args.device)
        result = {"status": record["status"], "experiment": "SW0121",
                  "stage": args.stage, "seed": args.seed, "lambda": record["lambda"]}
    elif args.stage == "train":
        train(args.seed, args.output, args.device)
        result = {"status": "complete", "experiment": "SW0121", "stage": "train"}
    else:
        evaluate(args.seed, args.checkpoint, args.output, args.device)
        result = {"status": "complete", "experiment": "SW0121", "stage": "evaluate"}
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
