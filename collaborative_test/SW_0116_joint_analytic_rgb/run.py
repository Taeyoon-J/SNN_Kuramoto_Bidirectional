"""Matched joint encoder/graph/core pilot with analytic partition RGB credit."""
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
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0115_analytic_partition_rgb import run as rgb_base
from collaborative_test.SW_0115_analytic_partition_rgb.loss import (
    batch_reconstruction_loss, production_partition,
)
from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_input_encoder
from snn_kuramoto_bidirectional.training.train_s2net_core import _forward_with_plv

SEEDS = (0,)
ARMS = ("control", "analytic_candidate")
BATCH = 16
UPDATES = 256
TRAIN_STEPS = 64
TRAIN_SETTLE = 32
CORE_GRAPH_LR = 3e-5
ENCODER_LR = 3e-6
CLIP_NORM = 1.0
OUT = ROOT / "trained_models/SW0116_joint_analytic_rgb"
ARCHIVE = HERE / "results_archive"
LAMBDA_PATH = ARCHIVE / "lambda_joint_seed0.json"
RGB_CACHE = rgb_base.RGB_CACHE
RGB_CACHE_MANIFEST = rgb_base.RGB_CACHE_MANIFEST
VAL_RGB_CACHE = ROOT / "data/SW_0106_spike_partition_rgb/validation_rgb_uint8.npy"
VAL_RGB_MANIFEST = ROOT / "data/SW_0106_spike_partition_rgb/validation_rgb_uint8.npy.complete.json"
ENCODER_PATH = base.ASSETS / "input_encoder/input_layer_encoder.pt"
STATS_PATH = base.ASSETS / "feature_preprocessing.pt"


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
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def implementation_fingerprint():
    files = [HERE / "run.py", HERE / "coordinator.py", HERE / "protocol.json",
             ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
             ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/run.py",
             ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/loss.py",
             ROOT / "collaborative_test/SW_0106_spike_partition_rgb/build_rgb_cache.py",
             ROOT / "collaborative_test/SW_0094_aligned_joint_pilot/run.py",
             ROOT / "snn_kuramoto_bidirectional/gamma_initializer.py",
             ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
             ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
             ROOT / "snn_kuramoto_bidirectional/evaluation.py",
             ROOT / "snn_kuramoto_bidirectional/loss_function.py",
             ROOT / "snn_kuramoto_bidirectional/training/train_gamma_initializer.py",
             ROOT / "snn_kuramoto_bidirectional/training/train_s2net_core.py"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files}


def source_contract(seed):
    checkpoint, manifest_path, manifest = base.source_paths(seed)
    ids, rows = base.train_indices(seed)
    rgb_base.validate_source_manifest(seed, manifest, ids)
    if sha(checkpoint) != base.EXPECTED_SOURCE_SHAS[seed]:
        raise AssertionError("SW0097 source core SHA differs from registered immutable checkpoint")
    return checkpoint, manifest_path, manifest, ids, rows


def load_rgb_training_cache():
    cache, manifest, cache_sha = rgb_base.validate_rgb_cache()
    return cache, manifest, cache_sha


def validate_rgb_validation_cache():
    if not VAL_RGB_CACHE.is_file() or not VAL_RGB_MANIFEST.is_file():
        raise FileNotFoundError("byte-verified SW0106 validation RGB cache is required")
    meta = json.loads(VAL_RGB_MANIFEST.read_text())
    source = base.DATASET
    stat = source.stat()
    expected_ids = np.arange(1320, 1640, dtype=np.int64)
    expected_ids_sha = hashlib.sha256(np.asarray(expected_ids, dtype="<i8").tobytes()).hexdigest()
    if (meta.get("status") != "complete" or meta.get("kind") != "validation"
            or meta.get("cache_shape") != [320, 128, 128, 3]
            or meta.get("cache_dtype") != "uint8"
            or meta.get("ids_mapping_sha256") != expected_ids_sha
            or meta.get("source_path") != str(source)
            or meta.get("source_size_bytes") != stat.st_size
            or meta.get("source_mtime_ns") != stat.st_mtime_ns
            or meta.get("source_image_shape") != [100000, 128, 128, 3]
            or meta.get("source_image_dtype") != "uint8"):
        raise AssertionError("validation RGB cache/source contract mismatch")
    cache = np.load(VAL_RGB_CACHE, mmap_mode="r")
    rgb_base.validate_npy_storage(VAL_RGB_CACHE, cache, (320, 128, 128, 3), np.uint8)
    cache_sha = rgb_base.audit_cache_blocks(VAL_RGB_CACHE, cache, meta.get("blocks", []), expected_ids)
    return cache, meta, cache_sha


def load_models(seed, device):
    source, source_manifest, manifest, ids, rows = source_contract(seed)
    if (sha(ENCODER_PATH) != base.EXPECTED_ENCODER_SHA256
            or sha(STATS_PATH) != base.EXPECTED_PREPROCESSING_SHA256):
        raise AssertionError("registered source encoder/preprocessing asset SHA mismatch")
    core = base.make_core(device, TRAIN_STEPS)
    core.load_state_dict(torch.load(source, map_location=device, weights_only=True), strict=True)
    # SW0110 loader freezes its graph for the XY test. Joint SW0116 explicitly
    # re-enables every source core/graph parameter for both matched arms.
    core.requires_grad_(True)
    core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
    if core.kuramoto.spike_pulse_gain is not None or core.graph_generator.uses_feedback:
        raise AssertionError("registered source has unsupported pulse/feedback dynamics")
    encoder = load_input_encoder(str(ENCODER_PATH), num_kernels=8, kernel_size=3,
                                 channels=3, device=device)
    encoder.requires_grad_(True)
    patcher = FeaturePatchGammaInitializer(grid_size=16).to(device)
    stats = torch.load(STATS_PATH, map_location="cpu", weights_only=True)
    mean, std, clip = preprocessing_tensors(stats, device)
    return core, encoder, patcher, mean, std, clip, source, source_manifest, manifest, ids, rows


def preprocessing_tensors(stats, device):
    if stats.get("mode") != "standardize":
        raise AssertionError("registered encoder preprocessing statistics are invalid")
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(stats["std"], dtype=torch.float32, device=device)
    if (not torch.isfinite(mean).all() or not torch.isfinite(std).all()
            or not (std > 0).all()):
        raise AssertionError("registered encoder mean/std contain invalid values")
    try:
        broadcast_shape = torch.broadcast_shapes((1, 8, 126, 126), tuple(mean.shape), tuple(std.shape))
    except RuntimeError as exc:
        raise AssertionError("registered preprocessing mean/std do not broadcast to encoder maps") from exc
    if broadcast_shape != (1, 8, 126, 126):
        raise AssertionError("registered preprocessing stats broadcast beyond expected encoder map shape")
    clip = float(stats.get("clip", 3.0))
    if not math.isfinite(clip) or clip <= 0:
        raise AssertionError("registered preprocessing clip is invalid")
    return mean, std, clip


def read_rgb(cache, rows, device):
    images = np.asarray(cache[np.asarray(rows, dtype=np.int64)]).copy()
    if images.shape != (len(rows), 128, 128, 3) or images.dtype != np.uint8:
        raise AssertionError("SW0106 native RGB batch has unexpected shape/dtype")
    return torch.from_numpy(images).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)


def encode_rgb(encoder, patcher, mean, std, clip, images):
    if images.dtype != torch.float32 or images.ndim != 4 or tuple(images.shape[1:]) != (3, 128, 128):
        raise ValueError("encoder input must be native RGB converted once to float [B,3,128,128]")
    features = encoder(images / 255.0)
    gamma = patcher(((features - mean) / std).clamp(-clip, clip))
    if gamma.shape != (images.shape[0], 8, 256) or not torch.isfinite(gamma).all():
        raise AssertionError("trained encoder gamma has invalid shape or values")
    return gamma


def criterion():
    return base.criterion()


def eligible_params(core, encoder):
    core_named = list(core.named_parameters())
    enc_named = list(encoder.named_parameters())
    if (not core_named or not enc_named
            or any(not p.requires_grad for _, p in core_named + enc_named)):
        raise AssertionError("joint pilot must train all core/graph and encoder parameters")
    graph = [p for name, p in core_named if name.startswith("graph_generator.")]
    nongraph = [p for name, p in core_named if not name.startswith("graph_generator.")]
    if not graph or not nongraph:
        raise AssertionError("joint core parameter families are empty")
    return {"encoder": [p for _, p in enc_named], "graph": graph, "core": nongraph,
            "all": [p for _, p in core_named] + [p for _, p in enc_named]}


def optimizer_for(core, encoder):
    return torch.optim.Adam([
        {"params": list(core.parameters()), "lr": CORE_GRAPH_LR},
        {"params": list(encoder.parameters()), "lr": ENCODER_LR},
    ])


def grad_norm(grads):
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            raise FloatingPointError("nonfinite SW0116 gradient")
        total += float(grad.detach().double().square().sum())
    return math.sqrt(total)


def gradients_by_family(loss, families, retain_graph):
    output = {}
    for family in ("encoder", "graph", "core"):
        params = families[family]
        grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
        norm = grad_norm(grads)
        output[family] = (params, grads, norm)
    return output


def cosine_between(left, right):
    lnorm, rnorm = grad_norm(left), grad_norm(right)
    if lnorm == 0 or rnorm == 0:
        return None
    dot = sum(float((a.detach().double() * b.detach().double()).sum())
              for a, b in zip(left, right) if a is not None and b is not None)
    return float(dot / (lnorm * rnorm))


def objective_parts(core, gamma, rgb_patches, lossfn):
    _, spikes, core_out, plv, theta = _forward_with_plv(
        core, gamma, lossfn, TRAIN_SETTLE, "phase", "mean")
    components = core.last_component_spikes
    if tuple(components.shape[1:]) != (4, 256, TRAIN_STEPS):
        raise AssertionError("SW0116 source component spike trace shape changed")
    q = spike_synchrony_affinity(components.mean(dim=1), components=components,
                                 settle=TRAIN_SETTLE, affinity_mode="spike")
    if q.shape != (gamma.shape[0], 256, 256) or not torch.isfinite(q).all():
        raise AssertionError("SW0116 Q must be the actual finite four-component product")
    primary, _ = lossfn(plv=plv, theta=theta)
    positive, _ = lossfn(plv=q)
    old = primary + 5.0 * positive
    labels, hard = production_partition(spikes, components, settle=TRAIN_SETTLE)
    rgb_loss, per_image, _, details = batch_reconstruction_loss(q, hard, rgb_patches)
    return old, primary, positive, rgb_loss, q, labels, hard, per_image, details, spikes, core_out, theta, components


def apply_grads(params, old_grads, rgb_grads=None, lam=0.0):
    for p, old_g, rgb_g in zip(params, old_grads, rgb_grads or [None] * len(params)):
        if old_g is None and rgb_g is None:
            p.grad = None
        else:
            g = torch.zeros_like(p) if old_g is None else old_g.detach().clone()
            if rgb_g is not None:
                g.add_(rgb_g.detach(), alpha=float(lam))
            p.grad = g


def collect_joint_gradients(old, rgb_loss, families, candidate):
    stats, old_by_family, rgb_by_family = {}, {}, {}
    all_old, all_rgb = [], []
    for family in ("encoder", "graph", "core"):
        params = families[family]
        old_grads = torch.autograd.grad(old, params, retain_graph=True, allow_unused=True)
        if candidate:
            rgb_grads = torch.autograd.grad(rgb_loss, params, retain_graph=True, allow_unused=True)
        else:
            rgb_grads = tuple(None for _ in params)
        old_by_family[family] = old_grads
        rgb_by_family[family] = rgb_grads
        old_norm, rgb_norm = grad_norm(old_grads), grad_norm(rgb_grads)
        stats[family] = {"old_gradient_norm": old_norm, "rgb_gradient_norm": rgb_norm,
                         "gradient_cosine": cosine_between(old_grads, rgb_grads) if candidate else None}
        all_old.extend(old_grads); all_rgb.extend(rgb_grads)
    stats["joint"] = {"old_gradient_norm": grad_norm(all_old),
                       "rgb_gradient_norm": grad_norm(all_rgb),
                       "gradient_cosine": cosine_between(all_old, all_rgb) if candidate else None}
    return stats, old_by_family, rgb_by_family


def apply_family_gradients(families, old_by_family, rgb_by_family, lam):
    for family in ("encoder", "graph", "core"):
        apply_grads(families[family], old_by_family[family], rgb_by_family[family], lam)


def clone_state(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def changed_trainable_groups(core_before, core_after, enc_before, enc_after):
    changed = {
        "graph": any(not torch.equal(core_before[key], core_after[key])
                     for key in core_before if key.startswith("graph_generator.")),
        "core": any(not torch.equal(core_before[key], core_after[key])
                    for key in core_before if not key.startswith("graph_generator.")),
        "encoder": any(not torch.equal(enc_before[key], enc_after[key]) for key in enc_before),
    }
    return changed


def validate_gradient_families(stats, arm, q_gradient_norm=0.0):
    for family in ("encoder", "graph", "core", "joint"):
        old_norm = float(stats[family]["old_gradient_norm"])
        if not math.isfinite(old_norm) or old_norm <= 0:
            raise AssertionError(f"old objective gradient is inert/nonfinite for {family}")
        if arm == "analytic_candidate":
            rgb_norm = float(stats[family]["rgb_gradient_norm"])
            if not math.isfinite(rgb_norm) or rgb_norm <= 0:
                raise AssertionError(f"RGB objective gradient is inert/nonfinite for {family}")
    if arm == "analytic_candidate" and (not math.isfinite(float(q_gradient_norm))
                                          or float(q_gradient_norm) <= 0):
        raise AssertionError("RGB objective gradient to actual Q is inert/nonfinite")


def compare_initial_rollout(core, reference, gamma, cached_gamma, rgb, lossfn):
    max_gamma = float((gamma.detach() - cached_gamma).abs().max())
    if not math.isfinite(max_gamma) or max_gamma > 2e-5:
        raise AssertionError(f"regenerated source gamma differs by {max_gamma}, above 2e-5")
    candidate = objective_parts(core, gamma, rgb, lossfn)
    with torch.no_grad():
        baseline = objective_parts(reference, cached_gamma, rgb, lossfn)
    exact = (torch.equal(candidate[5], baseline[5])
             and all(torch.equal(a, b) for a, b in zip(candidate[6], baseline[6])))
    if not exact:
        raise AssertionError("joint initial encoder changed production hard labels/H")
    max_trace = max(float((candidate[i].detach() - baseline[i]).abs().max())
                    for i in (9, 10, 11, 12))
    old_delta = abs(float(candidate[0].detach() - baseline[0]))
    if not math.isfinite(max_trace) or max_trace > 2e-5 or old_delta > 2e-5:
        raise AssertionError("joint source rollout/old objective exceeds registered gamma parity tolerance")
    return candidate, {"max_gamma_abs_diff": max_gamma, "max_trace_abs_diff": max_trace,
                       "old_loss_abs_diff": old_delta, "hard_labels_exact": True, "hard_H_exact": True}


def preflight(seed, arm, output, device="cuda"):
    if seed != 0 or arm not in ARMS:
        raise ValueError("SW0116 initial pilot is seed0 only")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing preflight; refusing overwrite: {output}")
    if arm == "analytic_candidate" and LAMBDA_PATH.exists():
        raise FileExistsError(f"preserve existing lambda artifact: {LAMBDA_PATH}")
    fingerprint = implementation_fingerprint()
    (core, encoder, patcher, mean, std, clip, source, source_manifest, source_record,
     ids, rows) = load_models(seed, device)
    cached_gamma, gamma_manifest = base.validate_gamma_cache(base.GAMMA_TRAIN, base.GAMMA_TRAIN_MANIFEST)
    rgb_cache, rgb_manifest, rgb_cache_sha = load_rgb_training_cache()
    families = eligible_params(core, encoder)
    initial_core = clone_state(core)
    initial_encoder = clone_state(encoder)
    reference = base.make_core(device, TRAIN_STEPS)
    reference.load_state_dict(torch.load(source, map_location=device, weights_only=True), strict=True)
    reference._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
    reference.eval()
    core.train(); core.graph_generator.train(); encoder.train(); reference.train()
    lossfn = criterion()
    ratios, records, real_values, scrambled_values = [], [], [], []
    fixed_perm = torch.as_tensor(np.random.default_rng(11501).permutation(256),
                                 device=device, dtype=torch.long)
    for batch_index in range(4):
        start = batch_index * BATCH
        batch_rows = rows[start:start + BATCH]
        row_tensor = torch.as_tensor(batch_rows, dtype=torch.long)
        cached = cached_gamma[row_tensor].to(device)
        image = read_rgb(rgb_cache, batch_rows, device)
        patches = rgb_base.rgb_patch_means(image / 255.0)
        gamma = encode_rgb(encoder, patcher, mean, std, clip, image)
        parts, parity = compare_initial_rollout(core, reference, gamma, cached, patches, lossfn)
        old, primary, positive, rloss, q, labels, hard, per_image, _, *_ = parts
        candidate_stats, old_by_family, rgb_by_family = collect_joint_gradients(
            old, rloss, families, candidate=(arm == "analytic_candidate"))
        q_grad = (torch.autograd.grad(rloss, q, retain_graph=True, allow_unused=True)[0]
                  if arm == "analytic_candidate" else None)
        q_gradient_norm = 0.0 if q_grad is None else grad_norm([q_grad])
        old_stats = {f: {"old_gradient_norm": candidate_stats[f]["old_gradient_norm"]}
                     for f in ("encoder", "graph", "core", "joint")}
        validate_gradient_families(candidate_stats, arm, q_gradient_norm)
        if arm == "analytic_candidate":
            ratios.append(old_stats["joint"]["old_gradient_norm"] /
                          candidate_stats["joint"]["rgb_gradient_norm"])
        scrambled_h = [h.index_select(0, fixed_perm) for h in hard]
        _, scrambled_per, _, _ = batch_reconstruction_loss(q, scrambled_h, patches)
        if not torch.isfinite(rloss) or not torch.isfinite(scrambled_per).all():
            raise FloatingPointError("nonfinite joint RGB preflight loss")
        real_values.extend(float(x.detach()) for x in per_image)
        scrambled_values.extend(float(x.detach()) for x in scrambled_per)
        records.append({"batch": batch_index, "global_ids": ids[start:start+BATCH].tolist(),
                        "gamma_rollout_parity": parity,
                        "old_loss": float(old.detach()), "primary": float(primary.detach()),
                        "positive_actual_spike_product": float(positive.detach()),
                        "rgb_loss": float(rloss.detach()),
                        "old_gradient_norms": {f: old_stats[f]["old_gradient_norm"] for f in ("encoder", "graph", "core", "joint")},
                        "rgb_gradient_norms": {f: candidate_stats[f]["rgb_gradient_norm"] for f in ("encoder", "graph", "core", "joint")},
                        "rgb_to_q_gradient_norm": q_gradient_norm,
                        "old_rgb_gradient_cosines": {f: candidate_stats[f]["gradient_cosine"] for f in ("encoder", "graph", "core", "joint")},
                        "group_counts": [int(h.shape[1]) for h in hard]})
        core.zero_grad(set_to_none=True); encoder.zero_grad(set_to_none=True)
    scramble_excess = rgb_base.mean_scramble_excess(real_values, scrambled_values)
    if not math.isfinite(scramble_excess) or scramble_excess <= 0:
        raise AssertionError("fixed count-preserving partition scramble did not increase RGB loss")
    if any(not torch.equal(initial_core[k], core.state_dict()[k]) for k in initial_core):
        raise AssertionError("preflight modified source core before the throwaway optimizer test")
    if any(not torch.equal(initial_encoder[k], encoder.state_dict()[k]) for k in initial_encoder):
        raise AssertionError("preflight modified source encoder before the throwaway optimizer test")
    lam = (0.25 * float(np.median(np.asarray(ratios, dtype=np.float64)))
           if arm == "analytic_candidate" else 0.0)
    if arm == "analytic_candidate" and (not math.isfinite(lam) or lam <= 0):
        raise AssertionError("invalid seed0 shared joint RGB coefficient")

    # The disposable update uses fresh copies and independent before/after
    # snapshots, so the guard detects actual updates rather than self-equality.
    update_core, update_encoder, update_patcher, update_mean, update_std, update_clip, *_ = load_models(seed, device)
    update_core.train(); update_core.graph_generator.train(); update_encoder.train()
    update_families = eligible_params(update_core, update_encoder)
    optimizer = optimizer_for(update_core, update_encoder)
    first_rows = rows[:BATCH]
    first_rgb = read_rgb(rgb_cache, first_rows, device)
    first_patches = rgb_base.rgb_patch_means(first_rgb / 255.0)
    first_gamma = encode_rgb(update_encoder, update_patcher, update_mean, update_std, update_clip, first_rgb)
    first_cached = cached_gamma[torch.as_tensor(first_rows, dtype=torch.long)].to(device)
    update_parts, _ = compare_initial_rollout(update_core, reference, first_gamma, first_cached,
                                               first_patches, lossfn)
    before_core, before_encoder = clone_state(update_core), clone_state(update_encoder)
    old, _, _, rloss, *_ = update_parts
    optimizer.zero_grad(set_to_none=True)
    _, old_by_family, rgb_by_family = collect_joint_gradients(
        old, rloss, update_families, candidate=(arm == "analytic_candidate"))
    apply_family_gradients(update_families, old_by_family, rgb_by_family, lam)
    gradnorm = torch.nn.utils.clip_grad_norm_(update_families["all"], CLIP_NORM)
    objective = old + lam * rloss if arm == "analytic_candidate" else old
    if not torch.isfinite(objective) or not torch.isfinite(gradnorm) or float(gradnorm) <= 0:
        raise FloatingPointError("throwaway joint B16 Adam update has invalid loss/gradient")
    optimizer.step()
    changed = changed_trainable_groups(before_core, update_core.state_dict(),
                                       before_encoder, update_encoder.state_dict())
    if not all(changed.values()):
        raise AssertionError(f"throwaway update failed to change encoder/graph/core groups: {changed}")

    record = {
        "status": "passed", "experiment": "SW0116", "seed": seed, "arm": arm,
        "source_core_sha256": sha(source), "source_manifest_sha256": sha(source_manifest),
        "implementation_fingerprint": fingerprint,
        "encoder_sha256": sha(ENCODER_PATH), "preprocessing_sha256": sha(STATS_PATH),
        "gamma_train_sha256": sha(base.GAMMA_TRAIN),
        "gamma_train_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
        "rgb_cache_sha256": rgb_cache_sha, "rgb_cache_manifest_sha256": sha(RGB_CACHE_MANIFEST),
        "rgb_cache_source_size_bytes": rgb_manifest.get("source_size_bytes"),
        "rgb_cache_source_mtime_ns": rgb_manifest.get("source_mtime_ns"),
        "training_ids": ids.tolist(),
        "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "matched_shuffle_seed": 117 + seed, "batch_size": BATCH,
        "time_steps": TRAIN_STEPS, "settle": TRAIN_SETTLE,
        "ground_truth_used": False, "batches": records,
        "lambda_joint": lam, "lambda_rule": "candidate seed0: 0.25 * median(old joint norm / RGB joint norm), first four registered TRAIN batches",
        "real_rgb_loss_mean": float(np.mean(real_values)),
        "scrambled_rgb_loss_mean": float(np.mean(scrambled_values)),
        "scrambled_minus_real_rgb_loss": scramble_excess,
        "throwaway_b16_adam_update": True,
        "throwaway_preclip_gradient_norm": float(gradnorm),
        "throwaway_parameter_groups_changed": changed,
        "initial_encoder_gamma_max_diff": max(row["gamma_rollout_parity"]["max_gamma_abs_diff"] for row in records),
        "initial_hard_labels_and_H_exact": all(row["gamma_rollout_parity"]["hard_labels_exact"] and row["gamma_rollout_parity"]["hard_H_exact"] for row in records),
    }
    write(output, record)
    if arm == "analytic_candidate":
        write(LAMBDA_PATH, {"status": "passed", "seed": 0, "lambda_joint": lam,
                            "preflight_sha256": sha(output),
                            "implementation_fingerprint": fingerprint})
    return record


def train(seed, arm, output, device="cuda", steps=UPDATES):
    if seed != 0 or arm not in ARMS or steps != UPDATES:
        raise ValueError("SW0116 only enables the registered seed0 256-update pilot")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing training output: {output}")
    pf_path = ARCHIVE / f"preflight_seed{seed}_{arm}.json"
    if not pf_path.is_file():
        raise FileNotFoundError(pf_path)
    pf = json.loads(pf_path.read_text())
    if (pf.get("status") != "passed" or pf.get("experiment") != "SW0116"
            or pf.get("seed") != seed or pf.get("arm") != arm
            or pf.get("implementation_fingerprint") != implementation_fingerprint()):
        raise AssertionError("successful current SW0116 preflight is required")
    lam, lambda_sha = 0.0, None
    if arm == "analytic_candidate":
        if not LAMBDA_PATH.is_file():
            raise FileNotFoundError(LAMBDA_PATH)
        data = json.loads(LAMBDA_PATH.read_text())
        lam = float(data.get("lambda_joint", float("nan")))
        if (data.get("status") != "passed" or data.get("preflight_sha256") != sha(pf_path)
                or data.get("implementation_fingerprint") != pf.get("implementation_fingerprint")
                or lam != float(pf.get("lambda_joint", float("nan")))
                or not math.isfinite(lam) or lam <= 0):
            raise AssertionError("shared lambda does not match immutable candidate preflight")
        lambda_sha = sha(LAMBDA_PATH)
    (core, encoder, patcher, mean, std, clip, source, source_manifest, source_record,
     ids, rows) = load_models(seed, device)
    gamma_cache, gamma_manifest = base.validate_gamma_cache(base.GAMMA_TRAIN, base.GAMMA_TRAIN_MANIFEST)
    rgb_cache, rgb_manifest, rgb_cache_sha = load_rgb_training_cache()
    expected_ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    expected_bindings = {
        "source_core_sha256": sha(source), "source_manifest_sha256": sha(source_manifest),
        "encoder_sha256": sha(ENCODER_PATH), "preprocessing_sha256": sha(STATS_PATH),
        "gamma_train_sha256": sha(base.GAMMA_TRAIN),
        "gamma_train_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
        "rgb_cache_sha256": rgb_cache_sha, "rgb_cache_manifest_sha256": sha(RGB_CACHE_MANIFEST),
        "training_ids_sha256": expected_ids_sha,
        "implementation_fingerprint": implementation_fingerprint(),
    }
    if any(pf.get(k) != v for k, v in expected_bindings.items()):
        raise AssertionError("source/cache/order/implementation changed after SW0116 preflight")
    families = eligible_params(core, encoder)
    optimizer = optimizer_for(core, encoder)
    lossfn = criterion()
    source_ids_sha = expected_ids_sha
    torch.manual_seed(117 + seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(117 + seed)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "training", "experiment": "SW0116", "seed": seed, "arm": arm,
        "source_core": str(source), "source_core_sha256": sha(source),
        "source_manifest_sha256": sha(source_manifest), "preflight_sha256": sha(pf_path),
        "implementation_fingerprint": implementation_fingerprint(),
        "encoder_source_sha256": sha(ENCODER_PATH), "preprocessing_sha256": sha(STATS_PATH),
        "gamma_train_sha256": sha(base.GAMMA_TRAIN),
        "gamma_train_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
        "rgb_cache_sha256": rgb_cache_sha, "rgb_cache_manifest_sha256": sha(RGB_CACHE_MANIFEST),
        "training_ids": ids.tolist(), "training_ids_sha256": source_ids_sha,
        "matched_shuffle_seed": 117 + seed, "updates": steps, "batch_size": BATCH,
        "core_graph_lr": CORE_GRAPH_LR, "encoder_lr": ENCODER_LR,
        "clip_norm": CLIP_NORM, "time_steps": TRAIN_STEPS, "settle": TRAIN_SETTLE,
        "lambda_joint": lam, "lambda_artifact_sha256": lambda_sha,
        "objective": "phase_primary + 5*positive_actual_spike_product + lambda_joint*analytic_partition_rgb_if_candidate",
        "ground_truth_used_for_training": False,
    }
    write(output / "manifest.json", manifest)
    history = []
    core.train(); core.graph_generator.train(); encoder.train()
    for update in range(steps):
        start = update * BATCH
        batch_rows = rows[start:start+BATCH]
        image = read_rgb(rgb_cache, batch_rows, device)
        patches = rgb_base.rgb_patch_means(image / 255.0)
        gamma = encode_rgb(encoder, patcher, mean, std, clip, image)
        parts = objective_parts(core, gamma, patches, lossfn)
        old, primary, positive, rloss, q, labels, hard, per_image, *_ = parts
        total = old + lam * rloss if arm == "analytic_candidate" else old
        if not torch.isfinite(total):
            raise FloatingPointError(f"nonfinite SW0116 objective at update {update+1}")
        optimizer.zero_grad(set_to_none=True)
        stats, old_by_family, rgb_by_family = collect_joint_gradients(
            old, rloss, families, candidate=(arm == "analytic_candidate"))
        apply_family_gradients(families, old_by_family, rgb_by_family, lam)
        gradnorm = torch.nn.utils.clip_grad_norm_(families["all"], CLIP_NORM)
        if not torch.isfinite(gradnorm) or float(gradnorm) <= 0:
            raise FloatingPointError(f"invalid joint gradient at update {update+1}")
        optimizer.step()
        history.append({"update": update+1, "total": float(total.detach()),
                        "old": float(old.detach()), "primary": float(primary.detach()),
                        "positive_actual_spike_product": float(positive.detach()),
                        "rgb": float(rloss.detach()), "lambda_joint": lam,
                        "gradient_norm_preclip": float(gradnorm),
                        "old_rgb_gradient_stats": stats,
                        "mean_groups": float(np.mean([h.shape[1] for h in hard])),
                        "mean_rgb_per_image": float(per_image.detach().mean())})
        if update == 0 or (update + 1) % 32 == 0:
            write(output / "progress.json", {"status": "training", "seed": seed,
                  "arm": arm, "update": update+1, "total_updates": steps})
    if any(not torch.isfinite(p).all() for p in list(core.parameters()) + list(encoder.parameters())):
        raise FloatingPointError("nonfinite final SW0116 model parameter")
    torch.save(core.state_dict(), output / "core.pt")
    torch.save(encoder.state_dict(), output / "encoder.pt")
    torch.save(optimizer.state_dict(), output / "optimizer.pt")
    write(output / "history.json", history)
    manifest.update(status="training_complete", completed=time.time(),
                    core_sha256=sha(output / "core.pt"), encoder_sha256=sha(output / "encoder.pt"),
                    optimizer_sha256=sha(output / "optimizer.pt"), history_sha256=sha(output / "history.json"))
    write(output / "manifest.json", manifest)
    (output / "TRAINING_COMPLETED").write_text("complete\n")


def evaluate(seed, arm, checkpoint, output, device="cuda"):
    if seed != 0 or arm not in ARMS:
        raise ValueError("only registered SW0116 seed0 evaluation is enabled")
    checkpoint, output = Path(checkpoint), Path(output)
    sidecar_path = output.parent / "evaluation_manifest.json"
    if output.exists() or sidecar_path.exists():
        raise FileExistsError("preserve existing SW0116 evaluation/sidecar; refusing overwrite")
    train_manifest_path = checkpoint.parent / "manifest.json"
    encoder_path = checkpoint.parent / "encoder.pt"
    if not train_manifest_path.is_file() or not encoder_path.is_file():
        raise FileNotFoundError("completed SW0116 core+encoder artifact is required")
    train_manifest = json.loads(train_manifest_path.read_text())
    if (train_manifest.get("status") != "training_complete" or train_manifest.get("seed") != seed
            or train_manifest.get("arm") != arm or train_manifest.get("core_sha256") != sha(checkpoint)
            or train_manifest.get("encoder_sha256") != sha(encoder_path)
            or train_manifest.get("encoder_source_sha256") != sha(ENCODER_PATH)
            or train_manifest.get("preprocessing_sha256") != sha(STATS_PATH)):
        raise AssertionError("evaluation checkpoint does not match a completed SW0116 training manifest")
    if (train_manifest.get("history_sha256") != sha(checkpoint.parent / "history.json")
            or train_manifest.get("optimizer_sha256") != sha(checkpoint.parent / "optimizer.pt")):
        raise AssertionError("training history/optimizer provenance is incomplete")
    rgb_cache, rgb_meta, rgb_cache_sha = validate_rgb_validation_cache()
    _, encoder, patcher, mean, std, clip, *_ = load_models(seed, device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True), strict=True)
    encoder.eval(); patcher.eval()
    gammas = []
    with torch.no_grad():
        for start in range(0, 320, 8):
            image = read_rgb(rgb_cache, np.arange(start, start+8), device)
            gammas.append(encode_rgb(encoder, patcher, mean, std, clip, image).cpu())
    gamma = torch.cat(gammas, dim=0)
    gamma_path = output.parent / "trained_encoder_gamma_validation.pt"
    gamma_manifest_path = output.parent / "trained_encoder_gamma_validation_manifest.json"
    if gamma_path.exists() or gamma_manifest_path.exists():
        raise FileExistsError("preserve existing trained-encoder validation gamma artifact")
    tmp_gamma = gamma_path.with_suffix(".pt.tmp")
    torch.save(gamma, tmp_gamma); tmp_gamma.replace(gamma_path)
    gamma_manifest = {
        "status": "complete", "kind": "validation", "image_ids": [1320, 1639],
        "gamma_sha256": sha(gamma_path), "encoder_sha256": sha(encoder_path),
        "preprocessing_sha256": sha(STATS_PATH), "rgb_cache_sha256": rgb_cache_sha,
        "rgb_cache_manifest_sha256": sha(VAL_RGB_MANIFEST),
        "ground_truth_used_for_gamma": False,
    }
    write(gamma_manifest_path, gamma_manifest)
    old_gamma, old_gamma_manifest, old_validator = base.GAMMA_VAL, base.GAMMA_VAL_MANIFEST, base.validate_gamma_cache

    def validate_trained_gamma(path, manifest_path, validation=False):
        path, manifest_path = Path(path), Path(manifest_path)
        m = json.loads(manifest_path.read_text())
        if (not validation or path != gamma_path or manifest_path != gamma_manifest_path
                or tuple(torch.load(path, map_location="cpu", weights_only=True).shape) != (320, 8, 256)
                or m.get("gamma_sha256") != sha(path)
                or m.get("encoder_sha256") != sha(encoder_path)
                or m.get("preprocessing_sha256") != sha(STATS_PATH)
                or m.get("image_ids") != [1320, 1639]):
            raise AssertionError("trained-encoder validation gamma binding is invalid")
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True), m

    try:
        base.GAMMA_VAL, base.GAMMA_VAL_MANIFEST = gamma_path, gamma_manifest_path
        base.validate_gamma_cache = validate_trained_gamma
        base.evaluate(seed, "control", checkpoint, output, device=device)
    finally:
        base.GAMMA_VAL, base.GAMMA_VAL_MANIFEST = old_gamma, old_gamma_manifest
        base.validate_gamma_cache = old_validator
    report = json.loads(output.read_text())
    report.update(experiment="SW0116", arm=arm, seed=seed,
                  training_manifest_sha256=sha(train_manifest_path),
                  checkpoint_sha256=sha(checkpoint), encoder_checkpoint_sha256=sha(encoder_path),
                  trained_encoder_gamma_sha256=sha(gamma_path),
                  trained_encoder_gamma_manifest_sha256=sha(gamma_manifest_path),
                  evaluation_runner_sha256=sha(RUNNER),
                  shared_evaluator_sha256=sha(ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py"),
                  ground_truth_used_for_prediction=False)
    write(output, report)
    write(sidecar_path, {
        "experiment": "SW0116", "seed": seed, "arm": arm,
        "evaluation_file": output.name, "evaluation_sha256": sha(output),
        "training_manifest_sha256": sha(train_manifest_path),
        "checkpoint_sha256": sha(checkpoint), "encoder_checkpoint_sha256": sha(encoder_path),
        "evaluation_runner_sha256": sha(RUNNER),
        "shared_evaluator_sha256": sha(ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py"),
        "trained_encoder_gamma_sha256": sha(gamma_path),
        "trained_encoder_gamma_manifest_sha256": sha(gamma_manifest_path),
        "evaluation_contract": {"ids": [1320, 1639], "images": 320, "batch_size": 8,
            "time_steps": 1024, "settle": 512, "membrane_vth": 0.06,
            "readout_threshold": 0.50, "min_group_size": 2,
            "background": "largest_component", "ground_truth_used_for_prediction": False},
    })


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train", "evaluate"):
        sp = sub.add_parser(name)
        sp.add_argument("--seed", type=int, choices=SEEDS, required=True)
        sp.add_argument("--arm", choices=ARMS, required=True)
        sp.add_argument("--device", default="cuda")
        if name in ("preflight", "train"):
            sp.add_argument("--output", type=Path, required=True)
        else:
            sp.add_argument("--checkpoint", type=Path, required=True)
            sp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        record = preflight(args.seed, args.arm, args.output, args.device)
        print(json.dumps({"status": record["status"], "command": args.command,
                          "seed": args.seed, "arm": args.arm}), flush=True)
    elif args.command == "train":
        train(args.seed, args.arm, args.output, args.device)
        print(json.dumps({"status": "complete", "command": args.command,
                          "seed": args.seed, "arm": args.arm}), flush=True)
    else:
        evaluate(args.seed, args.arm, args.checkpoint, args.output, args.device)
        print(json.dumps({"status": "complete", "command": args.command,
                          "seed": args.seed, "arm": args.arm}), flush=True)


if __name__ == "__main__":
    main()
