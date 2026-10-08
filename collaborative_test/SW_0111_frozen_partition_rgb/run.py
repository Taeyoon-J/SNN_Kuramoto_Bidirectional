"""SW0111 paired frozen-encoder/graph RGB-credit pilot (actual inference unchanged)."""
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
import h5py

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"), str(HERE)]
from SW_0094_aligned_joint_pilot.run import ASSETS, DATASET, GAMMA, VAL_GAMMA, hparams
from SW_0106_spike_partition_rgb.partition_rgb import (
    SharedRGBDecoder, groups_to_onehot, reconstruct_one, rgb_patch_means)
from snn_kuramoto_bidirectional.evaluation import (
    clevr_mask_patch, evaluate_patch_masks, spatial_components_to_patch_labels)
from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
from snn_kuramoto_bidirectional.loss_function import UnsupervisedS2NetLoss
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.spike_classifier import (
    spike_synchrony_affinity, spike_synchrony_components)
from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_input_encoder
from snn_kuramoto_bidirectional.training.train_s2net_core import _forward_with_plv
from collaborative_test.SW_0110_xy_graph_route import run as common

SOURCE_ROOT = ROOT / "trained_models/SW0097_graph_adaptation"
RGB_CACHE = ROOT / "data/SW_0106_spike_partition_rgb"
TRAIN_RGB = RGB_CACHE / "train_rgb_uint8.npy"
VAL_RGB = RGB_CACHE / "validation_rgb_uint8.npy"
FEATURE_STATS = ASSETS / "feature_preprocessing.pt"
ENCODER_SOURCE = ASSETS / "input_encoder/input_layer_encoder.pt"
OUT = ROOT / "trained_models/SW0111_frozen_partition_rgb"
ARCHIVE = HERE / "results_archive"
SEEDS = (0, 1, 2)
ARMS = ("control", "candidate")
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
LR_CORE, LR_DECODER, SHARED_WARMUP = 3e-5, 3e-4, 32
UPDATES, BATCH, STEPS, SETTLE = 256, 16, 64, 32
EVAL_STEPS, EVAL_SETTLE = 1024, 512


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def pool_indices(seed):
    generator = torch.Generator(device="cpu").manual_seed(117 + seed)
    chosen = torch.randperm(70000, generator=generator)[:4096].numpy().astype(np.int64)
    ids = np.where(chosen < 1000, chosen, chosen + 640).astype(np.int64)
    return chosen, ids


def verify_contract(seed):
    checkpoint, manifest_path, manifest = common.source_paths(seed)
    chosen, ids = pool_indices(seed)
    if manifest.get("training_ids") != ids.tolist():
        raise AssertionError("SW0097 ordered train IDs differ from deterministic registered selection")
    if (manifest.get("steps") != 256 or manifest.get("batch") != 16
            or manifest.get("train_steps") != 64 or manifest.get("train_settle") != 32):
        raise AssertionError("SW0097 source has mismatched training contract")
    control = SOURCE_ROOT / f"seed{seed}_positive_frozen"
    control_manifest = control / "manifest.json"
    if not control_manifest.is_file():
        raise FileNotFoundError(control_manifest)
    cm = json.loads(control_manifest.read_text())
    if (cm.get("training_ids") != ids.tolist() or cm.get("steps") != 256
            or cm.get("batch") != 16 or cm.get("seed") != 117 + seed):
        raise AssertionError("matched frozen control source/order is not exact")
    train_meta = json.loads((RGB_CACHE / "train_rgb_uint8.npy.complete.json").read_text())
    val_meta = json.loads((RGB_CACHE / "validation_rgb_uint8.npy.complete.json").read_text())
    expected_train = np.concatenate((np.arange(1000, dtype="<i8"), np.arange(1640, 70640, dtype="<i8")))
    expected_val = np.arange(1320, 1640, dtype="<i8")
    if (train_meta.get("status") != "complete" or train_meta.get("cache_shape") != [70000, 128, 128, 3]
            or train_meta.get("ids_mapping_sha256") != hashlib.sha256(expected_train.tobytes()).hexdigest()):
        raise AssertionError("registered SW0106 native RGB train cache contract failed")
    if (val_meta.get("status") != "complete" or val_meta.get("cache_shape") != [320, 128, 128, 3]
            or val_meta.get("ids_mapping_sha256") != hashlib.sha256(expected_val.tobytes()).hexdigest()):
        raise AssertionError("registered native validation RGB cache contract failed")
    gamma_meta_path = ROOT / "data/SW_0090_large_unique_scale/manifest.json"
    gamma_manifest = json.loads(gamma_meta_path.read_text())
    if (gamma_manifest.get("training_ids", {}).get("count") != 70000
            or gamma_manifest.get("training_ids", {}).get("segments") != [[0, 999], [1640, 70639]]):
        raise AssertionError("registered training gamma manifest changed")
    return chosen, ids, checkpoint, sha(checkpoint), sha(control / "core.pt"), manifest_path


def load_models(device, seed):
    checkpoint, _, _ = common.source_paths(seed)
    core = S2NetCore(hparams("raw").validate(), device=device).to(device)
    core.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    core.graph_generator.requires_grad_(False)
    core.graph_generator.eval()
    core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.shape[0])]
    if core.graph_generator.uses_feedback or core.kuramoto.spike_pulse_gain is not None:
        raise AssertionError("registered SW0097 feedback/pulse mode changed")
    encoder = load_input_encoder(str(ENCODER_SOURCE), num_kernels=8,
                                 kernel_size=3, channels=3, device=device)
    encoder.eval().requires_grad_(False)
    stats = torch.load(FEATURE_STATS, map_location=device, weights_only=True)
    mean, std, clip = stats["mean"].to(device), stats["std"].to(device), float(stats.get("clip", 3.))
    if stats.get("mode") != "standardize" or not bool((std > 0).all()):
        raise AssertionError("registered feature preprocessing invalid")
    patcher = FeaturePatchGammaInitializer(grid_size=16).to(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(111)
        decoder = SharedRGBDecoder().to(device)
    return core, encoder, patcher, mean, std, clip, decoder


def encode(encoder, patcher, mean, std, clip, images):
    return patcher(((encoder(images.float() / 255.) - mean) / std).clamp(-clip, clip))


def read_batch(cache, indices, device):
    array = np.asarray(cache[np.asarray(indices, dtype=np.int64)]).copy()
    if array.shape[1:] != (128, 128, 3) or array.dtype != np.uint8:
        raise AssertionError("RGB cache batch must remain native uint8 128x128x3")
    return torch.from_numpy(array).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)


def criterion():
    return UnsupervisedS2NetLoss(
        spike_rate_weight=0., spike_smooth_weight=0., spike_diversity_weight=0.,
        structural_weight=0., plv_bimodality_weight=6., plv_balance_weight=10.,
        plv_coherence_weight=.5, plv_collapse_weight=1., plv_target_density=.867,
        patch_grid_size=(16, 16))


def affinity(core, settle=SETTLE):
    components = core.last_component_spikes
    if components is None:
        raise RuntimeError("actual per-component spike history missing")
    return spike_synchrony_affinity(components.mean(dim=1), components, settle=settle)


def hard_partition(spikes, components):
    groups = spike_synchrony_components(
        spikes.detach().cpu(), synchrony_threshold=.5, min_group_size=2, settle=SETTLE,
        components=components.detach().cpu(), background="largest_component",
        affinity_mode="spike", spatial_grid_size=16)
    labels = spatial_components_to_patch_labels(groups, 16, device=spikes.device).reshape(spikes.shape[0], -1)
    onehot = [groups_to_onehot(groups_i, device=spikes.device, dtype=spikes.dtype) for groups_i in groups]
    if any(not torch.equal(h.argmax(-1), labels[i]) for i, h in enumerate(onehot)):
        raise AssertionError("hard H differs from unchanged production group-to-label conversion")
    return labels, onehot, groups


def forward_batch(core, encoder, patcher, mean, std, clip, images, lossfn):
    gamma = encode(encoder, patcher, mean, std, clip, images)
    out = _forward_with_plv(core, gamma, lossfn, SETTLE, "phase", "mean")
    groups, spikes, core_out, plv, theta = out
    components = core.last_component_spikes
    if tuple(components.shape) != (images.shape[0], 4, 256, STEPS):
        raise AssertionError("component spike shape mismatch")
    q = affinity(core)
    labels, hard, detected = hard_partition(spikes, components)
    targets = rgb_patch_means(images / 255.)
    return gamma, out, q, labels, hard, detected, targets


def recon_losses(q, hard, gamma, target, decoder, *, credit):
    losses, predictions, details = [], [], []
    for b in range(q.shape[0]):
        prediction, loss, diagnostic = reconstruct_one(
            q[b] if credit else q[b].detach(), hard[b], gamma[b].transpose(0, 1),
            target[b], decoder, assignment_credit=credit)
        predictions.append(prediction); losses.append(loss); details.append(diagnostic)
    return torch.stack(predictions), torch.stack(losses).mean(), details


def grad_norm(grads):
    total = 0.
    for grad in grads:
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            raise FloatingPointError("nonfinite gradient")
        total += float(grad.detach().double().square().sum())
    return math.sqrt(total)


def old_objective(result, q, lossfn):
    _, _, _, plv, theta = result
    primary, _ = lossfn(plv=plv, theta=theta)
    positive, _ = lossfn(plv=q)
    return primary + 5. * positive, primary, positive


def preflight(seed, device="cuda"):
    device = torch.device(device)
    chosen, ids, source, source_sha, control_sha, source_manifest = verify_contract(seed)
    if not TRAIN_RGB.is_file() or not VAL_RGB.is_file():
        raise FileNotFoundError("verified SW0106 native RGB caches are required")
    train_rgb = np.load(TRAIN_RGB, mmap_mode="r")
    if train_rgb.shape != (70000, 128, 128, 3):
        raise AssertionError("train RGB cache shape mismatch")
    gamma_cache = torch.load(GAMMA, map_location="cpu", weights_only=True, mmap=True)
    if tuple(gamma_cache.shape) != (70000, 8, 256):
        raise AssertionError("registered gamma train cache shape mismatch")
    core, encoder, patcher, mean, std, clip, decoder = load_models(device, seed)
    lossfn = criterion()
    core_params = [p for p in core.parameters() if p.requires_grad]
    decoder_params = list(decoder.parameters())
    init_core = {k: v.detach().clone() for k, v in core.state_dict().items()}
    init_encoder = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    trainable_ids = chosen[:4096]
    warm_opt = torch.optim.Adam(decoder_params, lr=LR_DECODER)
    warm_history = []
    core.eval(); encoder.eval(); decoder.train()
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for batch_id in range(SHARED_WARMUP):
        ix = trainable_ids[batch_id * BATCH:(batch_id + 1) * BATCH]
        images = read_batch(train_rgb, ix, device)
        with torch.no_grad():
            gamma, result, q, labels, hard, groups, targets = forward_batch(
                core, encoder, patcher, mean, std, clip, images, lossfn)
        # Keep only the expensive source forward detached; the shared decoder
        # must receive gradients during its registered warmup.
        _, rec, _ = recon_losses(q.detach(), hard, gamma.detach(), targets, decoder, credit=False)
        if not torch.isfinite(rec):
            raise FloatingPointError(f"nonfinite shared decoder warmup batch {batch_id}")
        warm_opt.zero_grad(set_to_none=True); rec.backward()
        norm = torch.nn.utils.clip_grad_norm_(decoder_params, 1.)
        if not torch.isfinite(norm) or float(norm) <= 0:
            raise FloatingPointError("invalid shared decoder warmup gradient")
        warm_opt.step(); warm_history.append(float(rec.detach()))
    if any(not torch.equal(v, core.state_dict()[k]) for k, v in init_core.items()):
        raise AssertionError("decoder warmup changed source core")
    if any(not torch.equal(v, encoder.state_dict()[k]) for k, v in init_encoder.items()):
        raise AssertionError("decoder warmup changed frozen encoder")
    # Registered gamma agreement, hard-forward identity and actual assignment-credit route.
    first_ids = trainable_ids[:BATCH]
    images = read_batch(train_rgb, first_ids, device)
    expected = gamma_cache[torch.as_tensor(first_ids)].to(device)
    with torch.no_grad():
        actual_gamma = encode(encoder, patcher, mean, std, clip, images)
    gamma_diff = float((actual_gamma - expected).abs().max())
    if gamma_diff > 2e-5:
        raise AssertionError(f"RGB/native gamma mismatch {gamma_diff}")
    gamma, result, q, labels, hard, groups, targets = forward_batch(
        core, encoder, patcher, mean, std, clip, images, lossfn)
    old, primary, spike = old_objective(result, q, lossfn)
    pred_c, rec_c, _ = recon_losses(q, hard, gamma, targets, decoder, credit=True)
    pred_h, rec_h, _ = recon_losses(q, hard, gamma, targets, decoder, credit=False)
    if not torch.equal(pred_c, pred_h) or not torch.equal(rec_c, rec_h):
        raise AssertionError("candidate/control hard-forward reconstruction is not exact")
    if not torch.isfinite(old + rec_c):
        raise FloatingPointError("nonfinite preflight objective")
    cand_grads = torch.autograd.grad(rec_c, core_params, retain_graph=True, allow_unused=True)
    cand_norm = grad_norm(cand_grads)
    if cand_norm <= 0:
        raise AssertionError("assignment-only reconstruction credit is inert for the core")
    ctrl_grads = torch.autograd.grad(rec_h, core_params, retain_graph=True, allow_unused=True)
    if any(g is not None and bool(torch.count_nonzero(g)) for g in ctrl_grads):
        raise AssertionError("control reconstruction leaked gradients into source core")
    # Shared lambda is measured only on seed0; other seeds reuse it exactly.
    calibrations, ratios = [], []
    for batch_i in range(4):
        ix = trainable_ids[batch_i * BATCH:(batch_i + 1) * BATCH]
        batch = read_batch(train_rgb, ix, device)
        gamma, result, q, _, hard, _, targets = forward_batch(
            core, encoder, patcher, mean, std, clip, batch, lossfn)
        old_i, _, _ = old_objective(result, q, lossfn)
        _, rec_i, _ = recon_losses(q, hard, gamma, targets, decoder, credit=True)
        old_g = torch.autograd.grad(old_i, core_params, retain_graph=True, allow_unused=True)
        rec_g = torch.autograd.grad(rec_i, core_params, retain_graph=False, allow_unused=True)
        old_norm, rec_norm = grad_norm(old_g), grad_norm(rec_g)
        if (not math.isfinite(old_norm) or not math.isfinite(rec_norm)
                or old_norm <= 0 or rec_norm <= 0):
            raise AssertionError(f"calibration batch{batch_i} has inert/nonfinite core gradient")
        ratio = .25 * old_norm / rec_norm
        ratios.append(ratio)
        calibrations.append({"batch": batch_i, "old_core_grad_norm": old_norm,
                             "reconstruction_core_grad_norm": rec_norm, "ratio": ratio})
    if seed == 0:
        shared_lambda = float(np.median(np.asarray(ratios, dtype=np.float64)))
    else:
        seed0_path = ARCHIVE / "preflight_seed0.json"
        if not seed0_path.is_file():
            raise FileNotFoundError("seed0 must freeze shared lambda before other seed preflights")
        seed0 = json.loads(seed0_path.read_text())
        if seed0.get("status") != "passed":
            raise AssertionError("seed0 shared calibration did not pass")
        shared_lambda = float(seed0["lambda"])
    if not math.isfinite(shared_lambda) or shared_lambda <= 0:
        raise AssertionError("shared lambda is invalid")
    # Per-image hard-mask row scramble on first four TRAIN batches, no GT and no optimizer.
    scramble = []
    with torch.no_grad():
        for batch_i in range(4):
            ix = trainable_ids[batch_i*BATCH:(batch_i+1)*BATCH]
            batch = read_batch(train_rgb, ix, device)
            gamma, _, q, _, hard, _, targets = forward_batch(
                core, encoder, patcher, mean, std, clip, batch, lossfn)
            deltas = []
            for b in range(BATCH):
                perm = torch.randperm(256, generator=torch.Generator(device="cpu").manual_seed(
                    106 + 16 * seed + batch_i * BATCH + b)).to(device)
                _, real_loss, _ = reconstruct_one(q[b], hard[b], gamma[b].T,
                                                  targets[b], decoder, assignment_credit=False)
                _, scrambled_loss, _ = reconstruct_one(q[b], hard[b][perm], gamma[b].T,
                                                       targets[b], decoder, assignment_credit=False)
                deltas.append(float(scrambled_loss - real_loss))
            scramble.append({"batch": batch_i, "mean_delta": float(np.mean(deltas)),
                             "positive_count": sum(x > 0 for x in deltas)})
    scramble_mean = float(np.mean([x["mean_delta"] for x in scramble]))
    if not math.isfinite(scramble_mean) or scramble_mean <= 0:
        raise AssertionError("registered hard-partition row scramble did not increase reconstruction loss")
    artifact = ARCHIVE / f"shared_decoder_seed{seed}.pt"
    if artifact.exists():
        raise FileExistsError(f"preserve previous shared warmup artifact {artifact}")
    torch.save({"decoder_state_dict": decoder.state_dict(),
                "decoder_optimizer_state_dict": warm_opt.state_dict(),
                "seed": seed, "warmup_steps": SHARED_WARMUP}, artifact)
    artifact_sha = sha(artifact)
    # Actual candidate/control gradient routing and separate-optimizer update smoke.
    joint_opt = torch.optim.Adam(core_params, lr=LR_CORE)
    dec_opt = torch.optim.Adam(decoder_params, lr=LR_DECODER)
    dec_opt.load_state_dict(warm_opt.state_dict())
    joint_opt.zero_grad(set_to_none=True); dec_opt.zero_grad(set_to_none=True)
    joint_grads = torch.autograd.grad(old + shared_lambda * rec_c, core_params,
                                      retain_graph=True, allow_unused=True)
    decoder_grads = torch.autograd.grad(rec_c, decoder_params, allow_unused=True)
    if grad_norm(joint_grads) <= 0 or grad_norm(decoder_grads) <= 0:
        raise AssertionError("throwaway split-optimizer path has empty gradient")
    for param, grad in zip(core_params, joint_grads):
        param.grad = None if grad is None else grad.detach().clone()
    for param, grad in zip(decoder_params, decoder_grads):
        param.grad = None if grad is None else grad.detach().clone()
    torch.nn.utils.clip_grad_norm_(core_params, 1.); torch.nn.utils.clip_grad_norm_(decoder_params, 1.)
    frozen_graph = {k: v.detach().clone() for k, v in core.graph_generator.state_dict().items()}
    frozen_enc = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    joint_opt.step(); dec_opt.step()
    if any(not torch.equal(v, core.graph_generator.state_dict()[k]) for k, v in frozen_graph.items()):
        raise AssertionError("throwaway update changed frozen graph")
    if any(not torch.equal(v, encoder.state_dict()[k]) for k, v in frozen_enc.items()):
        raise AssertionError("throwaway update changed frozen encoder")
    report = {"status": "passed", "seed": seed, "device": str(device),
              "source_core_sha256": source_sha, "source_manifest_sha256": sha(source_manifest),
              "matched_control_core_sha256": control_sha,
              "encoder_sha256": sha(ENCODER_SOURCE), "preprocessing_sha256": sha(FEATURE_STATS),
              "training_ids": ids.tolist(), "training_ids_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
              "shuffle_seed": 117 + seed, "batch_size": BATCH, "train_steps": STEPS,
              "settle": SETTLE, "registered_gamma_first_batch_max_abs_diff": gamma_diff,
              "warmup_steps": SHARED_WARMUP, "warmup_loss_first_last": [warm_history[0], warm_history[-1]],
              "warmup_artifact": str(artifact), "warmup_artifact_sha256": artifact_sha,
              "warmup_core_encoder_unchanged": True, "hard_forward_exact": True,
              "candidate_reconstruction_core_gradient_norm": cand_norm,
              "control_reconstruction_core_gradient_zero": True,
              "lambda_calibration": calibrations, "lambda": shared_lambda,
              "lambda_source_seed": 0, "row_scramble": scramble,
              "row_scramble_mean_delta": scramble_mean,
              "throwaway_split_optimizer_update": True, "ground_truth_used": False,
              "implementation_sha256": implementation_fingerprint()}
    return report


def train(seed, arm, device="cuda"):
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("invalid seed or arm")
    device = torch.device(device)
    out = OUT / f"{arm}_seed{seed}"
    if out.exists():
        raise FileExistsError(f"preserve existing training output: {out}")
    preflight_path = ARCHIVE / f"preflight_seed{seed}.json"
    if not preflight_path.is_file():
        raise FileNotFoundError(f"preflight is required: {preflight_path}")
    pf = json.loads(preflight_path.read_text())
    chosen, ids, source, source_sha, control_sha, source_manifest = verify_contract(seed)
    expected_ids = ids.tolist()
    if (pf.get("status") != "passed" or pf.get("seed") != seed
            or pf.get("source_core_sha256") != source_sha
            or pf.get("matched_control_core_sha256") != control_sha
            or pf.get("training_ids") != expected_ids
            or pf.get("implementation_sha256") != implementation_fingerprint()
            or pf.get("ground_truth_used") is not False):
        raise AssertionError("preflight source/order/code provenance mismatch")
    if seed > 0:
        pf0_path = ARCHIVE / "preflight_seed0.json"
        if not pf0_path.is_file():
            raise FileNotFoundError("seed0 calibration is required")
        pf0 = json.loads(pf0_path.read_text())
        if pf0.get("status") != "passed" or pf.get("lambda") != pf0.get("lambda"):
            raise AssertionError("shared seed0 lambda mismatch")
    artifact = Path(pf.get("warmup_artifact", ""))
    if not artifact.is_file() or sha(artifact) != pf.get("warmup_artifact_sha256"):
        raise AssertionError("shared decoder warmup artifact SHA mismatch")
    train_rgb = np.load(TRAIN_RGB, mmap_mode="r")
    gamma_cache = torch.load(GAMMA, map_location="cpu", weights_only=True, mmap=True)
    core, encoder, patcher, mean, std, clip, decoder = load_models(device, seed)
    core_params = [p for p in core.parameters() if p.requires_grad]
    decoder.load_state_dict(torch.load(artifact, map_location=device,
                                       weights_only=True)["decoder_state_dict"], strict=True)
    warm_state = torch.load(artifact, map_location=device, weights_only=True)
    core_opt = torch.optim.Adam(core_params, lr=LR_CORE)
    decoder_opt = torch.optim.Adam(decoder.parameters(), lr=LR_DECODER)
    decoder_opt.load_state_dict(warm_state["decoder_optimizer_state_dict"])
    lossfn = criterion()
    rows = chosen[:4096]
    initial_core = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
    started = time.time()
    frozen_graph = {k: v.detach().clone() for k, v in core.graph_generator.state_dict().items()}
    frozen_encoder = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    out.mkdir(parents=True)
    write(out / "progress.json", {"status": "training", "seed": seed, "arm": arm,
                                  "updates": 0, "total_updates": UPDATES,
                                  "source_core_sha256": source_sha,
                                  "preflight_sha256": sha(preflight_path),
                                  "implementation_sha256": implementation_fingerprint()})
    history = []
    core.train(); core.graph_generator.eval(); encoder.eval(); decoder.train()
    for step in range(UPDATES):
        idx = rows[step * BATCH:(step + 1) * BATCH]
        if len(idx) != BATCH:
            raise AssertionError("registered training sequence exhausted")
        images = read_batch(train_rgb, idx, device)
        expected_gamma = gamma_cache[torch.as_tensor(idx)].to(device)
        gamma, result, q, labels, hard, groups, targets = forward_batch(
            core, encoder, patcher, mean, std, clip, images, lossfn)
        if step == 0:
            gamma_diff = float((gamma.detach() - expected_gamma).abs().max())
            if not math.isfinite(gamma_diff) or gamma_diff > 2e-5:
                raise AssertionError(f"registered native gamma mismatch {gamma_diff}")
        old, primary, positive = old_objective(result, q, lossfn)
        _, rec, details = recon_losses(q, hard, gamma, targets, decoder, credit=(arm == "candidate"))
        core_objective = old + (float(pf["lambda"]) * rec if arm == "candidate" else 0.)
        if not bool(torch.isfinite(core_objective + rec)):
            raise FloatingPointError(f"nonfinite objective at update {step}")
        core_opt.zero_grad(set_to_none=True); decoder_opt.zero_grad(set_to_none=True)
        core_grads = torch.autograd.grad(core_objective, core_params, retain_graph=True,
                                         allow_unused=True)
        decoder_grads = torch.autograd.grad(rec, tuple(decoder.parameters()), allow_unused=True)
        for param, grad in zip(core_params, core_grads):
            param.grad = None if grad is None else grad.detach().clone()
        for param, grad in zip(decoder.parameters(), decoder_grads):
            param.grad = None if grad is None else grad.detach().clone()
        core_norm = grad_norm([p.grad for p in core_params])
        decoder_norm = grad_norm([p.grad for p in decoder.parameters()])
        if core_norm <= 0 or decoder_norm <= 0:
            raise AssertionError(f"empty optimizer gradient at update {step}")
        torch.nn.utils.clip_grad_norm_(core_params, 1.)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.)
        core_opt.step(); decoder_opt.step()
        if (step == 0 or (step + 1) % 32 == 0):
            history.append({"update": step + 1, "total": float(core_objective.detach()),
                            "primary": float(primary.detach()),
                            "positive_product_spike_unweighted": float(positive.detach()),
                            "reconstruction_unweighted": float(rec.detach()),
                            "core_grad_norm_preclip": core_norm,
                            "decoder_grad_norm_preclip": decoder_norm,
                            "predicted_groups_mean": float(np.mean([len(g) for g in groups])),
                            "max_K": max(d["K"] for d in details)})
            write(out / "progress.json", {"status": "training", "seed": seed, "arm": arm,
                                          "updates": step + 1, "total_updates": UPDATES,
                                          "updated": time.time()})
    if any(not torch.equal(v, core.graph_generator.state_dict()[k]) for k, v in frozen_graph.items()):
        raise AssertionError("training changed frozen legacy graph")
    if any(not torch.equal(v, encoder.state_dict()[k]) for k, v in frozen_encoder.items()):
        raise AssertionError("training changed frozen encoder")
    changed_core = [k for k, v in core.state_dict().items()
                    if not torch.equal(v.detach().cpu(), initial_core[k])]
    if not changed_core:
        raise AssertionError("training did not update any eligible core parameter")
    torch.save(core.state_dict(), out / "core.pt")
    torch.save(decoder.state_dict(), out / "decoder.pt")
    optimizer_path = out / "optimizer_state.pt"
    torch.save({"core_optimizer_state_dict": core_opt.state_dict(),
                "decoder_optimizer_state_dict": decoder_opt.state_dict(),
                "seed": seed, "arm": arm, "updates": UPDATES,
                "shared_lambda": float(pf["lambda"])}, optimizer_path)
    write(out / "history.json", history)
    write(out / "manifest.json", {"status": "complete", "seed": seed, "arm": arm,
        "source_core_sha256": source_sha, "source_manifest_sha256": sha(source_manifest),
        "matched_control_core_sha256": control_sha, "encoder_sha256": sha(ENCODER_SOURCE),
        "preprocessing_sha256": sha(FEATURE_STATS), "training_ids": expected_ids,
        "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "shuffle_seed": 117 + seed, "updates": UPDATES, "batch_size": BATCH,
        "train_steps": STEPS, "settle": SETTLE, "old_spike_aux_weight": 5.,
        "lambda": float(pf["lambda"]), "warmup_artifact_sha256": sha(artifact),
        "warmup_preflight_sha256": sha(preflight_path), "warmup_steps": SHARED_WARMUP,
        "optimizer_start": "fresh core Adam; decoder Adam resumed from shared per-seed warmup",
        "changed_core_keys": changed_core, "ground_truth_used_for_training": False,
        "core_sha256": sha(out / "core.pt"), "optimizer_state_sha256": sha(optimizer_path),
        "started": started, "completed": time.time(), "runner_sha256": sha(Path(__file__))})
    (out / "TRAINING_COMPLETED").write_text("SW0111 fixed256-update training complete\n", encoding="utf-8")
    return out


@torch.no_grad()
def predict(core, gamma, *, batch_size=8, device="cuda"):
    from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
    labels, groups, diagnostics = [], [], []
    core.eval()
    for start in range(0, gamma.shape[0], batch_size):
        g = gamma[start:start + batch_size].to(device)
        _, spikes, _ = core(g, return_core_out=True, num_time_steps=EVAL_STEPS)
        components = core.last_component_spikes
        if components is None or tuple(components.shape) != (g.shape[0], 4, 256, EVAL_STEPS):
            raise AssertionError("evaluation component-spike history shape mismatch")
        groups_i = __import__("snn_kuramoto_bidirectional.spike_classifier", fromlist=["spike_synchrony_components"]).spike_synchrony_components(
            spikes.detach().cpu(), synchrony_threshold=.5, min_group_size=2,
            settle=EVAL_SETTLE, components=components.detach().cpu(),
            background="largest_component", affinity_mode="spike", spatial_grid_size=16)
        label_i = spatial_components_to_patch_labels(groups_i, 16, device="cpu")
        labels.append(label_i)
        groups.extend(groups_i)
        diagnostics.extend({"groups": len(gi), "foreground_fraction": float((li != 0).float().mean())}
                           for gi, li in zip(groups_i, label_i))
    return torch.cat(labels), groups, diagnostics


def evaluate(seed, *, device="cuda"):
    from collaborative_test.SW_0110_xy_graph_route import run as common
    from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks
    from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_input_encoder
    from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
    from collaborative_test.SW_0094_aligned_joint_pilot.run import hparams
    if seed not in SEEDS:
        raise ValueError("invalid seed")
    val = np.load(VAL_RGB, mmap_mode="r")
    if val.shape != (320, 128, 128, 3):
        raise AssertionError("validation RGB cache shape mismatch")
    val16, _ = common.validate_gamma_cache(VAL_GAMMA, common.GAMMA_VAL_MANIFEST, validation=True)
    reference_path, reference_sha, reference = registered_source_evaluation(seed)
    out = ARCHIVE / f"evaluation_seed{seed}"
    if out.exists():
        raise FileExistsError(f"preserve existing evaluation output: {out}")
    out.mkdir(parents=True)
    frozen = {}
    regenerated_val_gamma_max_abs_diff = None
    source_core = common.source_paths(seed)[0]
    checkpoints = {"source": source_core,
                   "control": OUT / f"control_seed{seed}/core.pt",
                   "candidate": OUT / f"candidate_seed{seed}/core.pt"}
    for name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        core = S2NetCore(hparams("raw").validate(), device=device).to(device)
        core.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.shape[0])]
        # Source and both paired arms use registered fixed native 8-D gamma.
        if name == "source":
            gamma = val16
        else:
            encoder = load_input_encoder(str(ENCODER_SOURCE), num_kernels=8,
                                         kernel_size=3, channels=3, device=device)
            encoder.load_state_dict(torch.load(ENCODER_SOURCE, map_location=device,
                                               weights_only=True), strict=True)
            encoder.eval().requires_grad_(False)
            stats = torch.load(FEATURE_STATS, map_location=device, weights_only=True)
            mean, std, clip = stats["mean"].to(device), stats["std"].to(device), float(stats.get("clip", 3.))
            patcher = FeaturePatchGammaInitializer(grid_size=16).to(device)
            chunks = []
            with torch.no_grad():
                for start in range(0, 320, 8):
                    images = read_batch(val, range(start, start + 8), device)
                    chunks.append(encode(encoder, patcher, mean, std, clip, images).cpu())
            gamma = torch.cat(chunks)
            if regenerated_val_gamma_max_abs_diff is None:
                regenerated_val_gamma_max_abs_diff = float((gamma - val16).abs().max())
                if (not math.isfinite(regenerated_val_gamma_max_abs_diff)
                        or regenerated_val_gamma_max_abs_diff > 2e-5):
                    raise AssertionError("frozen encoder RGB validation gamma differs from registered cache")
        labels, groups, diag = predict(core, gamma, batch_size=8, device=device)
        frozen[name] = {"labels": labels, "groups": groups, "diagnostics": diag,
                        "checkpoint_sha256": sha(checkpoint)}
        del core
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    prediction_file = out / "frozen_predictions.pt"
    torch.save(frozen, prediction_file)
    # Only now read validation masks.
    ids = np.arange(1320, 1640, dtype=np.int64)
    with h5py.File(DATASET, "r") as h5:
        masks = h5["mask"][ids.tolist()]
    masks = torch.from_numpy(masks.copy())
    target = clevr_mask_patch(masks, 8)["patch_labels"]
    scores = {}
    for name in ("source", "control", "candidate"):
        result = evaluate_patch_masks(frozen[name]["labels"], target)
        scores[name] = {"mean": {k: (float(v) if math.isfinite(float(v)) else None)
                                  for k, v in result["mean"].items()},
                        "valid_count": {k: int(v) for k, v in result["valid_count"].items()},
                        "per_image": {k: [(float(x) if math.isfinite(float(x)) else None) for x in v]
                                      for k, v in result["per_image"].items()}}
    source_checks = {}
    for metric in METRICS:
        actual = np.asarray(scores["source"]["per_image"][metric], dtype=np.float64)
        expected = np.asarray(reference["per_image"][metric], dtype=np.float64)
        delta = float(np.max(np.abs(actual - expected)))
        source_checks[metric] = {"max_abs_difference": delta, "pass": bool(np.isfinite(actual).all()
                                  and np.isfinite(expected).all() and delta <= 1e-10)}
    report = {"status": "complete" if all(x["pass"] for x in source_checks.values()) else "invalid_source_reference",
        "seed": seed, "ids": [1320, 1639], "count": 320, "ground_truth_used_during_prediction": False,
        "source_reference_path": str(reference_path), "source_reference_sha256": reference_sha,
        "regenerated_val_gamma_max_abs_diff": regenerated_val_gamma_max_abs_diff,
        "source_reference_check": source_checks, "scores": scores,
        "checkpoint_sha256": {k: v["checkpoint_sha256"] for k, v in frozen.items()},
        "frozen_predictions_sha256": sha(prediction_file),
        "implementation_sha256": implementation_fingerprint()}
    write(out / "evaluation.json", report)
    if report["status"] != "complete":
        raise AssertionError(f"SW0097 exact source baseline mismatch: {source_checks}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=int, choices=SEEDS)
    parser.add_argument("--train", type=int, choices=SEEDS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--evaluate", type=int, choices=SEEDS)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.preflight is not None:
        result = save_preflight(args.preflight, args.device)
    elif args.train is not None:
        if not args.arm:
            parser.error("--train requires --arm")
        result = train(args.train, args.arm, args.device)
    elif args.evaluate is not None:
        result = evaluate(args.evaluate, device=args.device)
    elif args.summarize:
        from summarize import main as summarize_main
        summarize_main()
        return
    else:
        parser.error("choose --preflight, --train or --evaluate")
    print(json.dumps({"status": "complete", "result": str(result)}, default=str), flush=True)


def implementation_fingerprint():
    files = [HERE / "run.py", HERE / "protocol.json", ROOT / "collaborative_test/SW_0106_spike_partition_rgb/partition_rgb.py",
             ROOT / "snn_kuramoto_bidirectional/s2net_cls.py", ROOT / "snn_kuramoto_bidirectional/graph_generator.py",
             ROOT / "snn_kuramoto_bidirectional/loss_function.py", ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
             ROOT / "snn_kuramoto_bidirectional/training/train_s2net_core.py",
             ROOT / "collaborative_test/SW_0094_aligned_joint_pilot/run.py"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in files}


def save_preflight(seed, device="cuda"):
    path = ARCHIVE / f"preflight_seed{seed}.json"
    if path.exists():
        raise FileExistsError(f"preserve existing preflight {path}")
    result = preflight(seed, device=device)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    write(path, result)
    return result


if __name__ == "__main__":
    main()
