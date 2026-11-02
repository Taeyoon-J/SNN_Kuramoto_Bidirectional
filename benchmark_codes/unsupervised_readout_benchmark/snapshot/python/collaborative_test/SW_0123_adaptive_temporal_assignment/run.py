"""SW0123 registered real-data preflight and 256-update training runner."""
from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "results_archive"
OUT = ROOT / "trained_models/SW0123_adaptive_temporal_assignment"
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0115_analytic_partition_rgb import run as rgb_base
from collaborative_test.SW_0117_joint_analytic_rgb import run as sw117
from collaborative_test.SW_0122_joint_rgb_seed_replication import run as sw122
from collaborative_test.SW_0123_adaptive_temporal_assignment.adaptive_dynamics import install_adaptive_dynamics
from collaborative_test.SW_0123_adaptive_temporal_assignment.model import (
    AdaptiveTemporalRGBModel, image_from_patches, normalized_patch_centers,
)
from collaborative_test.SW_0123_adaptive_temporal_assignment.readout import assignment_to_labels

SEEDS = (0, 1, 2)
ARMS = ("legacy_full", "adaptive_full", "gate_only_control")
BATCH, UPDATES, TIME_STEPS, SETTLE = 16, 256, 64, 32
HEAD_WARMUP_UPDATES = 32
CORE_LR, ENCODER_LR, HEAD_LR, CLIP = 3e-5, 3e-6, 3e-4, 1.0
SOURCE_CORE_SHA = {
    0: "36f2481dd1fa51fa29bd4dc34275a876b71d8b76fbee13ac47a0a1bfe6766fbf",
    1: "76f379d5a7e4cd9d12fdf0b701f3dd9dbbf28b5eea270a9cee2d30d87b4dad98",
    2: "798ad3e9d4bf837b1bbeb1bd7c13900df511b5c76f5736d2b9f8b973d7fa5661",
}
ENCODER_PATH = sw117.ENCODER_PATH
STATS_PATH = sw117.STATS_PATH
RGB_CACHE = sw117.RGB_CACHE
RGB_CACHE_MANIFEST = sw117.RGB_CACHE_MANIFEST


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json_once(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"preserving existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd = __import__("os").open(path, __import__("os").O_WRONLY | __import__("os").O_CREAT | __import__("os").O_EXCL, 0o600)
    with __import__("os").fdopen(fd, "wb") as stream:
        stream.write(payload); stream.flush(); __import__("os").fsync(stream.fileno())


def implementation_fingerprint():
    paths = [HERE / n for n in ("run.py", "protocol.json", "model.py", "adaptive_dynamics.py", "readout.py", "calibrate.py")]
    paths += [ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
              ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/run.py",
              ROOT / "collaborative_test/SW_0117_joint_analytic_rgb/run.py",
              ROOT / "collaborative_test/SW_0122_joint_rgb_seed_replication/run.py",
              ROOT / "collaborative_test/SW_0106_spike_partition_rgb/build_rgb_cache.py",
              ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
              ROOT / "snn_kuramoto_bidirectional/membrane_layer.py",
              ROOT / "snn_kuramoto_bidirectional/dendric_layer.py",
              ROOT / "snn_kuramoto_bidirectional/sinusoidal_gating.py"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths}


def source_contract(seed):
    if seed not in SEEDS:
        raise ValueError(f"unregistered source seed {seed}")
    if seed in (1, 2):
        source, manifest_path, manifest, ids, rows = sw122.source_contract(seed)
    else:
        source, manifest_path, manifest = base.source_paths(seed)
        ids, rows = base.train_indices(seed)
        if (len(ids) != 4096 or len(np.unique(ids)) != 4096
                or manifest.get("status") != "complete"
                or manifest.get("unique_images_seen") != 4096
                or manifest.get("steps") != 256 or manifest.get("batch") != 16
                or manifest.get("seed") != 117
                or manifest.get("training_ids") != ids.tolist()
                or manifest.get("ground_truth_used_for_training") is not False):
            raise AssertionError("seed0 SW0097 source manifest/order contract mismatch")
    if sha(source) != SOURCE_CORE_SHA[seed]:
        raise AssertionError(f"seed{seed} SW0097 source core SHA mismatch")
    return source, manifest_path, manifest, ids, rows


def load_training_assets(seed, device):
    source, source_manifest, source_record, ids, rows = source_contract(seed)
    if (sha(ENCODER_PATH) != base.EXPECTED_ENCODER_SHA256
            or sha(STATS_PATH) != base.EXPECTED_PREPROCESSING_SHA256):
        raise AssertionError("registered encoder/preprocessing SHA mismatch")
    # SW0117 and SW0122 have equivalent strict source/encoder loaders; using the
    # seed-specific SW0122 loader avoids the seed-0-only SW0115 manifest guard.
    if seed in (1, 2):
        core, encoder, patcher, mean, std, clip, *_ = sw122.load_models(seed, device)
    else:
        core, encoder, patcher, mean, std, clip, *_ = sw117.load_models(seed, device)
    core.requires_grad_(True)
    encoder.requires_grad_(True)
    return core, encoder, patcher, mean, std, clip, source, source_manifest, source_record, ids, rows


def load_registered_data():
    gamma, gamma_manifest = base.validate_gamma_cache(base.GAMMA_TRAIN, base.GAMMA_TRAIN_MANIFEST)
    rgb_cache, rgb_meta, rgb_sha = sw117.load_rgb_training_cache()
    if tuple(gamma.shape) != (70000, 8, 256):
        raise AssertionError(f"registered gamma cache shape mismatch: {tuple(gamma.shape)}")
    if rgb_cache.shape != (70000, 128, 128, 3) or rgb_cache.dtype != np.uint8:
        raise AssertionError("registered native-RGB training cache shape/dtype mismatch")
    return gamma, gamma_manifest, rgb_cache, rgb_meta, rgb_sha


def load_calibration(seed):
    path = ARCHIVE / f"calibration_seed{seed}_20261009.json"
    if not path.is_file():
        raise FileNotFoundError(f"passed SW0123 seed-{seed} calibration record required: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    source, manifest_path, _, ids, rows = source_contract(seed)
    if (record.get("status") != "passed" or record.get("experiment") != "SW0123"
            or record.get("seed") != seed or record.get("stage") != "train_calibration_activity"
            or record.get("source_core_sha256") != sha(source)
            or record.get("source_manifest_sha256") != sha(manifest_path)
            or record.get("source_training_ids") != ids.tolist()
            or record.get("gamma_rows") != rows.tolist()
            or record.get("ground_truth_used") is not False
            or record.get("optimizer_updates") != 0
            or record.get("encoder_sha256") != base.EXPECTED_ENCODER_SHA256
            or record.get("preprocessing_sha256") != base.EXPECTED_PREPROCESSING_SHA256):
        raise AssertionError("calibration evidence does not bind current source/order/assets")
    calibration_fingerprint = record.get("implementation_fingerprint")
    if not isinstance(calibration_fingerprint, dict) or not calibration_fingerprint:
        raise AssertionError("calibration implementation fingerprint is missing")
    for relative, expected_sha in calibration_fingerprint.items():
        dependency_path = ROOT / Path(relative)
        if not dependency_path.is_file() or sha(dependency_path) != expected_sha:
            raise AssertionError(f"calibration implementation dependency changed: {relative}")
    gamma, gamma_manifest = base.validate_gamma_cache(base.GAMMA_TRAIN, base.GAMMA_TRAIN_MANIFEST)
    if (record.get("gamma_train_sha256") != sha(base.GAMMA_TRAIN)
            or record.get("gamma_train_manifest_sha256") != sha(base.GAMMA_TRAIN_MANIFEST)
            or gamma_manifest.get("status") != record.get("gamma_manifest_status")):
        raise AssertionError("calibration cached-gamma provenance changed")
    kappa = torch.tensor(record.get("kappa"), dtype=torch.float32)
    rms = torch.tensor(record.get("spike_rms"), dtype=torch.float32)
    if (kappa.shape != (4,) or rms.shape != (4,) or not torch.isfinite(kappa).all()
            or not torch.isfinite(rms).all() or (kappa <= 0).any() or (rms <= 0).any()):
        raise AssertionError("calibration kappa/RMS buffers are invalid")
    if record.get("kappa_sha256") != hashlib.sha256(kappa.numpy().astype("<f4").tobytes()).hexdigest():
        raise AssertionError("calibration kappa byte SHA mismatch")
    if record.get("spike_rms_sha256") != hashlib.sha256(rms.numpy().astype("<f4").tobytes()).hexdigest():
        raise AssertionError("calibration RMS byte SHA mismatch")
    activity = record.get("adaptive_event_activity") or {}
    occ = np.asarray(activity.get("occupancy_by_component", []), dtype=np.float64)
    mixed = np.asarray(activity.get("mixed_image_patch_fraction_by_component", []), dtype=np.float64)
    if (occ.shape != (4,) or mixed.shape != (4,) or not np.isfinite(occ).all()
            or not np.isfinite(mixed).all() or not ((occ > .01) & (occ < .80)).all()
            or not (mixed >= .10).all()):
        raise AssertionError("calibration activity eligibility guard is not passed")
    return path, record, kappa, rms


def _new_core_model(seed, arm, device, kappa, rms, head_state=None, decoder_state=None):
    torch.manual_seed(12300 + seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(12300 + seed)
    model = AdaptiveTemporalRGBModel().to(device)
    with torch.no_grad():
        model.assignment_head.spike_rms.copy_(rms.to(device))
    if head_state is not None:
        model.assignment_head.load_state_dict(head_state, strict=True)
    if decoder_state is not None:
        model.rgb_decoder.load_state_dict(decoder_state, strict=True)
    return model


def configure_arm(core, arm, kappa):
    if arm == "adaptive_full":
        install_adaptive_dynamics(core, kappa.to(next(core.parameters()).device))
    elif arm not in ("legacy_full", "gate_only_control"):
        raise ValueError(f"unknown SW0123 arm {arm}")
    return core


def optimizer_pair(core, encoder, model, head_optimizer_state=None):
    core_opt = torch.optim.Adam([
        {"params": list(core.parameters()), "lr": CORE_LR},
        {"params": list(encoder.parameters()), "lr": ENCODER_LR},
    ])
    head_opt = torch.optim.Adam(list(model.assignment_head.parameters()) +
                                list(model.rgb_decoder.parameters()), lr=HEAD_LR)
    if head_optimizer_state is not None:
        head_opt.load_state_dict(head_optimizer_state)
    return core_opt, head_opt


def rollout(core, gamma, model, arm, patch_xy, capture_events=False, core_no_grad=False):
    captured_gate = []
    gate_hook = None
    if arm == "gate_only_control":
        if getattr(core, "spike_per_component", False) is not True:
            raise AssertionError("gate-only control requires the registered four-row component fold")
        gate_hook = core.membrane_layer.register_forward_pre_hook(
            lambda _module, args: captured_gate.append(args[1]))
    scope = torch.no_grad() if core_no_grad else contextlib.nullcontext()
    try:
        with scope:
            _groups, _mean_spikes, _mean_out, theta = core(
                gamma, return_core_out=True, return_theta=True)
    finally:
        if gate_hook is not None:
            gate_hook.remove()
    components = core.last_component_spikes
    if components is None or tuple(components.shape) != (gamma.shape[0], 4, 256, TIME_STEPS):
        raise AssertionError("S2Net did not return registered [B,4,256,64] actual spikes")
    if capture_events:
        if arm != "adaptive_full" or core.membrane_layer.event_history is None:
            raise AssertionError("binary event capture is only valid for adaptive_full")
        events = torch.stack(core.membrane_layer.event_history, dim=-1).reshape(
            gamma.shape[0], 4, 256, TIME_STEPS)[:, :, :, SETTLE:]
    else:
        events = None
    if arm == "gate_only_control":
        if len(captured_gate) != TIME_STEPS:
            raise AssertionError(f"recorded membrane gate has {len(captured_gate)} steps, expected {TIME_STEPS}")
        gate_rows = torch.stack(captured_gate, dim=-1)
        if gate_rows.shape != (gamma.shape[0] * 4, 256, TIME_STEPS):
            raise AssertionError("recorded membrane gate does not match folded [B*4,N,T] layout")
        gate = gate_rows.reshape(gamma.shape[0], 4, 256, TIME_STEPS)
        if not all(torch.equal(gate[:, 0], gate[:, d]) for d in range(1, 4)):
            raise AssertionError("actual scalar membrane gate differs across folded components")
        head_input = gate[:, :, :, SETTLE:]
    else:
        head_input = components[:, :, :, SETTLE:]
    if core_no_grad:
        head_input = head_input.detach()
    out = model(head_input, patch_xy)
    return out, components, theta, events


def normalized_rgb_loss(reconstruction, rgb):
    if reconstruction.shape != rgb.shape or rgb.ndim != 4:
        raise ValueError("RGB reconstruction/reference must share [B,3,H,W]")
    per_image_mse = (reconstruction - rgb).square().mean(dim=(1, 2, 3))
    variance = rgb.var(dim=(2, 3), unbiased=False).mean(dim=1).detach().clamp_min(1e-6)
    per_image = per_image_mse / variance
    return per_image.mean(), per_image


def _batch_data(seed, rows, start, gamma_cache, rgb_cache, encoder, patcher, mean, std, clip, device,
                check_source_cache=True):
    batch_rows = np.asarray(rows[start:start + BATCH], dtype=np.int64)
    if len(batch_rows) != BATCH:
        raise AssertionError("registered training batch is incomplete")
    images = sw117.read_rgb(rgb_cache, batch_rows, device)
    rgb = images / 255.0
    gamma = sw117.encode_rgb(encoder, patcher, mean, std, clip, images)
    if not torch.isfinite(gamma).all() or gamma.shape != (BATCH, 8, 256):
        raise AssertionError("live encoder gamma batch has invalid shape or values")
    if check_source_cache:
        cached = gamma_cache[torch.as_tensor(batch_rows, dtype=torch.long)].to(device)
        if gamma.shape != cached.shape:
            raise AssertionError("live gamma/cache batch shape mismatch")
        maxdiff = float((gamma.detach() - cached).abs().max())
        if not math.isfinite(maxdiff) or maxdiff > 2e-5:
            raise AssertionError(f"live gamma differs from registered source cache by {maxdiff}")
    return batch_rows, images, rgb, gamma


def _param_families(core, encoder, model):
    named = list(core.named_parameters())
    groups = {
        "encoder": list(encoder.parameters()),
        "graph": [p for n, p in named if n.startswith("graph_generator.")],
        "dendrite": list(core.dendric_layer.parameters()),
        "membrane": list(core.membrane_layer.parameters()),
        "head": list(model.assignment_head.parameters()),
        "decoder": list(model.rgb_decoder.parameters()),
    }
    # The gamma-to-oscillator drive is a distinct trainable route from the
    # graph. Record and guard its credit separately when this source exposes it.
    drive_names = {"gamma_channel_proj.weight", "gamma_channel_proj.bias", "gamma_phase_gain"}
    groups["oscillator_drive"] = [p for name, p in named if name in drive_names]
    groups["core"] = list(core.parameters())
    if any(not groups[k] for k in ("encoder", "graph", "dendrite", "membrane", "head", "decoder")):
        raise AssertionError("SW0123 expected trainable parameter family is empty")
    return groups


def _gradient_norm(loss, parameters, retain=True):
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain, allow_unused=True)
    squared = 0.0
    for grad in grads:
        if grad is not None:
            if not torch.isfinite(grad).all():
                raise FloatingPointError("nonfinite reconstruction gradient")
            squared += float(grad.detach().double().square().sum())
    return math.sqrt(squared), grads


def _state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _group_changed(before, module, prefixes=None):
    after = module.state_dict()
    keys = list(before) if prefixes is None else [k for k in before if any(k.startswith(p) for p in prefixes)]
    return any(not torch.equal(before[k], after[k]) for k in keys)


def _warmup_path(seed, arm):
    return ARCHIVE / f"warmup_seed{seed}_{arm}.pt"


def _calibration_path(seed):
    return ARCHIVE / f"calibration_seed{seed}_20261009.json"


def _atomic_torch_save_once(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"preserving existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _norm_and_grads(loss, parameters):
    parameters = list(parameters)
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    total = 0.0
    present = 0
    for param, grad in zip(parameters, grads):
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            raise FloatingPointError("nonfinite SW0123 reconstruction gradient")
        total += float(grad.detach().double().square().sum())
        present += 1
    return math.sqrt(total), present


def _batch_gradient_record(loss, core, encoder, model, arm):
    groups = _param_families(core, encoder, model)
    result = {}
    for group in ("encoder", "graph", "oscillator_drive", "dendrite", "membrane", "head", "decoder"):
        if group == "oscillator_drive" and not groups[group]:
            continue
        norm, present = _norm_and_grads(loss, groups[group])
        result[group] = {"norm": norm, "parameters_with_grad": present,
                         "parameters": len(groups[group])}
        if not math.isfinite(norm):
            raise FloatingPointError(f"nonfinite {group} gradient")
        expected = group not in ("dendrite", "membrane") or arm != "gate_only_control"
        if expected and norm <= 0:
            raise AssertionError(f"expected {group} reconstruction credit is zero for {arm}")
    return result


def _freeze(module, frozen):
    module.requires_grad_(not frozen)


def warmup_optimizer_step(core, gamma, model, arm, patch_xy, rgb, optimizer):
    """Update only the warm classifier/decoder from a detached source rollout."""
    before_core = [p.detach().clone() for p in core.parameters()]
    optimizer.zero_grad(set_to_none=True)
    out, _, _, _ = rollout(core, gamma, model, arm, patch_xy, core_no_grad=True)
    loss, _ = normalized_rgb_loss(out["reconstructed_image"], rgb)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite head warmup loss")
    loss.backward()
    params = list(model.assignment_head.parameters()) + list(model.rgb_decoder.parameters())
    norm = torch.nn.utils.clip_grad_norm_(params, CLIP)
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise FloatingPointError("head/decoder warmup gradient is invalid")
    optimizer.step()
    if any(not torch.equal(before, after) for before, after in zip(before_core, core.parameters())):
        raise AssertionError("head warmup unexpectedly changed source core parameters")
    return loss.detach(), norm.detach()


def _event_guard_from_batches(event_batches):
    events = torch.cat(event_batches, dim=0)
    if events.shape != (64, 4, 256, SETTLE):
        raise AssertionError(f"adaptive event guard expected [64,4,256,32], got {tuple(events.shape)}")
    if not torch.isfinite(events).all() or not torch.logical_or(events == 0, events == 1).all():
        raise FloatingPointError("adaptive binary events are nonfinite or nonbinary")
    occupancy = events.float().mean(dim=(0, 2, 3))
    mixed_fraction = (events.bool().any(dim=-1) & (~events.bool()).any(dim=-1)).float().mean(dim=(0, 2))
    record = {"occupancy_by_component": occupancy.detach().cpu().tolist(),
              "mixed_image_patch_fraction_by_component": mixed_fraction.detach().cpu().tolist()}
    if not ((occupancy > .01) & (occupancy < .80)).all():
        raise AssertionError(f"adaptive event occupancy failed: {record['occupancy_by_component']}")
    if not (mixed_fraction >= .10).all():
        raise AssertionError(f"adaptive event variation failed: {record['mixed_image_patch_fraction_by_component']}")
    return record


def preflight(seed, arm, output, device="cuda"):
    """Run live-RGB activity, head warmup, gradient-credit and two-Adam checks."""
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("unregistered SW0123 seed/arm")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserving existing preflight: {output}")
    warm_path = _warmup_path(seed, arm)
    if warm_path.exists():
        raise FileExistsError(f"preserving existing warmup artifact: {warm_path}")
    if not str(device).startswith("cuda"):
        raise ValueError("SW0123 real B16 preflight requires a CUDA device")
    calibration_path, calibration, kappa_cpu, rms_cpu = load_calibration(seed)
    fingerprint = implementation_fingerprint()
    (core, encoder, patcher, mean, std, clip, source, source_manifest, source_record,
     ids, rows) = load_training_assets(seed, device)
    gamma_cache, gamma_meta, rgb_cache, rgb_meta, rgb_sha = load_registered_data()
    model = _new_core_model(seed, arm, device, kappa_cpu, rms_cpu)
    configure_arm(core, arm, kappa_cpu)
    core.train(); core.graph_generator.train(); encoder.train(); model.train()
    loss_records = []

    # Evaluate the fixed live-encoder event eligibility before paying for head warmup.
    activity_record = None
    if arm == "adaptive_full":
        core.membrane_layer.capture_event_history = True
        event_batches = []
        _freeze(core, True); _freeze(encoder, True)
        with torch.no_grad():
            for bi in range(4):
                _, _, rgb, gamma = _batch_data(seed, rows, bi * BATCH, gamma_cache,
                                                rgb_cache, encoder, patcher, mean, std, clip, device)
                _out, _components, _theta, events = rollout(
                    core, gamma, model, arm, normalized_patch_centers(device=device), capture_events=True)
                event_batches.append(events)
        activity_record = _event_guard_from_batches(event_batches)
        core.membrane_layer.capture_event_history = False
        _freeze(core, False); _freeze(encoder, False)

    # The head/decoder are initialized identically across arms and warmed for
    # exactly 32 B16 TRAIN batches while source/encoder/new neuron params freeze.
    torch.manual_seed(12300 + seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(12300 + seed)
    model = _new_core_model(seed, arm, device, kappa_cpu, rms_cpu)
    model.train()
    head_opt = torch.optim.Adam(list(model.assignment_head.parameters()) +
                                list(model.rgb_decoder.parameters()), lr=HEAD_LR)
    _freeze(core, True); _freeze(encoder, True)
    warm_losses = []
    centers = normalized_patch_centers(device=device)
    for update in range(HEAD_WARMUP_UPDATES):
        start = update * BATCH
        _, _, rgb, gamma = _batch_data(seed, rows, start, gamma_cache, rgb_cache,
                                        encoder, patcher, mean, std, clip, device)
        loss, _head_norm = warmup_optimizer_step(core, gamma, model, arm, centers, rgb, head_opt)
        warm_losses.append(float(loss))
    _freeze(core, False); _freeze(encoder, False)
    core.train(); core.graph_generator.train(); encoder.train()
    warm_model_states = {"assignment_head": _state(model.assignment_head),
                         "rgb_decoder": _state(model.rgb_decoder),
                         "head_optimizer": copy.deepcopy(head_opt.state_dict())}

    # Four real TRAIN minibatches check the live reconstruction path and all
    # preregistered gradient families before a disposable optimizer update.
    gradient_records = []
    if arm == "adaptive_full":
        core.membrane_layer.capture_event_history = True
    for bi in range(4):
        start = bi * BATCH
        _, _, rgb, gamma = _batch_data(seed, rows, start, gamma_cache, rgb_cache,
                                        encoder, patcher, mean, std, clip, device)
        out, _components, _theta, _events = rollout(core, gamma, model, arm, centers)
        loss, _ = normalized_rgb_loss(out["reconstructed_image"], rgb)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite SW0123 preflight loss")
        gradient_records.append({"batch": bi, "image_ids": ids[start:start+BATCH].tolist(),
                                 "loss": float(loss.detach()),
                                 "gradients": _batch_gradient_record(loss, core, encoder, model, arm)})
        core.zero_grad(set_to_none=True); encoder.zero_grad(set_to_none=True); model.zero_grad(set_to_none=True)

    # Use independent copies and optimizer states for the actual two-Adam
    # throwaway update; the saved warmup weights/moments remain immutable.
    (tc, te, tp, tm, ts, tclip, *_rest) = load_training_assets(seed, device)
    configure_arm(tc, arm, kappa_cpu)
    throw_model = _new_core_model(seed, arm, device, kappa_cpu, rms_cpu,
                                  warm_model_states["assignment_head"],
                                  warm_model_states["rgb_decoder"])
    throw_core_opt, throw_head_opt = optimizer_pair(tc, te, throw_model,
                                                     copy.deepcopy(warm_model_states["head_optimizer"]))
    throw_model.train(); tc.train(); tc.graph_generator.train(); te.train()
    first_images = sw117.read_rgb(rgb_cache, rows[:BATCH], device)
    first_rgb = first_images / 255.0
    first_gamma = sw117.encode_rgb(te, tp, tm, ts, tclip, first_images)
    throw_out, _, _, _ = rollout(tc, first_gamma, throw_model, arm, centers)
    throw_loss, _ = normalized_rgb_loss(throw_out["reconstructed_image"], first_rgb)
    _batch_gradient_record(throw_loss, tc, te, throw_model, arm)
    before_core, before_encoder = _state(tc), _state(te)
    before_head, before_decoder = _state(throw_model.assignment_head), _state(throw_model.rgb_decoder)
    throw_core_opt.zero_grad(set_to_none=True); throw_head_opt.zero_grad(set_to_none=True)
    throw_loss.backward()
    core_params = list(tc.parameters()) + list(te.parameters())
    head_params = list(throw_model.assignment_head.parameters()) + list(throw_model.rgb_decoder.parameters())
    core_norm = torch.nn.utils.clip_grad_norm_(core_params, CLIP)
    head_norm = torch.nn.utils.clip_grad_norm_(head_params, CLIP)
    if (not torch.isfinite(throw_loss) or not torch.isfinite(core_norm) or not torch.isfinite(head_norm)
            or float(core_norm) <= 0 or float(head_norm) <= 0):
        raise FloatingPointError("invalid two-Adam disposable B16 update")
    throw_core_opt.step(); throw_head_opt.step()
    changed = {
        "encoder": _group_changed(before_encoder, te),
        "graph": _group_changed(before_core, tc, ("graph_generator.",)),
        "core": _group_changed(before_core, tc),
        "head": _group_changed(before_head, throw_model.assignment_head),
        "decoder": _group_changed(before_decoder, throw_model.rgb_decoder),
        "dendrite": _group_changed(before_core, tc, ("dendric_layer.",)),
        "membrane": _group_changed(before_core, tc, ("membrane_layer.",)),
    }
    drive_keys = tuple(f"{name}" for name, _ in tc.named_parameters()
                       if name in {"gamma_channel_proj.weight", "gamma_channel_proj.bias", "gamma_phase_gain"})
    if drive_keys:
        changed["oscillator_drive"] = any(
            not torch.equal(before_core[key], tc.state_dict()[key]) for key in drive_keys)
    expected_change = {key: True for key in ("encoder", "graph", "core", "head", "decoder")}
    if arm != "gate_only_control":
        expected_change.update(dendrite=True, membrane=True)
    else:
        expected_change.update(dendrite=False, membrane=False)
    if drive_keys:
        expected_change["oscillator_drive"] = True
    if any(changed[key] != expected for key, expected in expected_change.items()):
        raise AssertionError(f"disposable Adam parameter-change guard failed: {changed}")

    fingerprint = implementation_fingerprint()
    warm_record = {"status": "passed", "experiment": "SW0123", "seed": seed, "arm": arm,
                   "head_seed": 12300 + seed, "warmup_updates": HEAD_WARMUP_UPDATES,
                   "batch_size": BATCH, "training_ids": ids[:512].tolist(),
                   "assignment_head": warm_model_states["assignment_head"],
                   "rgb_decoder": warm_model_states["rgb_decoder"],
                   "head_optimizer": warm_model_states["head_optimizer"],
                   "source_core_sha256": sha(source), "encoder_sha256": sha(ENCODER_PATH),
                   "preprocessing_sha256": sha(STATS_PATH),
                   "calibration_sha256": sha(calibration_path),
                   "implementation_fingerprint": fingerprint,
                   "ground_truth_used": False}
    _atomic_torch_save_once(warm_path, warm_record)
    record = {"status": "passed", "experiment": "SW0123", "stage": "preflight",
              "seed": seed, "arm": arm, "source_core_sha256": sha(source),
              "source_manifest_sha256": sha(source_manifest),
              "calibration_sha256": sha(calibration_path),
              "gamma_train_sha256": sha(base.GAMMA_TRAIN),
              "gamma_train_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
              "rgb_cache_sha256": rgb_sha, "rgb_cache_manifest_sha256": sha(RGB_CACHE_MANIFEST),
              "encoder_sha256": sha(ENCODER_PATH), "preprocessing_sha256": sha(STATS_PATH),
              "training_ids": ids.tolist(),
              "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
              "gamma_rows": rows.tolist(), "matched_shuffle_seed": 117 + seed,
              "batch_size": BATCH, "time_steps": TIME_STEPS, "settle": SETTLE,
              "head_warmup_updates": HEAD_WARMUP_UPDATES,
              "head_warmup_losses": warm_losses,
              "warmup_artifact": str(warm_path), "warmup_artifact_sha256": sha(warm_path),
              "implementation_fingerprint": fingerprint,
              "adaptive_event_activity_live_encoder": activity_record,
              "gradient_batches": gradient_records,
              "throwaway_b16_two_adam_update": True,
              "throwaway_core_gradient_norm_preclip": float(core_norm),
              "throwaway_head_gradient_norm_preclip": float(head_norm),
              "throwaway_parameter_groups_changed": changed,
              "expected_parameter_groups_changed": expected_change,
              "ground_truth_used": False}
    write_json_once(output, record)
    return record


def train(seed, arm, output, device="cuda", steps=UPDATES):
    if seed not in SEEDS or arm not in ARMS or steps != UPDATES:
        raise ValueError("SW0123 training is fixed to registered seeds/arms and 256 updates")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserving existing training output: {output}")
    pf_path = ARCHIVE / f"preflight_seed{seed}_{arm}.json"
    warm_path = _warmup_path(seed, arm)
    if not pf_path.is_file() or not warm_path.is_file():
        raise FileNotFoundError("successful current preflight and saved warmup are required")
    pf = json.loads(pf_path.read_text(encoding="utf-8"))
    fingerprint = implementation_fingerprint()
    if (pf.get("status") != "passed" or pf.get("experiment") != "SW0123"
            or pf.get("seed") != seed or pf.get("arm") != arm
            or pf.get("implementation_fingerprint") != fingerprint
            or pf.get("warmup_artifact_sha256") != sha(warm_path)):
        raise AssertionError("SW0123 training requires the exact reviewed preflight/warmup artifacts")
    calibration_path, calibration, kappa, rms = load_calibration(seed)
    (core, encoder, patcher, mean, std, clip, source, source_manifest, source_record,
     ids, rows) = load_training_assets(seed, device)
    gamma_cache, gamma_manifest, rgb_cache, rgb_meta, rgb_sha = load_registered_data()
    expected_ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    if (pf.get("source_core_sha256") != sha(source)
            or pf.get("source_manifest_sha256") != sha(source_manifest)
            or pf.get("calibration_sha256") != sha(calibration_path)
            or pf.get("gamma_train_sha256") != sha(base.GAMMA_TRAIN)
            or pf.get("gamma_train_manifest_sha256") != sha(base.GAMMA_TRAIN_MANIFEST)
            or pf.get("rgb_cache_sha256") != rgb_sha
            or pf.get("rgb_cache_manifest_sha256") != sha(RGB_CACHE_MANIFEST)
            or pf.get("encoder_sha256") != sha(ENCODER_PATH)
            or pf.get("preprocessing_sha256") != sha(STATS_PATH)
            or pf.get("training_ids_sha256") != expected_ids_sha
            or pf.get("matched_shuffle_seed") != 117 + seed):
        raise AssertionError("SW0123 source/data/calibration/order changed after preflight")
    warm = torch.load(warm_path, map_location="cpu", weights_only=False)
    if (warm.get("status") != "passed" or warm.get("seed") != seed or warm.get("arm") != arm
            or warm.get("source_core_sha256") != sha(source)
            or warm.get("calibration_sha256") != sha(calibration_path)
            or warm.get("implementation_fingerprint") != fingerprint):
        raise AssertionError("warmup artifact provenance mismatch")
    configure_arm(core, arm, kappa)
    model = _new_core_model(seed, arm, device, kappa, rms,
                            warm["assignment_head"], warm["rgb_decoder"])
    core_opt, head_opt = optimizer_pair(core, encoder, model, warm["head_optimizer"])
    centers = normalized_patch_centers(device=device)
    torch.manual_seed(117 + seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(117 + seed)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "training", "experiment": "SW0123", "seed": seed, "arm": arm,
                "source_core": str(source), "source_core_sha256": sha(source),
                "source_manifest_sha256": sha(source_manifest),
                "preflight_sha256": sha(pf_path), "calibration_sha256": sha(calibration_path),
                "warmup_artifact_sha256": sha(warm_path),
                "implementation_fingerprint": fingerprint,
                "encoder_source_sha256": sha(ENCODER_PATH), "preprocessing_sha256": sha(STATS_PATH),
                "gamma_train_sha256": sha(base.GAMMA_TRAIN),
                "gamma_train_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
                "rgb_cache_sha256": rgb_sha, "rgb_cache_manifest_sha256": sha(RGB_CACHE_MANIFEST),
                "training_ids": ids.tolist(), "training_ids_sha256": expected_ids_sha,
                "gamma_rows": rows.tolist(), "matched_shuffle_seed": 117 + seed,
                "updates": UPDATES, "batch_size": BATCH, "time_steps": TIME_STEPS,
                "settle": SETTLE, "core_graph_lr": CORE_LR, "encoder_lr": ENCODER_LR,
                "head_decoder_lr": HEAD_LR, "clip_norm": CLIP,
                "head_warmup_updates": HEAD_WARMUP_UPDATES,
                "kappa": calibration["kappa"], "kappa_sha256": calibration["kappa_sha256"],
                "spike_rms": calibration["spike_rms"], "spike_rms_sha256": calibration["spike_rms_sha256"],
                "objective": "per-image full-resolution native-RGB MSE / detached max(mean channel pixel variance,1e-6)",
                "ground_truth_used_for_training": False}
    write_json_once(output / "manifest.json", manifest)
    history = []
    core.train(); core.graph_generator.train(); encoder.train(); model.train()
    for update in range(UPDATES):
        start = update * BATCH
        _batch_rows, _images, rgb, gamma = _batch_data(seed, rows, start, gamma_cache,
                                                        rgb_cache, encoder, patcher, mean, std, clip, device,
                                                        check_source_cache=False)
        prediction, _components, _theta, _events = rollout(core, gamma, model, arm, centers)
        loss, per_image = normalized_rgb_loss(prediction["reconstructed_image"], rgb)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite reconstruction objective at update {update + 1}")
        core_opt.zero_grad(set_to_none=True); head_opt.zero_grad(set_to_none=True)
        loss.backward()
        core_parameters = list(core.parameters()) + list(encoder.parameters())
        head_parameters = list(model.assignment_head.parameters()) + list(model.rgb_decoder.parameters())
        core_norm = torch.nn.utils.clip_grad_norm_(core_parameters, CLIP)
        head_norm = torch.nn.utils.clip_grad_norm_(head_parameters, CLIP)
        if (not torch.isfinite(core_norm) or float(core_norm) <= 0
                or not torch.isfinite(head_norm) or float(head_norm) <= 0):
            raise FloatingPointError(f"invalid optimizer gradient at update {update + 1}")
        core_opt.step(); head_opt.step()
        history.append({"update": update + 1, "rgb_loss": float(loss.detach()),
                        "rgb_loss_per_image_mean": float(per_image.detach().mean()),
                        "core_gradient_norm_preclip": float(core_norm),
                        "head_gradient_norm_preclip": float(head_norm)})
        if update == 0 or (update + 1) % 32 == 0:
            write_json_once(output / f"progress_{update + 1:04d}.json",
                            {"status": "training", "seed": seed, "arm": arm,
                             "update": update + 1, "total_updates": UPDATES})
    if any(not torch.isfinite(p).all() for p in list(core.parameters()) + list(encoder.parameters())
           + list(model.parameters())):
        raise FloatingPointError("nonfinite final SW0123 parameter")
    checkpoint_paths = {"core": output / "core.pt", "encoder": output / "encoder.pt",
                        "assignment_head": output / "assignment_head.pt",
                        "rgb_decoder": output / "rgb_decoder.pt",
                        "core_optimizer": output / "core_optimizer.pt",
                        "head_optimizer": output / "head_optimizer.pt",
                        "history": output / "history.json"}
    torch.save(core.state_dict(), checkpoint_paths["core"])
    torch.save(encoder.state_dict(), checkpoint_paths["encoder"])
    torch.save(model.assignment_head.state_dict(), checkpoint_paths["assignment_head"])
    torch.save(model.rgb_decoder.state_dict(), checkpoint_paths["rgb_decoder"])
    torch.save(core_opt.state_dict(), checkpoint_paths["core_optimizer"])
    torch.save(head_opt.state_dict(), checkpoint_paths["head_optimizer"])
    write_json_once(checkpoint_paths["history"], history)
    for name, path in checkpoint_paths.items():
        manifest[f"{name}_sha256"] = sha(path)
    training_guard = _post_train_guards(seed, arm, core, encoder, patcher, mean, std, clip,
                                       model, rows, gamma_cache, rgb_cache, device, kappa,
                                       normalized_patch_centers(device=device))
    manifest.update(status="training_complete", completed=time.time(), final_training_guards=training_guard)
    # Replace only our own in-progress manifest; the terminal marker is written last.
    tmp_manifest = output / "manifest.json.tmp"
    tmp_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp_manifest.replace(output / "manifest.json")
    (output / "TRAINING_COMPLETED").write_text("complete\n", encoding="utf-8")
    return manifest


@torch.no_grad()
def _post_train_guards(seed, arm, core, encoder, patcher, mean, std, clip, model,
                       rows, gamma_cache, rgb_cache, device, kappa, centers):
    core.eval(); core.graph_generator.eval(); encoder.eval(); model.eval()
    event_batches, real_losses, shuffled_losses = [], [], []
    if arm == "adaptive_full":
        core.membrane_layer.capture_event_history = True
    permutation = torch.as_tensor(np.random.default_rng(12301).permutation(256),
                                  dtype=torch.long, device=device)
    for batch in range(4):
        _, _, rgb, gamma = _batch_data(seed, rows, batch * BATCH, gamma_cache,
                                        rgb_cache, encoder, patcher, mean, std, clip, device,
                                        check_source_cache=False)
        out, _components, _theta, events = rollout(
            core, gamma, model, arm, centers, capture_events=(arm == "adaptive_full"))
        real_loss, _ = normalized_rgb_loss(out["reconstructed_image"], rgb)
        shuffled_probability = out["assignment"].index_select(1, permutation)
        shuffled_patches = torch.einsum("bnk,bknhwc->bnhwc", shuffled_probability,
                                        out["decoded_slot_patches"])
        shuffled_loss, _ = normalized_rgb_loss(image_from_patches(shuffled_patches), rgb)
        if not torch.isfinite(real_loss) or not torch.isfinite(shuffled_loss):
            return {"status": "failed_nonfinite_assignment_guard",
                    "assignment_shuffle_seed": 12301,
                    "real_reconstruction_loss_mean_first4": None,
                    "shuffled_reconstruction_loss_mean_first4": None,
                    "shuffled_minus_real_loss": None,
                    "assignment_credit_guard": "failed_nonfinite"}
        real_losses.append(float(real_loss))
        shuffled_losses.append(float(shuffled_loss))
        if events is not None:
            event_batches.append(events)
    result = {"assignment_shuffle_seed": 12301,
              "real_reconstruction_loss_mean_first4": float(np.mean(real_losses)),
              "shuffled_reconstruction_loss_mean_first4": float(np.mean(shuffled_losses)),
              "shuffled_minus_real_loss": float(np.mean(shuffled_losses) - np.mean(real_losses))}
    result["assignment_credit_guard"] = (
        "passed" if result["shuffled_minus_real_loss"] > 0 else "failed_nonpositive_shuffle_excess")
    if arm == "adaptive_full":
        try:
            result["adaptive_event_activity"] = _event_guard_from_batches(event_batches)
            result["activity_guard"] = "passed"
        except (AssertionError, FloatingPointError) as exc:
            result["adaptive_event_activity"] = {"failed": str(exc)}
            result["activity_guard"] = "failed"
        core.membrane_layer.capture_event_history = False
    result["status"] = ("passed" if result["assignment_credit_guard"] == "passed"
                        and result.get("activity_guard", "passed") == "passed" else "failed")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("preflight", "train"):
        sp = sub.add_parser(stage)
        sp.add_argument("--seed", type=int, choices=SEEDS, required=True)
        sp.add_argument("--arm", choices=ARMS, required=True)
        sp.add_argument("--device", required=True)
        sp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage == "preflight":
        record = preflight(args.seed, args.arm, args.output, args.device)
        summary = {"status": record["status"], "stage": args.stage,
                   "seed": args.seed, "arm": args.arm, "output": str(args.output)}
    else:
        record = train(args.seed, args.arm, args.output, args.device)
        summary = {"status": record["status"], "stage": args.stage,
                   "seed": args.seed, "arm": args.arm, "output": str(args.output)}
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()




