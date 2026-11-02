"""SW0134 source-bound preflight and training entry point.

The native source, SW0130 integration adapter, and production loss/readout code
remain unchanged. SW0134 trains a spike-history binder and RGB decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"),
                str(ROOT / "collaborative_test")]

from collaborative_test.SW_0110_xy_graph_route import run as source97
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0134_native_spike_binding.binder import (
    NativeSpikeSlotBinder, RelativeSlotRGBDecoder, reconstruct_from_spikes,
)
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from snn_kuramoto_bidirectional.loss_function import phase_locking_value
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity

SEEDS = (0, 1, 2)
ARMS = ("actual_joint", "gate_joint", "actual_frozen")
BATCH = 16
TRAIN_STEPS = 1024
TRAIN_SETTLE = 512
TAIL = 64
WARMUP_UPDATES = 32
WARMUP_LR = 3e-4
CORE_LR = 3e-5
ENCODER_LR = 3e-6
HEAD_LR = 3e-4
CLIP = 1.0
ARCHIVE = HERE / "results_archive"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def implementation_fingerprint():
    files = dict(sw130.implementation_fingerprint())
    for path in (HERE / "binder.py", HERE / "rollout.py", HERE / "run.py",
                 HERE / "protocol.json"):
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    return files


def source_contract(seed):
    result = sw130.source_contract(seed)
    checkpoint, manifest_path, manifest, pool, ids, source_sha = result
    if manifest.get("steps") != 256 or manifest.get("source_model_seed") != seed:
        raise AssertionError("SW0134 requires the completed registered SW0097 source")
    return result


def load_models(seed, device):
    # Load and strict-check the immutable source first, then add SW0130's
    # opt-in integration parameters. All three SW0134 arms use the same phase
    # recurrence; gate_joint changes only what the binder receives.
    wrapped, encoder, patcher, mean, std, clip, _legacy_decoder, pool, ids, source_sha = \
        sw130.load_models(seed, device, "phase")
    binder = NativeSpikeSlotBinder(seed=134).to(device)
    decoder = RelativeSlotRGBDecoder(seed=106).to(device)
    return wrapped, encoder, patcher, mean, std, clip, binder, decoder, pool, ids, source_sha


def _batch_forward(wrapped, encoder, patcher, mean, std, clip, binder, decoder,
                   images, arm, *, live_tail_steps=TAIL, checkpoint_chunks=True):
    gamma = sw130.encode(encoder, patcher, mean, std, clip, images)
    trace = late_rollout(wrapped, gamma, total_steps=TRAIN_STEPS,
                         live_tail_steps=live_tail_steps)
    head_trace = trace["component_spikes"][..., -TRAIN_SETTLE:]
    if arm == "gate_joint":
        head_trace = trace["component_gates"][..., -TRAIN_SETTLE:]
    prediction, assignment, slots, _features, labels = reconstruct_from_spikes(
        head_trace, binder, decoder, chunk_size=1024,
        checkpoint_chunks=checkpoint_chunks)
    components = trace["component_spikes"]
    q = spike_synchrony_affinity(components.mean(dim=1), components,
                                 settle=TRAIN_SETTLE, affinity_mode="spike")
    plv = phase_locking_value(trace["theta"], settle=TRAIN_SETTLE)
    criterion = sw130.make_criterion()
    primary, _ = criterion(plv=plv, theta=trace["theta"])
    q_loss, _ = criterion(plv=q)
    old = primary + 5.0 * q_loss
    target = images.permute(0, 2, 3, 1).contiguous() / 255.0
    rgb_per_image = (prediction - target).square().mean(dim=(1, 2, 3))
    rgb = rgb_per_image.mean()
    return {"gamma": gamma, "trace": trace, "q": q, "plv": plv,
            "old": old, "rgb": rgb, "rgb_per_image": rgb_per_image,
            "prediction": prediction, "assignment": assignment, "slots": slots,
            "labels": labels, "target": target, "head_trace": head_trace}


def _unique_params(named):
    params = [parameter for _name, parameter in named]
    if len({id(value) for value in params}) != len(params):
        raise AssertionError("SW0134 joint gradient parameters contain duplicates")
    return params


def _grad_norm(grads):
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            raise FloatingPointError("nonfinite SW0134 gradient")
        total += float(grad.detach().double().square().sum())
    return math.sqrt(total)


def _save_torch_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(value, stream)
    return sha(path)


def _load_rgb_assets():
    assets = sw130.validate_rgb_assets()
    gamma_cache, gamma_manifest = source97.validate_gamma_cache(
        source97.GAMMA_TRAIN, source97.GAMMA_TRAIN_MANIFEST)
    return assets, gamma_cache, gamma_manifest


def _write_warmup(seed, arm, device, pool, ids, assets, train_cache, output_dir):
    wrapped, encoder, patcher, mean, std, clip, binder, decoder, *_ = load_models(seed, device)
    # The native encoder/core are frozen during head warmup. The actual and
    # frozen-backbone arms share the same actual-spike warm state; the gate arm
    # receives its own equally sized warmup artifact.
    wrapped.eval(); encoder.eval(); binder.train(); decoder.train()
    named_head = list(binder.named_parameters()) + [
        (f"decoder.{name}", value) for name, value in decoder.named_parameters()]
    head_params = _unique_params(named_head)
    optimizer = torch.optim.Adam(head_params, lr=WARMUP_LR)
    losses = []
    for step in range(WARMUP_UPDATES):
        rows = pool[step * BATCH:(step + 1) * BATCH].tolist()
        images = sw130.read_batch(train_cache, rows, device)
        with torch.no_grad():
            gamma = sw130.encode(encoder, patcher, mean, std, clip, images)
            trace = late_rollout(wrapped, gamma, total_steps=TRAIN_STEPS,
                                 live_tail_steps=0)
            head_trace = trace["component_spikes"][..., -TRAIN_SETTLE:]
            if arm == "gate_joint":
                head_trace = trace["component_gates"][..., -TRAIN_SETTLE:]
            target = images.permute(0, 2, 3, 1).contiguous() / 255.0
        optimizer.zero_grad(set_to_none=True)
        # Recompute head with gradients from the frozen full rollout traces.
        prediction, *_ = reconstruct_from_spikes(head_trace.detach(), binder, decoder,
                                                 chunk_size=1024,
                                                 checkpoint_chunks=True)
        loss = (prediction - target).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite SW0134 head warmup loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_params, CLIP)
        optimizer.step()
        losses.append(float(loss.detach()))
    path = Path(output_dir) / f"warm_{arm}_seed{seed}.pt"
    _checkpoint, manifest_path, _manifest, _pool, _ids, source_sha = source_contract(seed)
    payload = {"experiment": "SW0134_native_spike_binding", "seed": seed,
               "arm": arm, "updates": WARMUP_UPDATES, "batch_size": BATCH,
               "training_ids": [int(v) for v in ids[:512]],
               "training_ids_sha256": hashlib.sha256(np.asarray(
                   ids[:512], dtype="<i8").tobytes()).hexdigest(),
               "all_training_ids_sha256": hashlib.sha256(
                   np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
               "source_manifest_sha256": sha(manifest_path),
               "binder_state_dict": binder.state_dict(),
               "decoder_state_dict": decoder.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(),
               "loss_first_last": [losses[0], losses[-1]],
               "source_core_sha256": source_sha,
               "asset_hashes": assets,
               "implementation_fingerprint": implementation_fingerprint()}
    return path, _save_torch_once(path, payload), payload


def _read_warm(path, device, binder, decoder, *, seed, arm, ids, assets):
    warm = torch.load(path, map_location=device, weights_only=True)
    source = source_contract(seed)
    expected_ids = [int(v) for v in ids[:512]]
    expected_id_sha = hashlib.sha256(np.asarray(ids[:512], dtype="<i8").tobytes()).hexdigest()
    if (warm.get("updates") != WARMUP_UPDATES or warm.get("batch_size") != BATCH
            or warm.get("experiment") != "SW0134_native_spike_binding"
            or warm.get("seed") != seed or warm.get("arm") != arm
            or warm.get("training_ids") != expected_ids
            or warm.get("training_ids_sha256") != expected_id_sha
            or warm.get("all_training_ids_sha256") != hashlib.sha256(
                np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
            or warm.get("source_core_sha256") != source97.EXPECTED_SOURCE_SHAS[seed]
            or warm.get("source_manifest_sha256") != sha(source[1])
            or warm.get("asset_hashes") != assets
            or warm.get("implementation_fingerprint") != implementation_fingerprint()):
        raise AssertionError("registered SW0134 warmup artifact contract mismatch")
    states = warm.get("optimizer_state_dict", {}).get("state", {})
    steps = []
    for state in states.values():
        step = state.get("step")
        steps.append(int(step.item() if torch.is_tensor(step) else step))
    if len(steps) != len(list(binder.parameters()) + list(decoder.parameters())) \
            or set(steps) != {WARMUP_UPDATES}:
        raise AssertionError("warm head Adam must contain one step-32 state per parameter")
    for value in list(warm["binder_state_dict"].values()) + list(warm["decoder_state_dict"].values()):
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise FloatingPointError("warm head contains nonfinite tensors")
    binder.load_state_dict(warm["binder_state_dict"], strict=True)
    decoder.load_state_dict(warm["decoder_state_dict"], strict=True)
    return warm


def _calibrate_lambda(seed, device, pool, train_cache, warm_path, assets):
    wrapped, encoder, patcher, mean, std, clip, binder, decoder, *_ = load_models(seed, device)
    warm = _read_warm(warm_path, device, binder, decoder, seed=seed,
                      arm="actual_joint", ids=sw130.source_contract(seed)[4],
                      assets=assets)
    named, groups = sw130.joint_parameters(wrapped, encoder)
    params = _unique_params(named)
    records, ratios = [], []
    for index in range(4):
        rows = pool[index * BATCH:(index + 1) * BATCH].tolist()
        images = sw130.read_batch(train_cache, rows, device)
        out = _batch_forward(wrapped, encoder, patcher, mean, std, clip,
                             binder, decoder, images, "actual_joint")
        old_grads = torch.autograd.grad(out["old"], params, retain_graph=True,
                                        allow_unused=True)
        rgb_grads = torch.autograd.grad(out["rgb"], params, retain_graph=False,
                                        allow_unused=True)
        old_norm, rgb_norm = _grad_norm(old_grads), _grad_norm(rgb_grads)
        if not all(math.isfinite(v) and v > 0 for v in (old_norm, rgb_norm)):
            raise AssertionError("SW0134 lambda calibration requires finite nonzero old/R gradients")
        ratio = 0.25 * old_norm / rgb_norm
        ratios.append(ratio)
        records.append({"batch_index": index, "old_joint_norm": old_norm,
                        "rgb_joint_norm": rgb_norm, "lambda_ratio": ratio})
    return float(statistics.median(ratios)), records


def _validate_seed0_lambda(output_path, seed, assets, ids):
    if seed == 0:
        return None
    record_path = Path(output_path).parent / "preflight_seed0.json"
    if not record_path.is_file():
        raise RuntimeError("seed1/2 requires completed seed0 SW0134 calibration")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    source0 = source_contract(0)
    seed0_ids = source0[4]
    seed0_ids_sha = hashlib.sha256(np.asarray(seed0_ids, dtype="<i8").tobytes()).hexdigest()
    if (record.get("status") != "passed" or record.get("seed") != 0
            or record.get("experiment") != "SW0134_native_spike_binding"
            or record.get("implementation_fingerprint") != implementation_fingerprint()
            or record.get("source_core_sha256") != source97.EXPECTED_SOURCE_SHAS[0]
            or record.get("source_manifest_sha256") != sha(source0[1])
            or record.get("training_ids") != seed0_ids
            or record.get("training_ids_sha256") != seed0_ids_sha
            or record.get("pool_indices_sha256") != hashlib.sha256(
                np.asarray(source0[3], dtype="<i8").tobytes()).hexdigest()
            or record.get("asset_hashes") != assets):
        raise RuntimeError("seed0 SW0134 lambda provenance mismatch")
    rows = record.get("lambda_calibration")
    if not isinstance(rows, list) or len(rows) != 4:
        raise RuntimeError("seed0 SW0134 lambda calibration must contain four batches")
    ratios = []
    for index, row in enumerate(rows):
        old = float(row["old_joint_norm"]); rgb = float(row["rgb_joint_norm"])
        ratio = float(row["lambda_ratio"])
        if (row.get("batch_index") != index or not all(math.isfinite(v) and v > 0
                for v in (old, rgb, ratio)) or not math.isclose(ratio, .25 * old / rgb,
                                                                rel_tol=1e-12, abs_tol=0.0)):
            raise RuntimeError("seed0 SW0134 lambda ratios do not match recorded gradients")
        ratios.append(ratio)
    value = float(record.get("lambda", float("nan")))
    if not math.isfinite(value) or value <= 0 or value != float(statistics.median(ratios)):
        raise RuntimeError("seed0 SW0134 lambda is not the exact recorded median")
    warm = record.get("warm_artifacts", {}).get("actual_joint", {})
    warm_path = Path(warm.get("path", ""))
    if not warm_path.is_file() or sha(warm_path) != warm.get("sha256"):
        raise RuntimeError("seed0 actual-spike warm artifact is missing or changed")
    warm_payload = torch.load(warm_path, map_location="cpu", weights_only=True)
    expected_warm_ids = [int(v) for v in seed0_ids[:512]]
    if (warm_payload.get("updates") != WARMUP_UPDATES
            or warm_payload.get("training_ids") != expected_warm_ids
            or warm_payload.get("source_core_sha256") != source97.EXPECTED_SOURCE_SHAS[0]
            or warm_payload.get("source_manifest_sha256") != sha(source0[1])
            or warm_payload.get("asset_hashes") != assets
            or warm_payload.get("implementation_fingerprint") != implementation_fingerprint()):
        raise RuntimeError("seed0 actual-spike warm artifact provenance mismatch")
    return sha(record_path)


@torch.no_grad()
def _zero_init_late_parity(seed, gamma, device):
    checkpoint = source_contract(seed)[0]
    source_state = torch.load(checkpoint, map_location=device, weights_only=True)
    rows = []
    for steps, settle in ((64, 32), (1024, 512)):
        core = source97.make_core(device, steps).to(device)
        core.load_state_dict(source_state, strict=True)
        core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
        native = core(gamma, return_core_out=True, return_theta=True, num_time_steps=steps)
        native_components = core.last_component_spikes.detach().clone()
        native_membrane = core.last_component_out.detach().clone()
        native_q = spike_synchrony_affinity(native_components.mean(dim=1), native_components,
                                            settle=settle, affinity_mode="spike")
        native_h, _native_masks, _native_groups = sw130.hard_labels(
            native[1], native_components, settle)
        criterion = sw130.make_criterion()
        native_plv = phase_locking_value(native[3], settle=settle)
        native_primary, _ = criterion(plv=native_plv, theta=native[3])
        native_q_loss, _ = criterion(plv=native_q)
        native_old = native_primary + 5.0 * native_q_loss
        for arm in ("phase", "constant"):
            candidate = source97.make_core(device, steps).to(device)
            candidate.load_state_dict(source_state, strict=True)
            candidate._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
            wrapped = PhaseStateIntegration(candidate, arm)
            full = wrapped(gamma, return_core_out=True, return_theta=True,
                           num_time_steps=steps, capture_state=True)
            full_components = candidate.last_component_spikes.detach().clone()
            full_membrane = candidate.last_component_out.detach().clone()
            full_gates = candidate.last_gate_history.detach().clone()
            actual = late_rollout(wrapped, gamma, total_steps=steps, live_tail_steps=0)
            if (not torch.equal(actual["component_spikes"], native_components)
                    or not torch.equal(actual["component_membrane"], native_membrane)
                    or not torch.equal(actual["theta"], native[3])
                    or not torch.equal(actual["component_spikes"], full_components)
                    or not torch.equal(actual["component_membrane"], full_membrane)
                    or not torch.equal(actual["theta"], full[3])):
                raise AssertionError(f"SW0134 zero-init full trace parity failed: {arm}, T={steps}")
            if not torch.equal(actual["component_gates"], full_gates):
                raise AssertionError(f"SW0134 native gate parity failed: {arm}, T={steps}")
            q0 = spike_synchrony_affinity(native_components.mean(dim=1), native_components,
                                          settle=settle, affinity_mode="spike")
            q1 = spike_synchrony_affinity(actual["component_spikes"].mean(dim=1),
                                          actual["component_spikes"], settle=settle,
                                          affinity_mode="spike")
            if not torch.equal(q0, q1):
                raise AssertionError(f"SW0134 actual-Q parity failed: {arm}, T={steps}")
            candidate_h, _candidate_masks, _candidate_groups = sw130.hard_labels(
                full[1], full_components, settle)
            actual_plv = phase_locking_value(actual["theta"], settle=settle)
            actual_primary, _ = criterion(plv=actual_plv, theta=actual["theta"])
            actual_q_loss, _ = criterion(plv=q1)
            if (not torch.equal(candidate_h, native_h)
                    or not torch.equal(actual_primary + 5.0 * actual_q_loss, native_old)):
                raise AssertionError(f"SW0134 initial hard-H/old-loss parity failed: {arm}, T={steps}")
            rows.append({"arm": arm, "steps": steps, "settle": settle,
                         "full_component_trace_gate_q_h_oldloss_exact": True,
                         "production_adapter_gate_trace_exact": True})
    return rows


def preflight(seed, device, output):
    if seed not in SEEDS:
        raise ValueError("registered SW0134 seed must be 0, 1, or 2")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0134 preflight: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    assets, gamma_cache, gamma_manifest = _load_rgb_assets()
    source_path, source_manifest_path, source_manifest, pool, ids, source_sha = source_contract(seed)
    if len(ids) != 4096 or len(set(ids)) != len(ids):
        raise AssertionError("SW0134 requires the registered 4096 unique TRAIN IDs")
    seed0_lambda_sha = _validate_seed0_lambda(output, seed, assets, ids)
    train_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r")
    warm_artifacts = {}
    try:
        # Exact source/cache/live-encoder binding on the first registered batch.
        wrapped, encoder, patcher, mean, std, clip, binder, decoder, loaded_pool, loaded_ids, loaded_sha = \
            load_models(seed, device)
        if (loaded_sha != source_sha or loaded_ids != ids
                or not np.array_equal(loaded_pool, pool)):
            raise AssertionError("SW0134 loader differs from the registered source/order")
        first_rows = pool[:BATCH].tolist()
        first_images = sw130.read_batch(train_cache, first_rows, device)
        with torch.no_grad():
            cached = gamma_cache[torch.as_tensor(pool[:BATCH], dtype=torch.long)].to(device)
            live = sw130.encode(encoder, patcher, mean, std, clip, first_images)
            gamma_diff = float((live - cached).abs().max())
        if not math.isfinite(gamma_diff) or gamma_diff > 2e-5:
            raise AssertionError(f"SW0134 live RGB gamma differs from registered cache: {gamma_diff}")
        parity = _zero_init_late_parity(seed, cached, device)

        # Actual-spike and native-gate heads get equal 32-update warmups.
        # The actual-spike state is reused unchanged by actual_frozen.
        for arm in ("actual_joint", "gate_joint"):
            warm_path, warm_sha, warm = _write_warmup(seed, arm, device, pool, ids,
                                                       assets, train_cache, output.parent)
            warm_artifacts[arm] = {"path": str(warm_path.resolve()), "sha256": warm_sha,
                                   "loss_first_last": warm["loss_first_last"],
                                   "updates": WARMUP_UPDATES}
        warm_path = Path(warm_artifacts["actual_joint"]["path"])
        if seed == 0:
            fixed_lambda, calibration = _calibrate_lambda(seed, device, pool,
                                                            train_cache, warm_path, assets)
        else:
            record_path = output.parent / "preflight_seed0.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            fixed_lambda = float(record["lambda"])
            calibration = "reused immutable seed0 SW0134 first-four-batch calibration"

        # A fresh, real optimizer update on the registered 16-image batch
        # checks that warm optimizer/model states load and the RGB objective is live.
        disposable = []
        for arm in ARMS:
            disposable.append(_disposable_update(seed, arm, device, pool,
                                                  train_cache, fixed_lambda,
                                                  warm_artifacts, assets))
        report = {
            "status": "passed", "experiment": "SW0134_native_spike_binding",
            "seed": seed, "device": str(device), "created_unix": time.time(),
            "implementation_fingerprint": implementation_fingerprint(),
            "source_core_sha256": source_sha,
            "source_manifest_sha256": sha(source_manifest_path),
            "source_updates": int(source_manifest["steps"]),
            "training_ids": ids,
            "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
            "pool_indices_sha256": hashlib.sha256(np.asarray(pool, dtype="<i8").tobytes()).hexdigest(),
            "shuffle_seed": 117 + seed, "batch_size": BATCH,
            "train_time_steps": TRAIN_STEPS, "settle_steps": TRAIN_SETTLE,
            "live_tail_steps": TAIL, "asset_hashes": assets,
            "gamma_cache_sha256": gamma_manifest["gamma_sha256"],
            "gamma_manifest_sha256": sha(source97.GAMMA_TRAIN_MANIFEST),
            "live_gamma_cache_max_abs_diff_first_batch": gamma_diff,
            "native_zero_init_late_parity": parity,
            "warm_artifacts": warm_artifacts,
            "warm_actual_arms_shared": True,
            "lambda_source_seed": 0, "lambda": fixed_lambda,
            "lambda_seed0_record_sha256": seed0_lambda_sha,
            "lambda_calibration": calibration,
            "disposable_updates": disposable,
            "joint_training_updates": 0, "warmup_optimizer_updates_per_head": WARMUP_UPDATES,
            "throwaway_optimizer_updates_per_arm": 1,
            "ground_truth_used": False,
            "source_checkpoint_modified": False,
        }
        write_once(output, report)
        return report
    except Exception as exc:
        # Preserve a terminal failure receipt without rewriting any prior attempt.
        failure_path = output.with_name(output.stem + ".failure.json")
        if not failure_path.exists():
            write_once(failure_path, {"status": "failed_preflight", "seed": seed,
                "experiment": "SW0134_native_spike_binding", "error": repr(exc),
                "implementation_fingerprint": implementation_fingerprint(),
                "created_unix": time.time(), "ground_truth_used": False})
        raise


def _disposable_update(seed, arm, device, pool, train_cache, fixed_lambda,
                       warm_artifacts, assets):
    wrapped, encoder, patcher, mean, std, clip, binder, decoder, *_ = load_models(seed, device)
    state_key = "gate_joint" if arm == "gate_joint" else "actual_joint"
    ids = sw130.source_contract(seed)[4]
    warm = _read_warm(warm_artifacts[state_key]["path"], device, binder, decoder,
                      seed=seed, arm=state_key, ids=ids, assets=assets)
    named, groups = sw130.joint_parameters(wrapped, encoder)
    joint = _unique_params(named)
    head = list(binder.parameters()) + list(decoder.parameters())
    images = sw130.read_batch(train_cache, pool[:BATCH].tolist(), device)
    if arm == "actual_frozen":
        # The frozen arm must not update running buffers or activate dropout in
        # its source backbone, even though gradients still flow through the
        # forward pass into the trainable binder/decoder.
        wrapped.eval(); encoder.eval()
        wrapped.requires_grad_(False); encoder.requires_grad_(False)
        named = [(n, p) for n, p in named if p.requires_grad]
        groups = []
        joint = []
    else:
        wrapped.train(); encoder.train()
    binder.train(); decoder.train()
    frozen_source_before = ({k: v.detach().clone() for k, v in wrapped.state_dict().items()},
                            {k: v.detach().clone() for k, v in encoder.state_dict().items()})
    out = _batch_forward(wrapped, encoder, patcher, mean, std, clip,
                         binder, decoder, images, arm)
    total = out["old"] + float(fixed_lambda) * out["rgb"]
    if not torch.isfinite(total):
        raise FloatingPointError(f"nonfinite {arm} disposable total")
    head_optimizer = torch.optim.Adam(head, lr=HEAD_LR)
    head_optimizer.load_state_dict(warm["optimizer_state_dict"])
    head_grads = torch.autograd.grad(out["rgb"], head, retain_graph=True, allow_unused=True)
    if _grad_norm(head_grads) <= 0:
        raise AssertionError(f"{arm} RGB gradient to binder/decoder is empty")
    if arm == "actual_frozen":
        joint_grads = ()
        joint_optimizer = None
        rgb_family_norms = {}
    else:
        rgb_grads = torch.autograd.grad(out["rgb"], joint, retain_graph=True,
                                        allow_unused=True)
        rgb_family_norms = sw130.family_norm(named, rgb_grads)
        if any(not math.isfinite(value) for value in rgb_family_norms.values()):
            raise FloatingPointError(f"{arm} nonfinite RGB gradient family")
        if arm == "actual_joint":
            required = ("encoder", "graph", "oscillator_drive", "kuramoto",
                        "dendrite", "membrane", "a_d", "a_m", "b")
            missing = {key: rgb_family_norms.get(key, 0.0) for key in required
                       if rgb_family_norms.get(key, 0.0) <= 0.0}
            if missing:
                raise AssertionError(f"actual-spike RGB gradient lacks registered families: {missing}")
        if arm == "gate_joint":
            downstream = ("dendrite", "membrane", "a_d", "a_m", "b")
            leaked = {key: rgb_family_norms.get(key, 0.0) for key in downstream
                      if rgb_family_norms.get(key, 0.0) != 0.0}
            if leaked:
                raise AssertionError(f"native gate input received downstream RGB credit: {leaked}")
        joint_optimizer = torch.optim.Adam(groups)
        joint_grads = torch.autograd.grad(total, joint, retain_graph=False, allow_unused=True)
        if _grad_norm(joint_grads) <= 0:
            raise AssertionError(f"{arm} joint gradient is empty")
    before_head = [p.detach().clone() for p in head]
    before_joint = [p.detach().clone() for p in joint]
    for p, grad in zip(head, head_grads):
        p.grad = None if grad is None else grad.detach().clone()
    torch.nn.utils.clip_grad_norm_(head, CLIP)
    if joint_optimizer is not None:
        for p, grad in zip(joint, joint_grads):
            p.grad = None if grad is None else grad.detach().clone()
        torch.nn.utils.clip_grad_norm_(joint, CLIP)
    head_optimizer.step()
    if joint_optimizer is not None:
        joint_optimizer.step()
        wrapped.project_integrations_()
    head_changed = sum(not torch.equal(a, b) for a, b in zip(before_head, head))
    joint_changed = sum(not torch.equal(a, b) for a, b in zip(before_joint, joint))
    if not head_changed or (joint_optimizer is not None and not joint_changed):
        raise AssertionError(f"{arm} disposable step did not update registered parameters")
    if any(not torch.isfinite(p).all() for p in list(wrapped.parameters()) +
           list(encoder.parameters()) + list(binder.parameters()) + list(decoder.parameters())):
        raise FloatingPointError(f"{arm} disposable update produced nonfinite parameters")
    source_unchanged = (
        all(torch.equal(value, wrapped.state_dict()[key])
            for key, value in frozen_source_before[0].items()) and
        all(torch.equal(value, encoder.state_dict()[key])
            for key, value in frozen_source_before[1].items()))
    if arm == "actual_frozen" and not source_unchanged:
        raise AssertionError("actual_frozen disposable update modified source parameters")
    return {"arm": arm, "old_loss": float(out["old"].detach()),
            "rgb_loss": float(out["rgb"].detach()),
            "head_gradient_norm": _grad_norm(head_grads),
            "joint_gradient_norm": _grad_norm(joint_grads),
            "rgb_gradient_norms_by_source_family": rgb_family_norms,
            "head_changed_parameter_count": int(head_changed),
            "joint_changed_parameter_count": int(joint_changed),
            "source_frozen": arm == "actual_frozen",
            "source_training_mode": "eval" if arm == "actual_frozen" else "train",
            "source_parameters_unchanged": source_unchanged,
            "throwaway_only": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight",), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = preflight(args.seed, torch.device(args.device), args.output)
    print(json.dumps({"status": report["status"], "stage": args.stage,
                      "seed": args.seed, "output": str(args.output),
                      "lambda": report["lambda"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
