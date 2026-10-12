"""One disposable native32 boundary-Q credit and source-Q preservation probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"), str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0135_native32_spike_binding.foundation import late_rollout
from collaborative_test.SW_0142_native32_masked_completion import run as sw142
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity

NAME = "SW0145_native32_boundary_credit_probe"
SEED = 0
GRID, PATCH, IMAGE, N = 32, 4, 128, 1024
STEPS, SETTLE, TAIL = 1024, 512, 64
LOGICAL_BATCH, MICRO, SCALE_ROWS = 16, 4, 4096
GRAD_BATCHES = 4
OUTPUT = HERE / "results_archive" / "source0_boundary_credit_probe.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_validated_asset_record(provenance):
    record = provenance.get("rgb_asset_validation") if isinstance(provenance, dict) else None
    if not isinstance(record, dict) or not record.get("train_cache_sha256"):
        raise ValueError("native32 foundation was not loaded with verified RGB assets")
    return record


def boundary_pairs():
    pairs = []
    for row in range(GRID):
        for col in range(GRID - 1):
            pairs.append((row * GRID + col, row * GRID + col + 1))
    for row in range(GRID - 1):
        for col in range(GRID):
            pairs.append((row * GRID + col, (row + 1) * GRID + col))
    return np.asarray(pairs, dtype=np.int64)


def boundary_energy(rgb_uint8):
    """Compute mean pixel-pair MSE along every adjacent native32 patch seam."""
    image = np.asarray(rgb_uint8)
    if image.ndim != 4 or image.shape[1:] != (IMAGE, IMAGE, 3) or image.dtype != np.uint8:
        raise ValueError("boundary targets require NHWC uint8 native128 RGB")
    patches = image.reshape(len(image), GRID, PATCH, GRID, PATCH, 3)
    # One row of four RGB pixel pairs per neighboring patch seam.
    right_diff = (patches[:, :, :, :-1, -1, :].astype(np.float64)
                  - patches[:, :, :, 1:, 0, :].astype(np.float64)) / 255.0
    down_diff = (patches[:, :-1, -1, :, :, :].astype(np.float64)
                 - patches[:, 1:, 0, :, :, :].astype(np.float64)) / 255.0
    right = np.mean(right_diff ** 2, axis=(2, 4))  # [B,32,31]
    down = np.mean(down_diff ** 2, axis=(3, 4))    # [B,31,32]
    return np.concatenate((right.reshape(len(image), -1), down.reshape(len(image), -1)), axis=1)


def boundary_scale_from_train(rgb_cache, pool_rows, *, chunk=64):
    rows = np.asarray(pool_rows, dtype=np.int64)
    if rows.shape != (SCALE_ROWS,) or len(np.unique(rows)) != SCALE_ROWS:
        raise ValueError("scale calibration requires exactly4096 unique registered TRAIN rows")
    total, count = 0.0, 0
    for start in range(0, SCALE_ROWS, chunk):
        batch = np.asarray(rgb_cache[rows[start:start + chunk]])
        energies = boundary_energy(batch)
        if not np.isfinite(energies).all():
            raise FloatingPointError("nonfinite clean TRAIN boundary energies")
        total += float(energies.sum(dtype=np.float64))
        count += energies.size
    scale = total / count
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("registered4096 TRAIN boundary scale must be positive and finite")
    return scale


def boundary_q_loss(q, targets, pairs=None):
    """Mean squared match between actual-spike local Q and exp(-edge/s)."""
    if not torch.is_tensor(q) or q.ndim != 3 or q.shape[-1] != q.shape[-2]:
        raise ValueError("Q must be [B,N,N]")
    if not torch.isfinite(q).all():
        raise FloatingPointError("Q contains nonfinite values")
    if pairs is None:
        pairs = torch.as_tensor(boundary_pairs(), device=q.device)
    else:
        pairs = torch.as_tensor(pairs, device=q.device, dtype=torch.long)
    target = torch.as_tensor(targets, device=q.device, dtype=q.dtype)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or target.shape != (q.shape[0], pairs.shape[0]):
        raise ValueError("boundary pair/target shape mismatch")
    if not torch.isfinite(target).all() or (target < 0).any() or (target > 1).any():
        raise ValueError("boundary target must be finite and within [0,1]")
    values = q[:, pairs[:, 0], pairs[:, 1]]
    return (values - target).square().mean()


def source_q_preservation_loss(student_q, teacher_q):
    if (student_q.ndim != 3 or student_q.shape[-1] != student_q.shape[-2]
            or teacher_q.shape != student_q.shape):
        raise ValueError("source-Q preservation requires matching [B,N,N] matrices")
    if not torch.isfinite(student_q).all() or not torch.isfinite(teacher_q).all():
        raise FloatingPointError("source-Q preservation received nonfinite affinities")
    offdiag = ~torch.eye(student_q.shape[-1], dtype=torch.bool, device=student_q.device)
    return (student_q[:, offdiag] - teacher_q.detach()[:, offdiag]).square().mean()


def _actual_q(trace):
    expected = (trace["spikes"].shape[0], 4, N, STEPS)
    if tuple(trace["component_spikes"].shape) != expected:
        raise ValueError("native32 component spike trace shape mismatch")
    q = spike_synchrony_affinity(trace["spikes"], trace["component_spikes"],
                                 settle=SETTLE, affinity_mode="spike")
    if tuple(q.shape) != (expected[0], N, N) or not torch.isfinite(q).all():
        raise FloatingPointError("actual-spike Q affinity is invalid")
    return q


def _families(foundation):
    core = foundation.wrapped.core
    return {
        "encoder": list(foundation.encoder.parameters()),
        "graph": list(core.graph_generator.parameters()),
        "native_core": [p for n, p in core.named_parameters() if not n.startswith("graph_generator.")],
        "integration_scalars": [foundation.wrapped.a_d, foundation.wrapped.a_m, foundation.wrapped.b],
    }


def _optimizer(foundation, groups):
    optimizer_groups = [
        {"params": groups["encoder"], "lr": 3e-6, "name": "encoder"},
        {"params": groups["graph"], "lr": 3e-5, "name": "graph"},
        {"params": groups["native_core"], "lr": 3e-5, "name": "native_core"},
        {"params": groups["integration_scalars"], "lr": 1e-3, "name": "integration_scalars"},
    ]
    params = [p for group in optimizer_groups for p in group["params"]]
    if len({id(p) for p in params}) != len(params) or any(not p.requires_grad for p in params):
        raise ValueError("probe optimizer must contain unique live source-family parameters")
    return torch.optim.Adam(optimizer_groups), params


def _q_and_trace(foundation, rgb):
    gamma = foundation.encode(rgb)
    trace = late_rollout(foundation.wrapped, gamma, total_steps=STEPS, live_tail_steps=TAIL)
    q = _actual_q(trace)
    return q, trace


def _teacher_q_first64(foundation, raw, rows, device):
    result = []
    for start in range(0, GRAD_BATCHES * LOGICAL_BATCH, MICRO):
        rgb = torch.from_numpy(np.asarray(raw[rows[start:start + MICRO]]).copy()).to(device)
        with torch.no_grad():
            q, _ = _q_and_trace(foundation, rgb)
        result.append(q.detach().cpu())
    return torch.cat(result, dim=0)


def _targets_for_batch(rgb, scale, *, device):
    energies = boundary_energy(rgb.detach().cpu().numpy())
    target = np.exp(-energies / scale)
    if not np.isfinite(target).all():
        raise FloatingPointError("nonfinite exp(-boundary/s) targets")
    return torch.as_tensor(target, device=device, dtype=torch.float32)


def _event_rates(trace, wrapped):
    threshold = (0.06 * torch.exp(wrapped.b.detach())).view(1, 4, 1, 1)
    events = trace["component_membrane"].detach() > threshold
    return {"all_steps": events.float().mean((0, 2, 3)).cpu().tolist(),
            "settled512": events[..., SETTLE:].float().mean((0, 2, 3)).cpu().tolist()}


def _state_hash(foundation, groups):
    return sw142._parameter_group_sha(groups)


def _gradient_norms(grads, parameters, family_names):
    by_id = {id(parameter): gradient for parameter, gradient in zip(parameters, grads)}
    result = {}
    for name, params in family_names.items():
        squares = []
        for p in params:
            grad = by_id[id(p)]
            if grad is not None:
                if not torch.isfinite(grad).all():
                    raise FloatingPointError("nonfinite diagnostic gradient")
                squares.append(grad.double().square().sum())
        result[name] = math.sqrt(float(torch.stack(squares).sum().cpu())) if squares else 0.0
    return result


def _accumulate_weighted_gradients(accumulated, gradients, weight):
    if len(accumulated) != len(gradients) or not math.isfinite(float(weight)) or weight <= 0:
        raise ValueError("gradient accumulation shape/weight is invalid")
    for index, gradient in enumerate(gradients):
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("nonfinite diagnostic microbatch gradient")
        value = gradient.detach() * float(weight)
        accumulated[index] = value.clone() if accumulated[index] is None else accumulated[index] + value
    return accumulated


def run_probe(*, device="cuda:0", output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0145 probe output: {output}")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0145 actual source probe requires CUDA")
    torch.cuda.set_device(dev); torch.cuda.init(); torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    started = time.perf_counter()
    source = sw130.source_contract(SEED)
    checkpoint, manifest_path, source_manifest, pool, image_ids, source_sha = source
    if len(image_ids) != SCALE_ROWS or len(set(image_ids)) != SCALE_ROWS:
        raise ValueError("seed0 source contract must provide exactly4096 ordered TRAIN IDs")
    raw = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    scale = boundary_scale_from_train(raw, pool[:SCALE_ROWS])
    foundation = sw135.load_native32_foundation(SEED, dev, verify_rgb_assets=True)
    if (foundation.provenance["source_core_sha256"] != source_sha
            or list(map(int, foundation.image_ids)) != image_ids
            or not np.array_equal(foundation.pool_indices, np.asarray(pool))):
        raise ValueError("native32 foundation does not match registered source0 IDs/checkpoint")
    rgb_validation = require_validated_asset_record(foundation.provenance)
    for parameter in foundation.wrapped.parameters():
        parameter.requires_grad_(True)
    foundation.encoder.requires_grad_(True)
    foundation.wrapped.train(); foundation.encoder.train()
    groups = _families(foundation)
    optimizer, params = _optimizer(foundation, groups)
    pre_hashes = _state_hash(foundation, groups)
    initial_scalar = {name: getattr(foundation.wrapped, name).detach().cpu().tolist()
                      for name in ("a_d", "a_m", "b")}
    # Cache source Q for the common64 before the disposable student update.
    teacher_q = _teacher_q_first64(foundation, raw, np.asarray(pool[:64]), dev)
    pairs_np = boundary_pairs()
    pairs = torch.as_tensor(pairs_np, device=dev)

    # One logical B-only Adam step on the first16 registered source0 images.
    optimizer.zero_grad(set_to_none=True)
    b_loss_values, b_event_values = [], []
    initial_q_exact = True
    initial_q_max_abs = 0.0
    for start in range(0, LOGICAL_BATCH, MICRO):
        rgb = torch.from_numpy(np.asarray(raw[pool[start:start + MICRO]]).copy()).to(dev)
        q, trace = _q_and_trace(foundation, rgb)
        expected_q = teacher_q[start:start + MICRO].to(dev)
        initial_q_exact = initial_q_exact and torch.equal(q.detach(), expected_q)
        initial_q_max_abs = max(initial_q_max_abs,
                                float((q.detach() - expected_q).abs().max().cpu()))
        if not initial_q_exact:
            raise AssertionError("fresh source/students do not match the saved source-Q baseline")
        targets = _targets_for_batch(rgb, scale, device=dev)
        loss = boundary_q_loss(q, targets, pairs)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        for p, grad in zip(params, grads):
            if grad is None:
                continue
            if not torch.isfinite(grad).all():
                raise FloatingPointError("nonfinite B-only gradient")
            grad = grad.detach() / (LOGICAL_BATCH // MICRO)
            p.grad = grad.clone() if p.grad is None else p.grad + grad
        b_loss_values.append(float(loss.detach().cpu()))
        b_event_values.append(_event_rates(trace, foundation.wrapped))
        del q, trace, loss, grads
    pre_grad_norms = {}
    for name, family in groups.items():
        norm_sq = sum(float(p.grad.detach().double().square().sum().cpu())
                      for p in family if p.grad is not None)
        pre_grad_norms[name] = math.sqrt(norm_sq)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    with torch.no_grad():
        foundation.wrapped.a_d.clamp_(0, 1)
        foundation.wrapped.a_m.clamp_(0, 1)
    post_hashes = _state_hash(foundation, groups)
    scalar_after = {name: getattr(foundation.wrapped, name).detach().cpu().tolist()
                    for name in ("a_d", "a_m", "b")}

    # Compare boundary B and full off-diagonal source-Q preservation A on the
    # same four B4 minibatches after the single disposable update.
    enc_graph = groups["encoder"] + groups["graph"]
    batch_ratios = []
    family_names = {"encoder": groups["encoder"], "graph": groups["graph"]}
    for start in range(0, GRAD_BATCHES * LOGICAL_BATCH, LOGICAL_BATCH):
        grad_b_sum = [None] * len(enc_graph)
        grad_a_sum = [None] * len(enc_graph)
        micro_b_values, micro_a_values = [], []
        for micro_start in range(start, start + LOGICAL_BATCH, MICRO):
            row = micro_start
            rgb = torch.from_numpy(np.asarray(raw[pool[row:row + MICRO]]).copy()).to(dev)
            q, _trace = _q_and_trace(foundation, rgb)
            targets = _targets_for_batch(rgb, scale, device=dev)
            loss_b = boundary_q_loss(q, targets, pairs)
            teacher = teacher_q[micro_start:micro_start + MICRO].to(dev)
            loss_a = source_q_preservation_loss(q, teacher)
            grad_b = torch.autograd.grad(loss_b, enc_graph, retain_graph=True, allow_unused=True)
            grad_a = torch.autograd.grad(loss_a, enc_graph, retain_graph=False, allow_unused=True)
            weight = rgb.shape[0] / LOGICAL_BATCH
            _accumulate_weighted_gradients(grad_b_sum, grad_b, weight)
            _accumulate_weighted_gradients(grad_a_sum, grad_a, weight)
            micro_b_values.append(float(loss_b.detach().cpu()))
            micro_a_values.append(float(loss_a.detach().cpu()))
            del q, _trace, loss_b, loss_a, grad_b, grad_a, teacher
        b_norms = _gradient_norms(grad_b_sum, enc_graph, family_names)
        a_norms = _gradient_norms(grad_a_sum, enc_graph, family_names)
        bn = math.hypot(b_norms["encoder"], b_norms["graph"])
        an = math.hypot(a_norms["encoder"], a_norms["graph"])
        index = start // LOGICAL_BATCH
        ratio = bn / an if (math.isfinite(bn) and math.isfinite(an) and bn > 0 and an > 0) else None
        batch_ratios.append({"logical_batch": index, "boundary_encoder_graph_norm": bn,
                             "source_q_encoder_graph_norm": an,
                             "boundary_family_norms": b_norms,
                             "source_q_family_norms": a_norms,
                             "boundary_microbatch_losses": micro_b_values,
                             "source_q_microbatch_losses": micro_a_values,
                             "ratio_B_over_A": ratio})
    ratios = [row["ratio_B_over_A"] for row in batch_ratios if row["ratio_B_over_A"] is not None]
    lambda_a = 0.25 * float(np.median(ratios)) if len(ratios) == GRAD_BATCHES else None
    if lambda_a is not None and (not math.isfinite(lambda_a) or lambda_a <= 0):
        lambda_a = None

    # Same first16 data after update: record threshold-event occupancy only.
    post_events = []
    for start in range(0, LOGICAL_BATCH, MICRO):
        rgb = torch.from_numpy(np.asarray(raw[pool[start:start + MICRO]]).copy()).to(dev)
        with torch.no_grad():
            _q, trace = _q_and_trace(foundation, rgb)
            post_events.append(_event_rates(trace, foundation.wrapped))
    torch.cuda.synchronize(dev)
    record = {
        "status": "disposable_probe_complete",
        "experiment": NAME,
        "seed": SEED,
        "resource_only": True,
        "training_admission": False,
        "ground_truth_used": False,
        "optimizer_updates": 1,
        "scope": "one disposable clean-input source0 boundary-Q update; no saved model or checkpoint",
        "source": {"checkpoint_sha256": source_sha,
                   "manifest_sha256": sha256_file(manifest_path),
                   "training_ids_sha256": hashlib.sha256(np.asarray(image_ids, dtype="<i8").tobytes()).hexdigest(),
                   "training_ids": image_ids, "rgb_asset_validation": rgb_validation,
                   "native32_provenance": foundation.provenance},
        "scale": {"value": scale, "definition": "mean seam RGB pixel-pair MSE over first4096 registered clean source0 TRAIN IDs",
                  "target": "exp(-e/s)",
                  "role": "fixed scale used only in this disposable probe; any later training calibration requires separate prospective experiment registration"},
        "update": {"logical_batch": LOGICAL_BATCH, "microbatch": MICRO,
                   "loss": "mean squared difference of actual-spike Q on all local4-neighbor pairs and exp(-e/s)",
                   "initial_source_q_baseline_exact_on_first16": initial_q_exact,
                   "initial_source_q_max_abs_difference": initial_q_max_abs,
                   "microbatch_losses": b_loss_values,
                   "preclip_family_gradient_norms": pre_grad_norms,
                   "parameter_group_lr": {"encoder": 3e-6, "graph": 3e-5,
                                           "native_core": 3e-5, "integration_scalars": 1e-3},
                   "global_clip_norm": 1.0,
                   "pre_parameter_group_sha256": pre_hashes,
                   "post_parameter_group_sha256": post_hashes,
                   "changed_parameter_groups": {key: pre_hashes[key] != post_hashes[key] for key in pre_hashes},
                   "integration_scalars_before": initial_scalar,
                   "integration_scalars_after": scalar_after,
                   "event_rates_during_update": b_event_values,
                   "event_rates_after_update_same16": post_events},
        "gradient_comparison_after_disposable_step": {
            "scope": "first4 logical B16 batches (first64 same registered clean TRAIN IDs); same encoder+graph parameter set",
            "per_logical_batch_gradient_norms": batch_ratios,
            "ratios": batch_ratios,
            "proposed_lambda_A": lambda_a,
            "lambda_rule": ".25 * median(B/A) across four common-input batches; null if any required norm is zero/nonfinite",
            "source_Q_A": "full off-diagonal MSE between student actual-spike Q and detached Q saved from the initial source on the same64 inputs"},
        "cuda": {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
                 "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev)),
                 "elapsed_seconds": time.perf_counter() - started},
        "implementation_sha256": sha256_file(HERE / "run.py"),
        "limitations": ["Disposable one-step diagnostic only; no GT, validation, segmentation score, or saved checkpoint.",
                        "It does not establish that boundary targets improve object masks or that any gate is necessary.",
                        "Any future training objective and admission require separate registration and review."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, allow_nan=False); stream.write("\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    row = run_probe(device=args.device, output=args.output)
    print(json.dumps({"status": row["status"], "training_admission": False,
                      "output": str(args.output)}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
