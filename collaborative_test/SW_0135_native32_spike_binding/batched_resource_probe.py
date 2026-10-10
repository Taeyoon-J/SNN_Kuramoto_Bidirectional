"""Batched 4x4 native32 resource measurement; never training admission."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for item in (str(ROOT), str(ROOT / "collaborative_test"), str(ROOT / "snn_kuramoto_bidirectional")):
    if item not in sys.path:
        sys.path.insert(0, item)

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding.binder import (
    NativeSpikeSlotBinder, RelativeSlotRGBDecoder,
)
from collaborative_test.SW_0135_native32_spike_binding.foundation import (
    forward_batch, load_native32_foundation, sha256_file,
)

LOGICAL_BATCH = 16
MICRO_BATCH = 4
RESOURCE_LAMBDA = 1.0
MAIN_UPDATES = 4096
WARMUP_UPDATES = 32
B1_REPORT = HERE / "results_archive" / "resource_probe_seed0.json"
B1_REPORT_SHA256 = "dcccd773cc303eb0bbb843b686e1bd61d9d9f998bc8f752a85f0b0b57a15445c"


def _unique_parameters(parameters, name):
    values = list(parameters)
    identities = [id(parameter) for parameter in values]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{name} parameter list contains duplicates")
    if any(not parameter.requires_grad for parameter in values):
        raise ValueError(f"{name} contains a frozen parameter")
    return values


def _named_joint_parameters(named_parameters, optimizer_groups):
    """Preserve named-family order while validating optimizer-group membership."""
    named = list(named_parameters)
    ordered = _unique_parameters([parameter for _, parameter in named], "named joint")
    grouped = _unique_parameters(
        [parameter for group in optimizer_groups for parameter in group["params"]],
        "joint optimizer")
    if {id(parameter) for parameter in ordered} != {id(parameter) for parameter in grouped}:
        raise AssertionError("named joint order and optimizer groups have different membership")
    return named, ordered


def _accumulate(loss, parameters, accumulator, *, retain_graph):
    """Add one microbatch autograd vector, preserving None as unused credit."""
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph,
                                    allow_unused=True)
    for index, gradient in enumerate(gradients):
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("resource probe encountered a nonfinite gradient")
        detached = gradient.detach()
        accumulator[index] = detached.clone() if accumulator[index] is None else accumulator[index] + detached


def _norm(gradients):
    square = 0.0
    for gradient in gradients:
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("resource probe gradient norm is nonfinite")
        square += float(gradient.detach().double().square().sum().cpu())
    return math.sqrt(square)


def _install_average_grad(parameters, accumulated, scale):
    for parameter, gradient in zip(parameters, accumulated):
        parameter.grad = None if gradient is None else gradient.mul(scale).clone()


def _parameter_delta(parameters, before):
    changed = 0
    finite = True
    norm_sq = 0.0
    for parameter, initial in zip(parameters, before):
        delta = parameter.detach() - initial
        finite = finite and bool(torch.isfinite(parameter).all())
        if torch.count_nonzero(delta).item():
            changed += 1
        norm_sq += float(delta.double().square().sum().cpu())
    return {"changed_tensors": changed, "tensor_count": len(parameters),
            "delta_l2": math.sqrt(norm_sq), "parameters_finite": finite}


def _named_family(name):
    if name.startswith("encoder."):
        return "encoder"
    if name.startswith("core.a_d"):
        return "integration_a_d"
    if name.startswith("core.a_m"):
        return "integration_a_m"
    if name.startswith("core.b"):
        return "integration_b"
    if name.startswith("core.core.graph_generator."):
        return "graph"
    if name.startswith(("core.core.gamma_channel_proj.", "core.core.gamma_phase_gain.")):
        return "oscillator_drive"
    if name.startswith("core.core.kuramoto."):
        return "kuramoto"
    if name.startswith("core.core.dendric_layer."):
        return "dendrite"
    if name.startswith("core.core.membrane_layer."):
        return "membrane"
    return "core_other"


def _scaled_family_norm(named_parameters, gradients, scale):
    squares = {}
    for (name, _), gradient in zip(named_parameters, gradients):
        if gradient is None:
            continue
        family = _named_family(name)
        squares[family] = squares.get(family, 0.0) + float(
            gradient.detach().double().square().sum().cpu())
    return {name: math.sqrt(value) * scale for name, value in squares.items()}


def _delta_families(named_parameters, before):
    changed, squares = {}, {}
    for (name, parameter), initial in zip(named_parameters, before):
        family = _named_family(name)
        delta = parameter.detach() - initial
        squares[family] = squares.get(family, 0.0) + float(delta.double().square().sum().cpu())
        changed[family] = changed.get(family, 0) + int(torch.count_nonzero(delta).item() > 0)
    return {"changed_tensors": changed,
            "delta_l2": {name: math.sqrt(value) for name, value in squares.items()}}


def run_probe(seed=0, device="cuda:0", output=None):
    if int(seed) != 0:
        raise ValueError("the registered B1 comparison is seed0 only")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("resource probe requires an exclusive CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(device)
    torch.cuda.init()
    if output is None:
        raise ValueError("--output is required for a create-once resource record")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing resource probe output: {output}")

    foundation = load_native32_foundation(seed, device, verify_rgb_assets=True)
    wrapped, encoder = foundation.wrapped, foundation.encoder
    binder = NativeSpikeSlotBinder(seed=135).to(device)
    decoder = RelativeSlotRGBDecoder(seed=106).to(device)
    # Native32 alone uses the preregistered coherence1.0 value.
    from collaborative_test.SW_0135_native32_spike_binding.foundation import make_criterion32
    criterion = make_criterion32()
    joint_named, optimizer_groups = sw130.joint_parameters(wrapped, encoder)
    joint_named, joint_parameters = _named_joint_parameters(joint_named, optimizer_groups)
    named_ids = [id(parameter) for _, parameter in joint_named]
    if (len(set(named_ids)) != len(named_ids)
            or set(named_ids) != {id(p) for p in joint_parameters}):
        raise AssertionError("registered joint parameter/norm union is not unique and complete")
    head_named = [(f"binder.{name}", parameter) for name, parameter in binder.named_parameters()]
    head_named += [(f"decoder.{name}", parameter) for name, parameter in decoder.named_parameters()]
    head_parameters = _unique_parameters([parameter for _, parameter in head_named], "head/decoder")
    if {id(p) for p in joint_parameters} & {id(p) for p in head_parameters}:
        raise AssertionError("joint and head optimizers must have disjoint parameters")

    joint_optimizer = torch.optim.Adam(optimizer_groups)
    head_optimizer = torch.optim.Adam([{"params": head_parameters, "lr": sw130.DECODER_LR}])
    joint_before = [p.detach().clone() for p in joint_parameters]
    head_before = [p.detach().clone() for p in head_parameters]
    old_accum = [None] * len(joint_parameters)
    rgb_joint_accum = [None] * len(joint_parameters)
    rgb_head_accum = [None] * len(head_parameters)
    micro_seconds = []

    train_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r")
    rows = foundation.pool_indices[:LOGICAL_BATCH].copy()
    global_ids = [foundation.image_ids[int(index)] for index in range(LOGICAL_BATCH)]
    b1_sha = sha256_file(B1_REPORT)
    if b1_sha != B1_REPORT_SHA256:
        raise ValueError("registered B1 resource report SHA changed; preserve and investigate")
    b1 = json.loads(B1_REPORT.read_text(encoding="utf-8"))
    if (b1.get("seed") != int(seed)
            or b1.get("image_ids") != global_ids
            or b1.get("source_core_sha256") != foundation.provenance["source_core_sha256"]
            or b1.get("source_manifest_sha256") != foundation.provenance["source_manifest_sha256"]
            or b1.get("status") != "resource_probe_complete"
            or b1.get("logical_batch_size") != LOGICAL_BATCH):
        raise ValueError("B1 comparison report does not match this source and logical batch")
    b1_gradients = b1.get("gradient_norms", {})
    required_norms = ("old_joint_accumulated", "rgb_joint_accumulated",
                      "rgb_joint_mean_by_family", "old_joint_mean_by_family",
                      "rgb_head_decoder_by_family")
    if any(key not in b1_gradients for key in required_norms):
        raise ValueError("B1 comparison report is missing registered gradient norms")
    torch.cuda.reset_peak_memory_stats(device)
    start_all = time.perf_counter()
    for start in range(0, LOGICAL_BATCH, MICRO_BATCH):
        stop = min(start + MICRO_BATCH, LOGICAL_BATCH)
        images = torch.from_numpy(np.asarray(train_cache[rows[start:stop]]).copy()).to(device)
        torch.cuda.synchronize(device)
        start_micro = time.perf_counter()
        result = forward_batch(wrapped, encoder, foundation.patcher,
                               foundation.feature_mean, foundation.feature_std,
                               foundation.feature_clip, binder, decoder, images,
                               criterion=criterion, total_steps=1024, settle=512,
                               live_tail_steps=64)
        count = stop - start
        # Each criterion is a microbatch mean. Multiply by its actual image
        # count so the final divide by 16 reproduces the logical mean exactly.
        _accumulate(result["old"] * count, joint_parameters, old_accum, retain_graph=True)
        _accumulate(result["rgb"] * count, joint_parameters, rgb_joint_accum, retain_graph=True)
        _accumulate(result["rgb"] * count, head_parameters, rgb_head_accum, retain_graph=False)
        torch.cuda.synchronize(device)
        micro_seconds.append(time.perf_counter() - start_micro)
        del images, result
    del train_cache

    scale = 1.0 / LOGICAL_BATCH
    old_accum_norm = _norm(old_accum) * scale
    rgb_joint_norm = _norm(rgb_joint_accum) * scale
    joint_accum = [None if old is None and rgb is None else
                   (torch.zeros_like(rgb if old is None else old)
                    if old is None else old) +
                   (torch.zeros_like(old if rgb is None else rgb)
                    if rgb is None else rgb)
                   for old, rgb in zip(old_accum, rgb_joint_accum)]
    rgb_joint_families = _scaled_family_norm(joint_named, rgb_joint_accum, scale)
    old_joint_families = _scaled_family_norm(joint_named, old_accum, scale)
    binder_count = len(list(binder.parameters()))
    rgb_head_families = {
        "binder": _norm(rgb_head_accum[:binder_count]) * scale,
        "decoder": _norm(rgb_head_accum[binder_count:]) * scale,
    }

    joint_optimizer.zero_grad(set_to_none=True)
    head_optimizer.zero_grad(set_to_none=True)
    _install_average_grad(joint_parameters, joint_accum, scale)
    _install_average_grad(head_parameters, rgb_head_accum, scale)
    joint_clip = float(torch.nn.utils.clip_grad_norm_(joint_parameters, sw130.CLIP_NORM))
    head_clip = float(torch.nn.utils.clip_grad_norm_(head_parameters, sw130.CLIP_NORM))
    joint_optimizer.step()
    head_optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_all
    total_allocated = int(torch.cuda.max_memory_allocated(device))
    total_reserved = int(torch.cuda.max_memory_reserved(device))
    joint_delta = _parameter_delta(joint_parameters, joint_before)
    head_delta = _parameter_delta(head_parameters, head_before)
    joint_delta["by_family"] = _delta_families(joint_named, joint_before)
    binder_parameters = list(binder.parameters())
    decoder_parameters = list(decoder.parameters())
    head_delta["by_family"] = {
        "binder": _parameter_delta(binder_parameters, head_before[:len(binder_parameters)]),
        "decoder": _parameter_delta(decoder_parameters, head_before[len(binder_parameters):]),
    }
    if joint_optimizer.state and any(int(state["step"].item()) != 1 for state in joint_optimizer.state.values()):
        raise AssertionError("resource-only joint Adam must perform exactly one disposable step")
    if head_optimizer.state and any(int(state["step"].item()) != 1 for state in head_optimizer.state.values()):
        raise AssertionError("resource-only head Adam must perform exactly one disposable step")

    current_gradients = {
        "old_joint_accumulated": old_accum_norm,
        "rgb_joint_accumulated": rgb_joint_norm,
        "rgb_joint_mean_by_family": rgb_joint_families,
        "old_joint_mean_by_family": old_joint_families,
        "rgb_head_decoder_by_family": rgb_head_families,
    }
    b1_gradients = b1["gradient_norms"]
    norm_comparison = {}
    for key, current in current_gradients.items():
        prior = b1_gradients.get(key)
        if isinstance(current, dict):
            norm_comparison[key] = {
                name: {"b1": prior.get(name), "b4": value,
                       "absolute_difference": abs(float(value) - float(prior[name]))}
                for name, value in current.items() if prior is not None and name in prior
            }
        else:
            norm_comparison[key] = {"b1": prior, "b4": current,
                                    "absolute_difference": abs(float(current) - float(prior))}
    report = {
        "experiment": "SW0135_native32_spike_binding",
        "status": "batched_resource_probe_complete",
        "microbatch_images": MICRO_BATCH,
        "microbatch_count": LOGICAL_BATCH // MICRO_BATCH,
        "b1_comparison": {"path": str(B1_REPORT), "sha256": b1_sha,
                          "norm_comparison": norm_comparison,
                          "comparison_note": "Numerical norm comparison only; reduction order differs and bitwise equality is not claimed."},
        "resource_only": True,
        "training_admission": False,
        "scientific_lambda": None,
        "resource_probe_lambda": RESOURCE_LAMBDA,
        "seed": int(seed),
        "source_core_sha256": foundation.provenance["source_core_sha256"],
        "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
        "implementation_fingerprint": foundation.provenance["implementation_fingerprint"],
        "rgb_assets": foundation.provenance["rgb_asset_validation"],
        "image_ids": global_ids,
        "image_id_count": len(global_ids),
        "ground_truth_used": False,
        "optimizer_updates": 0,
        "resource_optimizer_steps": {"joint": 1, "head_decoder": 1},
        "microbatch_size": MICRO_BATCH,
        "logical_batch_size": LOGICAL_BATCH,
        "microbatch_seconds": micro_seconds,
        "logical_update_seconds": elapsed,
        "resource_forecast_seconds": {
            "warm32": elapsed * WARMUP_UPDATES,
            "main4096": elapsed * MAIN_UPDATES,
            "combined": elapsed * (WARMUP_UPDATES + MAIN_UPDATES),
        },
        "cuda_peak_memory_bytes": {"allocated": total_allocated, "reserved": total_reserved},
        "gradient_norms": {
            "old_joint_accumulated": old_accum_norm,
            "rgb_joint_accumulated": rgb_joint_norm,
            "rgb_joint_mean_by_family": rgb_joint_families,
            "old_joint_mean_by_family": old_joint_families,
            "rgb_head_decoder_by_family": rgb_head_families,
            "combined_joint_clip_preclip_norm": joint_clip,
            "rgb_head_decoder_clip_preclip_norm": head_clip,
        },
        "disposable_parameter_changes": {"joint": joint_delta, "head_decoder": head_delta},
        "limitations": [
            "One resource-only logical update with lambda1; not the calibrated objective.",
            "No warmup, scientific calibration, optimizer checkpoint, or candidate training was run.",
            "Timing forecast is linear extrapolation from this one logical batch.",
            "Zero RGB family credit is reported diagnostically and does not constitute admission.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_probe(args.seed, args.device, args.output)
    print(json.dumps({"status": report["status"], "logical_update_seconds": report[
        "logical_update_seconds"], "cuda_peak_memory_bytes": report["cuda_peak_memory_bytes"]},
        allow_nan=False))
    return report


if __name__ == "__main__":
    main()
