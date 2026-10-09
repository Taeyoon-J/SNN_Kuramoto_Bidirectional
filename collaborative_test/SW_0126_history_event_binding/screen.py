"""Read-only first-64 TRAIN screen for the SW0126 history-event mechanism."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0115_analytic_partition_rgb import run as rgb_base
from collaborative_test.SW_0117_joint_analytic_rgb import run as sw117
from collaborative_test.SW_0125_late_rollout_credit import run as sw125
from collaborative_test.SW_0125_late_rollout_credit import late_rollout as late_rollout_module
from collaborative_test.SW_0126_history_event_binding.binder import TemporalSlotRGBBinder
from collaborative_test.SW_0126_history_event_binding.history_event import (
    attach_history_event_membrane,
    head_trace_for_arm,
)

STEPS = 1024
SETTLE = 512
TAIL = 64
BATCH = 16
SCREEN_BATCHES = 4
ACTIVITY_BOUNDS = (0.02, 0.98)
MIN_MIXED_UNIT_FRACTION = 0.10


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def implementation_fingerprint():
    paths = [HERE / "history_event.py", HERE / "binder.py", HERE / "screen.py",
             HERE / "protocol.json", sw125.RUNNER, sw125.HERE / "late_rollout.py",
             ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
             ROOT / "snn_kuramoto_bidirectional/membrane_layer.py",
             ROOT / "snn_kuramoto_bidirectional/dendric_layer.py",
             ROOT / "snn_kuramoto_bidirectional/sinusoidal_gating.py"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in paths}


def source_and_cache_contract(seed):
    source, source_manifest, source_meta, ids, rows = sw125.source_contract(seed)
    if sha(source) != base.EXPECTED_SOURCE_SHAS[seed]:
        raise AssertionError("immutable SW0097 core SHA mismatch")
    if (sha(sw117.ENCODER_PATH) != base.EXPECTED_ENCODER_SHA256
            or sha(sw117.STATS_PATH) != base.EXPECTED_PREPROCESSING_SHA256):
        raise AssertionError("registered encoder/preprocessing SHA mismatch")
    gamma, gamma_meta = base.validate_gamma_cache(base.GAMMA_TRAIN,
                                                   base.GAMMA_TRAIN_MANIFEST)
    rgb, rgb_meta, rgb_sha = sw117.load_rgb_training_cache()
    return (source, source_manifest, source_meta, ids, rows,
            gamma, gamma_meta, rgb, rgb_meta, rgb_sha)


def _full_rollout(core, gamma, live_tail):
    """Static source rollout, with an optional detached 960-step prefix."""
    if (core.gamma_drive_mode != "static" or core.osc_dim != 4
            or core.spike_per_component is not True
            or core.graph_generator.uses_feedback
            or core.kuramoto.spike_pulse_gain is not None):
        raise ValueError("unsupported source dynamics for SW0126 activity screen")
    if gamma.ndim != 3 or tuple(gamma.shape[1:]) != (core.T, core.in_dim):
        raise ValueError("gamma must use the registered static [B,8,256] contract")
    if int(live_tail) not in (0, TAIL):
        raise ValueError("screen supports only a no-grad reference or 64-step live tail")
    gamma = gamma.to(core.device)
    boundary = STEPS - int(live_tail)
    batch = gamma.shape[0]
    with torch.no_grad():
        sc = core.graph_generator(gamma)
        drive = core.gamma_to_drive(gamma, core.gamma_channel_proj, core.gamma_phase_gain)
        theta = core._init_theta(drive, batch)
        coupling = None
        if core.kuramoto_backend == "factorized":
            coupling = core.kuramoto.prepare_coupling(
                sc, batch_size=batch, num_units=core.in_dim, device=gamma.device)
    core.dendric_layer.set_neuron_state(batch * 4)
    core.membrane_layer.set_neuron_state(batch * 4)
    theta_history, carrier_frames, gate_frames = [], [], []
    membrane_frames, spike_frames = [], []
    for t in range(STEPS):
        if live_tail and t == boundary:
            theta_history = [value.detach() for value in theta_history]
            theta = theta.detach()
            core.dendric_layer.h = core.dendric_layer.h.detach()
            if hasattr(core.membrane_layer, "detach_rollout_boundary"):
                core.membrane_layer.detach_rollout_boundary()
            else:
                core.membrane_layer.mem = core.membrane_layer.mem.detach()
                core.membrane_layer.spike = core.membrane_layer.spike.detach()
        with torch.set_grad_enabled(bool(live_tail and t >= boundary)):
            theta = core.kuramoto(theta, drive, A=sc, spike=None, coupling=coupling)
            theta_history.append(theta)
            carrier, gate = late_rollout_module.sinusoidal_gating(
                theta_history, t, core.phase_delay_steps, gate_mode=core.gate_mode)
            folded = carrier.permute(0, 2, 1).reshape(batch * 4, core.in_dim, 1)
            folded_gate = gate.repeat_interleave(4, dim=0)
            h_wave = core.dendric_layer(folded, core.membrane_layer.spike)
            mem, spike = core.membrane_layer(h_wave, folded_gate)
            membrane_frames.append(mem.reshape(batch, 4, core.in_dim))
            spike_frames.append(spike.reshape(batch, 4, core.in_dim))
        carrier_frames.append(carrier)
        gate_frames.append(gate)
    return {
        "theta": torch.stack(theta_history, dim=-1),
        "carrier": torch.stack(carrier_frames, dim=-1),
        "gate": torch.stack(gate_frames, dim=-1),
        "component_membrane": torch.stack(membrane_frames, dim=-1),
        "component_spikes": torch.stack(spike_frames, dim=-1),
    }


def assert_phase_carrier_gate_parity(reference, adaptive):
    checks = {key: torch.equal(reference[key], adaptive[key])
              for key in ("theta", "carrier", "gate")}
    if not all(checks.values()):
        raise AssertionError(f"frozen-source theta/carrier/gate parity failed: {checks}")
    return checks


def assert_native_source_rollout_parity(core, gamma, reference):
    """Check copied rollout against the production forward on the same source."""
    with torch.no_grad():
        _, _, _, theta = core(gamma, num_time_steps=STEPS, return_theta=True,
                              return_core_out=True)
        native_membrane = core.last_component_out
        native_spikes = core.last_component_spikes
    checks = {
        "theta": torch.equal(reference["theta"], theta.permute(0, 2, 3, 1)),
        "component_membrane": torch.equal(reference["component_membrane"], native_membrane),
        "component_spikes": torch.equal(reference["component_spikes"], native_spikes),
    }
    if not all(checks.values()):
        raise AssertionError(f"manual rollout differs from production source forward: {checks}")
    return checks


def activity_summary(binary_events):
    if binary_events.ndim != 4 or binary_events.shape[1] != 4 or binary_events.shape[-1] != SETTLE:
        raise ValueError("events must have shape [B,4,N,512]")
    if not torch.isfinite(binary_events).all() or not torch.logical_or(
            binary_events == 0, binary_events == 1).all():
        raise ValueError("event trace must be finite and binary")
    occupancy = binary_events.float().mean(dim=(0, 2, 3))
    mixed_units = ((binary_events.amin(dim=-1) == 0)
                   & (binary_events.amax(dim=-1) == 1)).float().mean(dim=-1)
    low, high = ACTIVITY_BOUNDS
    return {
        "occupancy_by_component": occupancy.detach().cpu().tolist(),
        "occupancy_pass_by_component": ((occupancy >= low) & (occupancy <= high)).detach().cpu().tolist(),
        "minimum_mixed_unit_fraction_by_image_component": mixed_units.amin(dim=0).detach().cpu().tolist(),
        "mixed_unit_pass_by_image_component": (mixed_units >= MIN_MIXED_UNIT_FRACTION).detach().cpu().tolist(),
        "pass": bool(((occupancy >= low) & (occupancy <= high)).all()
                     and (mixed_units >= MIN_MIXED_UNIT_FRACTION).all()),
    }


def _norm(grads):
    values = [g.detach().float().norm() for g in grads if g is not None]
    if not values:
        return 0.0
    return float(torch.stack(values).norm().cpu())


def _gradient_report(loss, assignment, layer, dendritic_params, require_credit):
    p_grad, = torch.autograd.grad(loss, (assignment,), retain_graph=True,
                                  allow_unused=True)
    beta_grads = torch.autograd.grad(loss, (layer.beta_logits,), retain_graph=True,
                                     allow_unused=True)
    membrane_grads = torch.autograd.grad(loss, (layer.tau_m,), retain_graph=True,
                                         allow_unused=True)
    dendrite_grads = torch.autograd.grad(loss, dendritic_params, retain_graph=True,
                                        allow_unused=True)
    result = {"assignment_norm": _norm((p_grad,)),
              "beta_norm": _norm(beta_grads),
              "beta_component_abs": (beta_grads[0].detach().abs().cpu().tolist()
                                     if beta_grads[0] is not None else [0.0] * 4),
              "membrane_tau_m_norm": _norm(membrane_grads),
              "dendritic_norm": _norm(dendrite_grads)}
    if require_credit:
        expected = [result["assignment_norm"], result["beta_norm"],
                    result["membrane_tau_m_norm"], result["dendritic_norm"]]
        if any(not math.isfinite(value) or value <= 0 for value in expected):
            raise AssertionError(f"actual-event reconstruction lost required gradient credit: {result}")
        if any(not math.isfinite(v) or v <= 0 for v in result["beta_component_abs"]):
            raise AssertionError(f"one or more beta components received no credit: {result}")
    return result


def _run_batch(seed, batch_index, rows, ids, rgb_cache, gamma_cache, encoder,
               patcher, mean, std, clip, source_core, adaptive_core, layer, binders, device):
    row_slice = rows[batch_index * BATCH:(batch_index + 1) * BATCH]
    global_ids = ids[batch_index * BATCH:(batch_index + 1) * BATCH]
    images = sw117.read_rgb(rgb_cache, row_slice, device)
    cached_gamma = gamma_cache[torch.as_tensor(row_slice, dtype=torch.long)].to(device)
    with torch.no_grad():
        live_gamma = sw117.encode_rgb(encoder, patcher, mean, std, clip, images)
    gamma_delta = float((live_gamma - cached_gamma).abs().max().cpu())
    if not math.isfinite(gamma_delta) or gamma_delta > 2e-5:
        raise AssertionError(f"seed{seed} batch{batch_index}: cached/live gamma mismatch {gamma_delta}")

    reference = _full_rollout(source_core, live_gamma, live_tail=0)
    native_source_parity = None
    if batch_index == 0:
        native_source_parity = assert_native_source_rollout_parity(
            source_core, live_gamma, reference)
    adaptive = _full_rollout(adaptive_core, live_gamma, live_tail=TAIL)
    parity = assert_phase_carrier_gate_parity(reference, adaptive)
    component_events = layer.component_event_trace(BATCH)
    component_gates = layer.component_gate_trace(BATCH)
    settled = {name: value[..., -SETTLE:] for name, value in adaptive.items()
               if name in ("component_spikes", "component_membrane")}
    spikes = settled["component_spikes"]
    gates = component_gates[..., -SETTLE:]
    events = component_events[..., -SETTLE:]
    if not torch.equal(spikes, gates * events):
        raise AssertionError("emitted actual spikes are not exactly gate times binary event")
    if not torch.logical_or(events == 0, events == 1).all():
        raise AssertionError("event trace is not binary in the settled screen window")

    target = images.permute(0, 2, 3, 1).contiguous() / 255.0
    arm_reports = {}
    for arm, head_input in (("history_event", head_trace_for_arm(spikes, gates, "history_event")),
                            ("gate_only", head_trace_for_arm(spikes, gates, "gate_only"))):
        binder = binders[arm]
        output = binder(head_input)
        reconstruction = output["reconstruction"]
        if reconstruction.shape != target.shape or not torch.isfinite(reconstruction).all():
            raise AssertionError(f"{arm} decoder returned invalid native RGB reconstruction")
        assignment = output["assignment"]
        if not torch.isfinite(assignment).all():
            raise AssertionError(f"{arm} binder returned nonfinite assignments")
        assignment_error = float((assignment.sum(dim=1) - 1).abs().max().detach().cpu())
        if assignment_error > 1e-6:
            raise AssertionError(f"{arm} assignments do not form a slot distribution")
        loss = torch.nn.functional.mse_loss(reconstruction, target)
        if not torch.isfinite(loss):
            raise AssertionError(f"{arm} RGB reconstruction loss is nonfinite")
        arm_reports[arm] = {
            "rgb_mse": float(loss.detach().cpu()),
            "assignment_finite": bool(torch.isfinite(output["assignment"]).all()),
            "assignment_patch_mass_max_error": assignment_error,
            "gradient_credit": _gradient_report(
                loss, output["assignment"], layer, tuple(adaptive_core.dendric_layer.parameters()),
                require_credit=(arm == "history_event")),
        }
    return {
        "batch_index": batch_index,
        "image_ids": [int(x) for x in global_ids],
        "cached_live_gamma_max_abs_diff": gamma_delta,
        "theta_carrier_gate_exact": parity,
        "native_source_rollout_exact_first_batch": native_source_parity,
        "component_membrane_max_abs_diff_allowed": float(
            (reference["component_membrane"] - adaptive["component_membrane"]).abs().max().detach().cpu()),
        "event_fraction_differs_from_continuous_gate": int((spikes != gates).sum().detach().cpu()),
        "activity": activity_summary(events.detach()),
        "centered_event_trace_variance_by_component": (
            events.float() - events.float().mean(dim=-1, keepdim=True)
        ).square().mean(dim=(0, 2, 3)).detach().cpu().tolist(),
        "arms": arm_reports,
    }, events.detach().cpu()


def _reset_cuda_peak_stats(device):
    """Initialize the selected CUDA context before querying allocator peaks."""
    target = torch.device(device)
    torch.cuda.set_device(target)
    # reset_peak_memory_stats queries the allocator and requires this context.
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(target)


def screen_seed(seed, output, device="cuda:0"):
    if seed not in (0, 1, 2):
        raise ValueError("SW0126 screen supports source seeds 0,1,2")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserving existing screen result: {output}")
    if str(device).startswith("cuda"):
        _reset_cuda_peak_stats(device)
    (source, source_manifest, source_meta, ids, rows, gamma_cache, gamma_meta,
     rgb_cache, rgb_meta, rgb_sha) = source_and_cache_contract(seed)
    if len(ids) != 4096 or len(rows) != 4096:
        raise AssertionError("screen requires the exact registered 4096 TRAIN-ID order")
    encoder = sw117.load_input_encoder(str(sw117.ENCODER_PATH), num_kernels=8,
                                       kernel_size=3, channels=3, device=device)
    encoder.eval().requires_grad_(False)
    patcher = sw117.FeaturePatchGammaInitializer(grid_size=16).to(device).eval()
    stats = torch.load(sw117.STATS_PATH, map_location="cpu", weights_only=True)
    mean, std, clip = sw117.preprocessing_tensors(stats, device)
    source_core = base.make_core(device, steps=64)
    source_core.load_state_dict(torch.load(source, map_location=device, weights_only=True), strict=True)
    source_core.requires_grad_(False).eval()
    source_core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.shape[0])]
    adaptive_core = base.make_core(device, steps=64)
    adaptive_core.load_state_dict(torch.load(source, map_location=device, weights_only=True), strict=True)
    original_state = {key: value.detach().clone() for key, value in adaptive_core.state_dict().items()}
    layer = attach_history_event_membrane(adaptive_core, capture_gate_trace=True)
    adaptive_core.eval()
    adaptive_core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.shape[0])]
    for parameter in adaptive_core.parameters():
        parameter.requires_grad_(False)
    for parameter in adaptive_core.dendric_layer.parameters():
        parameter.requires_grad_(True)
    for parameter in layer.parameters():
        parameter.requires_grad_(True)
    binders = {}
    init_seed = 1260 + seed
    event_binder = TemporalSlotRGBBinder(init_seed=init_seed).to(device)
    gate_binder = TemporalSlotRGBBinder(init_seed=init_seed).to(device)
    gate_binder.load_state_dict(event_binder.state_dict(), strict=True)
    binders.update(history_event=event_binder, gate_only=gate_binder)

    batches, event_batches = [], []
    for batch_index in range(SCREEN_BATCHES):
        record, events = _run_batch(seed, batch_index, rows, ids, rgb_cache, gamma_cache,
                                    encoder, patcher, mean, std, clip, source_core,
                                    adaptive_core, layer, binders, device)
        batches.append(record)
        event_batches.append(events)
    full_events = torch.cat(event_batches, dim=0)
    total_activity = activity_summary(full_events)
    all_image_component_mixed = all(
        bool(torch.as_tensor(batch["activity"]["mixed_unit_pass_by_image_component"]).all())
        for batch in batches)
    aggregate_occupancy = all(total_activity["occupancy_pass_by_component"])
    if not aggregate_occupancy:
        status = "failed_activity_guard"
    elif not all_image_component_mixed:
        status = "failed_per_batch_activity_guard"
    else:
        status = "passed_feasibility_screen"
    changed_source_params = [key for key, value in adaptive_core.state_dict().items()
                             if key in original_state and not torch.equal(value, original_state[key])]
    if changed_source_params:
        raise AssertionError(f"screen unexpectedly modified source parameters: {changed_source_params}")
    report = {
        "status": status, "experiment": "SW0126", "seed": seed,
        "device": str(device), "optimizer_updates": 0,
        "ground_truth_used_for_prediction_or_training": False,
        "source_core_sha256": sha(source), "source_manifest_sha256": sha(source_manifest),
        "source_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "matched_train_ids": [int(v) for v in ids[:BATCH * SCREEN_BATCHES]],
        "gamma_cache_sha256": sha(base.GAMMA_TRAIN),
        "gamma_cache_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
        "rgb_cache_sha256": rgb_sha,
        "rgb_cache_manifest_sha256": sha(sw117.RGB_CACHE_MANIFEST),
        "encoder_sha256": sha(sw117.ENCODER_PATH),
        "preprocessing_sha256": sha(sw117.STATS_PATH),
        "implementation_fingerprint": implementation_fingerprint(),
        "cuda_peak_memory_bytes": ({
            "allocated": int(torch.cuda.max_memory_allocated(torch.device(device))),
            "reserved": int(torch.cuda.max_memory_reserved(torch.device(device))),
        } if str(device).startswith("cuda") else None),
        "activity_thresholds": {"component_occupancy": list(ACTIVITY_BOUNDS),
                                "minimum_mixed_unit_fraction_per_image_component": MIN_MIXED_UNIT_FRACTION},
        "aggregate_activity": total_activity,
        "batches": batches,
        "screen_interpretation": "Feasibility only; a pass does not establish segmentation value or promote the mechanism.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = screen_seed(args.seed, args.output, args.device)
    print(json.dumps({"status": record["status"], "seed": record["seed"],
                      "output": str(args.output), "optimizer_updates": 0}, allow_nan=False),
          flush=True)


if __name__ == "__main__":
    main()
