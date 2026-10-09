"""Read-only TRAIN calibration and adaptive event-activity eligibility probe.

This script does not update model parameters, optimizers, checkpoints, or labels.
It uses the registered cached gamma rows for the first four ordered B16 TRAIN
batches and writes one create-once calibration/eligibility record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0117_joint_analytic_rgb.run import _forward_with_plv
from collaborative_test.SW_0123_adaptive_temporal_assignment.adaptive_dynamics import install_adaptive_dynamics
from collaborative_test.SW_0123_adaptive_temporal_assignment.model import fit_spike_rms

SEEDS = (0, 1, 2)
BATCH = 16
TIME_STEPS = 64
SETTLE = 32
KAPPA_FACTOR = 4.0
KAPPA_QUANTILE = 0.90
SOURCE_SHA256 = {
    0: "36f2481dd1fa51fa29bd4dc34275a876b71d8b76fbee13ac47a0a1bfe6766fbf",
    1: "76f379d5a7e4cd9d12fdf0b701f3dd9dbbf28b5eea270a9cee2d30d87b4dad98",
    2: "798ad3e9d4bf837b1bbeb1bd7c13900df511b5c76f5736d2b9f8b973d7fa5661",
}
ENCODER_PATH = base.ASSETS / "input_encoder/input_layer_encoder.pt"
PREPROCESSING_PATH = base.ASSETS / "feature_preprocessing.pt"


class EligibilityFailure(RuntimeError):
    """A registered scientific activity guard failed; no tuning is allowed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    paths = [Path(__file__), HERE / "protocol.json", HERE / "adaptive_dynamics.py",
             HERE / "model.py", HERE / "readout.py",
             ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
             ROOT / "collaborative_test/SW_0117_joint_analytic_rgb/run.py",
             ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
             ROOT / "snn_kuramoto_bidirectional/membrane_layer.py",
             ROOT / "snn_kuramoto_bidirectional/dendric_layer.py"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in paths}


def same_ordered_indices(left, right):
    return np.array_equal(np.asarray(left), np.asarray(right))


def calibration_arrays(component_out: torch.Tensor, component_spikes: torch.Tensor):
    """Return registered kappa and RMS for [64,4,256,32] settled source traces."""
    expected = (64, 4, 256, SETTLE)
    if tuple(component_out.shape) != expected or tuple(component_spikes.shape) != expected:
        raise ValueError(f"calibration traces must both have shape {expected}")
    if not torch.isfinite(component_out).all() or not torch.isfinite(component_spikes).all():
        raise FloatingPointError("source calibration traces contain nonfinite values")
    margin = (component_out - 0.06).clamp_min(0)
    by_component = margin.permute(1, 0, 2, 3).reshape(4, -1)
    kappa = KAPPA_FACTOR * torch.quantile(by_component, KAPPA_QUANTILE, dim=1)
    rms = fit_spike_rms(component_spikes)
    if not torch.isfinite(kappa).all() or not torch.isfinite(rms).all():
        raise FloatingPointError("calibration produced nonfinite kappa/RMS")
    return kappa.detach(), rms.detach()


def activity_summary(binary_events: torch.Tensor):
    """Summarize [64,4,256,32] binary events under fixed activity guards."""
    if binary_events.shape != (64, 4, 256, SETTLE):
        raise ValueError("event traces must have shape [64,4,256,32]")
    if not torch.isfinite(binary_events).all():
        raise FloatingPointError("adaptive event traces contain nonfinite values")
    if not torch.logical_or(binary_events == 0, binary_events == 1).all():
        raise ValueError("event traces must be binary flags")
    events = binary_events.bool()
    occupancy = events.float().mean(dim=(0, 2, 3))
    mixed = (events.any(dim=-1) & (~events).any(dim=-1)).float().mean(dim=(0, 2))
    result = {"occupancy_by_component": occupancy.detach().cpu().tolist(),
              "mixed_image_patch_fraction_by_component": mixed.detach().cpu().tolist()}
    if not ((occupancy > 0.01) & (occupancy < 0.80)).all():
        raise EligibilityFailure(f"adaptive binary event occupancy guard failed: {result['occupancy_by_component']}")
    if not (mixed >= 0.10).all():
        raise EligibilityFailure(f"adaptive image-patch temporal variation guard failed: {result['mixed_image_patch_fraction_by_component']}")
    return result


def _load_source(seed: int, device):
    checkpoint, manifest_path, manifest = base.source_paths(seed)
    ids, rows = base.train_indices(seed)
    if len(ids) != 4096 or len(np.unique(ids)) != 4096:
        raise AssertionError("registered source training IDs are not 4096 unique ordered images")
    if (manifest.get("status") != "complete" or manifest.get("unique_images_seen") != 4096
            or manifest.get("steps") != 256 or manifest.get("batch") != 16
            or manifest.get("seed") != 117 + seed
            or manifest.get("training_ids") != ids.tolist()
            or manifest.get("ground_truth_used_for_training") is not False):
        raise AssertionError(f"SW0097 seed-{seed} source manifest/order contract mismatch")
    actual_source_sha = sha256(checkpoint)
    if actual_source_sha != SOURCE_SHA256[seed]:
        raise AssertionError(f"SW0097 seed-{seed} checkpoint SHA mismatch: {actual_source_sha}")
    core = base.make_core(device, TIME_STEPS)
    core.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    # Evaluation grouping is unused by this rollout and can invoke a CPU clique solver.
    core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.shape[0])]
    if core.kuramoto.spike_pulse_gain is not None or core.graph_generator.uses_feedback:
        raise AssertionError("SW0097 source has nonregistered pulse/feedback dynamics")
    return core, checkpoint, manifest_path, manifest, ids, rows


def _forward_four(core, gamma, rows, device, adaptive=False):
    pieces = []
    for batch_index in range(4):
        lo, hi = batch_index * BATCH, (batch_index + 1) * BATCH
        batch = gamma[torch.as_tensor(rows[lo:hi], dtype=torch.long)].to(device)
        if batch.shape != (BATCH, 8, 256) or not torch.isfinite(batch).all():
            raise AssertionError("registered cached gamma batch has invalid shape or values")
        _forward_with_plv(core, batch, base.criterion(), SETTLE, "phase", "mean")
        component_out = core.last_component_out
        component_spikes = core.last_component_spikes
        if component_out is None or component_spikes is None:
            raise AssertionError("source core did not expose four component traces")
        expected = (BATCH, 4, 256, TIME_STEPS)
        if tuple(component_out.shape) != expected or tuple(component_spikes.shape) != expected:
            raise AssertionError(f"unexpected core trace shape: {tuple(component_out.shape)}")
        if not torch.isfinite(component_out).all() or not torch.isfinite(component_spikes).all():
            raise FloatingPointError("source/adaptive forward produced nonfinite traces")
        if adaptive:
            history = core.membrane_layer.event_history
            if history is None or len(history) != TIME_STEPS:
                raise AssertionError("adaptive membrane event history was not captured")
            # S2Net folds rows b*4+d and stores each step as [B*4,256].
            events = torch.stack(history, dim=-1).reshape(BATCH, 4, 256, TIME_STEPS)
            pieces.append(events[:, :, :, SETTLE:].detach())
        else:
            pieces.append((component_out[:, :, :, SETTLE:].detach(),
                           component_spikes[:, :, :, SETTLE:].detach()))
    if adaptive:
        return torch.cat(pieces, dim=0)
    return (torch.cat([piece[0] for piece in pieces], dim=0),
            torch.cat([piece[1] for piece in pieces], dim=0))


def calibrate(seed: int, device: str, output: Path):
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserving existing calibration record: {output}")
    if not str(device).startswith("cuda"):
        raise ValueError("the registered B16/T64 calibration eligibility run requires a real CUDA device")
    source, checkpoint, source_manifest_path, manifest, ids, rows = _load_source(seed, device)
    encoder_sha = sha256(ENCODER_PATH)
    preprocessing_sha = sha256(PREPROCESSING_PATH)
    if (encoder_sha != base.EXPECTED_ENCODER_SHA256
            or preprocessing_sha != base.EXPECTED_PREPROCESSING_SHA256):
        raise AssertionError("registered encoder/preprocessing asset SHA mismatch")
    gamma, gamma_manifest = base.validate_gamma_cache(base.GAMMA_TRAIN, base.GAMMA_TRAIN_MANIFEST)
    if tuple(gamma.shape) != (70000, 8, 256):
        raise AssertionError(f"registered TRAIN gamma cache shape mismatch: {tuple(gamma.shape)}")

    with torch.no_grad():
        legacy_out, legacy_spikes = _forward_four(source, gamma, rows, device, adaptive=False)
    kappa, rms = calibration_arrays(legacy_out, legacy_spikes)

    # Strict-load the same immutable source again, then install only the opt-in
    # adaptive neuron state. No parameters or optimizer are updated by this probe.
    status, failure, activity = "passed", None, None
    if (kappa <= 0).any():
        status, failure = "failed_kappa_guard", f"nonpositive TRAIN-calibrated kappa: {kappa.tolist()}"
    else:
        adaptive, _, _, _, adaptive_ids, adaptive_rows = _load_source(seed, device)
        if not same_ordered_indices(ids, adaptive_ids) or not same_ordered_indices(rows, adaptive_rows):
            raise AssertionError("source ordered IDs/rows changed between calibration arms")
        install_adaptive_dynamics(adaptive, kappa.to(device))
        adaptive.membrane_layer.capture_event_history = True
        with torch.no_grad():
            events = _forward_four(adaptive, gamma, rows, device, adaptive=True)
        try:
            activity = activity_summary(events)
        except EligibilityFailure as exc:
            # A scientific guard failure is durable; there is no recalibration.
            status, failure = "failed_activity_guard", str(exc)
            occupancy = events.float().mean(dim=(0, 2, 3))
            mixed = (events.bool().any(dim=-1) & (~events.bool()).any(dim=-1)).float().mean(dim=(0, 2))
            activity = {"occupancy_by_component": occupancy.cpu().tolist(),
                        "mixed_image_patch_fraction_by_component": mixed.cpu().tolist()}

    return_record = {
        "status": status, "failure_reason": failure,
        "experiment": "SW0123", "stage": "train_calibration_activity",
        "seed": seed, "source_core_sha256": sha256(checkpoint),
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_training_manifest_status": manifest.get("status"),
        "source_training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "source_training_ids": ids.tolist(), "gamma_rows": rows.tolist(),
        "gamma_train_sha256": sha256(base.GAMMA_TRAIN),
        "gamma_train_manifest_sha256": sha256(base.GAMMA_TRAIN_MANIFEST),
        "gamma_manifest_status": gamma_manifest.get("status"),
        "encoder_sha256": encoder_sha,
        "preprocessing_sha256": preprocessing_sha,
        "batch_size": BATCH, "time_steps": TIME_STEPS, "settle": SETTLE,
        "calibration_image_ids": ids[:4 * BATCH].tolist(),
        "calibration_gamma_rows": rows[:4 * BATCH].tolist(),
        "kappa_rule": "4*quantile_0.90(clamp_min(legacy_component_out_settled-0.06,0))",
        "kappa": kappa.cpu().tolist(),
        "kappa_sha256": hashlib.sha256(kappa.cpu().numpy().astype("<f4").tobytes()).hexdigest(),
        "spike_rms_rule": "legacy actual component spikes over same64images*256patches*32settled frames; <=1e-8 maps to1",
        "spike_rms": rms.cpu().tolist(),
        "spike_rms_sha256": hashlib.sha256(rms.cpu().numpy().astype("<f4").tobytes()).hexdigest(),
        "adaptive_event_activity": activity,
        "ground_truth_used": False, "optimizer_created": False,
        "optimizer_updates": 0, "source_checkpoint_modified": False,
        "adaptive_probe_core_created": True,
        "implementation_fingerprint": implementation_fingerprint(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(return_record, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return return_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = calibrate(args.seed, args.device, args.output)
    print(json.dumps({"status": result["status"], "stage": result["stage"],
                      "seed": args.seed, "output": str(args.output)}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
