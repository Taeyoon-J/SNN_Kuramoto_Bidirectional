"""Native32 mapped-source versus zero-initialized SW0130 adapter parity check."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for item in (str(ROOT), str(ROOT / "collaborative_test"), str(ROOT / "snn_kuramoto_bidirectional")):
    if item not in sys.path:
        sys.path.insert(0, item)

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding.foundation import (
    encode_native32, make_criterion32, sha256_file,
)
from collaborative_test.SW_0135_native32_spike_binding.resolution import build_core
from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
from snn_kuramoto_bidirectional.loss_function import phase_locking_value
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_input_encoder

PARITY_ATOL = 0.0
PARITY_RTOL = 0.0
HORIZONS = ((64, 32), (1024, 512))
ALLOWED_MICROBATCH = (1, 2, 4, 8, 16)


def implementation_fingerprint():
    """Fingerprint the parity runner and its complete foundation dependency set."""
    from collaborative_test.SW_0135_native32_spike_binding.foundation import implementation_fingerprint as foundation_fp
    result = dict(foundation_fp())
    result["collaborative_test/SW_0135_native32_spike_binding/parity.py"] = sha256_file(HERE / "parity.py")
    result["collaborative_test/SW_0130_phase_state_integration/model.py"] = sha256_file(
        ROOT / "collaborative_test/SW_0130_phase_state_integration/model.py")
    result["collaborative_test/SW_0134_native_spike_binding/rollout.py"] = sha256_file(
        ROOT / "collaborative_test/SW_0134_native_spike_binding/rollout.py")
    return result


def _tensor_difference(reference, actual):
    if tuple(reference.shape) != tuple(actual.shape):
        raise ValueError(f"parity tensor shape mismatch: {tuple(reference.shape)} != {tuple(actual.shape)}")
    ref = reference.detach().to(device="cpu")
    got = actual.detach().to(device="cpu")
    if not torch.isfinite(ref).all() or not torch.isfinite(got).all():
        raise FloatingPointError("native parity contains nonfinite values")
    ref64 = ref.to(torch.float64)
    got64 = got.to(torch.float64)
    delta = (got64 - ref64).abs()
    nonzero_reference = ref64 != 0
    denominator = ref64.abs().clamp_min(1e-300)
    relative = (delta[nonzero_reference] / denominator[nonzero_reference]
                if nonzero_reference.any() else torch.zeros(1, dtype=torch.float64))
    zero_reference_max_abs = (float(delta[~nonzero_reference].max())
                              if (~nonzero_reference).any() else 0.0)
    return {
        "shape": list(ref.shape),
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "max_relative_nonzero_reference": float(relative.max()) if relative.numel() else 0.0,
        "zero_reference_max_abs": zero_reference_max_abs,
        "relative_denominator_floor": 1e-300,
        "exact": bool(torch.equal(ref, got)),
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
    }


def _validate_gamma(gamma, batch, nodes=1024):
    if (not torch.is_tensor(gamma) or gamma.ndim != 3
            or tuple(gamma.shape) != (batch, 8, nodes)
            or not torch.isfinite(gamma).all()):
        raise ValueError("native32 parity requires finite registered gamma [B,8,1024]")


def registered_batch(pool_indices, image_ids, batch_size):
    """Return cache rows and global IDs in registered order (different index spaces)."""
    if (batch_size not in ALLOWED_MICROBATCH or len(pool_indices) < batch_size
            or len(image_ids) < batch_size):
        raise ValueError("registered parity batch is outside available ordered TRAIN source")
    rows = np.asarray(pool_indices[:batch_size], dtype=np.int64)
    ids = [int(value) for value in image_ids[:batch_size]]
    if len(set(rows.tolist())) != batch_size or len(set(ids)) != batch_size:
        raise ValueError("registered parity batch rows and global IDs must be unique")
    return rows, ids


def _loss_values(theta, q, settle, criterion):
    theta_band = theta[:, settle:]
    phase_plv = phase_locking_value(theta_band, combine="mean")
    phase_loss, _ = criterion(plv=phase_plv, theta=theta_band)
    q_loss, _ = criterion(plv=q)
    old = phase_loss + 5.0 * q_loss
    values = {"phase_plv": phase_plv, "phase_loss": phase_loss,
              "positive_q_loss": q_loss, "old_objective": old}
    if any(not torch.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("native32 parity objective contains nonfinite values")
    return values


def _native_capture(core, gamma, steps, settle, criterion):
    with torch.no_grad():
        output = core(gamma, return_core_out=True, return_theta=True,
                      num_time_steps=steps)
        groups, mean_spikes, mean_membrane, theta = output
        components = core.last_component_spikes.detach().clone()
        component_membrane = core.last_component_out.detach().clone()
        q = spike_synchrony_affinity(mean_spikes, components, settle=settle,
                                     affinity_mode="spike")
        losses = _loss_values(theta, q, settle, criterion)
        record = {
            "mean_spikes": mean_spikes.detach().cpu().clone(),
            "mean_membrane": mean_membrane.detach().cpu().clone(),
            "theta": theta.detach().cpu().clone(),
            "component_spikes": components.cpu(),
            "component_membrane": component_membrane.cpu(),
            "q": q.detach().cpu().clone(),
            "losses": {key: value.detach().cpu().clone() for key, value in losses.items()},
        }
    del groups
    return record


def _adapter_capture(wrapped, gamma, steps, settle, criterion):
    with torch.no_grad():
        trace = late_rollout(wrapped, gamma, total_steps=steps, live_tail_steps=0)
        q = spike_synchrony_affinity(trace["spikes"], trace["component_spikes"],
                                     settle=settle, affinity_mode="spike")
        losses = _loss_values(trace["theta"], q, settle, criterion)
        return {
            "mean_spikes": trace["spikes"].detach(),
            "mean_membrane": trace["membrane"].detach(),
            "theta": trace["theta"].detach(),
            "component_spikes": trace["component_spikes"].detach(),
            "component_membrane": trace["component_membrane"].detach(),
            "q": q.detach(),
            "losses": losses,
        }


def _compare_records(reference, actual):
    keys = ("theta", "mean_membrane", "mean_spikes", "component_membrane",
            "component_spikes", "q")
    differences = {key: _tensor_difference(reference[key], actual[key]) for key in keys}
    for key in ("phase_plv", "phase_loss", "positive_q_loss", "old_objective"):
        differences[key] = _tensor_difference(reference["losses"][key], actual["losses"][key])
    return differences


def write_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_parity(seed=0, device="cuda:0", output=None, microbatch_size=4):
    seed = int(seed)
    if seed not in sw130.SEEDS:
        raise ValueError(f"seed must be one of {sw130.SEEDS}")
    if int(microbatch_size) not in ALLOWED_MICROBATCH:
        raise ValueError(f"microbatch size must be one of {ALLOWED_MICROBATCH}")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("registered native32 parity runner requires an available exclusive CUDA device")
    if output is None:
        raise ValueError("--output is required; parity records are create-once")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing parity record: {output}")
    torch.cuda.set_device(device)
    torch.cuda.init()

    checkpoint, source_manifest, manifest, pool, ids, source_sha = sw130.source_contract(seed)
    asset_validation = sw130.validate_rgb_assets()
    if len(pool) != 4096 or len(ids) != 4096:
        raise AssertionError("native32 parity source order must contain the registered 4096 TRAIN images")
    rows, image_ids = registered_batch(pool, ids, int(microbatch_size))
    train_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r")
    rgb_np = np.asarray(train_cache[rows]).copy()
    del train_cache
    if rgb_np.dtype != np.uint8 or tuple(rgb_np.shape) != (int(microbatch_size), 128, 128, 3):
        raise ValueError("registered TRAIN cache returned unexpected native128 RGB batch")
    rgb = torch.from_numpy(rgb_np).to(device)
    encoder = load_input_encoder(str(sw130.ENCODER_SOURCE), num_kernels=8,
                                 kernel_size=3, channels=3, device=device)
    encoder.eval()
    stats = torch.load(sw130.FEATURE_STATS, map_location=device, weights_only=True)
    if stats.get("mode") != "standardize":
        raise AssertionError("registered feature preprocessing mode mismatch")
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    clip = float(stats.get("clip", 3.0))
    patcher = FeaturePatchGammaInitializer(grid_size=32).to(device)
    with torch.no_grad():
        gamma = encode_native32(encoder, patcher, mean, std, clip, rgb).detach()
    _validate_gamma(gamma, int(microbatch_size))
    gamma_sha = hashlib.sha256(gamma.cpu().contiguous().numpy().tobytes()).hexdigest()
    del encoder, patcher, mean, std, rgb
    torch.cuda.empty_cache()

    source_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    criterion = make_criterion32()
    reference_core = build_core(source_state, device=device).eval()
    reference_core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
    reference = {}
    for steps, settle in HORIZONS:
        reference[str(steps)] = _native_capture(reference_core, gamma, steps, settle, criterion)
    del reference_core
    torch.cuda.empty_cache()

    arms = {}
    all_exact = True
    for arm in ("phase", "constant"):
        mapped_core = build_core(source_state, device=device).eval()
        mapped_core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
        wrapped = PhaseStateIntegration(mapped_core, arm)
        if any(torch.count_nonzero(value).item() for value in (wrapped.a_d, wrapped.a_m, wrapped.b)):
            raise AssertionError("zero-initialized SW0130 integration parameters are not zero")
        arm_checks = []
        for steps, settle in HORIZONS:
            actual = _adapter_capture(wrapped, gamma, steps, settle, criterion)
            differences = _compare_records(reference[str(steps)], actual)
            exact = all(value["exact"] for value in differences.values())
            all_exact = all_exact and exact
            arm_checks.append({"steps": steps, "settle": settle,
                               "live_tail_steps": 0,
                               "differences": differences,
                               "exact_parity": exact})
        arms[arm] = arm_checks
        del wrapped, mapped_core
        torch.cuda.empty_cache()

    report = {
        "experiment": "SW0135_native32_spike_binding",
        "stage": "native32_zero_adapter_parity",
        "status": "native32_zero_adapter_parity_passed" if all_exact else "native32_zero_adapter_parity_failed",
        "source_seed": seed,
        "source_core_path": str(checkpoint),
        "source_core_sha256": source_sha,
        "source_manifest_path": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_steps": int(manifest["steps"]),
        "training_image_ids": image_ids,
        "training_image_ids_sha256": hashlib.sha256(np.asarray(image_ids, dtype="<i8").tobytes()).hexdigest(),
        "registered_train_order_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "rgb_asset_validation": asset_validation,
        "encoder_sha256": sha256_file(sw130.ENCODER_SOURCE),
        "feature_preprocessing_sha256": sha256_file(sw130.FEATURE_STATS),
        "gamma_shape": list(gamma.shape),
        "gamma_sha256": gamma_sha,
        "microbatch_size": int(microbatch_size),
        "horizons": [{"steps": steps, "settle": settle} for steps, settle in HORIZONS],
        "arms": arms,
        "ground_truth_used": False,
        "masks_read": False,
        "optimizer_updates": 0,
        "training_admission": False,
        "tolerance": {"atol": PARITY_ATOL, "rtol": PARITY_RTOL,
                      "policy": "exact torch.equal parity; no relaxed tolerance"},
        "implementation_fingerprint": implementation_fingerprint(),
        "limitations": ["Parity only; no warmup, lambda calibration, or scientific training admission."],
    }
    write_once(output, report)
    if not all_exact:
        raise AssertionError(f"native32 zero-adapter parity failed; differences preserved at {output}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=sw130.SEEDS, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch-size", type=int, choices=ALLOWED_MICROBATCH, default=4)
    args = parser.parse_args(argv)
    result = run_parity(args.seed, args.device, args.output, args.microbatch_size)
    print(json.dumps({"status": result["status"], "seed": result["source_seed"],
                      "image_ids": result["training_image_ids"],
                      "microbatch_size": result["microbatch_size"]}, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
