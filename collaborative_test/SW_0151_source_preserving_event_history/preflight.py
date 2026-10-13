"""Read-only actual-source parity/calibration preflight for prospective SW0151."""
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
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _path in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0145_native32_boundary_credit_probe import run as sw145
from collaborative_test.SW_0150_spike_spatial_mixture import preflight as sw150_preflight
from collaborative_test.SW_0151_source_preserving_event_history.adaptive_rollout import (
    adaptive_rollout,
)
from collaborative_test.SW_0151_source_preserving_event_history.event_history import (
    AdaptiveEventHistory,
)


SEED = 1
FIRST_TRAIN_COUNT = 64
MICRO_BATCH = 4
STEPS = 1024
SETTLE = 512
TRACE_FIELDS = ("component_membrane", "component_spikes", "component_gates",
                "membrane", "spikes", "theta")
PARITY_PATH = (ROOT / "collaborative_test/SW_0135_native32_spike_binding"
               / "results_archive/native32_zero_adapter_parity_seed1_b4.json")
OUTPUT_DEFAULT = HERE / "results_archive" / "adaptive_source_preflight_seed1.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint() -> dict[str, str]:
    paths = (
        HERE / "event_history.py", HERE / "adaptive_rollout.py", HERE / "preflight.py",
        HERE / "protocol.json",
        ROOT / "collaborative_test/SW_0135_native32_spike_binding/parity.py",
        ROOT / "collaborative_test/SW_0145_native32_boundary_credit_probe/run.py",
        ROOT / "collaborative_test/SW_0150_spike_spatial_mixture/preflight.py",
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"preflight dependency missing: {missing[0]}")
    fingerprint = dict(sw135.implementation_fingerprint())
    fingerprint.update({path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths})
    return fingerprint


def validate_registered_train_order(image_ids, pool_rows, *, count: int = 4096,
                                   expected_ids=None, expected_rows=None) -> tuple[list[int], np.ndarray]:
    """Fail closed unless IDs and source-contract rows form the ordered TRAIN pool."""
    ids = [int(value) for value in image_ids]
    rows = np.asarray(pool_rows, dtype=np.int64)
    if (len(ids) != count or len(set(ids)) != count or rows.shape != (count,)
            or len(np.unique(rows)) != count or np.any(rows < 0) or np.any(rows >= 70000)):
        raise ValueError("registered source must contain 4096 unique ordered TRAIN IDs and rows")
    if expected_ids is not None and ids != [int(value) for value in expected_ids]:
        raise ValueError("source image IDs do not match the registered order")
    if expected_rows is not None and not np.array_equal(rows, np.asarray(expected_rows, dtype=np.int64)):
        raise ValueError("source pool rows do not match the registered order")
    return ids, rows


def validate_train_rgb_schema(cache) -> None:
    """Check the registered full RGB cache header without reading its contents."""
    if tuple(cache.shape) != (70000, 128, 128, 3) or cache.dtype != np.uint8:
        raise ValueError("TRAIN RGB cache must have registered uint8 shape [70000,128,128,3]")


def module_state_digest(modules: dict[str, torch.nn.Module]) -> dict[str, str]:
    """Hash all parameters and buffers, including nonpersistent buffers."""
    result = {}
    for module_name, module in modules.items():
        digest = hashlib.sha256()
        values = [(f"parameter:{name}", value) for name, value in module.named_parameters()]
        values += [(f"buffer:{name}", value) for name, value in module.named_buffers()]
        for name, tensor in sorted(values, key=lambda item: item[0]):
            value = tensor.detach().contiguous().cpu()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        result[module_name] = digest.hexdigest()
    return result


def new_rms_accumulator() -> tuple[np.ndarray, int]:
    return np.zeros(4, dtype=np.float64), 0


def accumulate_threshold_rms(accumulator, component_membrane: torch.Tensor,
                             base_threshold: torch.Tensor, *, settle: int = SETTLE):
    """Accumulate per-component float64 RMS(M - native threshold), settled only."""
    sums, count = accumulator
    if (not torch.is_tensor(component_membrane) or component_membrane.ndim != 4
            or component_membrane.shape[1] != 4 or component_membrane.shape[-1] <= settle
            or component_membrane.dtype != torch.float32):
        raise ValueError("native trace must be float32 [B,4,N,T] with settled samples")
    threshold = torch.as_tensor(base_threshold)
    if threshold.shape != (4,) or not torch.isfinite(threshold).all():
        raise ValueError("base threshold must be finite [4]")
    if not torch.isfinite(component_membrane).all():
        raise FloatingPointError("native membrane trace is nonfinite")
    diff = (component_membrane[..., settle:].to(torch.float64)
            - threshold.to(device=component_membrane.device, dtype=torch.float64).view(1, 4, 1, 1))
    sums += diff.square().sum(dim=(0, 2, 3)).detach().cpu().numpy()
    count += int(diff.shape[0] * diff.shape[2] * diff.shape[3])
    return sums, count


def finish_threshold_rms(accumulator) -> list[float]:
    sums, count = accumulator
    sums = np.asarray(sums, dtype=np.float64)
    if count <= 0 or sums.shape != (4,) or not np.isfinite(sums).all():
        raise ValueError("threshold RMS requires finite accumulated samples")
    values = np.sqrt(sums / count)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("per-component settled membrane RMS calibration is nonpositive or nonfinite")
    return [float(value) for value in values]


def assert_exact_trace_parity(native: dict, adaptive: dict, native_q: torch.Tensor,
                              adaptive_q: torch.Tensor) -> None:
    for key in TRACE_FIELDS:
        left, right = native.get(key), adaptive.get(key)
        if not torch.is_tensor(left) or not torch.is_tensor(right) or not torch.equal(left, right):
            raise AssertionError(f"kappa-zero native/adaptive trace mismatch: {key}")
    if not torch.is_tensor(native_q) or not torch.is_tensor(adaptive_q) or not torch.equal(native_q, adaptive_q):
        raise AssertionError("kappa-zero native/adaptive actual-spike Q mismatch")


def event_summaries(trace: dict) -> dict:
    events = trace.get("component_events")
    history = trace.get("component_event_history")
    if (not torch.is_tensor(events) or events.ndim != 4 or events.shape[1] != 4
            or events.shape[-1] != STEPS or not torch.isfinite(events).all()):
        raise ValueError("adaptive event trace must be finite [B,4,N,1024]")
    if (not torch.is_tensor(history) or history.shape != events.shape
            or not torch.isfinite(history).all()):
        raise ValueError("adaptive event-history trace must match events and be finite")
    settled_events = events[..., SETTLE:].to(torch.float64)
    settled_history = history[..., SETTLE:].to(torch.float64)
    return {
        "settled_event_on_fraction_by_component": settled_events.mean(dim=(0, 2, 3)).cpu().tolist(),
        "settled_event_exact_zero_fraction_by_component": (settled_events == 0).to(torch.float64).mean(dim=(0, 2, 3)).cpu().tolist(),
        "settled_event_exact_one_fraction_by_component": (settled_events == 1).to(torch.float64).mean(dim=(0, 2, 3)).cpu().tolist(),
        "settled_history_variance_by_component": settled_history.var(dim=(0, 2, 3), unbiased=False).cpu().tolist(),
    }


def merge_history_moments(mean: np.ndarray, m2: np.ndarray, count: int,
                          values: torch.Tensor) -> tuple[np.ndarray, np.ndarray, int]:
    """Stable pooled float64 population moments for [B,4,N,T] history values."""
    if (not torch.is_tensor(values) or values.ndim != 4 or values.shape[1] != 4
            or values.dtype != torch.float64 or not torch.isfinite(values).all()):
        raise ValueError("history moment input must be finite float64 [B,4,N,T]")
    batch_count = int(values.shape[0] * values.shape[2] * values.shape[3])
    if batch_count <= 0 or count < 0 or np.asarray(mean).shape != (4,) or np.asarray(m2).shape != (4,):
        raise ValueError("invalid history moment accumulator")
    batch_mean_t = values.mean(dim=(0, 2, 3))
    centered = values - batch_mean_t.view(1, 4, 1, 1)
    batch_m2 = centered.square().sum(dim=(0, 2, 3)).detach().cpu().numpy()
    batch_mean = batch_mean_t.detach().cpu().numpy()
    if count == 0:
        return batch_mean.copy(), batch_m2, batch_count
    total = count + batch_count
    delta = batch_mean - mean
    merged_mean = mean + delta * (batch_count / total)
    merged_m2 = m2 + batch_m2 + np.square(delta) * (count * batch_count / total)
    return merged_mean, merged_m2, total


def _rng_snapshot(device):
    return torch.random.get_rng_state(), torch.cuda.get_rng_state(device)


def _rng_restore(snapshot, device):
    torch.random.set_rng_state(snapshot[0])
    torch.cuda.set_rng_state(snapshot[1], device)


def _sync(device):
    torch.cuda.synchronize(device)


def run_preflight(*, device="cuda:0", output=OUTPUT_DEFAULT):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("actual-source SW0151 preflight requires an available CUDA device")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preflight output already exists: {output}")

    foundation = sw135.load_native32_foundation(SEED, device, verify_rgb_assets=True)
    if (foundation.seed != SEED or len(foundation.image_ids) != 4096
            or len(foundation.pool_indices) != 4096
            or foundation.provenance.get("ground_truth_used") is not False):
        raise ValueError("loaded source1 foundation does not match registered source contract")
    checkpoint, source_manifest_path, manifest, source_rows, source_ids, source_sha = sw130.source_contract(SEED)
    source_ids, source_rows = validate_registered_train_order(source_ids, source_rows)
    foundation_ids, foundation_rows = validate_registered_train_order(
        foundation.image_ids, foundation.pool_indices)
    if (foundation_ids != source_ids or not np.array_equal(foundation_rows, source_rows)
            or foundation.provenance["source_core_sha256"] != source_sha):
        raise ValueError("foundation and registered source1 order/checkpoint do not agree")
    asset_record = sw130.validate_rgb_assets()
    if foundation.provenance.get("rgb_asset_validation") != asset_record:
        raise ValueError("foundation did not bind the current validated RGB cache bytes")
    parity, parity_sha = sw150_preflight.load_and_validate_parity(
        PARITY_PATH, source_sha=source_sha,
        source_manifest_sha=sw135.sha256_file(source_manifest_path),
        training_ids=source_ids, asset_validation=asset_record)
    if parity.get("status") != "native32_zero_adapter_parity_passed":
        raise ValueError("saved SW0135 source1 native32 parity is not valid")

    # SW0135 parity constructs the encoder and both cores in eval mode; preserve
    # that registered inference behavior and prove no parameters/buffers change.
    foundation.encoder.eval().requires_grad_(False)
    foundation.patcher.eval().requires_grad_(False)
    foundation.wrapped.eval().requires_grad_(False)
    if foundation.encoder.training or foundation.wrapped.training or foundation.patcher.training:
        raise AssertionError("source parity modules must remain in registered eval mode")
    event_history = AdaptiveEventHistory().to(device).eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in event_history.parameters()):
        raise AssertionError("event module unexpectedly requires gradients for kappa-zero parity")
    with torch.no_grad():
        event_history.kappa_raw.zero_()
    modules = {"encoder": foundation.encoder, "patcher": foundation.patcher,
               "wrapped": foundation.wrapped, "event_history": event_history}
    model_state_before = module_state_digest(modules)

    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r")
    validate_train_rgb_schema(cache)
    ids64 = source_ids[:FIRST_TRAIN_COUNT]
    rows64 = np.asarray(source_rows[:FIRST_TRAIN_COUNT], dtype=np.int64)
    scale_accumulator = new_rms_accumulator()
    event_counts = None
    history_mean = None
    history_m2 = None
    history_count = 0
    native_elapsed = adaptive_elapsed = 0.0
    exact_trace_batches = exact_q_batches = 0
    torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, FIRST_TRAIN_COUNT, MICRO_BATCH):
        batch_rows = rows64[start:start + MICRO_BATCH]
        rgb = np.asarray(cache[batch_rows]).copy()
        if rgb.shape != (MICRO_BATCH, 128, 128, 3) or rgb.dtype != np.uint8:
            raise ValueError("first64 source TRAIN read did not produce registered uint8 RGB microbatches")
        rgb_tensor = torch.from_numpy(rgb).to(device=device)
        with torch.no_grad():
            gamma = foundation.encode(rgb_tensor)
            if tuple(gamma.shape) != (MICRO_BATCH, 8, 1024) or gamma.dtype != torch.float32:
                raise AssertionError("native32 source encoder returned an unexpected gamma contract")
            rng = _rng_snapshot(device)
            _sync(device)
            started = time.perf_counter()
            native = foundation.rollout(gamma, total_steps=STEPS, live_tail_steps=0)
            _sync(device)
            native_elapsed += time.perf_counter() - started
            scale_accumulator = accumulate_threshold_rms(
                scale_accumulator, native["component_membrane"],
                (0.06 * torch.exp(foundation.wrapped.b.detach())).float(), settle=SETTLE)
            native_q = sw145._actual_q(native)

            # With exact kappa=0 the scale is mathematically inactive. Use fixed
            # positive unit scale for parity while calibrating the registered s
            # from the same native first64 TRAIN traces without a second rollout.
            _rng_restore(rng, device)
            _sync(device)
            started = time.perf_counter()
            adaptive = adaptive_rollout(
                foundation.wrapped, gamma, event_history, total_steps=STEPS,
                live_tail_steps=0, scale=1.0)
            _sync(device)
            adaptive_elapsed += time.perf_counter() - started
            adaptive_q = sw145._actual_q(adaptive)
            assert_exact_trace_parity(native, adaptive, native_q, adaptive_q)
            exact_trace_batches += 1
            exact_q_batches += 1
            summaries = event_summaries(adaptive)
            if event_counts is None:
                event_counts = {key: np.zeros(4, dtype=np.float64)
                                for key in ("on", "zero", "one")}
                history_mean = np.zeros(4, dtype=np.float64)
                history_m2 = np.zeros(4, dtype=np.float64)
            n = MICRO_BATCH * 1024 * (STEPS - SETTLE)
            event_counts["on"] += np.asarray(summaries["settled_event_on_fraction_by_component"]) * n
            event_counts["zero"] += np.asarray(summaries["settled_event_exact_zero_fraction_by_component"]) * n
            event_counts["one"] += np.asarray(summaries["settled_event_exact_one_fraction_by_component"]) * n
            h = adaptive["component_event_history"][..., SETTLE:].to(torch.float64)
            history_mean, history_m2, history_count = merge_history_moments(
                history_mean, history_m2, history_count, h)
            del rgb, rgb_tensor, gamma, native, adaptive, native_q, adaptive_q, h

    scale = finish_threshold_rms(scale_accumulator)
    model_state_after = module_state_digest(modules)
    if model_state_after != model_state_before:
        raise AssertionError("source parameters or persistent buffers changed during read-only preflight")
    if exact_trace_batches != FIRST_TRAIN_COUNT // MICRO_BATCH or exact_q_batches != exact_trace_batches:
        raise AssertionError("not all registered first64 microbatches passed exact trace/Q parity")
    event_stats = {
        "settled_event_on_fraction_by_component": (event_counts["on"] / history_count).tolist(),
        "settled_event_exact_zero_fraction_by_component": (event_counts["zero"] / history_count).tolist(),
        "settled_event_exact_one_fraction_by_component": (event_counts["one"] / history_count).tolist(),
        "settled_history_variance_by_component": (history_m2 / history_count).tolist(),
    }
    if (not np.isfinite(np.asarray(list(event_stats.values()), dtype=np.float64)).all()
            or not all(math.isfinite(value) for value in scale)):
        raise FloatingPointError("preflight diagnostics/calibration are nonfinite")
    report = {
        "experiment": "SW0151_source_preserving_event_history",
        "stage": "actual_source_native32_parity_and_scale_preflight",
        "status": "passed_kappa_zero_actual_source_parity",
        "seed": SEED,
        "training_admission": False,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source": {
            "core_checkpoint_path": str(checkpoint),
            "core_checkpoint_sha256": source_sha,
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": sw135.sha256_file(source_manifest_path),
            "source_manifest_updates": int(manifest["steps"]),
            "ordered_train_ids_sha256": hashlib.sha256(np.asarray(source_ids, dtype="<i8").tobytes()).hexdigest(),
            "first64_train_ids": ids64,
            "first64_pool_rows": rows64.tolist(),
            "rgb_asset_validation": asset_record,
            "foundation_implementation_fingerprint": foundation.provenance["implementation_fingerprint"],
            "saved_native32_parity_path": str(PARITY_PATH),
            "saved_native32_parity_sha256": parity_sha,
        },
        "model": {
            "native32_grid": [32, 32], "patches": 1024,
            "steps": STEPS, "settle": SETTLE, "live_tail_steps": 0,
            "base_threshold": "0.06*exp(existing wrapped.b); same base threshold for event and unchanged reset",
            "new_parameters": 8,
            "lambda_initial": .95, "kappa_initial": 0.0,
            "rollout_source_sha256": adaptive_rollout.__globals__["SW0134_ROLLOUT_SHA256"],
            "dependency_sha256": adaptive_rollout.__globals__["SOURCE_DEPENDENCY_SHA256"],
            "registered_module_state_sha256_before": model_state_before,
            "registered_module_state_sha256_after": model_state_after,
            "registered_module_state_unchanged": True,
            "module_mode": "eval, matching SW0135 native32 parity runner",
        },
        "parity": {"microbatch": MICRO_BATCH, "batches": exact_trace_batches,
                   "exact_trace_batches": exact_trace_batches, "exact_production_q_batches": exact_q_batches,
                   "all_six_trace_fields_exact": True, "production_actual_spike_q_exact": True,
                   "kappa_zero_scale_for_parity": 1.0},
        "calibration": {"method": "float64 RMS of settled native component_membrane minus existing 0.06*exp(b), exact first64 source TRAIN images",
                        "fixed_per_component_scale": scale,
                        "source_rollouts_reused_for_calibration": True},
        "descriptive_events": event_stats,
        "timing_seconds": {"native_rollout_total": native_elapsed,
                           "adaptive_rollout_total": adaptive_elapsed,
                           "adaptive_minus_native": adaptive_elapsed - native_elapsed},
        "peak_cuda_memory_bytes": {"allocated": int(torch.cuda.max_memory_allocated(device)),
                                    "reserved": int(torch.cuda.max_memory_reserved(device))},
        "implementation_fingerprint": implementation_fingerprint(),
        "ground_truth_used": False,
        "masks_read": False,
        "validation_or_test_ids_read": False,
        "optimizer_updates": 0,
        "claims": ["synthetic helper tests and this source parity/calibration preflight do not demonstrate learned-history efficacy or segmentation improvement"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args(argv)
    report = run_preflight(device=args.device, output=args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "training_admission": False}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
