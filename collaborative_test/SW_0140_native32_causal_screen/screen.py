"""GT-free, no-update causal screen for the registered native32 source."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _path in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding.foundation import load_native32_foundation
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0135_native32_spike_binding.foundation import sha256_file
from collaborative_test.SW_0140_native32_causal_screen.causal_rollout import (
    CONDITIONS, TRACE_FIELDS, assert_none_parity, causal_rollout,
)

SCREEN_IMAGES = 4
STEPS = 1024


def _state_sha(module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def screen_fingerprint(base_fingerprint):
    result = dict(base_fingerprint)
    for path in (HERE / "causal_rollout.py", HERE / "screen.py", HERE / "protocol.json"):
        result[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return result


def first_train_records(pool, ids):
    """Validate and return the registered ordered pool rows and their image IDs."""
    pool = np.asarray(pool, dtype=np.int64)
    ids = np.asarray(ids, dtype=np.int64)
    if pool.shape != (4096,) or ids.shape != (4096,):
        raise ValueError("source contract must provide 4096 ordered TRAIN rows and IDs")
    if ((pool < 0).any() or (pool >= 70000).any()
            or len(set(pool.tolist())) != 4096 or len(set(ids.tolist())) != 4096):
        raise ValueError("registered source TRAIN rows/IDs must be unique and in range")
    return pool[:SCREEN_IMAGES].copy(), ids[:SCREEN_IMAGES].copy()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _write_once(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_safe(record), indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def _assert_close_parity(expected, actual):
    for key in TRACE_FIELDS:
        if not torch.equal(expected[key], actual[key]):
            raise AssertionError(f"NONE differs from frozen late_rollout in {key}")


def _assert_finite_trace(trace):
    keys = (*TRACE_FIELDS, "component_carrier", "component_events",
            "component_natural_threshold_events")
    missing = [name for name in TRACE_FIELDS if name not in trace]
    if missing:
        raise KeyError(f"SW0140 trace is missing registered fields: {missing}")
    keys = [name for name in keys if name in trace]
    checks = [torch.isfinite(trace[name]).all() for name in keys]
    if not bool(torch.stack(checks).all().item()):
        raise FloatingPointError("SW0140 trajectory contains nonfinite trace values")


@torch.no_grad()
def run_screen(seed: int, device: str, output: str | Path, *, foundation_loader=None,
               train_cache=None, _test_only_allow_cpu=False):
    """Run the four ordered TRAIN images; never opens labels or performs updates."""
    seed = int(seed)
    if seed not in sw130.SEEDS:
        raise ValueError(f"seed must be one of {sw130.SEEDS}")
    device_obj = torch.device(device)
    if device_obj.type != "cuda" and not _test_only_allow_cpu:
        raise ValueError("SW0140 registered screen requires an explicitly leased CUDA device")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SW0140 screen output: {output}")

    checkpoint, manifest_path, manifest, pool, ids, source_sha = sw130.source_contract(seed)
    if int(manifest.get("steps", -1)) != 256:
        raise AssertionError("SW0140 expects the registered 256-step SW0097 source")
    first_pool, first_ids = first_train_records(pool, ids)

    if foundation_loader is None:
        foundation_loader = load_native32_foundation
    foundation = foundation_loader(seed, device_obj, verify_rgb_assets=True)
    if foundation.provenance["source_core_sha256"] != source_sha:
        raise AssertionError("foundation source checkpoint differs from source contract")
    if (not np.array_equal(foundation.pool_indices, pool)
            or list(map(int, foundation.image_ids)) != list(map(int, ids))):
        raise AssertionError("native32 foundation TRAIN ordering differs from SW0097")
    if train_cache is None:
        train_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r")
    if tuple(train_cache.shape) != (70000, 128, 128, 3) or train_cache.dtype != np.uint8:
        raise AssertionError("registered TRAIN RGB cache must be uint8 [70000,128,128,3]")

    wrapped = foundation.wrapped
    wrapped.eval()
    wrapped.requires_grad_(False)
    foundation.encoder.eval()
    foundation.encoder.requires_grad_(False)
    core_sha_before = _state_sha(wrapped)
    encoder_sha_before = _state_sha(foundation.encoder)
    original_k = float(wrapped.core.kuramoto.K)
    if device_obj.type == "cuda":
        torch.cuda.set_device(device_obj)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device_obj)
        torch.cuda.synchronize(device_obj)
    started = time.perf_counter()
    batches = []
    for ordinal, (pool_row, image_id) in enumerate(zip(first_pool.tolist(), first_ids.tolist())):
        rgb = torch.from_numpy(np.asarray(train_cache[pool_row:pool_row + 1]).copy()).to(device_obj)
        gamma = foundation.encode(rgb).detach()
        if tuple(gamma.shape) != (1, 8, 1024) or not torch.isfinite(gamma).all():
            raise AssertionError("native32 TRAIN gamma must be finite [1,8,1024]")
        frozen = late_rollout(wrapped, gamma, total_steps=STEPS, live_tail_steps=0)
        _assert_finite_trace(frozen)
        trace = causal_rollout(wrapped, gamma, intervention="none", total_steps=STEPS)
        _assert_finite_trace(trace)
        _assert_close_parity(frozen, trace)
        conditions = {"none": {"parity": True, "diagnostics": trace["diagnostics"]}}
        for intervention in CONDITIONS[1:]:
            changed = causal_rollout(wrapped, gamma, intervention=intervention,
                                     total_steps=STEPS)
            _assert_finite_trace(changed)
            if (intervention != "kuramoto_interaction_off"
                    and not torch.equal(changed["theta"], frozen["theta"])):
                raise AssertionError(f"downstream intervention changed source theta: {intervention}")
            record = {"diagnostics": changed["diagnostics"]}
            for field in TRACE_FIELDS:
                delta = (changed[field].float() - frozen[field].float()).abs()
                record[field] = {
                    "max_abs_difference": float(delta.max().item()),
                    "mean_abs_difference": float(delta.mean().item()),
                    "changed_elements": int(torch.count_nonzero(delta).item()),
                }
            if intervention == "binary_events_forced_on":
                record["natural_threshold_event_occupancy"] = changed["diagnostics"][
                    "natural_threshold_event_occupancy"]
                record["imposed_event_occupancy"] = changed["diagnostics"]["event_occupancy"]
            conditions[intervention] = record
        batches.append({"ordinal": ordinal, "pool_row": int(pool_row),
                        "image_id": int(image_id), "gamma_shape": list(gamma.shape),
                        "conditions": conditions})
    if device_obj.type == "cuda":
        torch.cuda.synchronize(device_obj)
    elapsed = time.perf_counter() - started
    core_sha_after = _state_sha(wrapped)
    encoder_sha_after = _state_sha(foundation.encoder)
    if core_sha_after != core_sha_before or encoder_sha_after != encoder_sha_before:
        raise AssertionError("SW0140 screen changed frozen source or encoder state")
    if float(wrapped.core.kuramoto.K) != original_k:
        raise AssertionError("Kuramoto K was not restored after the intervention screen")

    record = {
        "experiment": "SW0140_native32_causal_screen",
        "status": "screen_complete",
        "seed": seed,
        "device": str(device_obj),
        "steps": STEPS,
        "images": SCREEN_IMAGES,
        "image_ids": [int(v) for v in first_ids],
        "pool_rows": [int(v) for v in first_pool],
        "ground_truth_used": False,
        "optimizer_updates": 0,
        "conditions": list(CONDITIONS),
        "condition_counts": {"none": 1, "interventions": 6},
        "source_core_sha256": source_sha,
        "source_manifest_sha256": sha256_file(manifest_path),
        "encoder_sha256": foundation.provenance["encoder_sha256"],
        "feature_preprocessing_sha256": foundation.provenance["feature_preprocessing_sha256"],
        "train_rgb_sha256": foundation.provenance["rgb_asset_validation"]["train_cache_sha256"],
        "implementation_fingerprint": screen_fingerprint(
            foundation.provenance["implementation_fingerprint"]),
        "source_state_before_sha256": core_sha_before,
        "source_state_after_sha256": core_sha_after,
        "kuramoto_K_before": original_k,
        "kuramoto_K_after": float(wrapped.core.kuramoto.K),
        "encoder_state_before_sha256": encoder_sha_before,
        "encoder_state_after_sha256": encoder_sha_after,
        "elapsed_seconds": elapsed,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device_obj)),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device_obj)),
        "batches": batches,
        "interpretation_limit": "GT-free causal trace screen only; not a segmentation metric or gate-necessity proof",
    }
    _write_once(output, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=sw130.SEEDS, required=True)
    parser.add_argument("--device", required=True, help="leased CUDA device, e.g. cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_screen(args.seed, args.device, args.output)
    print(json.dumps({"status": result["status"], "seed": result["seed"],
                      "output": str(args.output), "elapsed_seconds": result["elapsed_seconds"]},
                     allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
