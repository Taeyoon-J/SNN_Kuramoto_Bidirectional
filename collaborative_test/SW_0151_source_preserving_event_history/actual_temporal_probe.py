"""One-image read-only actual-source temporal-credit diagnostic for SW0151.

Requires a completed, strictly validated SW0151 source parity/calibration
report. This records feasibility diagnostics only: it is not training
admission, an objective, a performance test, or evidence of segmentation gain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

# The existing queue module hides CUDA in its supervisor. Preserve the child
# mapping while importing its strict artifact validator; Torch is imported
# first and the environment is restored before the first CUDA query.
_CUDA_VISIBILITY = os.environ.get("CUDA_VISIBLE_DEVICES")
from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0151_source_preserving_event_history import preflight
from collaborative_test.SW_0151_source_preserving_event_history import preflight_queue
from collaborative_test.SW_0151_source_preserving_event_history.temporal_credit_probe import (
    run_temporal_credit_probe,
)
if _CUDA_VISIBILITY is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _CUDA_VISIBILITY
else:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)


SEED = 1
TRAIN_COUNT = 4096
PROBE_KAPPA = 0.1
STEPS = 1024
SETTLE = 512
LIVE_TAIL = 64
OUTPUT_DEFAULT = HERE / "results_archive" / "actual_temporal_credit_seed1_row0.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint() -> dict[str, str]:
    paths = (
        HERE / "actual_temporal_probe.py",
        HERE / "temporal_credit_probe.py",
        HERE / "event_history.py",
        HERE / "adaptive_rollout.py",
        HERE / "preflight.py",
        HERE / "preflight_queue.py",
        HERE / "protocol.json",
    )
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("an SW0151 actual temporal probe dependency is missing")
    result = dict(preflight.implementation_fingerprint())
    result.update({p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths})
    return result


def validated_preflight(path: str | Path, *, validator=None) -> tuple[dict, str]:
    """Apply the existing strict SW0151 queue validator and return bound report."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required actual-source parity/calibration report missing: {path}")
    if validator is None:
        validator = preflight_queue.validate_report
    if validator(path) is not True:
        raise ValueError("existing SW0151 strict preflight validator rejected the report")
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if (report.get("status") != "passed_kappa_zero_actual_source_parity"
            or report.get("training_admission") is not False
            or report.get("ground_truth_used") is not False
            or report.get("optimizer_updates") != 0):
        raise ValueError("SW0151 preflight is not a completed read-only parity/calibration record")
    scales = report.get("calibration", {}).get("fixed_per_component_scale")
    if (not isinstance(scales, list) or len(scales) != 4
            or not np.isfinite(np.asarray(scales, dtype=np.float64)).all()
            or np.any(np.asarray(scales, dtype=np.float64) <= 0)):
        raise ValueError("strictly validated preflight lacks its fixed positive component scales")
    return report, sha256_file(path)


def validate_first_row(cache, *, expected_id: int, expected_row: int,
                       registered_ids, registered_rows) -> np.ndarray:
    """Read one exact first TRAIN row, with no conversion before encode()."""
    ids = [int(v) for v in registered_ids]
    rows = np.asarray(registered_rows, dtype=np.int64)
    if len(ids) != TRAIN_COUNT or rows.shape != (TRAIN_COUNT,):
        raise ValueError("source contract must contain the registered ordered 4096 TRAIN rows")
    if ids[0] != int(expected_id) or int(rows[0]) != int(expected_row):
        raise ValueError("preflight first TRAIN ID/row does not match current source contract")
    image = np.asarray(cache[int(expected_row)])
    if image.shape != (128, 128, 3) or image.dtype != np.uint8:
        raise ValueError("registered source row must be uint8 RGB [128,128,3]")
    if not np.isfinite(image).all():
        raise ValueError("source RGB row contains invalid values")
    return image.copy()


def encode_once(foundation, image_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    """Pass the source-cache uint8 row once to the registered /255 encoder."""
    if image_uint8.shape != (128, 128, 3) or image_uint8.dtype != np.uint8:
        raise ValueError("encode_once requires one unmodified uint8 native128 RGB image")
    tensor = torch.from_numpy(np.ascontiguousarray(image_uint8[None])).to(device=device)
    with torch.no_grad():
        gamma = foundation.encode(tensor)
    if gamma.shape != (1, 8, 1024) or gamma.dtype != torch.float32 or not torch.isfinite(gamma).all():
        raise ValueError("registered source encoder did not return finite float32 [1,8,1024]")
    return gamma.detach()


def run_probe(*, device="cuda:0", preflight_path=preflight.OUTPUT_DEFAULT,
              output=OUTPUT_DEFAULT):
    device = torch.device(device)
    output, preflight_path = Path(output), Path(preflight_path)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("actual-source temporal-credit probe requires an available CUDA device")
    if output.exists():
        raise FileExistsError(f"actual temporal-probe output already exists: {output}")

    parity_report, parity_sha = validated_preflight(preflight_path)
    source_ids = [int(v) for v in parity_report["source"]["first64_train_ids"]]
    source_rows = [int(v) for v in parity_report["source"]["first64_pool_rows"]]
    if len(source_ids) != 64 or len(source_rows) != 64:
        raise ValueError("validated source preflight must bind exactly 64 first TRAIN IDs/rows")

    foundation = sw135.load_native32_foundation(SEED, device, verify_rgb_assets=True)
    checkpoint, manifest_path, manifest, pool, ids, source_sha = sw130.source_contract(SEED)
    ids, pool = preflight.validate_registered_train_order(ids, pool)
    if (ids[:64] != source_ids or pool[:64].tolist() != source_rows
            or foundation.image_ids != ids or not np.array_equal(foundation.pool_indices, pool)
            or foundation.provenance.get("source_core_sha256") != source_sha
            or foundation.provenance.get("rgb_asset_validation") != sw130.validate_rgb_assets()
            or foundation.provenance.get("source_manifest_sha256") != sw135.sha256_file(manifest_path)):
        raise ValueError("current seed1 foundation/source order/assets differ from the validated preflight")
    if parity_report["source"].get("core_checkpoint_sha256") != source_sha:
        raise ValueError("preflight is bound to a different SW0097 source checkpoint")
    if parity_report["source"].get("foundation_implementation_fingerprint") != sw135.implementation_fingerprint():
        raise ValueError("preflight source foundation implementation fingerprint is stale")
    if parity_report["implementation_fingerprint"] != preflight.implementation_fingerprint():
        raise ValueError("preflight helper/protocol fingerprint is stale")

    for module in (foundation.encoder, foundation.patcher, foundation.wrapped):
        module.eval().requires_grad_(False)
    modules_before = preflight.module_state_digest({
        "encoder": foundation.encoder,
        "patcher": foundation.patcher,
        "wrapped": foundation.wrapped,
    })
    rgb_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r")
    preflight.validate_train_rgb_schema(rgb_cache)
    image = validate_first_row(rgb_cache, expected_id=source_ids[0],
                               expected_row=source_rows[0], registered_ids=ids,
                               registered_rows=pool)
    input_sha = hashlib.sha256(image.tobytes(order="C")).hexdigest()
    gamma = encode_once(foundation, image, device)
    gamma_sha = hashlib.sha256(gamma.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
    scales = parity_report["calibration"]["fixed_per_component_scale"]

    torch.cuda.reset_peak_memory_stats(device)
    result = run_temporal_credit_probe(
        foundation.wrapped, gamma, scale=scales, positive_kappa=PROBE_KAPPA,
        total_steps=STEPS, live_tail_steps=LIVE_TAIL, settle_steps=SETTLE)
    if (result["status"] != "probe_complete"
            or result["zero_kappa_trace_exact_native"] is not True
            or result["zero_kappa_actual_spike_q_exact_native"] is not True
            or result["positive_live_vs_detached_forward_exact"] is not True
            or result["positive_live_vs_detached_q_exact"] is not True):
        raise AssertionError("actual source temporal-credit probe failed required parity/control checks")
    for branch in ("positive_live_q_gradient", "positive_detached_control_q_gradient"):
        if any(metric.get("finite") is not True for metric in result[branch].values()):
            raise FloatingPointError(f"nonfinite {branch} measurement")
    modules_after = preflight.module_state_digest({
        "encoder": foundation.encoder,
        "patcher": foundation.patcher,
        "wrapped": foundation.wrapped,
    })
    if modules_before != modules_after or result["source_model_state_unchanged"] is not True:
        raise AssertionError("read-only probe changed source model state")

    report = {
        "experiment": "SW0151_source_preserving_event_history",
        "stage": "single_registered_train_image_temporal_credit_probe",
        "status": "probe_complete",
        "training_admission": False,
        "seed": SEED,
        "scope": {"split": "registered TRAIN only", "image_count": 1,
                  "image_id": source_ids[0], "pool_row": source_rows[0],
                  "selection": "first image in the immutable registered seed1 TRAIN order",
                  "ground_truth_used": False, "masks_read": False,
                  "optimizer_updates": 0},
        "preflight": {"path": str(preflight_path), "sha256": parity_sha,
                      "status": parity_report["status"],
                      "fixed_per_component_scale": scales},
        "source": {"core_checkpoint_path": str(checkpoint),
                   "core_checkpoint_sha256": source_sha,
                   "source_manifest_path": str(manifest_path),
                   "source_manifest_sha256": sw135.sha256_file(manifest_path),
                   "ordered_train_ids_sha256": hashlib.sha256(
                       np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                   "rgb_asset_validation": foundation.provenance["rgb_asset_validation"],
                   "input_rgb_uint8_sha256": input_sha,
                   "gamma_float32_sha256": gamma_sha,
                   "foundation_implementation_fingerprint": foundation.provenance["implementation_fingerprint"]},
        "probe": {"positive_kappa": PROBE_KAPPA, "steps": STEPS,
                  "settle_steps_for_q": SETTLE, "live_tail_steps": LIVE_TAIL,
                  "forward_control": "same positive-kappa forward values; auxiliary event/history states detached for backward control only",
                  "measurements": result},
        "model_state_sha256_before": modules_before,
        "model_state_sha256_after": modules_after,
        "implementation_fingerprint": implementation_fingerprint(),
        "gpu_used": True,
        "training_or_optimizer_updates": False,
        "claims": ["one-image source-specific feasibility diagnostic only",
                   "zero gradient/event/Q changes are descriptive, not an admission decision",
                   "no training objective, model-performance, or segmentation claim"],
        "peak_cuda_memory_bytes": {"allocated": int(torch.cuda.max_memory_allocated(device)),
                                    "reserved": int(torch.cuda.max_memory_reserved(device))},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", type=Path, default=preflight.OUTPUT_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args(argv)
    report = run_probe(device=args.device, preflight_path=args.preflight, output=args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "training_admission": False}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
