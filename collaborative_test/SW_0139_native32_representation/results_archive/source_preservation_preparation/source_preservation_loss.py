"""Small SW0139 source-Q anchor contract; this is not a cache or trainer.

Importing this module does not initialize CUDA. Q is the production
``spike_synchrony_affinity(..., settle=512, affinity_mode='spike')`` output:
float32, [B,1024,1024], symmetric, in [0,1]. The anchor is a per-image,
off-diagonal mean squared error against a detached source teacher.
"""
from __future__ import annotations

import hashlib
import math
import statistics
from typing import Mapping, Sequence

import torch


N_PATCHES = 1024
Q_STEPS = 1024
Q_SETTLE = 512
Q_BATCH = 4
LAMBDA_BATCH_NORMS = 4
_HEX = frozenset("0123456789abcdef")
_CACHE_SOURCE_HASH_KEYS = (
    "source_core_sha256", "source_manifest_sha256", "encoder_sha256",
    "feature_statistics_sha256", "rgb_cache_manifest_sha256",
    "implementation_fingerprint_sha256",
)


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in _HEX for char in value.lower()))


def _require_sha(value: object, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a 64-character SHA-256")
    return str(value).lower()


def image_ids_sha256(image_ids: Sequence[int]) -> str:
    """Hash ordered IDs in the same little-endian int64 representation as SW0139."""
    values = [int(value) for value in image_ids]
    payload = b"".join(value.to_bytes(8, "little", signed=True) for value in values)
    return hashlib.sha256(payload).hexdigest()


def validate_q_matrix(q: torch.Tensor, *, name: str,
                      expected_batch: int | None = None,
                      require_grad: bool | None = None) -> None:
    """Fail closed on tensors outside the production affinity contract."""
    if not torch.is_tensor(q):
        raise TypeError(f"{name} must be a torch.Tensor")
    if q.dtype != torch.float32:
        raise TypeError(f"{name} must preserve production float32 Q values")
    if q.ndim != 3 or tuple(q.shape[1:]) != (N_PATCHES, N_PATCHES):
        raise ValueError(f"{name} must have shape [B,1024,1024]")
    if q.shape[0] < 1 or (expected_batch is not None and q.shape[0] != expected_batch):
        raise ValueError(f"{name} has an unexpected image batch")
    if require_grad is not None and q.requires_grad != require_grad:
        raise ValueError(f"{name} requires_grad does not match its registered role")
    if not torch.isfinite(q).all():
        raise ValueError(f"{name} contains nonfinite values")
    # Production affinity is a product of positive-clamped correlations.
    # Allow only roundoff-scale range excursions, without clamping or mutation.
    if (q < -1e-6).any() or (q > 1.0 + 1e-6).any():
        raise ValueError(f"{name} is outside the production [0,1] affinity range")
    if not torch.allclose(q, q.transpose(-1, -2), rtol=0.0, atol=2e-6):
        raise ValueError(f"{name} is not symmetric")


def source_q_anchor(student_q: torch.Tensor, teacher_q: torch.Tensor) -> torch.Tensor:
    """Return the registered symmetric off-diagonal Q MSE.

    The diagonal is excluded exactly. Each image contributes its own mean over
    1024*1023 ordered off-diagonal entries; the batch is then averaged. The
    teacher must already be detached; neither input is modified.
    """
    validate_q_matrix(student_q, name="student_q", require_grad=True)
    validate_q_matrix(teacher_q, name="teacher_q", expected_batch=student_q.shape[0],
                      require_grad=False)
    if student_q.device != teacher_q.device:
        raise ValueError("student and teacher Q must be on the same device")
    difference = student_q - teacher_q
    off_diagonal = ~torch.eye(N_PATCHES, dtype=torch.bool, device=student_q.device)
    squared = difference.square().masked_fill(~off_diagonal.unsqueeze(0), 0.0)
    per_image = squared.sum(dim=(-2, -1)) / float(N_PATCHES * (N_PATCHES - 1))
    loss = per_image.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("source-Q anchor loss is nonfinite")
    return loss


def validate_teacher_cache_manifest(
    manifest: Mapping[str, object], *,
    expected_image_ids: Sequence[int],
    expected_pool_rows: Sequence[int],
    expected_source_hashes: Mapping[str, str],
    expected_cache_sha256: str,
    expected_chunk_sha256: Sequence[str],
) -> None:
    """Validate identity/order/provenance before a teacher-Q cache is consumed.

    The cache writer is intentionally out of scope. Its ledger schema is fixed
    here so a later writer/reader cannot silently reorder rows or change B4
    grouping. ``chunk_sha256`` has one digest per contiguous B4 matrix chunk.
    """
    ids = [int(value) for value in expected_image_ids]
    rows = [int(value) for value in expected_pool_rows]
    if not ids or len(ids) != len(rows) or len(set(ids)) != len(ids) or len(set(rows)) != len(rows):
        raise ValueError("expected TRAIN IDs and pool rows must be nonempty, unique, and aligned")
    if len(ids) % Q_BATCH:
        raise ValueError("registered teacher cache must contain complete contiguous B4 groups")
    if manifest.get("schema") != "sw0139_source_q_teacher_v1":
        raise ValueError("unknown source-Q cache schema")
    if manifest.get("status") != "complete" or manifest.get("ground_truth_used") is not False:
        raise ValueError("source-Q teacher cache must be complete and ground-truth free")
    if manifest.get("dtype") != "float32" or manifest.get("q_shape") != [len(ids), N_PATCHES, N_PATCHES]:
        raise ValueError("source-Q cache dtype or shape differs from the registered full matrix")
    if manifest.get("steps") != Q_STEPS or manifest.get("settle") != Q_SETTLE:
        raise ValueError("source-Q cache rollout horizon differs from the registered contract")
    if manifest.get("affinity_mode") != "spike" or manifest.get("patch_order") != "native32_row_major":
        raise ValueError("source-Q cache affinity or patch ordering is not registered")
    if manifest.get("batch_size") != Q_BATCH:
        raise ValueError("source-Q cache must preserve actual B4 rollout grouping")
    if manifest.get("image_ids") != ids or manifest.get("pool_rows") != rows:
        raise ValueError("source-Q cache image IDs or pool-row ordering mismatch")
    expected_id_sha = image_ids_sha256(ids)
    expected_row_sha = image_ids_sha256(rows)
    if manifest.get("image_ids_sha256") != expected_id_sha or manifest.get("pool_rows_sha256") != expected_row_sha:
        raise ValueError("source-Q cache ordered ID/row digest mismatch")
    groups = manifest.get("groups")
    want_groups = [
        {"start": start, "stop": start + Q_BATCH,
         "image_ids": ids[start:start + Q_BATCH], "pool_rows": rows[start:start + Q_BATCH]}
        for start in range(0, len(ids), Q_BATCH)
    ]
    if groups != want_groups:
        raise ValueError("source-Q cache B4 grouping is not the exact registered contiguous order")
    got_hashes = manifest.get("source_hashes")
    if not isinstance(got_hashes, Mapping):
        raise ValueError("source-Q cache is missing source provenance hashes")
    if set(expected_source_hashes) != set(_CACHE_SOURCE_HASH_KEYS):
        raise ValueError("expected source hashes must bind core, manifest, encoder, statistics, RGB cache and code")
    for key in _CACHE_SOURCE_HASH_KEYS:
        expected = _require_sha(expected_source_hashes[key], f"expected_source_hashes[{key}]")
        if got_hashes.get(key) != expected:
            raise ValueError(f"source-Q cache source hash mismatch for {key}")
    expected_cache = _require_sha(expected_cache_sha256, "expected_cache_sha256")
    if manifest.get("cache_sha256") != expected_cache:
        raise ValueError("source-Q cache payload hash does not match its validated file")
    chunk_hashes = manifest.get("chunk_sha256")
    expected_chunks = [_require_sha(value, "expected_chunk_sha256") for value in expected_chunk_sha256]
    if (not isinstance(chunk_hashes, list) or len(chunk_hashes) != len(want_groups)
            or len(expected_chunks) != len(want_groups)
            or any(not _is_sha256(value) for value in chunk_hashes)
            or chunk_hashes != expected_chunks):
        raise ValueError("source-Q cache chunk hashes do not match the validated contiguous B4 chunks")


def calibrate_anchor_lambda(batch_norms: Sequence[Mapping[str, object]],
                            metadata: Mapping[str, object], *,
                            expected_calibration_image_ids: Sequence[int],
                            expected_source_hashes: Mapping[str, str],
                            expected_lambda_c_record_sha256: str) -> dict[str, object]:
    """Apply the fixed four-batch post-probe median rule; never epsilon-rescue.

    Each row supplies norms of the accumulated ``lambda_C*C`` and ``A``
    gradients. Metadata proves this is source seed0 after exactly one disposable
    C-only update from fresh source weights and fresh Adam; the probe is discarded
    and actual arms must restart fresh. This function does not perform that probe.
    """
    if (metadata.get("experiment") != "SW0139_source_q_anchor_calibration"
            or metadata.get("seed") != 0
            or metadata.get("probe_stage") != "after_first_C_only_update"
            or metadata.get("probe_optimizer_updates") != 1
            or metadata.get("lambda_c_source_seed") != 0
            or metadata.get("calibration_logical_batches") != LAMBDA_BATCH_NORMS
            or metadata.get("logical_batch_size") != 16
            or metadata.get("microbatch_size") != 4):
        raise ValueError("lambda calibration is not the registered source0 fresh-start probe")
    if any(metadata.get(key) is not expected for key, expected in (
            ("fresh_source_start", True), ("fresh_adam_start", True),
            ("probe_discarded", True), ("actual_arms_restart_fresh", True),
            ("source_unchanged", True), ("ground_truth_used", False))):
        raise ValueError("lambda calibration probe state/usage metadata is invalid")
    for key in ("source_core_sha256", "source_manifest_sha256",
                "source_training_ids_sha256", "probe_start_state_sha256",
                "probe_end_state_sha256", "probe_optimizer_state_sha256",
                "lambda_c_source_record_sha256"):
        _require_sha(metadata.get(key), key)
    for key in ("source_core_sha256", "source_manifest_sha256", "source_training_ids_sha256"):
        expected = _require_sha(expected_source_hashes.get(key), f"expected_source_hashes[{key}]")
        if str(metadata[key]).lower() != expected:
            raise ValueError(f"lambda calibration source hash mismatch for {key}")
    expected_c_record = _require_sha(expected_lambda_c_record_sha256,
                                     "expected_lambda_c_record_sha256")
    if str(metadata["lambda_c_source_record_sha256"]).lower() != expected_c_record:
        raise ValueError("lambda_C source0 calibration record hash mismatch")
    lambda_c = float(metadata.get("lambda_c_value", float("nan")))
    if not math.isfinite(lambda_c) or lambda_c <= 0:
        raise ValueError("fixed source0 lambda_C must be finite and positive")
    if not isinstance(batch_norms, Sequence) or len(batch_norms) != LAMBDA_BATCH_NORMS:
        raise ValueError("lambda anchor calibration requires exactly four fixed batches")
    expected_ids = [int(value) for value in expected_calibration_image_ids]
    if len(expected_ids) != 64 or len(set(expected_ids)) != 64:
        raise ValueError("expected calibration IDs must be the fixed unique source0 first64")
    if metadata.get("calibration_image_ids") != expected_ids:
        raise ValueError("lambda calibration IDs differ from registered source0 first64")
    expected_batches = [expected_ids[i * 16:(i + 1) * 16]
                        for i in range(LAMBDA_BATCH_NORMS)]
    if len(metadata.get("calibration_image_ids", [])) != 64:
        raise ValueError("lambda calibration must bind the ordered first64 source TRAIN IDs")
    if (len(set(metadata["calibration_image_ids"])) != 64
            or metadata.get("calibration_image_ids_sha256") != image_ids_sha256(metadata["calibration_image_ids"])):
        raise ValueError("lambda calibration TRAIN ID ordering/hash is invalid")
    ratios: list[float] = []
    normalized_rows: list[dict[str, float | int | list[int]]] = []
    for index, (row, ids) in enumerate(zip(batch_norms, expected_batches)):
        if row.get("logical_batch") != index or row.get("image_ids") != ids:
            raise ValueError("lambda calibration batch IDs/order do not match the fixed first64")
        c_norm = float(row.get("weighted_c_gradient_norm", float("nan")))
        a_norm = float(row.get("anchor_gradient_norm", float("nan")))
        if not (math.isfinite(c_norm) and math.isfinite(a_norm) and c_norm > 0 and a_norm > 0):
            raise FloatingPointError("zero or nonfinite source-Q calibration norm; no epsilon fallback")
        ratio = c_norm / a_norm
        if not math.isfinite(ratio) or ratio <= 0:
            raise FloatingPointError("source-Q calibration ratio is invalid")
        ratios.append(ratio)
        normalized_rows.append({"logical_batch": index, "image_ids": list(ids),
                                "weighted_c_gradient_norm": c_norm,
                                "anchor_gradient_norm": a_norm, "ratio": ratio})
    value = float(statistics.median(ratios))
    if not math.isfinite(value) or value <= 0:
        raise FloatingPointError("calibrated lambda_A must be finite and positive")
    return {
        "experiment": "SW0139_source_q_anchor_calibration",
        "status": "calibrated",
        "lambda_A": value,
        "formula": "median_batches(||grad(lambda_C*C)||_2 / ||grad(A)||_2)",
        "source_seed": 0,
        "probe_stage": "after_first_C_only_update",
        "probe_optimizer_updates": 1,
        "calibration_batches": normalized_rows,
        "calibration_image_ids": expected_ids,
        "calibration_image_ids_sha256": metadata["calibration_image_ids_sha256"],
        "source_core_sha256": metadata["source_core_sha256"],
        "source_manifest_sha256": metadata["source_manifest_sha256"],
        "source_training_ids_sha256": metadata["source_training_ids_sha256"],
        "lambda_c_source_seed": 0,
        "lambda_c_value": lambda_c,
        "lambda_c_source_record_sha256": metadata["lambda_c_source_record_sha256"],
        "probe_start_state_sha256": metadata["probe_start_state_sha256"],
        "probe_end_state_sha256": metadata["probe_end_state_sha256"],
        "probe_optimizer_state_sha256": metadata["probe_optimizer_state_sha256"],
        "fresh_source_start": True,
        "fresh_adam_start": True,
        "probe_discarded": True,
        "actual_arms_restart_fresh": True,
        "source_unchanged": True,
        "ground_truth_used": False,
    }
