"""Local SW0141 native32 source-relation preservation candidate driver.

This candidate driver reuses the frozen SW0135/SW0139 native32 foundation and
SW0139 contrastive-view implementation. It does not launch jobs or initialize
CUDA on import; root admission is still required before queue execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_repo_override = os.environ.get("PATCH_V2_REPO")
_candidates = ([Path(_repo_override).resolve()] if _repo_override else []) + [
    parent / "patch_v2_sw" for parent in HERE.parents
]
ROOT = next((p for p in _candidates if (p / "collaborative_test/SW_0139_native32_representation/run.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("Set PATCH_V2_REPO to the patch_v2_sw checkout")
for _p in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional",
           HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract32
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136_eval
from collaborative_test.SW_0137_native32_source_qcc import run as sw137
from collaborative_test.SW_0139_native32_representation import run as sw139
from collaborative_test.SW_0139_native32_representation import evaluate as sw139_eval
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity

try:
    import source_preservation_loss as anchor
except ImportError:
    from collaborative_private_runtime_20261008.SW0134_deployment_20261009 import source_preservation_loss as anchor

CACHE_SCHEMA = "sw0139_source_q_teacher_v1"
PILOT_NAME = "SW0141_native32_source_preservation"
SEED = 1
SOURCE0_SEED = 0
TRAIN_N = 4096
CAL_N = 64
LOGICAL_BATCH = 16
MICROBATCH = 4
MICROS = LOGICAL_BATCH // MICROBATCH
STEPS = 1024
SETTLE = 512
TAIL = 64
ARMS = ("c_only", "c_plus_anchor")
TRAIN_ROOT = HERE / "training"
CACHE_ROOT = HERE / "teacher_cache"
RESULT_ROOT = HERE / "results"
LAMBDA_C_PATH = ROOT / "collaborative_test/SW_0139_native32_representation/results_archive/lambda_seed0.json"
LAMBDA_A_PATH = HERE / "lambda_anchor_source0.json"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _write_once(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _state_sha(modules) -> str:
    digest = hashlib.sha256()
    for prefix, module in modules:
        for name, tensor in sorted(module.state_dict().items()):
            value = tensor.detach().cpu().contiguous().numpy()
            digest.update(prefix.encode()); digest.update(name.encode())
            digest.update(str(value.dtype).encode()); digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
            digest.update(value.tobytes())
    return digest.hexdigest()


def implementation_fingerprint() -> dict[str, str]:
    # Compose the same complete dependency inventories used by the native32
    # foundation and SW0139 trainer/evaluator. This intentionally binds the
    # actual core, graph, dendrite, membrane, oscillator, gate and loaders.
    dependency_sets = (
        ("sw135", sw135.implementation_fingerprint()),
        ("sw139_train", sw139.implementation_fingerprint()),
        ("sw139_eval", sw139_eval.implementation_fingerprint()),
        ("sw137", sw137.implementation_fingerprint()),
    )
    merged = {f"{namespace}/{name}": digest
              for namespace, rows in dependency_sets for name, digest in rows.items()}
    files = (
        HERE / "pilot.py",
        HERE.parent / "source_preservation_loss.py",
    )
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"source-preservation dependency absent: {missing[0]}")
    merged.update({p.relative_to(ROOT).as_posix() if ROOT in p.parents else f"private/{p.name}": sha256_file(p)
                   for p in files})
    return merged


def fingerprint_sha256(fp=None) -> str:
    return _canonical_sha(implementation_fingerprint() if fp is None else fp)


def source_bundle(seed: int, device="cpu", *, verify_rgb_assets=True):
    if seed not in (0, 1):
        raise ValueError("local preparation supports source0 calibration and the registered seed1 pilot only")
    foundation = sw135.load_native32_foundation(seed, device, verify_rgb_assets=verify_rgb_assets)
    # Cross-check source order against the validated SW0130 contract, without
    # depending on the SW0139 pilot's seed1-only private guard.
    source = sw130.source_contract(seed)
    if (list(map(int, foundation.image_ids)) != list(map(int, source[4]))
            or not np.array_equal(foundation.pool_indices, np.asarray(source[3], dtype=np.int64))):
        raise AssertionError("native32 foundation TRAIN IDs/rows differ from SW0130 source contract")
    return foundation


def _cache_expected_hashes(foundation, fingerprint=None):
    assets = foundation.provenance["rgb_asset_validation"]
    fp = implementation_fingerprint() if fingerprint is None else fingerprint
    return {
        "source_core_sha256": foundation.provenance["source_core_sha256"],
        "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
        "encoder_sha256": foundation.provenance["encoder_sha256"],
        "feature_statistics_sha256": foundation.provenance["feature_preprocessing_sha256"],
        "rgb_cache_manifest_sha256": assets["train_manifest_sha256"],
        "implementation_fingerprint_sha256": _canonical_sha(fp),
    }


def _build_cache_manifest(*, seed, ids, rows, source_hashes, cache_sha, chunk_hashes):
    groups = [{"start": start, "stop": start + MICROBATCH,
               "image_ids": [int(x) for x in ids[start:start + MICROBATCH]],
               "pool_rows": [int(x) for x in rows[start:start + MICROBATCH]]}
              for start in range(0, len(ids), MICROBATCH)]
    return {
        "schema": CACHE_SCHEMA, "status": "complete", "seed": int(seed),
        "ground_truth_used": False, "optimizer_updates": 0,
        "dtype": "float32", "q_shape": [len(ids), 1024, 1024],
        "steps": STEPS, "settle": SETTLE, "affinity_mode": "spike",
        "patch_order": "native32_row_major", "batch_size": MICROBATCH,
        "image_ids": list(map(int, ids)), "pool_rows": list(map(int, rows)),
        "image_ids_sha256": anchor.image_ids_sha256(ids),
        "pool_rows_sha256": anchor.image_ids_sha256(rows),
        "groups": groups, "source_hashes": dict(source_hashes),
        # The helper schema pins the manifest SHA; this additional full record
        # pins the RGB payload bytes too, including the large train cache.
        "rgb_asset_validation": None,
        "foundation_implementation_fingerprint": None,
        "cache_sha256": cache_sha, "chunk_sha256": list(chunk_hashes),
        "implementation_fingerprint": implementation_fingerprint(),
    }


def write_teacher_cache(seed: int, output: str | Path, device="cuda:0", *, count=None):
    """Create immutable float32 source-Q cache in contiguous registered B4 chunks.

    Seed0 is exactly the first64 source TRAIN IDs (for lambda calibration);
    seed1 is all4096 source TRAIN IDs (shared by both student arms).
    """
    if seed not in (0, 1):
        raise ValueError("teacher cache supports only source seed0 or seed1")
    count = (CAL_N if seed == 0 else TRAIN_N) if count is None else int(count)
    expected = CAL_N if seed == 0 else TRAIN_N
    if count != expected:
        raise ValueError(f"source{seed} teacher cache row count is fixed at {expected}")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing teacher cache directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    foundation = source_bundle(seed, device, verify_rgb_assets=True)
    ids = np.asarray(foundation.image_ids[:count], dtype=np.int64)
    rows = np.asarray(foundation.pool_indices[:count], dtype=np.int64)
    if len(ids) % MICROBATCH or len(rows) != len(ids):
        raise AssertionError("teacher rows must preserve contiguous B4 source groups")
    core_path = Path(foundation.provenance["source_core_path"])
    before_core_sha = sha256_file(core_path)
    source_hashes = _cache_expected_hashes(foundation)
    data_path = output / "teacher_q.f32"
    payload_bytes = len(ids) * 1024 * 1024 * 4
    with data_path.open("xb") as stream:
        stream.truncate(payload_bytes)
    matrix = np.memmap(data_path, mode="r+", dtype=np.float32, shape=(len(ids), 1024, 1024))
    rgb_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    chunk_hashes = []
    foundation.wrapped.eval(); foundation.encoder.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(ids), MICROBATCH):
            end = start + MICROBATCH
            rgb = sw139.read_rgb_rows(rows[start:end], device, rgb_cache)
            gamma = sw139._features_to_gamma(foundation, rgb, None)
            trace = foundation.rollout(gamma, total_steps=STEPS, live_tail_steps=0)
            q = spike_synchrony_affinity(trace["spikes"], trace["component_spikes"],
                                         settle=SETTLE, affinity_mode="spike")
            anchor.validate_q_matrix(q, name="source teacher Q", expected_batch=MICROBATCH,
                                     require_grad=False)
            values = q.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError(f"nonfinite teacher Q at source rows {start}:{end}")
            matrix[start:end] = values
            chunk_hashes.append(hashlib.sha256(values.tobytes(order="C")).hexdigest())
            if end % 64 == 0 or end == len(ids):
                print(json.dumps({"stage": "teacher_cache", "seed": seed,
                                  "completed": end, "total": len(ids)}, allow_nan=False), flush=True)
    matrix.flush(); del matrix
    if sha256_file(core_path) != before_core_sha:
        raise AssertionError("source checkpoint bytes changed during read-only cache generation")
    data_sha = sha256_file(data_path)
    record = _build_cache_manifest(seed=seed, ids=ids, rows=rows,
                                   source_hashes=source_hashes, cache_sha=data_sha,
                                   chunk_hashes=chunk_hashes)
    record["rgb_asset_validation"] = foundation.provenance["rgb_asset_validation"]
    record["foundation_implementation_fingerprint"] = foundation.provenance["implementation_fingerprint"]
    record["elapsed_seconds"] = time.perf_counter() - started
    record["cache_file"] = data_path.name
    _write_once(output / "manifest.json", record)  # manifest is the completion marker
    return record


class TeacherCache:
    """Strict read-only cache view; each B4 payload chunk verifies before use."""
    def __init__(self, directory, foundation, *, expected_count):
        self.directory = Path(directory)
        self.manifest_path = self.directory / "manifest.json"
        self.data_path = self.directory / "teacher_q.f32"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.count = int(expected_count)
        if self.manifest.get("seed") != int(foundation.seed):
            raise ValueError("teacher cache source seed differs from the loaded foundation")
        ids = list(map(int, foundation.image_ids[:self.count]))
        rows = list(map(int, foundation.pool_indices[:self.count]))
        expected_hashes = _cache_expected_hashes(foundation)
        self.fp = implementation_fingerprint()
        if self.manifest.get("implementation_fingerprint") != self.fp:
            raise ValueError("teacher cache implementation dependencies changed")
        if self.manifest.get("rgb_asset_validation") != foundation.provenance.get("rgb_asset_validation"):
            raise ValueError("teacher cache RGB payload/assets changed since cache creation")
        if self.manifest.get("foundation_implementation_fingerprint") != foundation.provenance.get("implementation_fingerprint"):
            raise ValueError("teacher cache native foundation dependencies changed")
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.anchor_module_sha256 = sha256_file(HERE.parent / "source_preservation_loss.py")
        anchor.validate_teacher_cache_manifest(
            self.manifest, expected_image_ids=ids, expected_pool_rows=rows,
            expected_source_hashes=expected_hashes,
            expected_cache_sha256=self.manifest.get("cache_sha256"),
            expected_chunk_sha256=self.manifest.get("chunk_sha256", []))
        expected_bytes = self.count * 1024 * 1024 * 4
        if not self.data_path.is_file() or self.data_path.stat().st_size != expected_bytes:
            raise ValueError("teacher cache payload is missing or has wrong byte length")
        if self.manifest.get("cache_file") != self.data_path.name:
            raise ValueError("teacher cache manifest names a different payload")
        self._matrix = np.memmap(self.data_path, mode="r", dtype=np.float32,
                                 shape=(self.count, 1024, 1024))
        self._verified_chunks = set()

    def read_b4(self, start: int, device):
        if start < 0 or start % MICROBATCH or start + MICROBATCH > self.count:
            raise ValueError("teacher cache reads must be aligned complete contiguous B4 groups")
        index = start // MICROBATCH
        value = np.asarray(self._matrix[start:start + MICROBATCH])
        digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        if digest != self.manifest["chunk_sha256"][index]:
            raise ValueError(f"teacher cache chunk {index} failed its SHA-256 check")
        self._verified_chunks.add(index)
        result = torch.from_numpy(np.array(value, copy=True)).to(device=device)
        anchor.validate_q_matrix(result, name="loaded teacher Q", expected_batch=MICROBATCH,
                                 require_grad=False)
        return result

    def verify_complete_payload(self, *, require_all_chunks=True):
        # Each consumed B4 was SHA-checked before use. This avoids a second
        # 16-GiB disk pass while still requiring every registered chunk when
        # the full pilot/calibration pass completes.
        expected_chunks = set(range(self.count // MICROBATCH))
        if require_all_chunks and self._verified_chunks != expected_chunks:
            raise ValueError("teacher cache pass did not verify every registered B4 chunk")
        if sha256_file(self.manifest_path) != self.manifest_sha256:
            raise ValueError("teacher cache manifest changed during use")

    def close(self):
        matrix = self._matrix
        self._matrix = None
        mmap_file = getattr(matrix, "_mmap", None)
        if mmap_file is not None:
            mmap_file.close()


def _q_for_trace(trace):
    return spike_synchrony_affinity(trace["spikes"], trace["component_spikes"],
                                    settle=SETTLE, affinity_mode="spike")


def compose_objective(c_loss, *, lambda_c, anchor_loss=None, lambda_a=0.0):
    """Exact two-arm objective; deliberately contains no phase/Q prior term."""
    if not torch.is_tensor(c_loss) or c_loss.ndim != 0:
        raise TypeError("C must be a scalar tensor")
    if not math.isfinite(float(lambda_c)) or float(lambda_c) <= 0:
        raise ValueError("lambda_C must be finite and positive")
    result = float(lambda_c) * c_loss
    if anchor_loss is not None:
        if not torch.is_tensor(anchor_loss) or anchor_loss.ndim != 0:
            raise TypeError("A must be a scalar tensor")
        if not math.isfinite(float(lambda_a)) or float(lambda_a) <= 0:
            raise ValueError("lambda_A must be finite and positive when A is included")
        result = result + float(lambda_a) * anchor_loss
    elif float(lambda_a) != 0.0:
        raise ValueError("lambda_A cannot be nonzero without the source-Q anchor")
    if not torch.isfinite(result):
        raise FloatingPointError("source-preservation objective is nonfinite")
    return result


def _q_cache_parity(current, cached):
    if tuple(current.shape) != tuple(cached.shape):
        raise ValueError("teacher-cache/source Q shapes differ")
    if not (torch.isfinite(current).all() and torch.isfinite(cached).all()):
        raise FloatingPointError("teacher-cache/source Q contains nonfinite values")
    difference = (current.detach() - cached).abs()
    max_abs = float(difference.max().cpu())
    return {"exact": bool(torch.equal(current.detach(), cached)),
            "allclose_atol_2e-6_rtol_1e-6": bool(torch.allclose(
                current.detach(), cached, atol=2e-6, rtol=1e-6)),
            "max_abs_difference": max_abs}


def _paired_contrast(foundation, canonical_trace, rgb, *, nonce):
    view, contrast, offset = sw139._paired_view(rgb, seed=nonce)
    gamma_view = sw139._features_to_gamma(foundation, view, None)
    view_trace = foundation.rollout(gamma_view, total_steps=STEPS, live_tail_steps=TAIL)
    value = sw139.symmetric_temporal_infonce(
        canonical_trace["component_spikes"][..., SETTLE:],
        view_trace["component_spikes"][..., SETTLE:], seed=nonce + 1,
        negatives=sw139.NEGATIVES)
    return value, view_trace, contrast, offset


def _trainable(foundation):
    for name, parameter in foundation.wrapped.named_parameters():
        parameter.requires_grad_(name.startswith("core.graph_generator."))
    foundation.encoder.requires_grad_(True)
    foundation.encoder.train(); foundation.wrapped.train()
    params = [p for p in foundation.wrapped.parameters() if p.requires_grad] + list(foundation.encoder.parameters())
    if not params or len({id(p) for p in params}) != len(params):
        raise AssertionError("student optimizer parameters must be a nonempty unique graph+encoder union")
    return params


def _source_hashes(foundation):
    return {"source_core_sha256": foundation.provenance["source_core_sha256"],
            "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
            "source_training_ids_sha256": foundation.provenance["source_training_ids_sha256"]}


def _grad_norm(grads):
    values = [g.detach().double().square().sum() for g in grads if g is not None]
    if not values:
        return 0.0
    return math.sqrt(float(torch.stack(values).sum()))


def _optimizer_state_sha(optimizer):
    digest = hashlib.sha256()
    state = optimizer.state_dict()
    digest.update(json.dumps(state["param_groups"], sort_keys=True, default=str,
                             separators=(",", ":")).encode())
    for key, row in sorted(state["state"].items(), key=lambda pair: int(pair[0])):
        digest.update(str(key).encode())
        for name, value in sorted(row.items()):
            digest.update(name.encode())
            if torch.is_tensor(value):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode())
                digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
                digest.update(tensor.numpy().tobytes())
            else:
                digest.update(repr(value).encode())
    return digest.hexdigest()


def calibrate_lambda_a(*, source0_cache, output=LAMBDA_A_PATH, device="cuda:0"):
    """Run the registered source0 C-only disposable probe and freeze lambda_A."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing source-preservation lambda: {output}")
    foundation = source_bundle(SOURCE0_SEED, device)
    cache = TeacherCache(source0_cache, foundation, expected_count=CAL_N)
    lambda_c_record = sw139.load_lambda_record(LAMBDA_C_PATH)
    lambda_c_sha = sha256_file(LAMBDA_C_PATH)
    lambda_c = float(lambda_c_record["lambda"])
    params = _trainable(foundation)
    optimizer = torch.optim.Adam([{"params": [p for p in foundation.wrapped.parameters() if p.requires_grad], "lr": sw139.CORE_LR},
                                  {"params": list(foundation.encoder.parameters()), "lr": sw139.ENCODER_LR}])
    rgb_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    start_sha = _state_sha((("wrapped", foundation.wrapped), ("encoder", foundation.encoder)))
    probe_grad_norm = None
    source0_cache_parity = []
    logical_ids = list(map(int, foundation.image_ids[:LOGICAL_BATCH]))
    optimizer.zero_grad(set_to_none=True)
    for micro in range(MICROS):
        index = micro * MICROBATCH
        rgb = sw139.read_rgb_rows(foundation.pool_indices[index:index + MICROBATCH], device, rgb_cache)
        gamma = sw139._features_to_gamma(foundation, rgb, None)
        trace = foundation.rollout(gamma, total_steps=STEPS, live_tail_steps=TAIL)
        current_q = _q_for_trace(trace).detach()
        cached_q = cache.read_b4(micro * MICROBATCH, device)
        parity = _q_cache_parity(current_q, cached_q)
        source0_cache_parity.append(parity)
        if not parity["allclose_atol_2e-6_rtol_1e-6"]:
            raise AssertionError(f"source0 teacher cache does not reproduce initial B4 Q micro{micro}")
        c, _, _, _ = _paired_contrast(foundation, trace, rgb,
                                      nonce=139000 + 0 * 100000 + 1 * 100 + micro)
        loss = compose_objective(c, lambda_c=lambda_c)
        if not torch.isfinite(loss):
            raise FloatingPointError("source0 C-only calibration update loss is nonfinite")
        (loss / MICROS).backward()
    grads = [p.grad for p in params]
    if any(g is not None and not torch.isfinite(g).all() for g in grads):
        raise FloatingPointError("source0 calibration disposable gradients are nonfinite")
    probe_grad_norm = _grad_norm(grads)
    if probe_grad_norm <= 0:
        raise FloatingPointError("source0 calibration C-only update has no gradient")
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    end_sha = _state_sha((("wrapped", foundation.wrapped), ("encoder", foundation.encoder)))
    optimizer_sha = _optimizer_state_sha(optimizer)
    source_core_sha = sha256_file(foundation.provenance["source_core_path"])
    norms = []
    names = [(f"graph.{n}", p) for n, p in foundation.wrapped.core.graph_generator.named_parameters()]
    names += [(f"encoder.{n}", p) for n, p in foundation.encoder.named_parameters()]
    for batch in range(4):
        accum_c = [torch.zeros_like(p) for p in params]
        accum_a = [torch.zeros_like(p) for p in params]
        batch_ids = list(map(int, foundation.image_ids[batch * 16:(batch + 1) * 16]))
        for micro in range(MICROS):
            pos = batch * 16 + micro * 4
            rgb = sw139.read_rgb_rows(foundation.pool_indices[pos:pos + 4], device, rgb_cache)
            gamma = sw139._features_to_gamma(foundation, rgb, None)
            trace = foundation.rollout(gamma, total_steps=STEPS, live_tail_steps=TAIL)
            c, _, _, _ = _paired_contrast(foundation, trace, rgb,
                                          nonce=139000 + batch * 100 + micro)
            q_student = _q_for_trace(trace)
            q_teacher = cache.read_b4(pos, device)
            c_grads = torch.autograd.grad(lambda_c * c, params, retain_graph=True, allow_unused=True)
            a_loss = anchor.source_q_anchor(q_student, q_teacher)
            a_grads = torch.autograd.grad(a_loss, params, allow_unused=True)
            for i, (gc, ga) in enumerate(zip(c_grads, a_grads)):
                if gc is not None: accum_c[i].add_(gc.detach(), alpha=1.0 / MICROS)
                if ga is not None: accum_a[i].add_(ga.detach(), alpha=1.0 / MICROS)
        cn, an = _grad_norm(accum_c), _grad_norm(accum_a)
        dot = sum(float((x.detach().double() * y.detach().double()).sum())
                  for x, y in zip(accum_c, accum_a))
        cosine = dot / (cn * an) if cn > 0 and an > 0 else float("nan")
        if not all(math.isfinite(x) and x > 0 for x in (cn, an)) or not math.isfinite(cosine):
            raise FloatingPointError(f"source0 fixed lambda_A probe has invalid norms/cosine batch{batch}")
        norms.append({"logical_batch": batch, "image_ids": batch_ids,
                      "weighted_c_gradient_norm": cn, "anchor_gradient_norm": an,
                      "gradient_cosine": cosine})
    cache.verify_complete_payload()
    cache.close()
    metadata = {
        "experiment": "SW0139_source_q_anchor_calibration", "seed": 0,
        "probe_stage": "after_first_C_only_update", "probe_optimizer_updates": 1,
        "lambda_c_source_seed": 0, "lambda_c_value": lambda_c,
        "calibration_logical_batches": 4, "logical_batch_size": 16, "microbatch_size": 4,
        "fresh_source_start": True, "fresh_adam_start": True, "probe_discarded": True,
        "actual_arms_restart_fresh": True, "source_unchanged": source_core_sha == foundation.provenance["source_core_sha256"],
        "ground_truth_used": False,
        "source_core_sha256": foundation.provenance["source_core_sha256"],
        "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
        "source_training_ids_sha256": foundation.provenance["source_training_ids_sha256"],
        "probe_start_state_sha256": start_sha, "probe_end_state_sha256": end_sha,
        "probe_optimizer_state_sha256": optimizer_sha,
        "lambda_c_source_record_sha256": lambda_c_sha,
        "calibration_image_ids": list(map(int, foundation.image_ids[:64])),
        "calibration_image_ids_sha256": anchor.image_ids_sha256(foundation.image_ids[:64]),
    }
    # Helper's per-batch expected IDs are four B16 groups from the fixed first64.
    rows = [{"logical_batch": row["logical_batch"], "image_ids": row["image_ids"],
             "weighted_c_gradient_norm": row["weighted_c_gradient_norm"],
             "anchor_gradient_norm": row["anchor_gradient_norm"]} for row in norms]
    source_hashes = _source_hashes(foundation)
    result = anchor.calibrate_anchor_lambda(
        rows, metadata, expected_calibration_image_ids=foundation.image_ids[:64],
        expected_source_hashes=source_hashes,
        expected_lambda_c_record_sha256=lambda_c_sha)
    result.update({"cosine_by_batch": [r["gradient_cosine"] for r in norms],
                   "probe_initial_update_gradient_norm": probe_grad_norm,
                   "source0_teacher_first_logical_batch_b4_parity": source0_cache_parity,
                   "teacher_cache_manifest_sha256": cache.manifest_sha256,
                   "teacher_cache_sha256": cache.manifest["cache_sha256"],
                   "foundation_implementation_fingerprint": foundation.provenance["implementation_fingerprint"],
                   "rgb_asset_validation": foundation.provenance["rgb_asset_validation"],
                   "implementation_fingerprint": implementation_fingerprint(),
                   "implementation_fingerprint_sha256": fingerprint_sha256(),
                   "status": "calibrated_local_unregistered"})
    result["self_sha256"] = _canonical_sha(result)
    _write_once(output, result)
    return result


def _optimizer_summary(state):
    summary = {"param_groups": state.get("param_groups", []), "state": {}}
    for key, row in state.get("state", {}).items():
        summary["state"][str(key)] = {name: (value.detach().cpu().tolist() if torch.is_tensor(value) else value)
                                       for name, value in row.items()}
    return summary


def _load_lambda_a(path=LAMBDA_A_PATH):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = dict(record); digest = payload.pop("self_sha256", None)
    if digest != _canonical_sha(payload):
        raise ValueError("source-preservation lambda_A self hash mismatch")
    if record.get("status") != "calibrated_local_unregistered" or record.get("ground_truth_used") is not False:
        raise ValueError("lambda_A artifact is incomplete or GT contaminated")
    if record.get("implementation_fingerprint") != implementation_fingerprint():
        raise ValueError("lambda_A implementation dependencies changed")
    val = float(record.get("lambda_A", float("nan")))
    if not math.isfinite(val) or val <= 0:
        raise ValueError("lambda_A must be finite positive; no epsilon fallback")
    source0 = sw130.source_contract(SOURCE0_SEED)
    foundation0 = source_bundle(SOURCE0_SEED, "cpu", verify_rgb_assets=True)
    expected_ids = list(map(int, source0[4][:CAL_N]))
    current_lambda_c = sw139.load_lambda_record(LAMBDA_C_PATH)
    if (record.get("source_core_sha256") != source0[5]
            or record.get("source_manifest_sha256") != sha256_file(source0[1])
            or record.get("source_training_ids_sha256") != hashlib.sha256(
                np.asarray(source0[4], dtype="<i8").tobytes()).hexdigest()
            or record.get("calibration_image_ids") != expected_ids
            or record.get("calibration_image_ids_sha256") != anchor.image_ids_sha256(expected_ids)
            or record.get("lambda_c_source_record_sha256") != sha256_file(LAMBDA_C_PATH)
            or record.get("lambda_c_value") != float(current_lambda_c["lambda"])
            or record.get("foundation_implementation_fingerprint") != foundation0.provenance["implementation_fingerprint"]
            or record.get("rgb_asset_validation") != foundation0.provenance["rgb_asset_validation"]):
        raise ValueError("lambda_A calibration is not bound to current source0 IDs/source/lambda_C")
    return record


def preflight_arm(arm, *, device="cuda:0", teacher_cache=None,
                  lambda_a_record=None, output_root=TRAIN_ROOT):
    """One fresh-source optimizer smoke per arm; its state is always discarded."""
    if arm not in ARMS:
        raise ValueError("unregistered pilot arm")
    if arm == "c_plus_anchor" and (teacher_cache is None or lambda_a_record is None):
        raise ValueError("C+anchor preflight requires the complete frozen teacher and lambda_A records")
    output = Path(output_root) / f"preflight_{arm}_seed{SEED}.json"
    if output.exists(): raise FileExistsError(f"preserve existing arm preflight: {output}")
    foundation = source_bundle(SEED, device)
    cache = TeacherCache(teacher_cache, foundation, expected_count=TRAIN_N) if arm == "c_plus_anchor" else None
    lambda_c = float(sw139.load_lambda_record(LAMBDA_C_PATH)["lambda"])
    lambda_a = _load_lambda_a(lambda_a_record) if arm == "c_plus_anchor" else None
    params = _trainable(foundation)
    graph_params = [p for p in foundation.wrapped.core.graph_generator.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([{"params": graph_params, "lr": sw139.CORE_LR},
                                  {"params": list(foundation.encoder.parameters()), "lr": sw139.ENCODER_LR}])
    rgb_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    pre_sha = _state_sha((("wrapped", foundation.wrapped), ("encoder", foundation.encoder)))
    optimizer.zero_grad(set_to_none=True)
    c_losses, a_losses, ids = [], [], []
    cache_parity = []
    for micro in range(MICROS):
        pos = micro * MICROBATCH
        rgb = sw139.read_rgb_rows(foundation.pool_indices[pos:pos + MICROBATCH], device, rgb_cache)
        gamma = sw139._features_to_gamma(foundation, rgb, None)
        trace = foundation.rollout(gamma, total_steps=STEPS, live_tail_steps=TAIL)
        cached_q = None
        student_q = None
        if cache is not None:
            student_q = _q_for_trace(trace)
            expected_q = cache.read_b4(pos, device)
            cached_q = expected_q
            cache_parity.append(_q_cache_parity(student_q.detach(), expected_q))
            if not cache_parity[-1]["allclose_atol_2e-6_rtol_1e-6"]:
                raise AssertionError(f"seed1 teacher cache does not reproduce initial source Q micro{micro}")
        nonce = 139000 + SEED * 100000 + 1 * 100 + micro
        c, _, _, _ = _paired_contrast(foundation, trace, rgb, nonce=nonce)
        loss = compose_objective(c, lambda_c=lambda_c)
        c_losses.append(float(c.detach()))
        if cache is not None:
            if student_q is None: student_q = _q_for_trace(trace)
            if cached_q is None: cached_q = cache.read_b4(pos, device)
            a = anchor.source_q_anchor(student_q, cached_q)
            loss = compose_objective(c, lambda_c=lambda_c,
                                     anchor_loss=a, lambda_a=float(lambda_a["lambda_A"]))
            a_losses.append(float(a.detach()))
        if not torch.isfinite(loss): raise FloatingPointError("pilot disposable preflight loss nonfinite")
        (loss / MICROS).backward()
        ids.extend(int(v) for v in foundation.image_ids[pos:pos + MICROBATCH])
    grads = [p.grad for p in params]
    norm = _grad_norm(grads)
    if not math.isfinite(norm) or norm <= 0 or any(g is not None and not torch.isfinite(g).all() for g in grads):
        raise FloatingPointError("pilot disposable preflight gradient invalid")
    torch.nn.utils.clip_grad_norm_(params, sw139.CLIP)
    before = [p.detach().clone() for p in params]
    optimizer.step()
    changed = sum(not torch.equal(old, new) for old, new in zip(before, params))
    if changed == 0: raise AssertionError("pilot disposable Adam step changed no graph/encoder parameter")
    post_sha = _state_sha((("wrapped", foundation.wrapped), ("encoder", foundation.encoder)))
    if cache is not None:
        cache.verify_complete_payload(require_all_chunks=False); cache.close()
    record = {"status": "disposable_preflight_complete_local_unregistered",
              "seed": SEED, "arm": arm, "optimizer_updates": 1,
              "ground_truth_used": False, "source_core_sha256": foundation.provenance["source_core_sha256"],
              "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
              "source_training_ids_sha256": foundation.provenance["source_training_ids_sha256"],
              "first_batch_ids": ids, "first_batch_ids_sha256": anchor.image_ids_sha256(ids),
              "first_batch_pool_rows": list(map(int, foundation.pool_indices[:LOGICAL_BATCH])),
              "asset_hashes": foundation.provenance["rgb_asset_validation"],
              "foundation_implementation_fingerprint": foundation.provenance["implementation_fingerprint"],
              "lambda_c": lambda_c, "lambda_c_record_sha256": sha256_file(LAMBDA_C_PATH),
              "lambda_a": 0.0 if lambda_a is None else lambda_a["lambda_A"],
              "lambda_a_sha256": None if lambda_a is None else sha256_file(lambda_a_record),
              "teacher_cache_manifest_sha256": None if cache is None else cache.manifest_sha256,
              "source_q_cache_first_logical_batch_b4_parity": cache_parity,
              "c_micro_losses": c_losses, "anchor_micro_losses": a_losses,
              "gradient_norm_before_clip": norm, "changed_graph_encoder_parameters": changed,
              "initial_state_sha256": pre_sha, "disposable_updated_state_sha256": post_sha,
              "probe_discarded": True, "actual_training_starts_fresh": True,
              "implementation_fingerprint": implementation_fingerprint()}
    _write_once(output, record)
    return record


def _teacher_cache_path(seed, count):
    return CACHE_ROOT / f"source{seed}_n{count}"


def _train_arm(seed, arm, *, device, teacher_cache=None, lambda_a_record=None,
               output_root=TRAIN_ROOT, updates=None):
    if seed != SEED or arm not in ARMS:
        raise ValueError("pilot only supports the fixed seed1 C-only and C+anchor arms")
    updates = int(sw139.UPDATES if updates is None else updates)
    if updates < 1 or updates > sw139.UPDATES:
        raise ValueError("updates must be between1 and the registered256 (CPU smoke may request fewer)")
    preflight_path = Path(output_root) / f"preflight_{arm}_seed{seed}.json"
    if not preflight_path.is_file():
        raise FileNotFoundError(f"required fresh-source preflight is absent: {preflight_path}")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (preflight.get("status") != "disposable_preflight_complete_local_unregistered"
            or preflight.get("seed") != seed or preflight.get("arm") != arm
            or preflight.get("implementation_fingerprint") != implementation_fingerprint()
            or preflight.get("ground_truth_used") is not False
            or preflight.get("optimizer_updates") != 1 or preflight.get("probe_discarded") is not True
            or preflight.get("actual_training_starts_fresh") is not True):
        raise ValueError("arm preflight is incomplete or bound to different trainer/source code")
    target = Path(output_root) / f"{arm}_seed{seed}"
    if target.exists(): raise FileExistsError(f"preserve existing pilot output: {target}")
    lambda_c_record = sw139.load_lambda_record(LAMBDA_C_PATH)
    lambda_c = float(lambda_c_record["lambda"])
    lambda_a = None
    cache = None
    if arm == "c_plus_anchor":
        if teacher_cache is None or lambda_a_record is None:
            raise ValueError("C+anchor arm requires verified seed1 teacher cache and frozen lambda_A")
        lambda_a = _load_lambda_a(lambda_a_record)
        foundation = source_bundle(seed, device)
        cache = TeacherCache(teacher_cache, foundation, expected_count=TRAIN_N)
    else:
        foundation = source_bundle(seed, device)
    source = sw130.source_contract(seed)
    if (list(map(int, foundation.image_ids)) != list(map(int, source[4]))
            or not np.array_equal(foundation.pool_indices, np.asarray(source[3], dtype=np.int64))):
        raise AssertionError("student source order mismatch")
    if (preflight.get("source_core_sha256") != foundation.provenance["source_core_sha256"]
            or preflight.get("source_manifest_sha256") != foundation.provenance["source_manifest_sha256"]
            or preflight.get("source_training_ids_sha256") != foundation.provenance["source_training_ids_sha256"]
            or preflight.get("lambda_c_record_sha256") != sha256_file(LAMBDA_C_PATH)
            or preflight.get("lambda_c") != lambda_c):
        raise ValueError("arm preflight source or lambda_C provenance differs from current inputs")
    if preflight.get("asset_hashes") != foundation.provenance["rgb_asset_validation"]:
        raise ValueError("arm preflight RGB/statistics assets differ from current validated source")
    if preflight.get("foundation_implementation_fingerprint") != foundation.provenance["implementation_fingerprint"]:
        raise ValueError("arm preflight native foundation dependencies differ")
    expected_cache_sha = None if cache is None else cache.manifest_sha256
    expected_lambda_a_sha = None if lambda_a is None else sha256_file(lambda_a_record)
    if (preflight.get("teacher_cache_manifest_sha256") != expected_cache_sha
            or preflight.get("lambda_a_sha256") != expected_lambda_a_sha
            or (lambda_a is not None and preflight.get("lambda_a") != lambda_a["lambda_A"])):
        raise ValueError("arm preflight does not bind the current source-Q cache/lambda_A artifacts")
    params = _trainable(foundation)
    graph_params = [p for p in foundation.wrapped.core.graph_generator.parameters() if p.requires_grad]
    encoder_params = list(foundation.encoder.parameters())
    optimizer = torch.optim.Adam([{"params": graph_params, "lr": sw139.CORE_LR},
                                  {"params": encoder_params, "lr": sw139.ENCODER_LR}])
    initial_trainable_state_sha = _state_sha((("wrapped", foundation.wrapped),
                                               ("encoder", foundation.encoder)))
    rgb_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    target.mkdir(parents=True, exist_ok=False)
    history, ids = [], list(map(int, foundation.image_ids[:updates * LOGICAL_BATCH]))
    started = time.time()
    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        losses = {"c": [], "anchor": []}
        cache_parity_rows = []
        logical_ids = []
        for micro in range(MICROS):
            pos = update * LOGICAL_BATCH + micro * MICROBATCH
            rgb = sw139.read_rgb_rows(foundation.pool_indices[pos:pos + MICROBATCH], device, rgb_cache)
            gamma = sw139._features_to_gamma(foundation, rgb, None)
            trace = foundation.rollout(gamma, total_steps=STEPS, live_tail_steps=TAIL)
            nonce = 139000 + seed * 100000 + (update + 1) * 100 + micro
            c_loss, _, _, _ = _paired_contrast(foundation, trace, rgb, nonce=nonce)
            loss = compose_objective(c_loss, lambda_c=lambda_c)
            losses["c"].append(float(c_loss.detach()))
            if arm == "c_plus_anchor":
                assert cache is not None
                student_q = _q_for_trace(trace)
                teacher_q = cache.read_b4(pos, device)
                if update == 0:
                    parity = _q_cache_parity(student_q, teacher_q)
                    cache_parity_rows.append(parity)
                    if not parity["allclose_atol_2e-6_rtol_1e-6"]:
                        raise AssertionError(f"training first-B16 cache parity failed in B4 micro{micro}")
                a_loss = anchor.source_q_anchor(student_q, teacher_q)
                loss = compose_objective(c_loss, lambda_c=lambda_c,
                                         anchor_loss=a_loss,
                                         lambda_a=float(lambda_a["lambda_A"]))
                losses["anchor"].append(float(a_loss.detach()))
            if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss at update {update + 1}")
            (loss / MICROS).backward()
            logical_ids.extend(int(v) for v in foundation.image_ids[pos:pos + MICROBATCH])
        gradients = [p.grad for p in params]
        if any(g is not None and not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError(f"nonfinite gradient at update {update + 1}")
        norm = _grad_norm(gradients)
        if norm <= 0: raise FloatingPointError(f"empty gradient at update {update + 1}")
        torch.nn.utils.clip_grad_norm_(params, sw139.CLIP)
        optimizer.step()
        history.append({"update": update + 1, "image_ids": logical_ids,
                        "image_ids_sha256": anchor.image_ids_sha256(logical_ids),
                        "c_micro_mean": losses["c"], "anchor_micro_mean": losses["anchor"],
                        "first_update_teacher_cache_b4_parity": cache_parity_rows,
                        "weighted_c_lambda": lambda_c,
                        "lambda_a": 0.0 if lambda_a is None else lambda_a["lambda_A"],
                        "gradient_norm_before_clip": norm})
        if (update + 1) % 8 == 0 or update + 1 == updates:
            print(json.dumps({"stage": "train", "arm": arm, "seed": seed,
                              "update": update + 1, "updates": updates}, allow_nan=False), flush=True)
    if cache is not None:
        cache.verify_complete_payload(); cache.close()
    # Preserve scientific results even if a later endpoint fails. Training
    # artifacts are create-only and the marker is written last.
    ckpt = target / "checkpoint.pt"
    optimizer_path = target / "optimizer.pt"
    torch.save(optimizer.state_dict(), optimizer_path)
    pool_rows = list(map(int, foundation.pool_indices[:updates * LOGICAL_BATCH]))
    torch.save({"experiment": PILOT_NAME, "seed": seed, "arm": arm,
                "source_core_sha256": foundation.provenance["source_core_sha256"],
                "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
                "source_training_ids_sha256": foundation.provenance["source_training_ids_sha256"],
                "training_ids": ids, "training_ids_sha256": anchor.image_ids_sha256(ids),
                "training_pool_rows": pool_rows,
                "training_pool_rows_sha256": anchor.image_ids_sha256(pool_rows),
                "asset_hashes": foundation.provenance["rgb_asset_validation"],
                "foundation_implementation_fingerprint": foundation.provenance["implementation_fingerprint"],
                "wrapped_state_dict": foundation.wrapped.state_dict(),
                "encoder_state_dict": foundation.encoder.state_dict(),
                "lambda_c": lambda_c, "lambda_a": 0.0 if lambda_a is None else lambda_a["lambda_A"],
                "lambda_a_sha256": None if lambda_a is None else sha256_file(lambda_a_record),
                "implementation_fingerprint": implementation_fingerprint()}, ckpt)
    hist = target / "history.json"
    with hist.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"updates": updates, "records": history}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    manifest = {"experiment": PILOT_NAME, "status": "training_complete_local_unregistered",
                "seed": seed, "arm": arm, "updates": updates, "passes": 1,
                "unique_training_images": updates * LOGICAL_BATCH,
                "total_image_exposures": updates * LOGICAL_BATCH * 2,
                "views_per_image": 2,
                "extra_teacher_rollouts": 0 if arm == "c_only" else TRAIN_N,
                "batch_size": LOGICAL_BATCH, "microbatch_size": MICROBATCH,
                "steps": STEPS, "settle": SETTLE, "live_tail": TAIL,
                "source_core_sha256": foundation.provenance["source_core_sha256"],
                "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
                "source_training_ids_sha256": foundation.provenance["source_training_ids_sha256"],
                "initial_wrapped_encoder_state_sha256": initial_trainable_state_sha,
                "ordered_training_ids": ids, "ordered_training_ids_sha256": anchor.image_ids_sha256(ids),
                "ordered_pool_rows": pool_rows,
                "ordered_pool_rows_sha256": anchor.image_ids_sha256(pool_rows),
                "asset_hashes": foundation.provenance["rgb_asset_validation"],
                "foundation_implementation_fingerprint": foundation.provenance["implementation_fingerprint"],
                "lambda_c": lambda_c, "lambda_c_record_sha256": sha256_file(LAMBDA_C_PATH),
                "lambda_a": 0.0 if lambda_a is None else lambda_a["lambda_A"],
                "lambda_a_record_sha256": None if lambda_a is None else sha256_file(lambda_a_record),
                "teacher_cache_manifest_sha256": None if cache is None else cache.manifest_sha256,
                "implementation_fingerprint": implementation_fingerprint(),
                "checkpoint_sha256": sha256_file(ckpt), "optimizer_sha256": sha256_file(optimizer_path),
                "history_sha256": sha256_file(hist), "preflight_sha256": sha256_file(preflight_path),
                "ground_truth_used_for_training": False, "elapsed_seconds": time.time() - started}
    _write_once(target / "manifest.json", manifest)
    _write_once(target / "TRAINING_COMPLETED.json", {
        "status": "training_complete_local_unregistered",
        "manifest_sha256": sha256_file(target / "manifest.json"),
        "checkpoint_sha256": sha256_file(ckpt), "updates": updates,
        "ground_truth_used_for_training": False})
    return manifest


def validate_pilot_checkpoint(folder, *, arm, seed=SEED):
    folder = Path(folder)
    manifest_path, checkpoint_path = folder / "manifest.json", folder / "checkpoint.pt"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker_path = folder / "TRAINING_COMPLETED.json"
    optimizer_path, history_path = folder / "optimizer.pt", folder / "history.json"
    if (manifest.get("experiment") != PILOT_NAME
            or manifest.get("status") != "training_complete_local_unregistered"
            or manifest.get("seed") != seed or manifest.get("arm") != arm
            or manifest.get("updates") != sw139.UPDATES
            or manifest.get("unique_training_images") != TRAIN_N
            or manifest.get("total_image_exposures") != TRAIN_N * 2
            or manifest.get("views_per_image") != 2
            or manifest.get("batch_size") != LOGICAL_BATCH
            or manifest.get("microbatch_size") != MICROBATCH
            or manifest.get("steps") != STEPS or manifest.get("settle") != SETTLE
            or manifest.get("live_tail") != TAIL or manifest.get("passes") != 1
            or manifest.get("ordered_training_ids_sha256") != anchor.image_ids_sha256(
                manifest.get("ordered_training_ids", []))
            or manifest.get("implementation_fingerprint") != implementation_fingerprint()
            or manifest.get("ground_truth_used_for_training") is not False
            or manifest.get("checkpoint_sha256") != sha256_file(checkpoint_path)
            or manifest.get("optimizer_sha256") != sha256_file(optimizer_path)
            or manifest.get("history_sha256") != sha256_file(history_path)):
        raise ValueError("pilot checkpoint/manifest provenance mismatch")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (marker.get("status") != "training_complete_local_unregistered"
            or marker.get("updates") != sw139.UPDATES
            or marker.get("ground_truth_used_for_training") is not False
            or marker.get("manifest_sha256") != sha256_file(manifest_path)
            or marker.get("checkpoint_sha256") != sha256_file(checkpoint_path)):
        raise ValueError("pilot training completion marker does not bind saved artifacts")
    foundation = source_bundle(seed, "cpu", verify_rgb_assets=True)
    expected_ids = list(map(int, foundation.image_ids))
    expected_rows = list(map(int, foundation.pool_indices))
    if (manifest.get("source_core_sha256") != foundation.provenance["source_core_sha256"]
            or manifest.get("source_manifest_sha256") != foundation.provenance["source_manifest_sha256"]
            or manifest.get("source_training_ids_sha256") != foundation.provenance["source_training_ids_sha256"]
            or manifest.get("ordered_training_ids") != expected_ids
            or manifest.get("ordered_pool_rows") != expected_rows
            or manifest.get("ordered_pool_rows_sha256") != anchor.image_ids_sha256(expected_rows)
            or manifest.get("asset_hashes") != foundation.provenance["rgb_asset_validation"]
            or manifest.get("foundation_implementation_fingerprint") != foundation.provenance["implementation_fingerprint"]
            or manifest.get("source_core_sha256") != sw130.source97.EXPECTED_SOURCE_SHAS[seed]):
        raise ValueError("pilot completion is not bound to the current strict source/order")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    records = history.get("records")
    if history.get("updates") != sw139.UPDATES or not isinstance(records, list) or len(records) != sw139.UPDATES:
        raise ValueError("pilot history is incomplete")
    for index, row in enumerate(records):
        expected = expected_ids[index * LOGICAL_BATCH:(index + 1) * LOGICAL_BATCH]
        cache_parity = row.get("first_update_teacher_cache_b4_parity", [])
        parity_valid = ((arm == "c_plus_anchor" and index == 0
                         and len(cache_parity) == MICROS
                         and all(item.get("allclose_atol_2e-6_rtol_1e-6") is True
                                 and math.isfinite(float(item.get("max_abs_difference", float("nan"))))
                                 for item in cache_parity))
                        or ((arm == "c_only" or index > 0) and cache_parity == []))
        if (row.get("update") != index + 1 or row.get("image_ids") != expected
                or row.get("image_ids_sha256") != anchor.image_ids_sha256(expected)
                or len(row.get("c_micro_mean", [])) != MICROS
                or not np.isfinite(np.asarray(row.get("c_micro_mean", []), dtype=np.float64)).all()
                or not math.isfinite(float(row.get("gradient_norm_before_clip", float("nan"))))
                or float(row["gradient_norm_before_clip"]) <= 0
                or (arm == "c_plus_anchor" and (len(row.get("anchor_micro_mean", [])) != MICROS
                    or not np.isfinite(np.asarray(row["anchor_micro_mean"], dtype=np.float64)).all()))
                or (arm == "c_only" and row.get("anchor_micro_mean") != [])
                or not parity_valid):
            raise ValueError(f"pilot update {index + 1} violates order/loss/history contract")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (state.get("seed") != seed or state.get("arm") != arm
            or state.get("implementation_fingerprint") != implementation_fingerprint()
            or state.get("training_ids_sha256") != manifest.get("ordered_training_ids_sha256")
            or state.get("training_pool_rows_sha256") != manifest.get("ordered_pool_rows_sha256")
            or state.get("asset_hashes") != foundation.provenance["rgb_asset_validation"]):
        raise ValueError("pilot checkpoint state identity mismatch")
    if state.get("foundation_implementation_fingerprint") != foundation.provenance["implementation_fingerprint"]:
        raise ValueError("pilot checkpoint native foundation dependencies differ")
    if (state.get("wrapped_state_dict") is None or state.get("encoder_state_dict") is None
            or any(torch.is_tensor(value) and not torch.isfinite(value).all()
                   for payload in (state["wrapped_state_dict"], state["encoder_state_dict"])
                   for value in payload.values())):
        raise ValueError("pilot checkpoint contains missing/nonfinite model tensors")
    optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    steps = []
    for row in optimizer.get("state", {}).values():
        value = row.get("step")
        steps.append(int(value.item() if torch.is_tensor(value) else value))
        if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in row.values()):
            raise ValueError("pilot Adam state contains nonfinite values")
    if not steps or set(steps) != {sw139.UPDATES}:
        raise ValueError("pilot Adam must be persisted at exactly256 updates")
    return {"manifest": manifest, "manifest_path": manifest_path,
            "manifest_sha256": sha256_file(manifest_path),
            "marker_path": marker_path, "checkpoint": checkpoint_path,
            "checkpoint_sha256": manifest["checkpoint_sha256"], "state": state}


def predict_arm(arm, *, seed=SEED, device="cuda:0", training_root=TRAIN_ROOT,
                output_root=RESULT_ROOT):
    if arm not in ARMS: raise ValueError("unregistered source-preservation arm")
    folder = Path(training_root) / f"{arm}_seed{seed}"
    verified = validate_pilot_checkpoint(folder, arm=arm, seed=seed)
    out = Path(output_root) / arm
    if out.exists(): raise FileExistsError(f"preserve existing prediction output: {out}")
    out.mkdir(parents=True, exist_ok=False)
    foundation = source_bundle(seed, device)
    state = torch.load(verified["checkpoint"], map_location=device, weights_only=True)
    foundation.wrapped.load_state_dict(state["wrapped_state_dict"], strict=True)
    foundation.encoder.load_state_dict(state["encoder_state_dict"], strict=True)
    foundation.wrapped.eval(); foundation.encoder.eval()
    cache = np.load(sw130.VAL_RGB, mmap_mode="r", allow_pickle=False)
    labels = np.empty((contract32.COUNT, 32, 32), dtype=np.int64)
    for i in range(contract32.COUNT):
        rgb = sw139_eval.read_eval_rgb(cache, i, device)
        gamma = sw139._features_to_gamma(foundation, rgb, None)
        trace = foundation.rollout(gamma, total_steps=STEPS, live_tail_steps=0)
        labels[i:i+1] = sw136_eval._qcc_labels(trace)
        if (i + 1) % 40 == 0:
            print(json.dumps({"stage": "predict", "arm": arm, "completed": i + 1,
                              "total": contract32.COUNT}), flush=True)
    if (labels.shape != (contract32.COUNT, 32, 32)
            or not np.issubdtype(labels.dtype, np.integer)
            or np.issubdtype(labels.dtype, np.bool_)
            or not np.isfinite(labels).all() or np.any(labels < 0) or np.any(labels > 1024)):
        raise ValueError("native QCC predictor emitted invalid fixed320 labels")
    protocol = _build_own_prediction_protocol(arm, seed, verified, labels)
    npz = out / "predictions.npz"
    with npz.open("xb") as stream:
        np.savez_compressed(stream, image_ids=np.asarray(contract32.IMAGE_IDS, dtype=np.int64),
                            qcc_labels=labels)
    protocol["prediction_sha256"] = sha256_file(npz)
    _write_once(out / "protocol.json", protocol)
    return protocol


OWN_READOUT = {"threshold": .50, "min_group_size": 8,
               "background": "largest connected component",
               "affinity": "native actual-spike product QCC"}


def _build_own_prediction_protocol(arm, seed, verified, labels):
    """Single producer for the exact protocol schema consumed by the scorer."""
    if arm not in ARMS or seed != SEED:
        raise ValueError("prediction protocol requires the registered seed1 arm")
    manifest = verified["manifest"]
    if np.asarray(labels).shape != (320, 32, 32):
        raise ValueError("prediction protocol requires fixed320 native32 labels")
    return {"experiment": PILOT_NAME, "status": "predictions_complete_local_unregistered",
            "seed": seed, "arm": arm, "image_ids": [1320, 1639], "count": 320,
            "prediction_shape": [320, 32, 32], "steps": STEPS, "settle": SETTLE,
            "readout": OWN_READOUT,
            "checkpoint_sha256": verified["checkpoint_sha256"],
            "training_manifest_sha256": verified["manifest_sha256"],
            "source_core_sha256": manifest["source_core_sha256"],
            "source_manifest_sha256": manifest["source_manifest_sha256"],
            "source_training_ids_sha256": manifest["source_training_ids_sha256"],
            "implementation_fingerprint": implementation_fingerprint(),
            "ground_truth_used_for_prediction": False, "optimizer_updates": 0}


def validate_own_prediction(arm, *, prediction_root=RESULT_ROOT, training_root=TRAIN_ROOT):
    """Validate prediction against the actual completed training artifacts."""
    if arm not in ARMS:
        raise ValueError("unregistered pilot arm")
    train_dir = Path(training_root) / f"{arm}_seed{SEED}"
    trained = validate_pilot_checkpoint(train_dir, arm=arm, seed=SEED)
    manifest = trained["manifest"]
    protocol_path = Path(prediction_root) / arm / "protocol.json"
    npz_path = protocol_path.parent / "predictions.npz"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    expected = {
        "experiment": PILOT_NAME,
        "status": "predictions_complete_local_unregistered",
        "seed": SEED, "arm": arm, "image_ids": [1320, 1639], "count": 320,
        "prediction_shape": [320, 32, 32], "steps": STEPS, "settle": SETTLE,
        "readout": OWN_READOUT,
        "checkpoint_sha256": trained["checkpoint_sha256"],
        "training_manifest_sha256": trained["manifest_sha256"],
        "source_core_sha256": manifest["source_core_sha256"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "source_training_ids_sha256": manifest["source_training_ids_sha256"],
        "implementation_fingerprint": implementation_fingerprint(),
        "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"{arm} prediction protocol mismatch for {key}")
    npz_sha = sha256_file(npz_path)
    if protocol.get("prediction_sha256") != npz_sha:
        raise ValueError(f"{arm} prediction archive SHA mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        if not {"image_ids", "qcc_labels"}.issubset(archive.files):
            raise ValueError(f"{arm} prediction archive is missing required arrays")
        ids = contract32.require_fixed_image_ids(archive["image_ids"])
        labels = np.asarray(archive["qcc_labels"])
    if (labels.shape != (320, 32, 32) or not np.issubdtype(labels.dtype, np.integer)
            or np.issubdtype(labels.dtype, np.bool_) or not np.isfinite(labels).all()
            or np.any(labels < 0) or np.any(labels > 1024)):
        raise ValueError(f"{arm} prediction labels violate fixed320 native32 schema")
    if (sha256_file(npz_path) != npz_sha
            or hashlib.sha256(protocol_path.read_bytes()).hexdigest()
            != hashlib.sha256(protocol_bytes).hexdigest()):
        raise ValueError(f"{arm} prediction artifacts changed during validation")
    return {"labels": labels.astype(np.int64, copy=False), "ids": ids,
            "npz": npz_path, "npz_sha": npz_sha,
            "protocol": protocol_path,
            "protocol_sha": hashlib.sha256(protocol_bytes).hexdigest(),
            "training": trained}


def score_pilot(*, prediction_root=RESULT_ROOT, source_root=sw137.OUTPUT_ROOT,
                old_c_root=sw139_eval.EVALUATION_ROOT, dataset=sw139_eval.DATASET,
                slot_root=None, training_root=TRAIN_ROOT,
                output=RESULT_ROOT / "evaluation.json", bootstrap_samples=10000):
    """Validate all fixed320 arrays/provenance before loading GT and scoring."""
    output = Path(output)
    if output.exists(): raise FileExistsError(f"preserve existing summary: {output}")
    loaded = {}
    # Include the two newly trained arms, same-seed source97, and completed
    # SW0139 old+C reference. A missing old+C reference blocks scoring honestly.
    for arm in ARMS:
        loaded[arm] = validate_own_prediction(arm, prediction_root=prediction_root,
                                              training_root=training_root)
    source = sw137.validate_prediction(sw137._prediction_dir(SEED, source_root), seed=SEED)
    loaded["source97"] = {"labels": source["labels"], "ids": source["ids"],
                          "npz": source["archive_path"], "npz_sha": source["prediction_sha256"],
                          "protocol": source["protocol_path"], "protocol_sha": source["protocol_sha256"]}
    old_row = sw139_eval.validate_arm_prediction("cross_view", seed=SEED,
                                                  root=old_c_root, training_root=sw139.OUT)
    loaded["old_c"] = {"labels": old_row["labels"], "ids": old_row["ids"],
                        "npz": old_row["prediction_path"], "npz_sha": old_row["prediction_sha256"],
                        "protocol": old_row["protocol_path"], "protocol_sha": old_row["protocol_sha256"]}
    # Same-seed own70k Slot baseline, pooled directly from frozen native128 labels.
    slot_root = ROOT / "trained_models/SW0092_slot_our70000_eval" if slot_root is None else Path(slot_root)
    slot_dir = slot_root / "seed1_epoch10"
    slot_npz, slot_protocol_path = slot_dir / "predictions.npz", slot_dir / "protocol.json"
    slot_protocol_bytes = slot_protocol_path.read_bytes()
    slot_protocol = json.loads(slot_protocol_bytes.decode("utf-8-sig"))
    contract32.validate_slot_protocol(slot_protocol, seed=SEED, prediction_path=slot_npz)
    with np.load(slot_npz, allow_pickle=False) as saved_slot:
        if not {"labels", "image_ids"}.issubset(saved_slot.files):
            raise ValueError("registered Slot prediction lacks labels/image_ids")
        slot_ids = contract32.require_fixed_image_ids(saved_slot["image_ids"], name="Slot image_ids")
        slot_pixels = np.asarray(saved_slot["labels"])
    slot32 = contract32.slot_native_pixels_to32(slot_pixels)
    loaded["slot32"] = {"labels": slot32, "ids": slot_ids, "npz": slot_npz,
                        "npz_sha": slot_protocol["prediction_sha256"],
                        "protocol": slot_protocol_path,
                        "protocol_sha": hashlib.sha256(slot_protocol_bytes).hexdigest()}
    for name, row in loaded.items():
        if not np.array_equal(row["ids"], np.asarray(contract32.IMAGE_IDS)):
            raise ValueError(f"{name} does not match the fixed320 image order")
        if sha256_file(row["npz"]) != row["npz_sha"] or sha256_file(row["protocol"]) != row["protocol_sha"]:
            raise ValueError(f"{name} frozen predictions changed before target access")
    # GT is opened only after every prediction has passed identity and SHA checks.
    with __import__("h5py").File(dataset, "r") as hdf:
        if "mask" not in hdf or hdf["mask"].shape[0] < 1640:
            raise ValueError("registered HDF5 target mask slice is unavailable")
        pixel_masks = np.asarray(hdf["mask"][1320:1640])
    target = contract32.modal_native32(pixel_masks)
    metrics = {name: contract32.production_metrics32(row["labels"], target)
               for name, row in loaded.items()}
    comparisons = {}
    for candidate in ARMS:
        comparisons[candidate] = {}
        for reference in ("source97", "old_c", "c_only", "slot32"):
            if candidate == reference: continue
            differences = (np.asarray(metrics[candidate]["fg_ari"]["per_image"])
                           - np.asarray(metrics[reference]["fg_ari"]["per_image"]))
            low, high = paired_bootstrap_ci(differences, samples=bootstrap_samples)
            comparisons[candidate][reference] = {"mean_fg_ari_delta": float(differences.mean()),
                                                   "paired_95_ci": [low, high]}
    summary = {"status": "complete_local_unregistered", "experiment": PILOT_NAME,
               "ground_truth_opened_after_all_prediction_hashes": True,
               "ground_truth_used_for_prediction_or_training": False,
               "scope": "same-seed1 four-way endpoint comparison; no three-seed or scaling claim",
               "bootstrap": {"samples": bootstrap_samples, "rng": "numpy RandomState(139)",
                             "paired_common_image_ids": [1320, 1639]},
               "artifacts": {name: {"prediction_sha256": row["npz_sha"],
                                    "protocol_sha256": row["protocol_sha"]}
                             for name, row in loaded.items()},
               "scores": metrics, "paired_fg_ari": comparisons}
    ca_row = comparisons["c_plus_anchor"]
    summary["registered_pilot_expansion_check"] = {
        "c_plus_anchor_fg_mean_exceeds_source97": (
            metrics["c_plus_anchor"]["fg_ari"]["mean"] > metrics["source97"]["fg_ari"]["mean"]),
        "c_plus_anchor_fg_mean_exceeds_c_only": (
            metrics["c_plus_anchor"]["fg_ari"]["mean"] > metrics["c_only"]["fg_ari"]["mean"]),
        "paired_ci_lower_above_zero_vs_source97": ca_row["source97"]["paired_95_ci"][0] > 0,
        "paired_ci_lower_above_zero_vs_c_only": ca_row["c_only"]["paired_95_ci"][0] > 0,
        "eligible_to_consider_seed_expansion": all((
            metrics["c_plus_anchor"]["fg_ari"]["mean"] > metrics["source97"]["fg_ari"]["mean"],
            metrics["c_plus_anchor"]["fg_ari"]["mean"] > metrics["c_only"]["fg_ari"]["mean"],
            ca_row["source97"]["paired_95_ci"][0] > 0,
            ca_row["c_only"]["paired_95_ci"][0] > 0)),
        "automatic_expansion_or_promotion": False,
        "three_seed_or_scaling_claim": False,
    }
    _write_once(output, summary)
    return summary


def paired_bootstrap_ci(differences, *, samples=10000, seed=139):
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (320,) or not np.isfinite(values).all() or samples < 1:
        raise ValueError("paired bootstrap requires 320 finite image-paired differences")
    rng = np.random.RandomState(seed)
    means = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        means[i] = values[rng.randint(0, values.size, size=values.size)].mean()
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    cache = sub.add_parser("cache", help="create one read-only source-Q cache")
    cache.add_argument("--seed", type=int, choices=(0, 1), required=True)
    cache.add_argument("--device", default="cuda:0")
    cache.add_argument("--output", type=Path, required=True)
    cal = sub.add_parser("calibrate", help="source0 disposable C-only update and lambda_A calibration")
    cal.add_argument("--source0-cache", type=Path, required=True)
    cal.add_argument("--device", default="cuda:0")
    cal.add_argument("--output", type=Path, default=LAMBDA_A_PATH)
    pre = sub.add_parser("preflight", help="one disposable fresh-source pilot update")
    pre.add_argument("--arm", choices=ARMS, required=True)
    pre.add_argument("--device", default="cuda:0")
    pre.add_argument("--teacher-cache", type=Path)
    pre.add_argument("--lambda-a", type=Path, default=LAMBDA_A_PATH)
    pre.add_argument("--output-root", type=Path, default=TRAIN_ROOT)
    train = sub.add_parser("train", help="run the fixed seed1 C-only or C+anchor one-pass pilot")
    train.add_argument("--arm", choices=ARMS, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--teacher-cache", type=Path)
    train.add_argument("--lambda-a", type=Path, default=LAMBDA_A_PATH)
    train.add_argument("--output-root", type=Path, default=TRAIN_ROOT)
    train.add_argument("--updates", type=int, default=sw139.UPDATES,
                       help="registered pilot uses256; smaller value is only for local smoke tests")
    pred = sub.add_parser("predict", help="freeze seed1 native32 actual-spike QCC predictions")
    pred.add_argument("--arm", choices=ARMS, required=True)
    pred.add_argument("--device", default="cuda:0")
    pred.add_argument("--training-root", type=Path, default=TRAIN_ROOT)
    pred.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    score = sub.add_parser("score", help="GT-last fixed320 score after all predictions validate")
    score.add_argument("--prediction-root", type=Path, default=RESULT_ROOT)
    score.add_argument("--source-root", type=Path, default=sw137.OUTPUT_ROOT)
    score.add_argument("--old-c-root", type=Path, default=sw139_eval.EVALUATION_ROOT)
    score.add_argument("--slot-root", type=Path,
                       default=ROOT / "trained_models/SW0092_slot_our70000_eval")
    score.add_argument("--dataset", type=Path, default=sw139_eval.DATASET)
    score.add_argument("--output", type=Path, default=RESULT_ROOT / "evaluation.json")
    args = parser.parse_args(argv)
    if args.stage == "cache": result = write_teacher_cache(args.seed, args.output, args.device)
    elif args.stage == "calibrate": result = calibrate_lambda_a(source0_cache=args.source0_cache,
                                                                  output=args.output, device=args.device)
    elif args.stage == "preflight": result = preflight_arm(args.arm, device=args.device,
                                                            teacher_cache=args.teacher_cache,
                                                            lambda_a_record=args.lambda_a,
                                                            output_root=args.output_root)
    elif args.stage == "train": result = _train_arm(SEED, args.arm, device=args.device,
                                                     teacher_cache=args.teacher_cache,
                                                     lambda_a_record=args.lambda_a,
                                                     output_root=args.output_root, updates=args.updates)
    elif args.stage == "predict": result = predict_arm(args.arm, device=args.device,
                                                        training_root=args.training_root,
                                                        output_root=args.output_root)
    else: result = score_pilot(prediction_root=args.prediction_root, source_root=args.source_root,
                               old_c_root=args.old_c_root, slot_root=args.slot_root,
                               dataset=args.dataset, output=args.output)
    print(json.dumps({"status": result.get("status"), "stage": args.stage,
                      "lambda_A": result.get("lambda_A")}, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
