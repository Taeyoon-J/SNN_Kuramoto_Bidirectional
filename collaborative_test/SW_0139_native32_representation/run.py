"""SW0139 controlled native32 representation pilot (training has not been run)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0139_native32_representation.model import ContextResidualEncoder
from snn_kuramoto_bidirectional.loss_function import phase_locking_value
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity

SEED = 1
SOURCE_SEED = 0
ARMS = ("control", "context_residual", "cross_view")
TRAIN_IMAGES = 4096
LOGICAL_BATCH = 16
MICROBATCH = 4
MICROS = LOGICAL_BATCH // MICROBATCH
UPDATES = TRAIN_IMAGES // LOGICAL_BATCH
STEPS = 1024
SETTLE = 512
TAIL = 64
TEMPERATURE = 0.10
NEGATIVES = 32
CORE_LR = 3e-5
ENCODER_LR = 3e-6
CONTEXT_LR = 3e-4
CLIP = 1.0
ARCHIVE = HERE / "results_archive"
OUT = ROOT / "trained_models/SW0139_native32_representation"
LAMBDA_PATH = ARCHIVE / "lambda_seed0.json"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_once(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(payload)


def implementation_fingerprint():
    paths = (
        HERE / "protocol.json", HERE / "model.py", HERE / "run.py",
        ROOT / "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        ROOT / "collaborative_test/SW_0135_native32_spike_binding/resolution.py",
        ROOT / "collaborative_test/SW_0130_phase_state_integration/run.py",
        ROOT / "collaborative_test/SW_0130_phase_state_integration/model.py",
        ROOT / "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
        ROOT / "collaborative_test/SW_0094_aligned_joint_pilot/run.py",
        ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
        ROOT / "snn_kuramoto_bidirectional/graph_generator.py",
        ROOT / "snn_kuramoto_bidirectional/loss_function.py",
        ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
        ROOT / "snn_kuramoto_bidirectional/gamma_initializer.py",
        ROOT / "snn_kuramoto_bidirectional/training/train_gamma_initializer.py",
        ROOT / "snn_kuramoto_bidirectional/dendric_layer.py",
        ROOT / "snn_kuramoto_bidirectional/membrane_layer.py",
        ROOT / "snn_kuramoto_bidirectional/kuramoto_layer.py",
        ROOT / "snn_kuramoto_bidirectional/sinusoidal_gating.py",
    )
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0139 dependency missing: {missing[0]}")
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def source_bundle(seed, device):
    foundation = sw135.load_native32_foundation(seed, device, verify_rgb_assets=True)
    if foundation.provenance.get("ground_truth_used") is not False:
        raise AssertionError("SW0139 source loader unexpectedly used ground truth")
    return foundation


def trainable_backbone(foundation, arm, context):
    wrapped, encoder = foundation.wrapped, foundation.encoder
    for name, p in wrapped.named_parameters():
        p.requires_grad_(name.startswith("core.graph_generator."))
    encoder.requires_grad_(True)
    encoder.train()
    wrapped.train()
    if context is not None:
        context.train()
        context.requires_grad_(True)
    return [p for p in wrapped.parameters() if p.requires_grad] + list(encoder.parameters()) + (
        list(context.parameters()) if context is not None else [])


def _features_to_gamma(foundation, rgb_float, context=None):
    """Encode canonical/augmented float RGB with registered clipping and pool32."""
    if rgb_float.ndim != 4 or rgb_float.shape[1:] != (3, 128, 128):
        raise ValueError("SW0139 encoder input must be NCHW native128 RGB")
    feature = foundation.encoder(rgb_float)
    mean = foundation.feature_mean.to(device=feature.device, dtype=feature.dtype)
    std = foundation.feature_std.to(device=feature.device, dtype=feature.dtype)
    normalized = ((feature - mean) / std).clamp(-foundation.feature_clip, foundation.feature_clip)
    if context is not None:
        normalized = context(normalized)
    gamma = foundation.patcher(normalized)
    if tuple(gamma.shape) != (rgb_float.shape[0], 8, 1024) or not torch.isfinite(gamma).all():
        raise ValueError("SW0139 native32 gamma must be finite [B,8,1024]")
    return gamma


def read_rgb_rows(rows, device, cache=None):
    cache_path = sw130.TRAIN_RGB if cache is None else cache
    images = (np.load(cache_path, mmap_mode="r", allow_pickle=False)
              if isinstance(cache_path, (str, Path)) else cache_path)
    array = np.asarray(images[np.asarray(rows, dtype=np.int64)]).copy()
    if array.dtype != np.uint8 or array.shape[1:] != (128, 128, 3):
        raise ValueError("SW0139 TRAIN RGB rows must be uint8 native128 images")
    return torch.from_numpy(array).to(device=device).permute(0, 3, 1, 2).float() / 255.0


def _old_objective(foundation, gamma):
    trace = foundation.rollout(gamma, total_steps=STEPS, live_tail_steps=TAIL)
    if tuple(trace["component_spikes"].shape) != (gamma.shape[0], 4, 1024, 1024):
        raise AssertionError("SW0139 full native32 spike trace has an unexpected shape")
    criterion = sw135.make_criterion32()
    q = spike_synchrony_affinity(trace["spikes"], trace["component_spikes"],
                                 settle=SETTLE, affinity_mode="spike")
    theta = trace["theta"][:, SETTLE:]
    plv = phase_locking_value(theta, combine="mean")
    phase, _ = criterion(plv=plv, theta=theta)
    positive_q, _ = criterion(plv=q)
    return phase + 5.0 * positive_q, trace


def _sample_other_image_negatives(batch, nodes, negatives, seed, device):
    if batch < 2:
        raise ValueError("cross-view negatives require at least two distinct images")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    images = torch.randint(batch - 1, (batch, nodes, negatives), generator=generator)
    rows = torch.arange(batch)[:, None, None]
    images = images + (images >= rows).long()
    patches = torch.randint(nodes, (batch, nodes, negatives), generator=generator)
    return images.to(device), patches.to(device)


def symmetric_temporal_infonce(spikes_a, spikes_b, *, seed, negatives=NEGATIVES,
                               temperature=TEMPERATURE, patch_chunk=16):
    """Symmetric same-patch cross-view InfoNCE with other-image-only negatives."""
    if (spikes_a.shape != spikes_b.shape or spikes_a.ndim != 4
            or tuple(spikes_a.shape[1:]) != (4, 1024, SETTLE)):
        raise ValueError("InfoNCE requires paired actual spikes [B,4,1024,512]")
    if temperature <= 0 or negatives < 1 or patch_chunk < 1:
        raise ValueError("invalid registered InfoNCE settings")
    batch, _, nodes, _ = spikes_a.shape
    if batch < 2:
        raise ValueError("cross-view negatives require different images")
    a = F.normalize(spikes_a.permute(0, 2, 1, 3).reshape(batch, nodes, 2048),
                    p=2, dim=-1, eps=1e-8)
    b = F.normalize(spikes_b.permute(0, 2, 1, 3).reshape(batch, nodes, 2048),
                    p=2, dim=-1, eps=1e-8)
    image_idx, patch_idx = _sample_other_image_negatives(batch, nodes, negatives, seed, a.device)
    chunks = []
    for start in range(0, nodes, patch_chunk):
        stop = min(start + patch_chunk, nodes)
        idx, pidx = image_idx[:, start:stop], patch_idx[:, start:stop]
        aa, bb = a[:, start:stop], b[:, start:stop]
        negatives_b = b[idx, pidx]
        negatives_a = a[idx, pidx]
        logits_ab = torch.cat(((aa * bb).sum(-1, keepdim=True),
                               torch.einsum("bcf,bcnf->bcn", aa, negatives_b)), dim=-1)
        logits_ba = torch.cat(((bb * aa).sum(-1, keepdim=True),
                               torch.einsum("bcf,bcnf->bcn", bb, negatives_a)), dim=-1)
        target = torch.zeros(batch * (stop - start), dtype=torch.long, device=a.device)
        loss = (F.cross_entropy(logits_ab.reshape(-1, negatives + 1) / temperature,
                                target, reduction="mean")
                + F.cross_entropy(logits_ba.reshape(-1, negatives + 1) / temperature,
                                  target, reduction="mean")) * 0.5
        chunks.append(loss * ((stop - start) / nodes))
    return torch.stack(chunks).sum()


def _paired_view(rgb, *, seed):
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("view augmentation expects [B,3,H,W]")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    b = rgb.shape[0]
    c = 0.9 + 0.2 * torch.rand((b, 1, 1, 1), generator=g)
    offset = -0.03 + 0.06 * torch.rand((b, 1, 1, 1), generator=g)
    c, offset = c.to(rgb.device, rgb.dtype), offset.to(rgb.device, rgb.dtype)
    return ((rgb - 0.5) * c + 0.5 + offset).clamp(0, 1), c.flatten(), offset.flatten()


def _parameter_groups(foundation, arm, context):
    wrapped, encoder = foundation.wrapped, foundation.encoder
    graph = [p for p in wrapped.core.graph_generator.parameters() if p.requires_grad]
    enc = [p for p in encoder.parameters() if p.requires_grad]
    groups = [{"params": graph, "lr": CORE_LR, "name": "graph"},
              {"params": enc, "lr": ENCODER_LR, "name": "encoder"}]
    if context is not None:
        groups.append({"params": list(context.parameters()), "lr": CONTEXT_LR, "name": "context_adapter"})
    parameters = [p for group in groups for p in group["params"]]
    if len({id(p) for p in parameters}) != len(parameters) or not parameters:
        raise AssertionError("SW0139 optimizer groups must be a nonempty unique parameter union")
    return groups, parameters


def _logical_losses(foundation, arm, context, rgb, image_ids, *, nonce):
    gamma = _features_to_gamma(foundation, rgb, context if arm == "context_residual" else None)
    old, trace = _old_objective(foundation, gamma)
    contrast = None
    aug_meta = None
    if arm == "cross_view":
        view, c, offset = _paired_view(rgb, seed=nonce)
        gamma_view = _features_to_gamma(foundation, view, None)
        trace_view = foundation.rollout(gamma_view, total_steps=STEPS, live_tail_steps=TAIL)
        contrast = symmetric_temporal_infonce(
            trace["component_spikes"][..., SETTLE:], trace_view["component_spikes"][..., SETTLE:],
            seed=nonce + 1, negatives=NEGATIVES)
        aug_meta = {"contrast": contrast, "contrast_view": trace_view,
                    "contrastive_view_c": c, "contrastive_view_offset": offset}
    return old, contrast, trace, gamma, aug_meta


def _logical_id_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _family_norms(named, grads):
    totals = {"encoder": 0.0, "graph": 0.0}
    for (name, _), grad in zip(named, grads):
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            raise FloatingPointError(f"nonfinite SW0139 gradient for {name}")
        family = "encoder" if name.startswith("encoder.") else "graph"
        totals[family] += float(grad.detach().double().square().sum())
    return {name: math.sqrt(value) for name, value in totals.items()}


def _feature_map_from_rgb(foundation, rgb):
    fmap = foundation.encoder(rgb)
    return ((fmap - foundation.feature_mean.to(fmap)) / foundation.feature_std.to(fmap)).clamp(
        -foundation.feature_clip, foundation.feature_clip)


def _lambda_calibration(device="cpu", *, batches=4):
    """Fresh seed0 lambda from four logical B16 batches; no model updates."""
    foundation = source_bundle(SOURCE_SEED, device)
    trainable_backbone(foundation, "cross_view", None)
    if batches != 4:
        raise ValueError("SW0139 lambda is registered to exactly four logical TRAIN batches")
    seed_ids = np.asarray(foundation.image_ids[:batches * LOGICAL_BATCH], dtype=np.int64)
    rows = np.asarray(foundation.pool_indices[:batches * LOGICAL_BATCH], dtype=np.int64)
    named = [(f"graph.{n}", p) for n, p in foundation.wrapped.core.graph_generator.named_parameters()
             if p.requires_grad]
    named += [(f"encoder.{n}", p) for n, p in foundation.encoder.named_parameters()]
    params = [p for _, p in named]
    if not params or any(not torch.isfinite(p).all() for p in params):
        raise FloatingPointError("invalid seed0 encoder/graph parameters for lambda calibration")
    norm_rows = []
    rgb_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    for batch_index in range(batches):
        old_acc = [torch.zeros_like(p) for p in params]
        contrast_acc = [torch.zeros_like(p) for p in params]
        micro_details = []
        for micro in range(MICROS):
            start = batch_index * LOGICAL_BATCH + micro * MICROBATCH
            rgb = read_rgb_rows(rows[start:start + MICROBATCH], device, rgb_cache)
            image_ids = seed_ids[start:start + MICROBATCH]
            nonce = 139000 + batch_index * 100 + micro
            old, contrast, _, _, _ = _logical_losses(
                foundation, "cross_view", None, rgb, image_ids, nonce=nonce)
            old_grads = torch.autograd.grad(old, params, retain_graph=True, allow_unused=True)
            contrast_grads = torch.autograd.grad(contrast, params, allow_unused=True)
            for i, (go, gc) in enumerate(zip(old_grads, contrast_grads)):
                if go is not None:
                    old_acc[i].add_(go.detach(), alpha=1.0 / MICROS)
                if gc is not None:
                    contrast_acc[i].add_(gc.detach(), alpha=1.0 / MICROS)
            micro_details.append({"image_ids": image_ids.tolist(), "nonce": nonce,
                                  "old_loss": float(old.detach()),
                                  "contrast_loss": float(contrast.detach())})
        old_norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in old_acc))
        contrast_norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in contrast_acc))
        family_norms = {"old": _family_norms(named, old_acc),
                        "contrastive": _family_norms(named, contrast_acc)}
        if not math.isfinite(old_norm) or not math.isfinite(contrast_norm) or contrast_norm <= 0:
            raise FloatingPointError("source0 accumulated old/InfoNCE gradient norms are invalid")
        if any(not math.isfinite(v) or v <= 0 for by_family in family_norms.values()
               for v in by_family.values()):
            raise FloatingPointError(f"source0 old/InfoNCE gradients must reach encoder and graph: {family_norms}")
        norm_rows.append({"logical_batch": batch_index, "image_ids": seed_ids[
            batch_index * LOGICAL_BATCH:(batch_index + 1) * LOGICAL_BATCH].tolist(),
            "image_ids_sha256": _logical_id_sha(seed_ids[
                batch_index * LOGICAL_BATCH:(batch_index + 1) * LOGICAL_BATCH]),
            "old_encoder_graph_grad_norm": old_norm,
            "contrast_encoder_graph_grad_norm": contrast_norm,
            "gradient_norms_by_family": family_norms,
            "ratio": old_norm / contrast_norm,
            "microbatches": micro_details})
    ratio = float(np.median([row["ratio"] for row in norm_rows]))
    value = 0.25 * ratio
    if not math.isfinite(value) or value <= 0:
        raise FloatingPointError("registered source0 lambda is nonfinite or nonpositive")
    return {"experiment": "SW0139_native32_representation", "status": "lambda_calibrated",
            "seed": 0, "source_core_sha256": foundation.provenance["source_core_sha256"],
            "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
            "source_training_ids_sha256": foundation.provenance["source_training_ids_sha256"],
            "first_four_logical_batches": norm_rows, "median_ratio": ratio,
            "lambda": value, "formula": "0.25*median(accumulated_old_norm/accumulated_cross_view_norm)",
            "batch_size": LOGICAL_BATCH, "microbatch": MICROBATCH,
            "implementation_fingerprint": implementation_fingerprint(),
            "asset_hashes": foundation.provenance["rgb_asset_validation"],
            "optimizer_updates": 0, "ground_truth_used": False}


def load_lambda_record(path=LAMBDA_PATH):
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("experiment") != "SW0139_native32_representation"
            or record.get("status") != "lambda_calibrated"
            or record.get("seed") != 0
            or record.get("implementation_fingerprint") != implementation_fingerprint()
            or record.get("ground_truth_used") is not False
            or record.get("optimizer_updates") != 0
            or len(record.get("first_four_logical_batches", [])) != 4):
        raise ValueError("SW0139 lambda artifact does not match the fixed no-update seed0 calibration")
    ratios = []
    for idx, row in enumerate(record["first_four_logical_batches"]):
        old = float(row.get("old_encoder_graph_grad_norm", float("nan")))
        contrast = float(row.get("contrast_encoder_graph_grad_norm", float("nan")))
        if row.get("logical_batch") != idx or min(old, contrast) <= 0 or not all(map(math.isfinite, (old, contrast))):
            raise ValueError("SW0139 lambda calibration has invalid accumulated gradient evidence")
        expected = old / contrast
        if not math.isclose(float(row.get("ratio", float("nan"))), expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("SW0139 lambda batch ratio does not reproduce")
        families = row.get("gradient_norms_by_family", {})
        for objective in ("old", "contrastive"):
            values = families.get(objective, {})
            if set(values) != {"encoder", "graph"} or any(
                    not math.isfinite(float(values[name])) or float(values[name]) <= 0
                    for name in ("encoder", "graph")):
                raise ValueError("SW0139 lambda must bind finite old/C credit to encoder and graph")
        ratios.append(expected)
    expected_median = float(np.median(ratios))
    if (not math.isclose(float(record.get("median_ratio", float("nan"))), expected_median,
                         rel_tol=1e-10, abs_tol=1e-12)
            or not math.isclose(float(record.get("lambda", float("nan"))), 0.25 * expected_median,
                                rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError("SW0139 fixed lambda arithmetic does not reproduce")
    digest = record.get("self_sha256")
    payload = dict(record)
    payload.pop("self_sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != digest:
        raise ValueError("SW0139 lambda artifact self hash mismatch")
    return record


def _optimizer_parameters(foundation, arm, context):
    trainable = trainable_backbone(foundation, arm, context)
    groups, parameters = _parameter_groups(foundation, arm, context)
    if {id(p) for p in trainable} != {id(p) for p in parameters}:
        raise AssertionError("SW0139 trainable parameters and optimizer union differ")
    return groups, parameters


def _train_micro(foundation, arm, context, rgb, image_ids, *, lambda_value, nonce):
    old, contrast, _, _, metadata = _logical_losses(
        foundation, arm, context, rgb, image_ids, nonce=nonce)
    loss = old if arm != "cross_view" else old + lambda_value * contrast
    return loss, {"old": old, "contrast": contrast, "view_c": None if metadata is None else metadata["contrastive_view_c"],
                  "view_offset": None if metadata is None else metadata["contrastive_view_offset"]}


def _capture_core_state(foundation):
    return {k: v.detach().clone() for k, v in foundation.wrapped.state_dict().items()}


def _source_contract(foundation, arm, seed):
    return {"seed": seed, "arm": arm,
            "source_core_sha256": foundation.provenance["source_core_sha256"],
            "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
            "source_training_ids_sha256": foundation.provenance["source_training_ids_sha256"],
            "training_ids": foundation.image_ids,
            "training_ids_sha256": _logical_id_sha(foundation.image_ids),
            "pool_indices_sha256": _logical_id_sha(foundation.pool_indices),
            "implementation_fingerprint": implementation_fingerprint(),
            "asset_hashes": foundation.provenance["rgb_asset_validation"]}


def _initial_manifest(foundation, arm, seed, lambda_record):
    exposures = TRAIN_IMAGES * (2 if arm == "cross_view" else 1)
    return {"experiment": "SW0139_native32_representation", "status": "training_complete",
            "seed": seed, "arm": arm, "updates": UPDATES, "passes": 1,
            "batch_size": LOGICAL_BATCH, "microbatch_size": MICROBATCH,
            "unique_training_images": TRAIN_IMAGES, "total_image_exposures": exposures,
            "views_per_image": 2 if arm == "cross_view" else 1,
            "shuffle_seed": 117 + seed, "optimizer": "fresh_adam",
            "learning_rates": {"graph": CORE_LR, "encoder": ENCODER_LR,
                                **({"context_adapter": CONTEXT_LR} if arm == "context_residual" else {})},
            "clip_norm_unique_union": CLIP,
            "lambda": 0.0 if arm != "cross_view" else float(lambda_record["lambda"]),
            "lambda_sha256": None if arm != "cross_view" else sha256_file(LAMBDA_PATH),
            "parameter_freeze": "native oscillator/dendrite/membrane/integration frozen; learned native32 graph and source encoder trainable",
            "ground_truth_used_for_training": False,
            **_source_contract(foundation, arm, seed)}


def _validate_seed_ids(foundation, seed):
    if seed != SEED:
        raise ValueError("SW0139 currently registers only the common representative pilot seed1")
    if len(foundation.image_ids) != TRAIN_IMAGES or len(set(foundation.image_ids)) != TRAIN_IMAGES:
        raise AssertionError("source TRAIN order must contain the registered 4096 unique IDs")
    registered = sw130.source_contract(seed)
    if (list(map(int, foundation.image_ids)) != list(map(int, registered[4]))
            or not np.array_equal(np.asarray(foundation.pool_indices),
                                  np.asarray(registered[3]))):
        raise AssertionError("SW0139 IDs/rows differ from the registered source-order contract")


def calibrate_lambda(output=LAMBDA_PATH, device="cpu"):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0139 lambda artifact: {output}")
    record = _lambda_calibration(device)
    record["path"] = str(output)
    # Compute a content hash over the immutable calibration payload without self-reference.
    content = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    record["self_sha256"] = hashlib.sha256(content).hexdigest()
    _write_once(output, record)
    return record


def disposable_update(seed, arm, device="cpu", *, lambda_path=LAMBDA_PATH):
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    lambda_record = load_lambda_record(lambda_path) if arm == "cross_view" else None
    foundation = source_bundle(seed, device)
    _validate_seed_ids(foundation, seed)
    context = ContextResidualEncoder(seed=139 + seed).to(device) if arm == "context_residual" else None
    context_initial_parity = None
    if context is not None:
        probe = read_rgb_rows(foundation.pool_indices[:1], device)
        standardized = _feature_map_from_rgb(foundation, probe)
        contextual = context(standardized)
        context_initial_parity = bool(torch.equal(contextual, standardized.clamp(-3.0, 3.0)))
        if not context_initial_parity:
            raise AssertionError("context adapter output must exactly preserve source features at initialization")
    groups, params = _optimizer_parameters(foundation, arm, context)
    optimizer = torch.optim.Adam(groups)
    rgb_cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    old_steps = []
    source_before = _capture_core_state(foundation)
    probe_device = torch.device(device)
    if probe_device.type == "cuda":
        torch.cuda.set_device(probe_device)
        torch.cuda.init()
        torch.cuda.synchronize(probe_device)
        torch.cuda.reset_peak_memory_stats(probe_device)
    started = time.perf_counter()
    ids = foundation.image_ids[:MICROBATCH]
    rows = foundation.pool_indices[:MICROBATCH]
    separate_credit = None
    for update in range(2 if arm == "context_residual" else 1):
        update_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logical_rows, logical_ids = [], []
        total_value = 0.0
        for micro in range(MICROS):
            start = micro * MICROBATCH
            micro_rows = foundation.pool_indices[start:start + MICROBATCH]
            micro_ids = foundation.image_ids[start:start + MICROBATCH]
            rgb = read_rgb_rows(micro_rows, device, rgb_cache)
            loss, detail = _train_micro(foundation, arm, context, rgb, micro_ids,
                                        lambda_value=0.0 if lambda_record is None else lambda_record["lambda"],
                                        nonce=139000 + update * 100 + micro)
            if arm == "cross_view" and micro == 0:
                old_grads = torch.autograd.grad(detail["old"], params, retain_graph=True,
                                                allow_unused=True)
                c_grads = torch.autograd.grad(detail["contrast"], params, retain_graph=True,
                                              allow_unused=True)
                named = [(f"graph.{n}", p) for n, p in foundation.wrapped.core.graph_generator.named_parameters()
                         if p.requires_grad]
                named += [(f"encoder.{n}", p) for n, p in foundation.encoder.named_parameters()]
                separate_credit = {"old": _family_norms(named, old_grads),
                                   "contrastive": _family_norms(named, c_grads)}
            if not torch.isfinite(loss):
                raise FloatingPointError("SW0139 disposable loss is nonfinite")
            (loss / MICROS).backward()
            total_value += float(loss.detach()) / MICROS
            logical_rows.extend(int(x) for x in micro_rows)
            logical_ids.extend(int(x) for x in micro_ids)
        grads = [p.grad for p in params]
        if any(g is not None and not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError("SW0139 disposable gradient is nonfinite")
        grad_norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads if g is not None))
        if grad_norm <= 0 or not math.isfinite(grad_norm):
            raise FloatingPointError("SW0139 disposable optimizer has no finite parameter credit")
        def family_norm(named_parameters):
            return math.sqrt(sum(float(p.grad.detach().double().square().sum())
                                 for _, p in named_parameters
                                 if p.grad is not None))
        family_gradients = {
            "graph": family_norm(list(foundation.wrapped.core.graph_generator.named_parameters())),
            "encoder": family_norm(list(foundation.encoder.named_parameters())),
        }
        torch.nn.utils.clip_grad_norm_(params, CLIP)
        before = [p.detach().clone() for p in params]
        optimizer.step()
        if probe_device.type == "cuda":
            torch.cuda.synchronize(probe_device)
        update_seconds = time.perf_counter() - update_started
        changed = sum(not torch.equal(old, new) for old, new in zip(before, params))
        old_steps.append({"update": update + 1, "loss": total_value,
                          "gradient_norm": grad_norm, "changed_parameters": changed,
                          "gradient_norm_by_family": family_gradients,
                          "elapsed_seconds": update_seconds,
                          "training_ids": logical_ids,
                          "training_ids_sha256": _logical_id_sha(logical_ids)})
        if changed == 0:
            raise AssertionError("SW0139 disposable Adam step changed no trainable parameter")
    if arm == "context_residual":
        hidden_grad = [p.grad for branch in context.branches for p in branch.parameters()]
        if not any(g is not None and torch.isfinite(g).all() and bool((g != 0).any()) for g in hidden_grad):
            raise AssertionError("context hidden branches received no credit after the registered second step")
    source_after = _capture_core_state(foundation)
    changed_source = [k for k in source_before if not torch.equal(source_before[k], source_after[k])]
    # Only graph parameters are allowed to change in the core.
    disallowed = [name for name in changed_source if not name.startswith("core.graph_generator.")]
    if disallowed:
        raise AssertionError(f"SW0139 disposable update changed frozen source parameters: {disallowed[:3]}")
    if probe_device.type == "cuda":
        torch.cuda.synchronize(probe_device)
        max_allocated = int(torch.cuda.max_memory_allocated(probe_device))
        max_reserved = int(torch.cuda.max_memory_reserved(probe_device))
    else:
        max_allocated = max_reserved = None
    elapsed = time.perf_counter() - started
    return {"experiment": "SW0139_native32_representation", "status": "disposable_update_complete",
            "seed": seed, "arm": arm, "steps": old_steps, "lambda": 0.0 if lambda_record is None else lambda_record["lambda"],
            "lambda_sha256": None if lambda_record is None else sha256_file(lambda_path),
            "separate_old_contrastive_gradient_families": separate_credit,
            "changed_source_state_keys": changed_source, "optimizer_updates": len(old_steps),
            "resource_measurement": {"device": str(probe_device), "elapsed_seconds": elapsed,
                                     "max_memory_allocated_bytes": max_allocated,
                                     "max_memory_reserved_bytes": max_reserved},
            "context_exact_initial_feature_parity": context_initial_parity,
            "source_unchanged_except_trainable_graph": True, "ground_truth_used": False,
            "implementation_fingerprint": implementation_fingerprint()}


def train(seed, arm, device="cpu", *, preflight_path=None, lambda_path=LAMBDA_PATH, output_root=OUT):
    """Execute the registered 256-update, one-pass seed1 arm; create-only artifacts."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    preflight_path = Path(preflight_path or ARCHIVE / f"preflight_{arm}_seed{seed}.json")
    if not preflight_path.is_file():
        raise FileNotFoundError(f"required passed SW0139 disposable preflight is absent: {preflight_path}")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (preflight.get("status") != "disposable_update_complete"
            or preflight.get("seed") != seed or preflight.get("arm") != arm
            or preflight.get("implementation_fingerprint") != implementation_fingerprint()
            or preflight.get("ground_truth_used") is not False):
        raise ValueError("SW0139 arm preflight does not bind the current seed/arm implementation")
    expected_disposable_updates = 2 if arm == "context_residual" else 1
    if (preflight.get("optimizer_updates") != expected_disposable_updates
            or preflight.get("source_unchanged_except_trainable_graph") is not True
            or len(preflight.get("steps", [])) != expected_disposable_updates
            or any(int(row.get("changed_parameters", 0)) <= 0
                   or not math.isfinite(float(row.get("gradient_norm", float("nan"))))
                   for row in preflight["steps"])):
        raise ValueError("SW0139 disposable preflight lacks valid optimizer/source evidence")
    if arm == "cross_view":
        family_rows = preflight.get("separate_old_contrastive_gradient_families")
        if not isinstance(family_rows, dict) or any(
                set(family_rows.get(objective, {})) != {"encoder", "graph"}
                or any(not math.isfinite(float(family_rows[objective][name]))
                       or float(family_rows[objective][name]) <= 0
                       for name in ("encoder", "graph"))
                for objective in ("old", "contrastive")):
            raise ValueError("SW0139 cross-view preflight lacks separate encoder/graph gradient credit")
    lambda_record = load_lambda_record(lambda_path) if arm == "cross_view" else None
    if lambda_record and preflight.get("lambda_sha256") != sha256_file(lambda_path):
        raise ValueError("SW0139 cross-view preflight does not bind source0 calibration")
    folder = Path(output_root) / f"{arm}_seed{seed}"
    if folder.exists():
        raise FileExistsError(f"preserve existing SW0139 output directory: {folder}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir(parents=True, exist_ok=False)
    foundation = source_bundle(seed, device)
    _validate_seed_ids(foundation, seed)
    context = ContextResidualEncoder(seed=139 + seed).to(device) if arm == "context_residual" else None
    groups, params = _optimizer_parameters(foundation, arm, context)
    optimizer = torch.optim.Adam(groups)
    history, pass_rows = [], []
    order = np.arange(TRAIN_IMAGES, dtype=np.int64)
    ids = np.asarray(foundation.image_ids, dtype=np.int64)[order]
    rows = np.asarray(foundation.pool_indices, dtype=np.int64)[order]
    pass_rows.append({"pass": 0, "image_ids_sha256": _logical_id_sha(ids),
                      "first_ids": ids[:8].tolist(), "last_ids": ids[-8:].tolist()})
    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    fixed_lambda = 0.0 if lambda_record is None else float(lambda_record["lambda"])
    step = 0
    started = time.time()
    for batch_start in range(0, TRAIN_IMAGES, LOGICAL_BATCH):
        step += 1
        optimizer.zero_grad(set_to_none=True)
        old_values, contrast_values, micro_ids, aug_rows = [], [], [], []
        for micro in range(MICROS):
            start = batch_start + micro * MICROBATCH
            micro_rows = rows[start:start + MICROBATCH]
            ids_now = ids[start:start + MICROBATCH]
            rgb = read_rgb_rows(micro_rows, device, cache)
            loss, detail = _train_micro(foundation, arm, context, rgb, ids_now,
                                        lambda_value=fixed_lambda,
                                        nonce=139000 + seed * 100000 + step * 100 + micro)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"SW0139 nonfinite loss at update {step}, micro {micro}")
            (loss / MICROS).backward()
            old_values.append(float(detail["old"].detach()))
            contrast_values.append(None if detail["contrast"] is None else float(detail["contrast"].detach()))
            if detail["view_c"] is not None:
                aug_rows.append({"c": detail["view_c"].tolist(), "b": detail["view_offset"].tolist()})
            micro_ids.extend(int(v) for v in ids_now)
        if len(micro_ids) != LOGICAL_BATCH:
            raise AssertionError("SW0139 logical B16 accumulation lost image rows")
        grad_values = [p.grad for p in params]
        if any(g is not None and not torch.isfinite(g).all() for g in grad_values):
            raise FloatingPointError(f"SW0139 nonfinite gradients at update {step}")
        norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in grad_values if g is not None))
        if not math.isfinite(norm) or norm <= 0:
            raise FloatingPointError(f"SW0139 empty gradient union at update {step}")
        torch.nn.utils.clip_grad_norm_(params, CLIP)
        optimizer.step()
        history.append({"update": step, "batch_start": batch_start,
                        "training_ids": micro_ids, "training_ids_sha256": _logical_id_sha(micro_ids),
                        "old_loss_micro_mean": old_values,
                        "contrastive_loss_micro_mean": contrast_values,
                        "gradient_norm_before_clip": norm,
                        "augmentation_parameters": aug_rows})
    checkpoint_path, optim_path, hist_path = folder / "checkpoint.pt", folder / "optimizer.pt", folder / "history.json"
    payload = {"experiment": "SW0139_native32_representation", "seed": seed, "arm": arm,
               "source_core_sha256": foundation.provenance["source_core_sha256"],
               "wrapped_state_dict": foundation.wrapped.state_dict(),
               "encoder_state_dict": foundation.encoder.state_dict(),
               "context_state_dict": None if context is None else context.state_dict(),
               "lambda": fixed_lambda,
               "training_ids_sha256": _logical_id_sha(ids),
               "implementation_fingerprint": implementation_fingerprint()}
    torch.save(payload, checkpoint_path)
    torch.save(optimizer.state_dict(), optim_path)
    history_record = {"records": history, "pass_orders": pass_rows, "updates": step}
    hist_path.write_text(json.dumps(history_record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {"experiment": "SW0139_native32_representation", "status": "training_complete",
                "checkpoint_sha256": sha256_file(checkpoint_path), "optimizer_sha256": sha256_file(optim_path),
                "history_sha256": sha256_file(hist_path), "updates": step,
                "passes": 1, "elapsed_seconds": time.time() - started,
                "preflight_sha256": sha256_file(preflight_path),
                "warmup": False, "lambda_sha256": None if lambda_record is None else sha256_file(lambda_path),
                **_initial_manifest(foundation, arm, seed, lambda_record)}
    manifest_path = folder / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    marker = {"status": "training_complete", "manifest_sha256": sha256_file(manifest_path),
              "checkpoint_sha256": sha256_file(checkpoint_path), "updates": step,
              "ground_truth_used_for_training": False}
    (folder / "TRAINING_COMPLETED.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lambda", "preflight", "train"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lambda-path", type=Path, default=LAMBDA_PATH)
    parser.add_argument("--preflight-path", type=Path)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.stage == "lambda":
        if args.seed != 0:
            raise ValueError("SW0139 lambda calibration is source seed0 only")
        result = calibrate_lambda(args.output or args.lambda_path, args.device)
    elif args.stage == "preflight":
        if args.seed != 1 or args.arm is None or args.output is None:
            raise ValueError("preflight requires --seed 1, --arm, and create-once --output")
        result = disposable_update(args.seed, args.arm, args.device, lambda_path=args.lambda_path)
        result["path"] = str(args.output)
        _write_once(args.output, result)
    else:
        if args.arm is None:
            raise ValueError("training requires an arm")
        result = train(args.seed, args.arm, args.device,
                       preflight_path=args.preflight_path,
                       lambda_path=args.lambda_path, output_root=args.output_root)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
