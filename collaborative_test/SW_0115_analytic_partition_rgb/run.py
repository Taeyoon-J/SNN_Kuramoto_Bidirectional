"""Seed-0 paired pilot for SW0115 analytic partition RGB credit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run.py"
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0115_analytic_partition_rgb.loss import (
    batch_reconstruction_loss,
    production_partition,
    rgb_patch_means,
)
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from snn_kuramoto_bidirectional.training.train_s2net_core import _forward_with_plv


SEEDS = (0,)
ARMS = ("control", "analytic_candidate")
BATCH = 16
UPDATES = 256
TRAIN_STEPS = 64
TRAIN_SETTLE = 32
LR = 3e-5
RGB_CACHE = ROOT / "data/SW_0106_spike_partition_rgb/train_rgb_uint8.npy"
RGB_CACHE_MANIFEST = ROOT / "data/SW_0106_spike_partition_rgb/train_rgb_uint8.npy.complete.json"
OUT = ROOT / "trained_models/SW0115_analytic_partition_rgb"
ARCHIVE = HERE / "results_archive"
LAMBDA_SEED0 = ARCHIVE / "lambda_seed0.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def implementation_fingerprint():
    files = [HERE / "run.py", HERE / "coordinator.py", HERE / "protocol.json",
             HERE / "loss.py", ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
             ROOT / "collaborative_test/SW_0094_aligned_joint_pilot/run.py",
             ROOT / "collaborative_test/SW_0106_spike_partition_rgb/build_rgb_cache.py",
             ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
             ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
             ROOT / "snn_kuramoto_bidirectional/evaluation.py",
             ROOT / "snn_kuramoto_bidirectional/loss_function.py",
             ROOT / "snn_kuramoto_bidirectional/training/train_s2net_core.py"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files}


def validate_source_manifest(seed, manifest, ids):
    if seed not in SEEDS or len(ids) != 4096 or len(np.unique(ids)) != 4096:
        raise AssertionError("SW0115 requires the matched SW0097 4096-image order")
    if (manifest.get("status") != "complete" or manifest.get("unique_images_seen") != 4096
            or manifest.get("steps") != 256 or manifest.get("batch") != 16
            or manifest.get("seed") != 117 + seed
            or manifest.get("train_steps") != 64 or manifest.get("train_settle") != 32
            or manifest.get("training_ids") != ids.tolist()
            or manifest.get("ground_truth_used_for_training") is not False):
        raise AssertionError("matched SW0097 source contract differs from SW0115")


def verify_registered_source(seed):
    checkpoint, manifest_path, manifest = base.source_paths(seed)
    ids, rows = base.train_indices(seed)
    validate_source_manifest(seed, manifest, ids)
    return checkpoint, manifest_path, manifest, ids, rows


def validate_npy_storage(path, array, expected_shape, expected_dtype):
    """Check an mmap array including its NPY header offset, not just payload bytes."""
    if tuple(array.shape) != tuple(expected_shape) or array.dtype != np.dtype(expected_dtype):
        raise AssertionError("RGB NPY shape/dtype mismatch")
    offset = getattr(array, "offset", None)
    if offset is None or path.stat().st_size != int(offset) + int(array.nbytes):
        raise AssertionError("RGB NPY physical size does not equal its header plus array payload")


def validate_preflight_binding(pf, seed, arm, source, source_manifest, ids,
                               gamma_sha, gamma_manifest_sha, rgb_sha, rgb_manifest_sha):
    ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    expected = {
        "seed": seed, "arm": arm,
        "source_core_sha256": sha(source),
        "source_manifest_sha256": sha(source_manifest),
        "gamma_train_sha256": gamma_sha,
        "gamma_train_manifest_sha256": gamma_manifest_sha,
        "rgb_cache_sha256": rgb_sha,
        "rgb_cache_manifest_sha256": rgb_manifest_sha,
        "training_ids_sha256": ids_sha,
        "matched_shuffle_seed": 117 + seed,
        "implementation_fingerprint": implementation_fingerprint(),
    }
    if any(pf.get(key) != value for key, value in expected.items()):
        raise AssertionError("source/cache/order/implementation changed since successful preflight")


def audit_cache_block_ledger(cache, blocks, expected_ids, file_digest=None):
    """Verify the SW0106 emitted pool ledger against mapped cache bytes."""
    expected_ids = np.asarray(expected_ids, dtype=np.int64)
    if cache.ndim < 1 or cache.shape[0] != len(expected_ids):
        raise AssertionError("RGB cache rows do not match the registered ID ledger")
    payload_digest = hashlib.sha256()
    cursor = 0
    for row in blocks:
        count = int(row.get("count", -1))
        pool_start = int(row.get("pool_start", -1))
        source_start = int(row.get("source_id_start", -1))
        source_chunk = row.get("source_chunk")
        if (count <= 0 or pool_start != cursor or cursor + count > len(expected_ids)
                or not np.array_equal(expected_ids[cursor:cursor + count],
                                      np.arange(source_start, source_start + count, dtype=np.int64))):
            raise AssertionError("RGB cache block ledger has a gap, overlap, or wrong source-ID mapping")
        if (not isinstance(source_chunk, list) or len(source_chunk) != 2
                or not (int(source_chunk[0]) <= source_start
                        and source_start + count <= int(source_chunk[1]))):
            raise AssertionError("RGB block source-chunk provenance is invalid")
        payload = memoryview(cache[cursor:cursor + count]).cast("B")
        block_digest = hashlib.sha256(payload).hexdigest()
        if block_digest != row.get("sha256"):
            raise AssertionError(f"RGB cache block bytes differ from recorded digest at row {cursor}")
        payload_digest.update(payload)
        if file_digest is not None:
            file_digest.update(payload)
        cursor += count
    if cursor != len(expected_ids):
        raise AssertionError("RGB cache block ledger does not cover all registered rows")
    return payload_digest.hexdigest()


def audit_cache_blocks(path, cache, blocks, expected_ids):
    with Path(path).open("rb") as stream:
        header = stream.read(int(cache.offset))
    if len(header) != int(cache.offset):
        raise AssertionError("truncated RGB NPY header")
    file_digest = hashlib.sha256()
    file_digest.update(header)
    audit_cache_block_ledger(cache, blocks, expected_ids, file_digest=file_digest)
    return file_digest.hexdigest()


def validate_rgb_cache_metadata(meta, source, source_size, source_mtime_ns):
    expected_ids = np.concatenate((np.arange(1000, dtype="<i8"),
                                   np.arange(1640, 70640, dtype="<i8")))
    expected_hash = hashlib.sha256(expected_ids.tobytes()).hexdigest()
    if (meta.get("status") != "complete" or meta.get("kind") != "train"
            or meta.get("cache_shape") != [70000, 128, 128, 3]
            or meta.get("cache_dtype") != "uint8"
            or meta.get("ids_mapping_sha256") != expected_hash
            or meta.get("source_path") != str(source)
            or meta.get("source_size_bytes") != source_size
            or meta.get("source_mtime_ns") != source_mtime_ns
            or meta.get("source_image_shape") != [100000, 128, 128, 3]
            or meta.get("source_image_dtype") != "uint8"):
        raise AssertionError("native RGB cache manifest/source identity mismatch")
    blocks = meta.get("blocks")
    if (not isinstance(blocks, list)
            or any(int(row.get("count", -1)) <= 0 for row in blocks)
            or sum(int(row["count"]) for row in blocks) != 70000):
        raise AssertionError("RGB cache block completion record is incomplete")


def validate_rgb_cache():
    if not RGB_CACHE.is_file() or not RGB_CACHE_MANIFEST.is_file():
        raise FileNotFoundError("the completed SW0106 native RGB training cache is required")
    meta = json.loads(RGB_CACHE_MANIFEST.read_text())
    source = base.DATASET
    if not source.is_file():
        raise FileNotFoundError(source)
    stat = source.stat()
    validate_rgb_cache_metadata(meta, source, stat.st_size, stat.st_mtime_ns)
    cache = np.load(RGB_CACHE, mmap_mode="r")
    validate_npy_storage(RGB_CACHE, cache, (70000, 128, 128, 3), np.uint8)
    expected_ids = np.concatenate((np.arange(1000, dtype=np.int64),
                                   np.arange(1640, 70640, dtype=np.int64)))
    cache_sha = audit_cache_blocks(RGB_CACHE, cache, meta["blocks"], expected_ids)
    return cache, meta, cache_sha


def load_core(seed, device):
    checkpoint, manifest_path, manifest, ids, rows = verify_registered_source(seed)
    core = base.make_core(device, TRAIN_STEPS)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    core.load_state_dict(state, strict=True)
    core.graph_generator.requires_grad_(False)
    core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
    if core.graph_generator.uses_feedback or core.kuramoto.spike_pulse_gain is not None:
        raise AssertionError("source graph feedback or spike pulse violates registered source contract")
    return core, checkpoint, manifest_path, manifest, ids, rows


def read_rgb(cache, rows, device):
    images = np.asarray(cache[np.asarray(rows, dtype=np.int64)]).copy()
    if images.shape != (len(rows), 128, 128, 3) or images.dtype != np.uint8:
        raise AssertionError("RGB batch does not match native uint8 contract")
    # One and only one conversion from native uint8 [0,255] to float [0,1].
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    tensor = tensor / 255.0
    return rgb_patch_means(tensor)


def eligible_parameters(core):
    params = [p for name, p in core.named_parameters()
              if p.requires_grad and not name.startswith("graph_generator.")]
    if not params:
        raise AssertionError("no eligible trainable non-graph core parameters")
    return params


def grad_norm(grads):
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            raise FloatingPointError("nonfinite objective gradient")
        total += float(grad.detach().double().square().sum())
    return math.sqrt(total)


def mean_scramble_excess(real_per_image, scrambled_per_image):
    real = np.asarray(real_per_image, dtype=np.float64)
    scrambled = np.asarray(scrambled_per_image, dtype=np.float64)
    if real.ndim != 1 or scrambled.shape != real.shape or real.size == 0:
        raise ValueError("real and scrambled per-image losses must be matching nonempty vectors")
    if not np.isfinite(real).all() or not np.isfinite(scrambled).all():
        raise FloatingPointError("nonfinite per-image scramble loss")
    return float(np.mean(scrambled - real))


def forward_parts(core, gamma, rgb, lossfn):
    _, spikes, core_out, plv, theta = _forward_with_plv(
        core, gamma, lossfn, TRAIN_SETTLE, "phase", "mean")
    components = core.last_component_spikes
    if tuple(components.shape[1:]) != (4, 256, TRAIN_STEPS):
        raise AssertionError(f"actual component spike shape mismatch: {tuple(components.shape)}")
    if spikes.shape != (gamma.shape[0], 256, TRAIN_STEPS):
        raise AssertionError("actual event trace shape mismatch")
    q = spike_synchrony_affinity(
        components.mean(dim=1), components=components, settle=TRAIN_SETTLE,
        affinity_mode="spike")
    if q.shape != (gamma.shape[0], 256, 256) or not torch.isfinite(q).all():
        raise AssertionError("actual four-component positive-Pearson Q is invalid")
    primary, _ = lossfn(plv=plv, theta=theta)
    positive, _ = lossfn(plv=q)
    old = primary + 5.0 * positive
    labels, hard = production_partition(spikes, components, settle=TRAIN_SETTLE)
    rgb_loss, per_image, _, details = batch_reconstruction_loss(q, hard, rgb)
    if not torch.isfinite(old) or not torch.isfinite(rgb_loss):
        raise FloatingPointError("nonfinite old or analytic partition RGB loss")
    return (old, primary, positive, rgb_loss, q, labels, hard, per_image, details,
            spikes, core_out, theta, components)


def preflight(seed, arm, output, device="cuda"):
    if arm not in ARMS or seed != 0:
        raise ValueError("initial SW0115 pilot is seed0 only")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing preflight; refusing overwrite: {output}")
    if arm == "analytic_candidate" and LAMBDA_SEED0.exists():
        raise FileExistsError(f"preserve existing coefficient artifact; refusing overwrite: {LAMBDA_SEED0}")
    fingerprint = implementation_fingerprint()
    core, source, source_manifest, source_record, ids, rows = load_core(seed, device)
    gamma, gamma_manifest = base.validate_gamma_cache(base.GAMMA_TRAIN, base.GAMMA_TRAIN_MANIFEST)
    rgb_cache, rgb_manifest, rgb_cache_sha = validate_rgb_cache()
    params = eligible_parameters(core)
    initial = {k: v.detach().clone() for k, v in core.state_dict().items()}
    criterion = base.criterion()
    reference_core, *_ = load_core(seed, device)
    reference_core.train()
    reference_core.graph_generator.eval()
    ratios, batch_records, real_losses, scrambled_losses = [], [], [], []
    core.train()
    core.graph_generator.eval()
    fixed_perm = torch.as_tensor(np.random.default_rng(11501).permutation(256),
                                 device=device, dtype=torch.long)
    for batch_index in range(4):
        start = batch_index * BATCH
        batch_rows = rows[start:start + BATCH]
        ix = torch.as_tensor(batch_rows, device="cpu", dtype=torch.long)
        g = gamma[ix].to(device)
        rgb = read_rgb(rgb_cache, batch_rows, device)
        (old, primary, positive, rloss, q, labels, hard, per_image, _,
         spikes, core_out, theta, components) = forward_parts(
            core, g, rgb, criterion)
        with torch.no_grad():
            ref_total, ref_primary, ref_positive = base.loss_parts(reference_core, g, criterion)
            _, ref_spikes, ref_core_out, _, ref_theta = _forward_with_plv(
                reference_core, g, criterion, TRAIN_SETTLE, "phase", "mean")
            ref_components = reference_core.last_component_spikes.detach().clone()
        if not (torch.equal(old.detach(), ref_total.detach())
                and torch.equal(primary.detach(), ref_primary.detach())
                and torch.equal(positive.detach(), ref_positive.detach())
                and torch.equal(spikes.detach(), ref_spikes)
                and torch.equal(core_out.detach(), ref_core_out)
                and torch.equal(theta.detach(), ref_theta)
                and torch.equal(components.detach(), ref_components)):
            raise AssertionError("SW0115 source rollout/old loss differs from independent SW0110 baseline")
        old_grads = torch.autograd.grad(old, params, retain_graph=True, allow_unused=True)
        rgb_grads = torch.autograd.grad(rloss, params, retain_graph=True, allow_unused=True)
        q_grads = torch.autograd.grad(rloss, q, retain_graph=True, allow_unused=True)[0]
        old_norm, rgb_norm = grad_norm(old_grads), grad_norm(rgb_grads)
        q_norm = 0.0 if q_grads is None else grad_norm([q_grads])
        if not all(math.isfinite(v) and v > 0 for v in (old_norm, rgb_norm, q_norm)):
            raise AssertionError(f"nonfinite/inert preflight gradient on batch {batch_index}")

        scrambled_h = [h.index_select(0, fixed_perm) for h in hard]
        scrambled, scrambled_per_image, _, _ = batch_reconstruction_loss(q, scrambled_h, rgb)
        if not torch.isfinite(scrambled):
            raise FloatingPointError("scrambled partition loss is nonfinite")
        ratios.append(old_norm / rgb_norm)
        real_losses.extend(float(v.detach()) for v in per_image)
        scrambled_losses.extend(float(v.detach()) for v in scrambled_per_image)
        batch_records.append({
            "batch": batch_index,
            "global_ids": ids[start:start + BATCH].tolist(),
            "old_loss": float(old.detach()), "primary": float(primary.detach()),
            "positive_actual_spike_product": float(positive.detach()),
            "rgb_loss": float(rloss.detach()),
            "old_eligible_core_gradient_norm": old_norm,
            "rgb_eligible_core_gradient_norm": rgb_norm,
            "rgb_to_q_gradient_norm": q_norm,
            "group_counts": [int(h.shape[1]) for h in hard],
        })
        core.zero_grad(set_to_none=True)
    scramble_excess = mean_scramble_excess(real_losses, scrambled_losses)
    if not math.isfinite(scramble_excess) or scramble_excess <= 0:
        raise AssertionError("fixed count-preserving H scramble did not increase RGB loss")
    changed = [key for key, value in initial.items()
               if not torch.equal(value, core.state_dict()[key])]
    if changed:
        raise AssertionError(f"read-only preflight modified source parameters: {changed}")
    del reference_core
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    lam = 0.25 * float(np.median(np.asarray(ratios, dtype=np.float64)))
    if not math.isfinite(lam) or lam <= 0:
        raise AssertionError("invalid seed0 shared RGB coefficient")

    # Validate the actual candidate update on a throwaway source copy.
    update_core, *_ = load_core(seed, device)
    update_params = eligible_parameters(update_core)
    opt = torch.optim.Adam(update_params, lr=LR)
    g = gamma[torch.as_tensor(rows[:BATCH], dtype=torch.long)].to(device)
    rgb = read_rgb(rgb_cache, rows[:BATCH], device)
    before = {k: v.detach().clone() for k, v in update_core.state_dict().items()}
    old, _, _, rloss, *_ = forward_parts(update_core, g, rgb, criterion)
    objective = old + lam * rloss if arm == "analytic_candidate" else old
    opt.zero_grad(set_to_none=True)
    objective.backward()
    gradnorm = torch.nn.utils.clip_grad_norm_(update_params, 1.0)
    if not torch.isfinite(objective) or not torch.isfinite(gradnorm) or float(gradnorm) <= 0:
        raise FloatingPointError("throwaway B16 Adam update has invalid objective/gradient")
    opt.step()
    after = update_core.state_dict()
    if not any(not torch.equal(before[name], after[name])
               for name, p in update_core.named_parameters()
               if p.requires_grad and not name.startswith("graph_generator.")):
        raise AssertionError("throwaway update changed no eligible core parameter")
    if any(not torch.equal(before[k], after[k]) for k in before if k.startswith("graph_generator.")):
        raise AssertionError("throwaway update changed frozen graph")

    record = {
        "status": "passed", "experiment": "SW0115", "seed": seed, "arm": arm,
        "source_core_sha256": sha(source), "source_manifest_sha256": sha(source_manifest),
        "implementation_fingerprint": fingerprint,
        "gamma_train_sha256": sha(base.GAMMA_TRAIN),
        "gamma_train_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
        "rgb_cache_sha256": rgb_cache_sha, "rgb_cache_manifest_sha256": sha(RGB_CACHE_MANIFEST),
        "rgb_cache_source_size_bytes": rgb_manifest.get("source_size_bytes"),
        "rgb_cache_source_mtime_ns": rgb_manifest.get("source_mtime_ns"),
        "training_ids": ids.tolist(), "training_ids_sha256": hashlib.sha256(
            np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "matched_shuffle_seed": 117 + seed, "batch_size": BATCH,
        "time_steps": TRAIN_STEPS, "settle": TRAIN_SETTLE, "ground_truth_used": False,
        "batches": batch_records, "fixed_row_permutation": fixed_perm.cpu().tolist(),
        "real_rgb_loss_mean": float(np.mean(real_losses)),
        "scrambled_rgb_loss_mean": float(np.mean(scrambled_losses)),
        "scrambled_minus_real_rgb_loss": scramble_excess,
        "lambda_rgb": lam, "lambda_rule": "0.25 * median(old eligible-core norm / RGB eligible-core norm) over first four seed0 TRAIN batches",
        "throwaway_b16_adam_update": True, "throwaway_preclip_gradient_norm": float(gradnorm),
        "graph_frozen": True, "actual_q": "4-component positive-Pearson product from last_component_spikes, settle=32",
        "classifier": "production .50/min2/largest_component; variable-K H includes all omitted patches in background",
    }
    write(output, record)
    if arm == "analytic_candidate":
        write(LAMBDA_SEED0, {"status": "passed", "seed": 0, "lambda_rgb": lam,
                              "preflight_sha256": sha(output),
                              "implementation_fingerprint": fingerprint})
    return record


def train(seed, arm, output, device="cuda", steps=UPDATES):
    if seed != 0 or arm not in ARMS or steps != UPDATES:
        raise ValueError("only registered seed0 256-update pilot is currently enabled")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing training output; refusing overwrite: {output}")
    pf_path = ARCHIVE / f"preflight_seed{seed}_{arm}.json"
    if not pf_path.is_file():
        raise FileNotFoundError(f"successful current preflight required: {pf_path}")
    pf = json.loads(pf_path.read_text())
    if (pf.get("status") != "passed" or pf.get("seed") != seed or pf.get("arm") != arm
            or pf.get("implementation_fingerprint") != implementation_fingerprint()):
        raise AssertionError("preflight missing or implementation changed since preflight")
    lam = 0.0
    lambda_sha = None
    if arm == "analytic_candidate":
        if not LAMBDA_SEED0.is_file():
            raise FileNotFoundError("shared seed0 RGB coefficient artifact missing")
        data = json.loads(LAMBDA_SEED0.read_text())
        if (data.get("status") != "passed" or data.get("seed") != seed
                or data.get("preflight_sha256") != sha(pf_path)
                or data.get("implementation_fingerprint") != pf.get("implementation_fingerprint")):
            raise AssertionError("seed0 shared RGB coefficient provenance mismatch")
        lam = float(data["lambda_rgb"])
        if not math.isfinite(lam) or lam <= 0 or lam != float(pf.get("lambda_rgb", float("nan"))):
            raise AssertionError("candidate lambda does not match its successful preflight")
        lambda_sha = sha(LAMBDA_SEED0)

    core, source, source_manifest, source_record, ids, rows = load_core(seed, device)
    gamma, _ = base.validate_gamma_cache(base.GAMMA_TRAIN, base.GAMMA_TRAIN_MANIFEST)
    rgb_cache, rgb_manifest, rgb_cache_sha = validate_rgb_cache()
    validate_preflight_binding(pf, seed, arm, source, source_manifest, ids,
                               sha(base.GAMMA_TRAIN), sha(base.GAMMA_TRAIN_MANIFEST),
                               rgb_cache_sha, sha(RGB_CACHE_MANIFEST))
    params = eligible_parameters(core)
    source_graph = {k: v.detach().clone() for k, v in core.graph_generator.state_dict().items()}
    opt = torch.optim.Adam(params, lr=LR)
    criterion = base.criterion()
    torch.manual_seed(117 + seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(117 + seed)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "training", "experiment": "SW0115", "seed": seed, "arm": arm,
        "source_core": str(source), "source_core_sha256": sha(source),
        "source_manifest_sha256": sha(source_manifest), "preflight_sha256": sha(pf_path),
        "implementation_fingerprint": implementation_fingerprint(),
        "gamma_train_sha256": sha(base.GAMMA_TRAIN),
        "gamma_train_manifest_sha256": sha(base.GAMMA_TRAIN_MANIFEST),
        "rgb_cache_sha256": rgb_cache_sha,
        "rgb_cache_manifest_sha256": sha(RGB_CACHE_MANIFEST), "training_ids": ids.tolist(),
        "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "matched_shuffle_seed": 117 + seed, "updates": steps, "batch_size": BATCH,
        "time_steps": TRAIN_STEPS, "settle": TRAIN_SETTLE, "learning_rate": LR,
        "clip_norm": 1.0, "lambda_rgb": lam, "lambda_artifact_sha256": lambda_sha,
        "objective": "phase_primary + 5*positive_actual_spike_product + lambda_rgb*analytic_partition_rgb_if_candidate",
        "frozen": ["registered_encoder_precomputed_gamma", "all_legacy_graph_parameters"],
        "ground_truth_used_for_training": False,
    }
    write(output / "manifest.json", manifest)
    history = []
    core.train(); core.graph_generator.eval()
    for update in range(steps):
        start = update * BATCH
        batch_rows = rows[start:start + BATCH]
        ix = torch.as_tensor(batch_rows, dtype=torch.long)
        g = gamma[ix].to(device)
        rgb = read_rgb(rgb_cache, batch_rows, device)
        old, primary, positive, rloss, q, _, hard, per_image, *_ = forward_parts(
            core, g, rgb, criterion)
        total = old + lam * rloss if arm == "analytic_candidate" else old
        if not torch.isfinite(total):
            raise FloatingPointError(f"nonfinite SW0115 loss at update {update + 1}")
        opt.zero_grad(set_to_none=True)
        total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            raise FloatingPointError(f"nonfinite SW0115 gradient at update {update + 1}")
        gradnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gradnorm) or float(gradnorm) <= 0:
            raise FloatingPointError(f"invalid SW0115 gradient norm at update {update + 1}")
        opt.step()
        history.append({"update": update + 1, "total": float(total.detach()),
                        "old": float(old.detach()), "primary": float(primary.detach()),
                        "positive_actual_spike_product": float(positive.detach()),
                        "rgb": float(rloss.detach()), "rgb_lambda": lam,
                        "gradient_norm_preclip": float(gradnorm),
                        "mean_groups": float(np.mean([h.shape[1] for h in hard])),
                        "mean_rgb_per_image": float(per_image.detach().mean())})
        if update == 255:
            torch.save(core.state_dict(), output / "prefix_256_core.pt")
        if update == 0 or (update + 1) % 32 == 0:
            write(output / "progress.json", {"status": "training", "seed": seed,
                  "arm": arm, "update": update + 1, "total_updates": steps})
    after_graph = core.graph_generator.state_dict()
    if any(not torch.equal(value, after_graph[key]) for key, value in source_graph.items()):
        raise AssertionError("frozen graph changed during SW0115 training")
    if any(not torch.isfinite(p).all() for p in core.parameters()):
        raise FloatingPointError("nonfinite parameter after SW0115 training")
    torch.save(core.state_dict(), output / "core.pt")
    write(output / "history.json", history)
    manifest["core_sha256"] = sha(output / "core.pt")
    manifest["history_sha256"] = sha(output / "history.json")
    manifest.update(status="training_complete", completed=time.time())
    write(output / "manifest.json", manifest)
    (output / "TRAINING_COMPLETED").write_text("complete\n")


def evaluate(seed, arm, checkpoint, output, device="cuda"):
    if seed != 0 or arm not in ARMS:
        raise ValueError("only registered seed0 pilot evaluation is enabled")
    train_manifest = Path(checkpoint).parent / "manifest.json"
    if not train_manifest.is_file():
        raise FileNotFoundError(train_manifest)
    data = json.loads(train_manifest.read_text())
    if data.get("status") != "training_complete" or data.get("arm") != arm:
        raise AssertionError("evaluation requires a completed matching SW0115 training artifact")
    if data.get("core_sha256") != sha(checkpoint) or data.get("history_sha256") != sha(train_manifest.parent / "history.json"):
        raise AssertionError("evaluation checkpoint/history differs from completed training manifest")
    # SW0110's reviewed evaluator core stubs unused internal groups and computes
    # production external connected components from the actual component spikes.
    # Reuse its exact .06, B8/1024/512, .50, min-2 contract unchanged.
    base.evaluate(seed, "control", checkpoint, output, device=device)
    report_path = Path(output)
    report = json.loads(report_path.read_text())
    report["experiment"] = "SW0115"
    report["arm"] = arm
    report["seed"] = seed
    report["training_manifest_sha256"] = sha(train_manifest)
    report["checkpoint_sha256"] = sha(checkpoint)
    report["evaluation_runner_sha256"] = sha(HERE / "run.py")
    report["shared_evaluator_sha256"] = sha(ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py")
    report["ground_truth_used_for_prediction"] = False
    write(report_path, report)
    sidecar = {
        "experiment": "SW0115", "seed": seed, "arm": arm,
        "evaluation_file": report_path.name,
        "evaluation_sha256": sha(report_path),
        "training_manifest_sha256": sha(train_manifest),
        "checkpoint_sha256": sha(checkpoint),
        "evaluation_runner_sha256": sha(HERE / "run.py"),
        "shared_evaluator_sha256": sha(ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py"),
        "internal_grouping": "SW0110 eval core stubs unused terminal groups; external actual-component synchrony CC unchanged",
        "evaluation_contract": {
            "ids": [1320, 1639], "images": 320, "batch_size": 8,
            "time_steps": 1024, "settle": 512, "membrane_vth": 0.06,
            "readout_threshold": 0.50, "min_group_size": 2,
            "background": "largest_component", "ground_truth_used_for_prediction": False,
        },
    }
    write(report_path.parent / "evaluation_manifest.json", sidecar)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train", "evaluate"):
        sp = sub.add_parser(name)
        sp.add_argument("--seed", type=int, choices=SEEDS, required=True)
        sp.add_argument("--arm", choices=ARMS, required=True)
        sp.add_argument("--device", default="cuda")
        if name == "preflight":
            sp.add_argument("--output", type=Path, required=True)
        elif name == "train":
            sp.add_argument("--output", type=Path, required=True)
            sp.add_argument("--steps", type=int, default=UPDATES)
        else:
            sp.add_argument("--checkpoint", type=Path, required=True)
            sp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        record = preflight(args.seed, args.arm, args.output, args.device)
        print(json.dumps({"status": record["status"], "command": args.command,
                          "seed": args.seed, "arm": args.arm}), flush=True)
    elif args.command == "train":
        train(args.seed, args.arm, args.output, args.device, args.steps)
        print(json.dumps({"status": "complete", "command": args.command,
                          "seed": args.seed, "arm": args.arm}), flush=True)
    else:
        evaluate(args.seed, args.arm, args.checkpoint, args.output, args.device)
        print(json.dumps({"status": "complete", "command": args.command,
                          "seed": args.seed, "arm": args.arm}), flush=True)


if __name__ == "__main__":
    main()
