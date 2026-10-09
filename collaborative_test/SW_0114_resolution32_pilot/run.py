"""Local SW0114 model/data/evaluation adapter. No shared model files are patched."""
from __future__ import annotations

import hashlib
import copy
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "results_archive"
SOURCE_ROOT = ROOT / "trained_models/SW0097_graph_adaptation"
TRAIN_GAMMA = ROOT / "data/SW_0090_large_unique_scale/gamma_train_70000.pt"
VAL_GAMMA = ROOT / "data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"
DATASET = Path("/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5")
SEEDS = (0, 1, 2)
ARMS = ("resolution16_control", "resolution32_candidate")
TRAIN_STEPS, TRAIN_SETTLE, EVAL_STEPS, EVAL_SETTLE = 64, 32, 1024, 512
UPDATES, BATCH, ACCUM = 256, 16, 16
LR = 3e-5
SOURCE_EVAL_ROOT = ROOT / "collaborative_test/SW_0097_graph_adaptation/results"
SOURCE_EVAL_SHAS = {
    0: "009da533e0f3e7fa86d9840a85d87640d9424acf660512f64232ad1669fe61e0",
    1: "3576ad632d1caba7d396ce65e0d094e86ff2eb9d812f8ef03d5c5e8a30a54278",
    2: "2d4eb351a530058f307c5c907aa078b9a197323c69fc4e44754d237531e9538c",
}
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"), str(HERE)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_source_core(seed, grid_size, device):
    if seed not in SEEDS:
        raise ValueError("only registered source seeds 0, 1, 2 are allowed")
    from collaborative_test.SW_0110_xy_graph_route import run as common
    folder = SOURCE_ROOT / f"seed{seed}_positive_frozen"
    checkpoint, manifest_path, manifest = common.source_paths(seed)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    from resolution import build_core
    core = build_core(raw, grid_size=grid_size, device=device)
    core.eval()
    if core.graph_generator is not None:
        core.graph_generator.requires_grad_(False)
        core.graph_generator.eval()
    return core, checkpoint, manifest_path, manifest


def implementation_fingerprint():
    from collaborative_test.SW_0110_xy_graph_route import run as common
    paths = [HERE / "run.py", HERE / "resolution.py", HERE / "gamma_cache.py",
             HERE / "protocol.json", ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
             ROOT / "snn_kuramoto_bidirectional/graph_generator.py",
             ROOT / "snn_kuramoto_bidirectional/kuramoto_layer.py",
             ROOT / "snn_kuramoto_bidirectional/dendric_layer.py",
             ROOT / "snn_kuramoto_bidirectional/membrane_layer.py",
             ROOT / "snn_kuramoto_bidirectional/loss_function.py",
             ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
             ROOT / "snn_kuramoto_bidirectional/training/train_s2net_core.py",
             ROOT / "collaborative_test/SW_0094_aligned_joint_pilot/run.py"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths}


def criterion(grid_size):
    from snn_kuramoto_bidirectional.loss_function import UnsupervisedS2NetLoss
    return UnsupervisedS2NetLoss(
        spike_rate_weight=0., spike_smooth_weight=0., spike_diversity_weight=0.,
        structural_weight=0., plv_bimodality_weight=6., plv_balance_weight=10.,
        plv_coherence_weight=.5 if grid_size == 16 else 1.,
        plv_collapse_weight=1., plv_target_density=.867,
        patch_grid_size=(grid_size, grid_size))


def loss_parts(core, gamma, lossfn):
    from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
    from snn_kuramoto_bidirectional.training.train_s2net_core import _forward_with_plv
    _, _, _, plv, theta = _forward_with_plv(core, gamma, lossfn, TRAIN_SETTLE,
                                             "phase", "mean")
    components = core.last_component_spikes
    expected_nodes = core.kuramoto.N
    if tuple(components.shape[1:]) != (4, expected_nodes, TRAIN_STEPS):
        raise AssertionError(f"component spike shape mismatch: {tuple(components.shape)}")
    q = spike_synchrony_affinity(components.mean(dim=1), components,
                                 settle=TRAIN_SETTLE)
    primary, _ = lossfn(plv=plv, theta=theta)
    positive, _ = lossfn(plv=q)
    return primary + 5. * positive, primary, positive


def _parameter_grad_vector(core):
    parts = [p.grad.detach().reshape(-1).double().cpu() for p in core.parameters()
             if p.requires_grad and p.grad is not None]
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.float64)


def audit_control_microbatch_equivalence(source_state, gamma16, *, device="cpu"):
    """Compare one registered B16 loss/update with the specified 16xB1 accumulation."""
    from resolution import build_core
    if tuple(gamma16.shape) != (16, 8, 256):
        raise ValueError("microbatch audit requires the first logical B16 gamma batch")
    lossfn = criterion(16)
    full = build_core(source_state, grid_size=16, device=device)
    micro = build_core(source_state, grid_size=16, device=device)
    fp = [p for p in full.parameters() if p.requires_grad]
    mp = [p for p in micro.parameters() if p.requires_grad]
    lf, _, _ = loss_parts(full, gamma16.to(device), lossfn)
    lf.backward()
    lm_values = []
    for idx in range(16):
        li, _, _ = loss_parts(micro, gamma16[idx:idx+1].to(device), lossfn)
        (li / 16.).backward()
        lm_values.append(float(li.detach()))
    vf, vm = float(lf.detach()), sum(lm_values) / 16.
    if not (math.isfinite(vf) and math.isfinite(vm)):
        raise FloatingPointError("nonfinite accumulation audit loss")
    gf = torch.cat([p.grad.detach().reshape(-1).double().cpu() for p in fp if p.grad is not None])
    gm = torch.cat([p.grad.detach().reshape(-1).double().cpu() for p in mp if p.grad is not None])
    if gf.shape != gm.shape:
        raise AssertionError("full and microbatch gradient parameter layouts differ")
    delta = gf - gm
    rel = float(delta.norm() / gf.norm().clamp_min(1e-30))
    max_abs = float(delta.abs().max()) if delta.numel() else 0.
    allowed = 1e-5 + 1e-4 * (float(gf.abs().max()) if gf.numel() else 0.)
    if rel > 1e-4 or max_abs > allowed:
        raise AssertionError(f"B16 accumulation gradient mismatch: rel={rel}, max={max_abs}, allowed={allowed}")
    optf, optm = torch.optim.Adam(fp, lr=LR), torch.optim.Adam(mp, lr=LR)
    nf = torch.nn.utils.clip_grad_norm_(fp, 1.)
    nm = torch.nn.utils.clip_grad_norm_(mp, 1.)
    if not torch.isfinite(nf) or not torch.isfinite(nm):
        raise FloatingPointError("nonfinite clipped audit gradient")
    optf.step(); optm.step()
    max_update = max(float((a.detach() - b.detach()).abs().max())
                     for a, b in zip(fp, mp))
    return {"full_b16_loss": vf, "mean_b1_loss": vm,
            "loss_abs_diff": abs(vf-vm), "gradient_relative_norm_error": rel,
            "gradient_max_abs_error": max_abs, "gradient_max_abs_tolerance": allowed,
            "one_adam_update_max_abs_diff": max_update,
            "pass": rel <= 1e-4 and max_abs <= allowed}


def preflight(seed, grid_size, *, device="cuda"):
    """Actual data first-four-batch finite-gradient check; no GT or optimizer step."""
    from collaborative_test.SW_0110_xy_graph_route import run as common
    from resolution import build_core
    ids, rows = common.train_indices(seed)
    if grid_size == 16:
        gamma_blob, _ = common.validate_gamma_cache(TRAIN_GAMMA, common.GAMMA_TRAIN_MANIFEST)
        gamma = gamma_blob[torch.as_tensor(rows)]
        cache_info = {"kind": "registered16", "sha256": sha(TRAIN_GAMMA)}
    elif grid_size == 32:
        p = ARCHIVE / f"gamma32_train_seed{seed}.pt"
        m = ARCHIVE / f"gamma32_train_seed{seed}.json"
        if not p.is_file() or not m.is_file():
            raise FileNotFoundError("verified selected-ID candidate gamma32 cache required")
        cache_info = json.loads(m.read_text())
        from gamma_cache import dataset_identity
        if cache_info.get("gamma32_sha256") != sha(p) or cache_info.get("ids") != ids.tolist():
            raise AssertionError("candidate gamma32 cache provenance mismatch")
        if (tuple(cache_info.get("dataset_identity", {}).get("image_shape", [])) != (100000, 128, 128, 3)
                or cache_info.get("registered_gamma16_max_abs_diff", float("inf")) > 2e-5
                or cache_info.get("dataset_identity") != dataset_identity()):
            raise AssertionError("candidate gamma32 source-data contract failed")
        gamma = torch.load(p, map_location="cpu", weights_only=True)
        if tuple(gamma.shape) != (4096, 8, 1024):
            raise AssertionError(f"candidate gamma32 shape mismatch: {tuple(gamma.shape)}")
    else:
        raise ValueError("grid size must be 16 or32")
    device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    core, source, source_manifest, manifest = load_source_core(seed, grid_size, device)
    source_state = torch.load(source, map_location=device, weights_only=True)
    if grid_size == 16 and seed == 0:
        microaudit = audit_control_microbatch_equivalence(
            source_state, gamma[:16], device=device)
        if not microaudit["pass"]:
            raise AssertionError("B16 vs16xB1 gradient audit failed")
    else:
        microaudit = None
    lossfn = criterion(grid_size)
    params = [p for p in core.parameters() if p.requires_grad]
    if not params:
        raise AssertionError("no eligible trainable core parameters")
    batches = []
    core.train(); core.graph_generator.eval()
    for b in range(4):
        batch = gamma[b*BATCH:(b+1)*BATCH].to(device)
        if len(batch) != BATCH:
            raise AssertionError("preflight does not have four full matched B16 batches")
        core.zero_grad(set_to_none=True)
        if grid_size == 16:
            total, primary, positive = loss_parts(core, batch, lossfn)
            total.backward()
        else:
            total_value = primary_value = positive_value = 0.
            for j in range(BATCH):
                part, primary, positive = loss_parts(core, batch[j:j+1], lossfn)
                (part / BATCH).backward()
                total_value += float(part.detach()) / BATCH
                primary_value += float(primary.detach()) / BATCH
                positive_value += float(positive.detach()) / BATCH
            total = torch.tensor(total_value, device=device)
            primary = torch.tensor(primary_value, device=device)
            positive = torch.tensor(positive_value, device=device)
        grads = [p.grad for p in params if p.grad is not None]
        if not torch.isfinite(total) or not grads or any(not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError(f"nonfinite/empty trainable gradients seed={seed} batch={b}")
        norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads))
        if not math.isfinite(norm) or norm <= 0:
            raise AssertionError("eligible core gradient is zero/nonfinite")
        batches.append({"batch": b, "global_ids": ids[b*BATCH:(b+1)*BATCH].tolist(),
                        "loss": float(total.detach()), "primary": float(primary.detach()),
                        "positive_actual_spike_product": float(positive.detach()),
                        "gradient_norm": norm})
    # Verify one complete logical update on a disposable model. Candidate uses
    # exactly the intended B1x16 accumulation path; control uses native B16.
    update_core, _, _, _ = load_source_core(seed, grid_size, device)
    update_params = [p for p in update_core.parameters() if p.requires_grad]
    update_opt = torch.optim.Adam(update_params, lr=LR)
    graph_before = {k: v.detach().clone() for k, v in update_core.graph_generator.state_dict().items()}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    update_t0 = time.perf_counter()
    if grid_size == 16:
        update_loss, _, _ = loss_parts(update_core, gamma[:BATCH].to(device), lossfn)
        update_opt.zero_grad(set_to_none=True)
        update_loss.backward()
    else:
        update_loss_value = 0.
        update_opt.zero_grad(set_to_none=True)
        for j in range(BATCH):
            part, _, _ = loss_parts(update_core, gamma[j:j+1].to(device), lossfn)
            (part / BATCH).backward()
            update_loss_value += float(part.detach()) / BATCH
        update_loss = torch.tensor(update_loss_value, device=device)
    grad_norm = torch.nn.utils.clip_grad_norm_(update_params, 1.)
    if not torch.isfinite(update_loss) or not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
        raise FloatingPointError("disposable logical Adam update has invalid loss/gradient")
    update_opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        update_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        update_peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        update_peak_allocated = update_peak_reserved = None
    update_seconds = time.perf_counter() - update_t0
    graph_after = update_core.graph_generator.state_dict()
    if any(not torch.equal(v, graph_after[k]) for k, v in graph_before.items()):
        raise AssertionError("disposable update changed frozen graph state")
    if not any(not torch.equal(before, after) for before, after in zip(
            (p.detach() for p in core.parameters() if p.requires_grad),
            (p.detach() for p in update_core.parameters() if p.requires_grad))):
        raise AssertionError("disposable update changed no eligible core parameter")
    # Measure the registered inference horizon on a real selected training image
    # before any branch can enter its long continuation.
    import time as _time
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = _time.perf_counter()
    core.eval()
    with torch.no_grad():
        _, infer_spikes, infer_core_out = core(gamma[:1].to(device), return_core_out=True,
                                               num_time_steps=EVAL_STEPS)
        infer_components = core.last_component_spikes
    if tuple(infer_spikes.shape) != (1, grid_size * grid_size, EVAL_STEPS):
        raise AssertionError("full-horizon inference shape mismatch")
    if (not torch.isfinite(infer_spikes).all() or not torch.isfinite(infer_core_out).all()
            or infer_components is None or not torch.isfinite(infer_components).all()):
        raise FloatingPointError("full-horizon inference produced nonfinite model outputs")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated = peak_reserved = None
    infer_seconds = _time.perf_counter() - t0
    if not math.isfinite(infer_seconds) or infer_seconds <= 0:
        raise AssertionError("full-horizon inference timing is invalid")
    return {"status": "passed", "seed": seed, "grid_size": grid_size,
            "source_core_sha256": sha(source), "source_manifest_sha256": sha(source_manifest),
            "source_manifest_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
            "implementation_sha256": implementation_fingerprint(),
            "cache": cache_info, "batches": batches,
            "microbatch_equivalence": microaudit, "ground_truth_used": False,
            "throwaway_logical_adam_update": True,
            "throwaway_update_gradient_norm_preclip": float(grad_norm),
            "throwaway_update_seconds": update_seconds,
            "throwaway_update_peak_allocated_bytes": update_peak_allocated,
            "throwaway_update_peak_reserved_bytes": update_peak_reserved,
            "full_horizon_inference": {"time_steps": EVAL_STEPS, "settle": EVAL_SETTLE,
                "seconds": infer_seconds, "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved},
            "time_steps": TRAIN_STEPS, "settle": TRAIN_SETTLE, "device": str(device)}


def save_preflight(seed, grid_size, *, device="cuda"):
    path = ARCHIVE / f"preflight_seed{seed}_grid{grid_size}.json"
    if path.exists():
        raise FileExistsError(f"preserve previous preflight: {path}")
    record = preflight(seed, grid_size, device=device)
    write(path, record)
    return record


def train(seed, grid_size, output, *, device="cuda", updates=UPDATES):
    """Matched 256-update continuation, with 16 microbatches per logical update."""
    from collaborative_test.SW_0110_xy_graph_route import run as common
    from resolution import build_core
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite training attempt {out}")
    pf_path = ARCHIVE / f"preflight_seed{seed}_grid{grid_size}.json"
    if not pf_path.is_file():
        raise FileNotFoundError("successful registered preflight is required")
    pf = json.loads(pf_path.read_text())
    if (pf.get("status") != "passed" or pf.get("seed") != seed
            or pf.get("grid_size") != grid_size or pf.get("ground_truth_used") is not False
            or pf.get("implementation_sha256") != implementation_fingerprint()):
        raise AssertionError("preflight provenance does not match current implementation/task")
    ids, rows = common.train_indices(seed)
    if grid_size == 16:
        gamma_blob, _ = common.validate_gamma_cache(TRAIN_GAMMA, common.GAMMA_TRAIN_MANIFEST)
        gamma = gamma_blob
        cache_sha = sha(TRAIN_GAMMA)
    elif grid_size == 32:
        cache_path = ARCHIVE / f"gamma32_train_seed{seed}.pt"
        cache_meta_path = ARCHIVE / f"gamma32_train_seed{seed}.json"
        cache_meta = json.loads(cache_meta_path.read_text())
        from gamma_cache import dataset_identity
        if (cache_meta.get("gamma32_sha256") != sha(cache_path)
                or cache_meta.get("ids") != ids.tolist()
                or cache_meta.get("registered_gamma16_max_abs_diff", float("inf")) > 2e-5
                or cache_meta.get("dataset_identity") != dataset_identity()):
            raise AssertionError("candidate gamma cache failed frozen source contract")
        gamma = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
        cache_sha = sha(cache_path)
    else:
        raise ValueError("grid size must be 16 or32")
    if tuple(gamma.shape) != (70000, 8, grid_size * grid_size) and grid_size == 16:
        raise AssertionError("registered gamma16 cache shape mismatch")
    if tuple(gamma.shape) != (4096, 8, 1024) and grid_size == 32:
        raise AssertionError("candidate gamma32 cache shape mismatch")
    device = torch.device(device)
    core, source, source_manifest, _ = load_source_core(seed, grid_size, device)
    params = [p for p in core.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    lossfn = criterion(grid_size)
    frozen_graph = {k: v.detach().clone() for k, v in core.graph_generator.state_dict().items()}
    out.mkdir(parents=True, exist_ok=False)
    write(out / "manifest.json", {"status": "training", "seed": seed,
          "grid_size": grid_size, "arm": "resolution16_control" if grid_size == 16 else "resolution32_candidate",
          "source_core": str(source), "source_core_sha256": sha(source),
          "source_manifest_sha256": sha(source_manifest), "training_ids": ids.tolist(),
          "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
          "training_cache_sha256": cache_sha, "preflight_sha256": sha(pf_path),
          "implementation_sha256": implementation_fingerprint(),
          "matched_shuffle_seed": 117 + seed, "updates": updates, "batch_size": BATCH,
          "microbatch_size": 1, "accumulation_steps": ACCUM, "lr": LR,
          "time_steps": TRAIN_STEPS, "settle": TRAIN_SETTLE,
          "coherence_weight": .5 if grid_size == 16 else 1.,
          "loss": "phase_primary + 5*actual_positive_product_spike_affinity",
          "ground_truth_used_for_training": False})
    history = []
    core.train(); core.graph_generator.eval()
    for step in range(updates):
        begin = step * BATCH
        if grid_size == 16:
            ix = rows[begin:begin+BATCH]
            if len(ix) != BATCH:
                raise AssertionError("registered training order exhausted early")
            batch = gamma[torch.as_tensor(ix)].to(device)
        else:
            batch = gamma[begin:begin+BATCH].to(device)
        opt.zero_grad(set_to_none=True)
        total_value = primary_value = positive_value = 0.
        for j in range(BATCH):
            loss, primary, positive = loss_parts(core, batch[j:j+1], lossfn)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss seed{seed} grid{grid_size} update{step+1}")
            (loss / BATCH).backward()
            total_value += float(loss.detach()) / BATCH
            primary_value += float(primary.detach()) / BATCH
            positive_value += float(positive.detach()) / BATCH
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
            raise FloatingPointError(f"invalid trainable gradient at update{step+1}")
        opt.step()
        history.append({"update": step+1, "loss": total_value,
                        "primary": primary_value, "positive_spike": positive_value,
                        "gradient_norm_preclip": float(grad_norm)})
        if (step + 1) % 16 == 0:
            write(out / "progress.json", {"status": "training", "seed": seed,
                  "grid_size": grid_size, "update": step+1, "updates": updates})
        if (step + 1) % 64 == 0 or step + 1 == updates:
            torch.save(core.state_dict(), out / f"core_step_{step+1}.pt")
    if any(not torch.equal(v, core.graph_generator.state_dict()[k])
           for k, v in frozen_graph.items()):
        raise AssertionError("frozen graph parameters changed during continuation")
    if any(not torch.isfinite(p).all() for p in core.parameters()):
        raise FloatingPointError("nonfinite final model weights")
    torch.save(core.state_dict(), out / "core.pt")
    write(out / "history.json", history)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest.update(status="complete", completed_utc=time.time(), core_sha256=sha(out / "core.pt"))
    write(out / "manifest.json", manifest)


def load_encoder(device):
    from SW_0094_aligned_joint_pilot.run import ASSETS
    from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_input_encoder
    encoder_path = ASSETS / "input_encoder/input_layer_encoder.pt"
    stats_path = ASSETS / "feature_preprocessing.pt"
    encoder = load_input_encoder(str(encoder_path), num_kernels=8, kernel_size=3,
                                 channels=3, device=device)
    encoder.eval().requires_grad_(False)
    stats = torch.load(stats_path, map_location=device, weights_only=True)
    return encoder, stats, encoder_path, stats_path


def regenerate_selected_gamma(global_ids, *, batch_size=16, device="cuda", allow_validation=False):
    """Build only the passed RGB IDs at both registered pooling resolutions; never masks."""
    from gamma_cache import encode_gamma, ordered_rgb_batch, ids_sha256
    if not DATASET.is_file():
        raise FileNotFoundError(DATASET)
    encoder, stats, encoder_path, stats_path = load_encoder(device)
    ids = np.asarray(global_ids, dtype=np.int64)
    values16, values32 = [], []
    rgb_hasher = hashlib.sha256()
    with h5py.File(DATASET, "r") as data, torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            rgb = ordered_rgb_batch(data, batch_ids, allow_validation=allow_validation)
            rgb_hasher.update(rgb.permute(0, 2, 3, 1).contiguous().numpy().tobytes())
            values16.append(encode_gamma(rgb, encoder, stats, 16, device).cpu())
            values32.append(encode_gamma(rgb, encoder, stats, 32, device).cpu())
        image_shape = list(data["image"].shape)
    stat = DATASET.stat()
    return {"ids": ids.tolist(), "ids_sha256": ids_sha256(ids),
            "gamma16": torch.cat(values16), "gamma32": torch.cat(values32),
            "encoder_sha256": sha(encoder_path), "preprocessing_sha256": sha(stats_path),
            "selected_rgb_sha256": rgb_hasher.hexdigest(),
            "dataset_identity": {"path": str(DATASET), "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "image_shape": image_shape}}


def build_training_cache(seed, *, batch_size=16, device="cuda"):
    """Regenerate candidate gamma32 in registered order and verify all gamma16 rows."""
    from collaborative_test.SW_0110_xy_graph_route import run as common
    from gamma_cache import cache_rows, ids_sha256
    ids, rows = common.train_indices(seed)
    fresh = regenerate_selected_gamma(ids, batch_size=batch_size, device=device)
    expected, train_meta = common.validate_gamma_cache(
        TRAIN_GAMMA, common.GAMMA_TRAIN_MANIFEST, validation=False)
    if (fresh["encoder_sha256"] != train_meta.get("encoder_sha256")
            or fresh["preprocessing_sha256"] != train_meta.get("preprocessing_sha256")):
        raise AssertionError("regenerated gamma source encoder/stats differ from registered training cache")
    aligned = expected[torch.as_tensor(cache_rows(ids))]
    maxdiff = float((fresh["gamma16"] - aligned).abs().max())
    if not np.isfinite(maxdiff) or maxdiff > 2e-5:
        raise AssertionError(f"regenerated 16-grid gamma differs from registered cache: {maxdiff}")
    if not np.array_equal(ids, np.asarray(common.source_paths(seed)[2]["training_ids"], dtype=np.int64)):
        raise AssertionError("ordered IDs diverged from immutable SW0097 source manifest")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    out = ARCHIVE / f"gamma32_train_seed{seed}.pt"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite cache: {out}")
    torch.save(fresh["gamma32"], out)
    record = {"seed": seed, "ids": ids.tolist(), "ids_sha256": ids_sha256(ids), "gamma32_path": str(out),
            "gamma32_sha256": sha(out), "registered_gamma16_max_abs_diff": maxdiff,
            "source_core_sha256": sha(common.source_paths(seed)[0]),
            "source_manifest_sha256": sha(common.source_paths(seed)[1]),
            "encoder_sha256": fresh["encoder_sha256"],
            "preprocessing_sha256": fresh["preprocessing_sha256"],
            "selected_rgb_sha256": fresh["selected_rgb_sha256"],
            "dataset_identity": fresh["dataset_identity"], "count": len(ids)}
    write(ARCHIVE / f"gamma32_train_seed{seed}.json", record)
    return record


def build_validation_cache(*, batch_size=16, device="cuda"):
    """Regenerate only fixed validation RGB and require exact16-grid contract agreement."""
    from collaborative_test.SW_0110_xy_graph_route import run as common
    from gamma_cache import ids_sha256
    _, manifest = common.validate_gamma_cache(VAL_GAMMA, common.GAMMA_VAL_MANIFEST,
                                               validation=True)
    ids = np.arange(1320, 1640, dtype=np.int64)
    fresh = regenerate_selected_gamma(ids, batch_size=batch_size, device=device,
                                      allow_validation=True)
    if (fresh["encoder_sha256"] != manifest.get("encoder_sha256")
            or fresh["preprocessing_sha256"] != manifest.get("preprocessing_sha256")):
        raise AssertionError("regenerated validation gamma source encoder/stats differ")
    registered = torch.load(VAL_GAMMA, map_location="cpu", weights_only=True)
    maxdiff = float((fresh["gamma16"] - registered).abs().max())
    if not np.isfinite(maxdiff) or maxdiff > 2e-5:
        raise AssertionError(f"validation 16-grid gamma mismatch: {maxdiff}")
    out = ARCHIVE / "gamma32_validation_1320_1639.pt"
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite cache: {out}")
    torch.save(fresh["gamma32"], out)
    record = {"ids": ids.tolist(), "ids_range": [1320, 1639], "ids_sha256": ids_sha256(ids),
            "gamma32_path": str(out), "gamma32_sha256": sha(out),
            "registered_gamma16_max_abs_diff": maxdiff,
            "registered_gamma16_sha256": sha(VAL_GAMMA),
            "registered_gamma16_manifest_sha256": sha(common.GAMMA_VAL_MANIFEST),
            "encoder_sha256": fresh["encoder_sha256"],
            "preprocessing_sha256": fresh["preprocessing_sha256"],
            "selected_rgb_sha256": fresh["selected_rgb_sha256"],
            "dataset_identity": fresh["dataset_identity"],
            "count": len(ids), "ground_truth_read": False}
    write(ARCHIVE / "gamma32_validation_1320_1639.json", record)
    return record


@torch.no_grad()
def predict_frozen(core, gamma, grid_size, *, time_steps=EVAL_STEPS, settle=EVAL_SETTLE):
    """Prediction-only existing q>=.50 classifier, before any labels are read."""
    from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components
    from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
    _, spikes, _ = core(gamma, return_core_out=True, num_time_steps=time_steps)
    components = core.last_component_spikes
    if tuple(components.shape[:3]) != (gamma.shape[0], 4, grid_size * grid_size):
        raise AssertionError(f"component-spike shape invalid: {tuple(components.shape)}")
    groups = spike_synchrony_components(
        spikes, synchrony_threshold=.5, min_group_size=(2 if grid_size == 16 else 8),
        settle=settle, components=components, background="largest_component",
        spatial_grid_size=grid_size, affinity_mode="spike")
    labels = spatial_components_to_patch_labels(groups, grid_size, device="cpu")
    return labels, groups, {"empty_images": sum(not g for g in groups),
                            "predicted_components": [len(g) for g in groups],
                            "foreground_fraction": float((labels != 0).float().mean())}


def score_frozen_predictions(source16, prediction16, prediction32, groups32, ids, *, dataset_path=DATASET):
    """Open GT only after both predictions are frozen; return paired target scores."""
    from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
    from snn_kuramoto_bidirectional.evaluation import evaluate_patch_masks
    from resolution import downsample_component_labels
    if source16 is None and prediction16 is None and prediction32 is None:
        raise ValueError("at least one frozen prediction is required")
    ids = np.asarray(ids, dtype=np.int64)
    if not np.array_equal(ids, np.arange(1320, 1640, dtype=np.int64)):
        raise AssertionError("only the preregistered validation IDs may be scored")
    primary32 = downsample_component_labels(prediction32, groups32) if prediction32 is not None else None
    # This is the first mask read; all supplied predictions already exist in memory.
    with h5py.File(dataset_path, "r") as data:
        # Global source image IDs are direct HDF5 indices; only packed gamma
        # cache rows use the separate ID-640 mapping.
        order = np.argsort(ids)
        raw = data["mask"][ids[order].tolist()]
        masks = raw[np.argsort(order)]
    masks = torch.from_numpy(masks.copy())
    target16 = clevr_mask_patch(masks, 8)["patch_labels"]
    target32 = clevr_mask_patch(masks, 4)["patch_labels"]
    result = {}
    if source16 is not None:
        result["source16_vs_modal8"] = scores_to_json(
            evaluate_patch_masks(source16, target16))
    if prediction16 is not None:
        scores = evaluate_patch_masks(prediction16, target16)
        result["control16_vs_modal8"] = scores_to_json(scores)
        control32 = prediction16.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        result["control_repeated32_vs_modal4"] = scores_to_json(
            evaluate_patch_masks(control32, target32))
    if prediction32 is not None:
        scores16 = evaluate_patch_masks(primary32, target16)
        scores32 = evaluate_patch_masks(prediction32, target32)
        result["candidate_pooled16_vs_modal8"] = scores_to_json(scores16)
        result["candidate_native32_vs_modal4"] = scores_to_json(scores32)
    return result


@torch.no_grad()
def generate_validation_predictions(core, gamma_cache, grid_size, *, device="cuda", batch_size=1):
    if tuple(gamma_cache.shape) != (320, 8, grid_size * grid_size):
        raise AssertionError("validation gamma cache does not contain exact320 rows/grid")
    labels, groups, diagnostics = [], [], []
    if batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    for idx in range(0, 320, batch_size):
        label, image_groups, diag = predict_frozen(
            core, gamma_cache[idx:idx+batch_size].to(device), grid_size,
            time_steps=EVAL_STEPS, settle=EVAL_SETTLE)
        labels.append(label.cpu())
        groups.extend(image_groups)
        diagnostics.append(diag)
    return torch.cat(labels), groups, diagnostics


def registered_source_evaluation(seed):
    path = SOURCE_EVAL_ROOT / f"seed{seed}_positive_frozen/evaluation.json"
    if not path.is_file() or sha(path) != SOURCE_EVAL_SHAS[seed]:
        raise AssertionError(f"immutable SW0097 per-image evaluation reference changed: {path}")
    report = json.loads(path.read_text())
    rows = report.get("sweep")
    if report.get("ids") != [1320, 1639] or report.get("images") != 320 or not rows:
        raise AssertionError("registered SW0097 reference is not the fixed validation endpoint")
    scored = rows[0].get("scored_targets", {}).get("our_hdf5")
    if scored is None or set(scored.get("per_image", {})) != {
            "fg_ari", "foreground_iou", "matched_object_iou"}:
        raise AssertionError("registered source reference is missing per-image metrics")
    for metric in ("fg_ari", "foreground_iou", "matched_object_iou"):
        values = np.asarray(scored["per_image"][metric], dtype=np.float64)
        if (values.shape != (320,) or not np.isfinite(values).all()
                or scored["valid_count"].get(metric) != 320
                or abs(float(values.mean()) - float(scored["metrics"][metric])) > 1e-12):
            raise AssertionError(f"registered source metric failed validation: {metric}")
    return path, sha(path), scored


def evaluate_seed(seed, *, control_checkpoint, candidate_checkpoint,
                  source_checkpoint=None, device="cuda"):
    """Freeze predictions for source/control/candidate before opening GT masks."""
    from collaborative_test.SW_0110_xy_graph_route import run as common
    if source_checkpoint is None:
        source_checkpoint = common.source_paths(seed)[0]
    val16, _ = common.validate_gamma_cache(VAL_GAMMA, common.GAMMA_VAL_MANIFEST,
                                            validation=True)
    val32_path = ARCHIVE / "gamma32_validation_1320_1639.pt"
    val32_meta = ARCHIVE / "gamma32_validation_1320_1639.json"
    if not val32_path.is_file() or not val32_meta.is_file():
        raise FileNotFoundError("verified native32 validation gamma cache required")
    val32 = torch.load(val32_path, map_location="cpu", weights_only=True)
    meta = json.loads(val32_meta.read_text())
    if (meta.get("gamma32_sha256") != sha(val32_path)
            or meta.get("ids") != list(range(1320, 1640))
            or meta.get("count") != 320 or meta.get("ground_truth_read") is not False):
        raise AssertionError("native32 validation cache provenance mismatch")
    output_dir = ARCHIVE / f"evaluation_seed{seed}"
    if output_dir.exists():
        raise FileExistsError(f"preserve previous evaluation attempt: {output_dir}")
    output_dir.mkdir(parents=True)
    frozen = {}
    for name, checkpoint, grid in (
            ("source", source_checkpoint, 16),
            ("control", control_checkpoint, 16),
            ("candidate", candidate_checkpoint, 32)):
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        core, _, _, _ = load_source_core(seed, grid, device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        core.load_state_dict(state, strict=True)
        core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.shape[0])]
        core.eval()
        gamma = val16 if grid == 16 else val32
        eval_batch = 8 if grid == 16 else 1
        labels, groups, diag = generate_validation_predictions(
            core, gamma, grid, device=device, batch_size=eval_batch)
        frozen[name] = {"labels": labels, "groups": groups if grid == 32 else None,
                        "diagnostics": diag, "checkpoint_sha256": sha(checkpoint),
                        "grid_size": grid, "batch_size": eval_batch}
        del core
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Persist exact predictions before any GT/mask is opened.
    pred_path = output_dir / "frozen_predictions.pt"
    torch.save({k: {kk: vv for kk, vv in v.items() if kk != "diagnostics"}
                for k, v in frozen.items()}, pred_path)
    write(output_dir / "prediction_diagnostics.json",
          {k: {"diagnostics": v["diagnostics"], "checkpoint_sha256": v["checkpoint_sha256"],
               "grid_size": v["grid_size"], "batch_size": v["batch_size"]}
           for k, v in frozen.items()})
    ids = np.arange(1320, 1640, dtype=np.int64)
    scores = score_frozen_predictions(frozen["source"]["labels"],
                                      frozen["control"]["labels"],
                                      frozen["candidate"]["labels"],
                                      frozen["candidate"]["groups"], ids)
    reference_path, reference_sha, reference = registered_source_evaluation(seed)
    baseline_checks = {}
    for metric in ("fg_ari", "foreground_iou", "matched_object_iou"):
        actual = np.asarray(scores["source16_vs_modal8"]["per_image"][metric], dtype=np.float64)
        expected = np.asarray(reference["per_image"][metric], dtype=np.float64)
        diff = float(np.max(np.abs(actual - expected)))
        baseline_checks[metric] = {"max_abs_difference": diff, "tolerance": 1e-10,
                                   "pass": bool(np.isfinite(actual).all() and diff <= 1e-10)}
    baseline_pass = all(row["pass"] for row in baseline_checks.values())
    write(output_dir / "source_baseline_equivalence.json",
          {"reference_path": str(reference_path), "reference_sha256": reference_sha,
           "seed": seed, "batch_size": 8, "checks": baseline_checks,
           "pass": baseline_pass})
    if not baseline_pass:
        raise AssertionError(f"SW0097 source endpoint did not reproduce fixed B8 scores: {baseline_checks}")
    report = {"status": "complete", "seed": seed, "ids": [1320, 1639],
              "count": 320, "ground_truth_used_during_prediction": False,
              "source_baseline_equivalence": {"reference_sha256": reference_sha,
                    "pass": baseline_pass, "checks": baseline_checks},
              "frozen_predictions_sha256": sha(pred_path), "scores": scores,
              "implementation_fingerprint": implementation_fingerprint(),
              "source_checkpoint_sha256": frozen["source"]["checkpoint_sha256"],
              "control_checkpoint_sha256": frozen["control"]["checkpoint_sha256"],
              "candidate_checkpoint_sha256": frozen["candidate"]["checkpoint_sha256"],
              "native32_gamma_sha256": sha(val32_path),
              "registered_native16_gamma_sha256": sha(VAL_GAMMA)}
    write(output_dir / "evaluation.json", report)
    return report


def scores_to_json(scores):
    as_float = lambda value: (float(value) if math.isfinite(float(value)) else None)
    return {"mean": {k: as_float(v) for k, v in scores["mean"].items()},
            "valid_count": {k: int(v) for k, v in scores["valid_count"].items()},
            "per_image": {k: [as_float(x) for x in v.cpu().tolist()]
                          for k, v in scores["per_image"].items()}}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-train-cache", type=int, choices=SEEDS)
    parser.add_argument("--build-validation-cache", action="store_true")
    parser.add_argument("--preflight", type=int, choices=SEEDS)
    parser.add_argument("--grid", type=int, choices=(16, 32), default=32)
    parser.add_argument("--train", type=int, choices=SEEDS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluate", type=int, choices=SEEDS)
    parser.add_argument("--control-checkpoint", type=Path)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.build_train_cache is not None:
        result = build_training_cache(args.build_train_cache, device=args.device)
    elif args.build_validation_cache:
        result = build_validation_cache(device=args.device)
    elif args.preflight is not None:
        result = save_preflight(args.preflight, args.grid, device=args.device)
    elif args.train is not None:
        if args.output is None:
            parser.error("--train requires --output")
        result = train(args.train, args.grid, args.output, device=args.device)
    elif args.evaluate is not None:
        if args.control_checkpoint is None or args.candidate_checkpoint is None:
            parser.error("--evaluate requires control and candidate checkpoints")
        result = evaluate_seed(args.evaluate, control_checkpoint=args.control_checkpoint,
                               candidate_checkpoint=args.candidate_checkpoint,
                               device=args.device)
    elif args.summarize:
        from summarize import summarize
        result = summarize(output=ARCHIVE / "summary.json")
    else:
        parser.error("select one operation")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
