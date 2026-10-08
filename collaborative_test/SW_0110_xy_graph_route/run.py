"""Matched XY-node-feature route pilot. Does not edit shared model code."""
import argparse
import hashlib
import importlib.util
import io
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
from SW_0094_aligned_joint_pilot.run import ASSETS, hparams
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.loss_function import UnsupervisedS2NetLoss
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from snn_kuramoto_bidirectional.training.train_s2net_core import _forward_with_plv
from collaborative_test.SW_0110_xy_graph_route.xy_graph import attach_xy_graph

SEEDS = (0, 1, 2)
ARMS = ("control", "xy_candidate")
BATCH = 16
UPDATES = 256
TRAIN_STEPS = 64
TRAIN_SETTLE = 32
LR = 3e-5
SOURCE_ROOT = ROOT / "trained_models/SW0097_graph_adaptation"
GAMMA_TRAIN = ROOT / "data/SW_0090_large_unique_scale/gamma_train_70000.pt"
GAMMA_TRAIN_MANIFEST = ROOT / "data/SW_0090_large_unique_scale/manifest.json"
GAMMA_VAL = ROOT / "data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"
GAMMA_VAL_MANIFEST = ROOT / "data/SW_0042_hdf5_aligned/manifest.json"
DATASET = Path("/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5")
OUT = ROOT / "trained_models/SW0110_xy_graph_route"
ARCHIVE = HERE / "results_archive"
SOURCE_AUDIT = HERE / "source_audit.json"
SOURCE_EVIDENCE_DIR = ROOT / "collaborative_test/SW_0107_official_full70k_transfer/results_archive"
EXPECTED_SOURCE_SHAS = {
    0: "36f2481dd1fa51fa29bd4dc34275a876b71d8b76fbee13ac47a0a1bfe6766fbf",
    1: "76f379d5a7e4cd9d12fdf0b701f3dd9dbbf28b5eea270a9cee2d30d87b4dad98",
    2: "798ad3e9d4bf837b1bbeb1bd7c13900df511b5c76f5736d2b9f8b973d7fa5661",
}
EXPECTED_ENCODER_SHA256 = "9b9c9c59725f872084a6c9bfae1ebbd1b89de40d18ef4f8ab6b8ec014c2ab868"
EXPECTED_PREPROCESSING_SHA256 = "a9ec3281b9603021c37f7c346baa6cf62513c5ce7edde867ab7eddf24b18e127"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def implementation_fingerprint(extra_files=()):
    files = [HERE / "run.py", HERE / "xy_graph.py", HERE / "protocol.json",
             ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
             ROOT / "snn_kuramoto_bidirectional/loss_function.py",
             ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
             ROOT / "snn_kuramoto_bidirectional/training/train_s2net_core.py",
             ROOT / "collaborative_test/SW_0094_aligned_joint_pilot/run.py"]
    files.extend(extra_files)
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
            for path in files}


def source_paths(seed):
    folder = SOURCE_ROOT / f"seed{seed}_positive_frozen"
    checkpoint, manifest = folder / "core.pt", folder / "manifest.json"
    if seed not in SEEDS or not checkpoint.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"incomplete registered SW0097 source seed {seed}: {folder}")
    m = json.loads(manifest.read_text())
    audit = json.loads(SOURCE_AUDIT.read_text())
    proof_path = SOURCE_EVIDENCE_DIR / f"seed{seed}_manifest.json"
    if not proof_path.is_file():
        raise FileNotFoundError(proof_path)
    proof = json.loads(proof_path.read_text())
    expected_sha = EXPECTED_SOURCE_SHAS[seed]
    if (audit.get("core_sha256", {}).get(str(seed)) != expected_sha
            or proof.get("source_core_sha256") != expected_sha
            or sha(checkpoint) != expected_sha):
        raise AssertionError(f"immutable SW0097 source checkpoint SHA mismatch for seed {seed}")
    encoder = ASSETS / "input_encoder/input_layer_encoder.pt"
    stats = ASSETS / "feature_preprocessing.pt"
    if sha(encoder) != EXPECTED_ENCODER_SHA256 or sha(stats) != EXPECTED_PREPROCESSING_SHA256:
        raise AssertionError("registered frozen encoder/preprocessing SHA mismatch")
    if (m.get("status") != "complete" or m.get("source_model_seed") != seed
            or m.get("unique_images_seen") != 4096 or m.get("steps") != 256
            or m.get("batch") != 16 or m.get("seed") != 117 + seed
            or m.get("ground_truth_used_for_training") is not False):
        raise AssertionError(f"SW0097 matched source contract failed for seed{seed}")
    ids = m.get("training_ids")
    if not isinstance(ids, list) or len(ids) != 4096 or len(set(ids)) != 4096:
        raise AssertionError(f"invalid matched source training IDs for seed{seed}")
    if any(not (0 <= int(i) < 1000 or 1640 <= int(i) < 70640) for i in ids):
        raise AssertionError("matched SW0097 training IDs leave registered train pool")
    return checkpoint, manifest, m


def gamma_rows(global_ids):
    ids = np.asarray(global_ids, dtype=np.int64)
    return np.where(ids < 1000, ids, ids - 640).astype(np.int64)


def make_core(device, steps=TRAIN_STEPS):
    hp = hparams("raw")
    hp.num_time_steps = int(steps)
    return S2NetCore(hp.validate(), device=device).to(device)


def load_source(seed, arm, device):
    checkpoint, manifest_path, manifest = source_paths(seed)
    core = make_core(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    core.load_state_dict(state, strict=True)
    if arm == "xy_candidate":
        attach_xy_graph(core, grid_size=16)
    elif arm != "control":
        raise ValueError(f"unknown arm {arm}")
    core.graph_generator.requires_grad_(False)
    if arm == "xy_candidate":
        core.graph_generator.xy_projection.requires_grad_(True)
    core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
    if core.graph_generator.uses_feedback or core.kuramoto.spike_pulse_gain is not None:
        raise AssertionError("source graph feedback or spike pulse is outside SW0110 contract")
    return core, checkpoint, manifest_path, manifest


def criterion():
    return UnsupervisedS2NetLoss(
        spike_rate_weight=0.0, spike_smooth_weight=0.0,
        spike_diversity_weight=0.0, structural_weight=0.0,
        plv_bimodality_weight=6.0, plv_balance_weight=10.0,
        plv_coherence_weight=0.5, plv_collapse_weight=1.0,
        plv_target_density=0.867, patch_grid_size=(16, 16))


def loss_parts(core, gamma, lossfn):
    _, _, _, plv, theta = _forward_with_plv(
        core, gamma, lossfn, TRAIN_SETTLE, "phase", "mean")
    components = core.last_component_spikes
    if tuple(components.shape[1:]) != (4, 256, TRAIN_STEPS):
        raise AssertionError(f"actual component spike shape mismatch: {tuple(components.shape)}")
    q = spike_synchrony_affinity(components.mean(dim=1), components, settle=TRAIN_SETTLE)
    primary, _ = lossfn(plv=plv, theta=theta)
    positive, _ = lossfn(plv=q)
    return primary + 5.0 * positive, primary, positive


def train_indices(seed):
    _, _, m = source_paths(seed)
    ids = np.asarray(m["training_ids"], dtype=np.int64)
    rows = gamma_rows(ids)
    if len(np.unique(rows)) != 4096 or rows.min() < 0 or rows.max() >= 70000:
        raise AssertionError("SW0097 registered image ordering does not map to gamma cache")
    return ids, rows


def validate_gamma_cache(path, manifest_path, validation=False):
    path, manifest_path = Path(path), Path(manifest_path)
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(path if not path.is_file() else manifest_path)
    m = json.loads(manifest_path.read_text())
    if m.get("gamma_sha256") != sha(path):
        raise AssertionError("registered gamma cache SHA does not match its manifest")
    encoder = ASSETS / "input_encoder/input_layer_encoder.pt"
    stats = ASSETS / "feature_preprocessing.pt"
    if not encoder.is_file() or not stats.is_file():
        raise FileNotFoundError(encoder if not encoder.is_file() else stats)
    if (m.get("encoder_sha256") != sha(encoder) or m.get("preprocessing_sha256") != sha(stats)
            or m.get("encoder_sha256") != EXPECTED_ENCODER_SHA256
            or m.get("preprocessing_sha256") != EXPECTED_PREPROCESSING_SHA256):
        raise AssertionError("gamma cache does not use the registered frozen encoder/preprocessing")
    if validation:
        if m.get("image_ids") != [1320, 1639]:
            raise AssertionError("validation cache must contain exactly the registered image range")
        shape = (320, 8, 256)
    else:
        ids = m.get("training_ids", {})
        if ids.get("segments") != [[0, 999], [1640, 70639]] or ids.get("count") != 70000:
            raise AssertionError("training cache manifest differs from the registered 70k IDs")
        shape = (70000, 8, 256)
    blob = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if tuple(blob.shape) != shape:
        raise AssertionError(f"gamma cache shape {tuple(blob.shape)} != {shape}")
    return blob, m


def preflight(seed, arm, gamma_path=GAMMA_TRAIN, device="cuda"):
    implementation = implementation_fingerprint()
    """Read-only real-data four-batch gradient audit; no optimizer or checkpoint writes."""
    device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    core, source, source_manifest, manifest = load_source(seed, arm, device)
    ids, rows = train_indices(seed)
    gamma, gamma_manifest = validate_gamma_cache(gamma_path, GAMMA_TRAIN_MANIFEST)
    # The zero-P candidate must be the exact legacy rollout before training.
    control, _, _, _ = load_source(seed, "control", device)
    baseline, _, _, _ = load_source(seed, "xy_candidate", device)
    probe = gamma[torch.as_tensor(rows[:BATCH])].to(device)
    control.eval(); baseline.eval()
    with torch.no_grad():
        control_drive = control.gamma_to_drive(probe, control.gamma_channel_proj,
                                               control.gamma_phase_gain)
        candidate_drive = baseline.gamma_to_drive(probe, baseline.gamma_channel_proj,
                                                   baseline.gamma_phase_gain)
        if not torch.equal(control_drive, candidate_drive):
            raise AssertionError("XY route changed native gamma-to-drive at zero initialization")
        control_graph = control.graph_generator(probe)
        candidate_graph = baseline.graph_generator(probe)
        if not torch.equal(control_graph, candidate_graph):
            raise AssertionError("zero-initialized XY graph differs from legacy graph output")
        control_out = control(probe, return_core_out=True, return_theta=True,
                              num_time_steps=TRAIN_STEPS)
        candidate_out = baseline(probe, return_core_out=True, return_theta=True,
                                 num_time_steps=TRAIN_STEPS)
        for label, left, right in zip(("spikes", "membrane", "theta"),
                                      control_out[1:], candidate_out[1:]):
            if not torch.equal(left, right):
                raise AssertionError(f"zero-initialized XY core changed legacy {label} rollout")
    roundtrip_buffer = io.BytesIO()
    torch.save(baseline.state_dict(), roundtrip_buffer)
    roundtrip_buffer.seek(0)
    restored, _, _, _ = load_source(seed, "xy_candidate", device)
    restored.load_state_dict(torch.load(roundtrip_buffer, map_location=device, weights_only=True), strict=True)
    restored.eval()
    with torch.no_grad():
        restored_out = restored(probe, return_core_out=True, return_theta=True,
                                num_time_steps=TRAIN_STEPS)
    for label, left, right in zip(("spikes", "membrane", "theta"),
                                  candidate_out[1:], restored_out[1:]):
        if not torch.equal(left, right):
            raise AssertionError(f"candidate strict checkpoint roundtrip changed {label}")
    del control, baseline, restored, control_out, candidate_out, restored_out
    initial = {k: v.detach().clone() for k, v in core.state_dict().items()}
    params = [p for p in core.parameters() if p.requires_grad]
    if not params:
        raise AssertionError("preflight has no trainable parameters")
    lossfn = criterion()
    batches = []
    core.train()
    core.graph_generator.eval()
    for i in range(4):
        batch = gamma[torch.as_tensor(rows[i * BATCH:(i + 1) * BATCH])].to(device)
        total, primary, positive = loss_parts(core, batch, lossfn)
        if not torch.isfinite(total):
            raise FloatingPointError(f"nonfinite loss seed={seed} arm={arm} batch={i}")
        core.zero_grad(set_to_none=True)
        total.backward()
        grads = [p.grad for p in params if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError(f"nonfinite/empty gradients seed={seed} arm={arm} batch={i}")
        pnorm = None
        if arm == "xy_candidate":
            pg = core.graph_generator.xy_projection.grad
            if pg is None or not torch.isfinite(pg).all() or float(pg.norm()) <= 0:
                raise AssertionError(f"coordinate route is inert/nonfinite seed={seed} batch={i}")
            pnorm = float(pg.norm())
        norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads))
        if not math.isfinite(norm) or norm <= 0:
            raise AssertionError("preflight core gradient is zero/nonfinite")
        batches.append({"batch": i, "global_ids": ids[i*BATCH:(i+1)*BATCH].tolist(),
                        "total_loss": float(total.detach()), "primary": float(primary.detach()),
                        "positive_actual_spike_product": float(positive.detach()),
                        "gradient_norm": norm, "xy_projection_gradient_norm": pnorm})
    changed = [k for k, v in initial.items() if not torch.equal(v, core.state_dict()[k])]
    if changed:
        raise AssertionError(f"read-only preflight unexpectedly updated parameters: {changed}")
    # Exercise one actual Adam update on a disposable copy of the source model.
    update_core, _, _, _ = load_source(seed, arm, device)
    update_params = [p for p in update_core.parameters() if p.requires_grad]
    update_opt = torch.optim.Adam(update_params, lr=LR)
    first_batch = gamma[torch.as_tensor(rows[:BATCH])].to(device)
    before_update = {k: v.detach().clone() for k, v in update_core.state_dict().items()}
    update_loss, _, _ = loss_parts(update_core, first_batch, lossfn)
    update_opt.zero_grad(set_to_none=True)
    update_loss.backward()
    update_norm = torch.nn.utils.clip_grad_norm_(update_params, 1.0)
    if not torch.isfinite(update_loss) or not torch.isfinite(update_norm) or float(update_norm) <= 0:
        raise FloatingPointError("throwaway real-data Adam preflight has invalid loss/gradient")
    update_opt.step()
    after_update = update_core.state_dict()
    if not any(not torch.equal(v, after_update[k]) for k, v in before_update.items()
               if not k.startswith("graph_generator.base_graph." if arm == "xy_candidate" else "graph_generator.")):
        raise AssertionError("throwaway Adam preflight changed no trainable model parameter")
    frozen_prefix = "graph_generator.base_graph." if arm == "xy_candidate" else "graph_generator."
    if any(not torch.equal(v, after_update[k]) for k, v in before_update.items()
           if k.startswith(frozen_prefix)):
        raise AssertionError("throwaway Adam preflight changed frozen legacy graph")
    if arm == "xy_candidate" and torch.equal(before_update["graph_generator.xy_projection"],
                                              after_update["graph_generator.xy_projection"]):
        raise AssertionError("throwaway Adam preflight did not update XY projection")
    record = {"status": "passed", "seed": seed, "arm": arm,
              "source_core_sha256": sha(source), "source_manifest_sha256": sha(source_manifest),
              "implementation_sha256": implementation,
              "source_audit_sha256": sha(SOURCE_AUDIT),
              "source_evidence_manifest_sha256": sha(SOURCE_EVIDENCE_DIR / f"seed{seed}_manifest.json"),
              "encoder_sha256": sha(ASSETS / "input_encoder/input_layer_encoder.pt"),
              "preprocessing_sha256": sha(ASSETS / "feature_preprocessing.pt"),
              "gamma_train_sha256": sha(gamma_path),
              "gamma_train_manifest_sha256": sha(GAMMA_TRAIN_MANIFEST),
              "training_ids": ids.tolist(),
              "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
              "matched_shuffle_seed": manifest["seed"], "batch_size": BATCH,
              "time_steps": TRAIN_STEPS, "settle": TRAIN_SETTLE,
              "ground_truth_used": False, "batches": batches,
              "zero_p_legacy_equivalence": True, "native_drive_bitwise_unchanged": True,
              "strict_checkpoint_roundtrip_bitwise": True,
              "throwaway_b16_adam_update": True,
              "throwaway_update_gradient_norm_preclip": float(update_norm),
              "graph_frozen": True,
              "zero_initial_xy_projection": arm != "xy_candidate" or bool(torch.count_nonzero(core.graph_generator.xy_projection) == 0)}
    return record


def train(seed, arm, output, device="cuda", steps=UPDATES):
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"preserve existing attempt without overwrite: {out}")
    source_preflight = ARCHIVE / f"preflight_seed{seed}_{arm}.json"
    if not source_preflight.is_file():
        raise FileNotFoundError(f"successful registered preflight required: {source_preflight}")
    pf = json.loads(source_preflight.read_text())
    if pf.get("status") != "passed" or pf.get("seed") != seed or pf.get("arm") != arm:
        raise AssertionError("preflight record does not match requested training task")
    if pf.get("implementation_sha256") != implementation_fingerprint():
        raise AssertionError("implementation changed since preflight; revalidate before training")
    torch.manual_seed(117 + seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(117 + seed)
    core, source, source_manifest, source_record = load_source(seed, arm, device)
    ids, rows = train_indices(seed)
    gamma, gamma_manifest = validate_gamma_cache(GAMMA_TRAIN, GAMMA_TRAIN_MANIFEST)
    params = [p for p in core.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    lossfn = criterion()
    source_graph = {k: v.detach().clone() for k, v in core.graph_generator.base_graph.state_dict().items()} if arm == "xy_candidate" else {k: v.detach().clone() for k, v in core.graph_generator.state_dict().items()}
    out.mkdir(parents=True, exist_ok=False)
    write(out / "manifest.json", {"status": "training", "seed": seed, "arm": arm,
          "source_core": str(source), "source_core_sha256": sha(source),
          "source_manifest_sha256": sha(source_manifest), "training_ids": ids.tolist(),
          "source_audit_sha256": sha(SOURCE_AUDIT),
          "source_evidence_manifest_sha256": sha(SOURCE_EVIDENCE_DIR / f"seed{seed}_manifest.json"),
          "encoder_sha256": sha(ASSETS / "input_encoder/input_layer_encoder.pt"),
          "preprocessing_sha256": sha(ASSETS / "feature_preprocessing.pt"),
          "gamma_train_sha256": sha(GAMMA_TRAIN),
          "gamma_train_manifest_sha256": sha(GAMMA_TRAIN_MANIFEST),
          "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
          "training_source_manifest": str(source_manifest), "matched_shuffle_seed": 117 + seed,
          "updates": steps, "batch_size": BATCH, "lr": LR, "time_steps": TRAIN_STEPS,
          "settle": TRAIN_SETTLE, "loss": "phase_primary + 5 * positive_actual_spike_product",
          "ground_truth_used_for_training": False,
          "xy_definition": "16x16 patch-center coordinates in [-1,1]; P[16,2] added before node-feature normalization",
          "xy_init": "exact zeros without RNG consumption" if arm == "xy_candidate" else None})
    history = []
    core.train()
    core.graph_generator.eval()
    for step in range(steps):
        ix = rows[step * BATCH:(step + 1) * BATCH]
        total, primary, positive = loss_parts(core, gamma[torch.as_tensor(ix)].to(device), lossfn)
        if not torch.isfinite(total):
            raise FloatingPointError(f"nonfinite loss seed={seed} arm={arm} update={step+1}")
        opt.zero_grad(set_to_none=True)
        total.backward()
        grads = [p for p in params if p.grad is not None]
        if not grads or any(not torch.isfinite(p.grad).all() for p in grads):
            raise FloatingPointError(f"nonfinite/empty gradient at update {step+1}")
        xy_norm = None
        if arm == "xy_candidate":
            pg = core.graph_generator.xy_projection.grad
            if pg is None or not torch.isfinite(pg).all():
                raise FloatingPointError("invalid coordinate projection gradient")
            xy_norm = float(pg.norm())
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(norm) or float(norm) <= 0:
            raise FloatingPointError("invalid clipped gradient norm")
        opt.step()
        history.append({"update": step + 1, "loss": float(total.detach()),
                        "primary": float(primary.detach()),
                        "positive_actual_spike_product": float(positive.detach()),
                        "gradient_norm_preclip": float(norm), "xy_gradient_norm": xy_norm})
        if step == 255:
            torch.save(core.state_dict(), out / "prefix_256_core.pt")
        if step == 0 or (step + 1) % 32 == 0:
            write(out / "progress.json", {"status": "training", "seed": seed,
                  "arm": arm, "update": step + 1, "total_updates": steps})
    graph_after = core.graph_generator.base_graph.state_dict() if arm == "xy_candidate" else core.graph_generator.state_dict()
    if any(not torch.equal(v, graph_after[k]) for k, v in source_graph.items()):
        raise AssertionError("frozen legacy graph parameters changed")
    if any(not torch.isfinite(p).all() for p in core.parameters()):
        raise FloatingPointError("nonfinite final model parameter")
    torch.save(core.state_dict(), out / "core.pt")
    write(out / "history.json", history)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest.update(status="training_complete", completed=time.time(),
                    changed_xy=arm == "xy_candidate" and bool(torch.count_nonzero(core.graph_generator.xy_projection).item()))
    write(out / "manifest.json", manifest)


def _eval_core(device, checkpoint, steps, xy):
    core = make_core(device, steps)
    if xy:
        attach_xy_graph(core, grid_size=16)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    core.load_state_dict(state, strict=True)
    core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
    return core.eval()


def evaluate(seed, arm, checkpoint, output, device="cuda"):
    """Run the shared fixed classifier/evaluator without changing its scoring code."""
    import evaluate_fixed_split as fixed
    old_factory, old_argv = fixed._core, sys.argv[:]
    fixed._core = lambda dev, ckpt, steps, *args, **kwargs: _eval_core(
        dev, ckpt, steps, xy=(arm == "xy_candidate"))
    evaluator_path = ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py"
    try:
        spec = importlib.util.spec_from_file_location("sw0110_shared_evaluate", evaluator_path)
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        val_gamma, val_manifest = validate_gamma_cache(GAMMA_VAL, GAMMA_VAL_MANIFEST, validation=True)
        del val_gamma
        sys.argv = [str(evaluator_path), "--checkpoint", str(checkpoint),
                "--gamma-path", str(GAMMA_VAL), "--gamma-global-start", "1320",
                "--gamma-manifest", str(GAMMA_VAL_MANIFEST),
                "--dataset-path", str(DATASET), "--output-path", str(output),
                "--start", "1320", "--count", "320", "--steps", "1024",
                "--settle", "512", "--thresholds", ".50", "--min-group-size", "2",
                "--background", "largest_component", "--dendritic-projection", "shared",
                "--graph-spatial-decay", ".35", "--geodesic-steps", "3",
                "--geodesic-radius", "1.5", "--geodesic-contrast", "2",
                "--geodesic-temperature", ".5", "--geodesic-cap", "16",
                "--kuramoto-backend", "factorized", "--gate-mode", "raw",
                "--batch-size", "8", "--membrane-vth", ".06", "--device", str(device)]
        evaluator.main()
    finally:
        fixed._core = old_factory
        sys.argv = old_argv


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
        record = preflight(args.seed, args.arm, device=args.device)
        write(args.output, record)
    elif args.command == "train":
        train(args.seed, args.arm, args.output, args.device, args.steps)
    else:
        evaluate(args.seed, args.arm, args.checkpoint, args.output, args.device)
    print(json.dumps({"status": "complete", "command": args.command,
                      "seed": args.seed, "arm": args.arm}, flush=True))


if __name__ == "__main__":
    main()
