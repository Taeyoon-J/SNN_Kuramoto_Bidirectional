"""Fixed-320 endpoint for SW0123; predictions are persisted before GT is opened."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0123_adaptive_temporal_assignment import run
from collaborative_test.SW_0123_adaptive_temporal_assignment.model import AdaptiveTemporalRGBModel
from collaborative_test.SW_0123_adaptive_temporal_assignment.readout import assignment_to_labels
from collaborative_test.SW_0115_analytic_partition_rgb.loss import production_partition
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks

EVAL_IDS = np.arange(1320, 1640, dtype=np.int64)
EVAL_BATCH, EVAL_STEPS, EVAL_SETTLE = 8, 1024, 512
DATASET = Path("/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5")
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def _write_once(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _save_torch_once(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _finite_score(score):
    try:
        if set(score["per_image"]) != set(METRICS) or set(score["metrics"]) != set(METRICS):
            return False
        for metric in METRICS:
            vals = np.asarray(score["per_image"][metric], dtype=np.float64)
            mean = float(score["metrics"][metric])
            if (vals.shape != (320,) or not np.isfinite(vals).all()
                    or int(score["valid_count"][metric]) != 320 or not math.isfinite(mean)
                    or abs(float(vals.mean()) - mean) > 1e-12):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def evaluation_fingerprint():
    files = [HERE / "evaluate.py", HERE / "run.py", HERE / "protocol.json", HERE / "model.py",
             HERE / "adaptive_dynamics.py", HERE / "readout.py",
             ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
             ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/loss.py",
             ROOT / "collaborative_test/SW_0117_joint_analytic_rgb/run.py",
             ROOT / "collaborative_test/SW_0122_joint_rgb_seed_replication/run.py",
             ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
             ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
             ROOT / "snn_kuramoto_bidirectional/evaluation.py",
             ROOT / "snn_kuramoto_bidirectional/membrane_layer.py",
             ROOT / "snn_kuramoto_bidirectional/dendric_layer.py",
             ROOT / "snn_kuramoto_bidirectional/sinusoidal_gating.py"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): run.sha(path) for path in files}


@torch.no_grad()
def _predict_batch(core, assignment_head, gamma, arm, settle=EVAL_SETTLE):
    """Variable-horizon rollout; keeps the training runner's 64-step API fixed."""
    horizon = int(getattr(core, "num_time_steps", -1))
    if horizon < 1 or gamma.ndim != 3 or tuple(gamma.shape[1:]) != (8, 256):
        raise ValueError("evaluation requires [B,8,256] gamma and a positive registered core horizon")
    if horizon <= settle or not getattr(core, "spike_per_component", False):
        raise ValueError("evaluation requires four-component traces longer than settle")
    gate_rows = []
    hook = None
    if arm == "gate_only_control":
        hook = core.membrane_layer.register_forward_pre_hook(
            lambda _module, args: gate_rows.append(args[1].detach()))
    try:
        _groups, _mean_spikes, _core_out, _theta = core(
            gamma, return_core_out=True, return_theta=True)
    finally:
        if hook is not None:
            hook.remove()
    components = core.last_component_spikes
    expected = (gamma.shape[0], 4, 256, horizon)
    if components is None or tuple(components.shape) != expected or not torch.isfinite(components).all():
        raise AssertionError(f"evaluation component-spike shape/finite check failed: {None if components is None else tuple(components.shape)}")
    if arm == "gate_only_control":
        if len(gate_rows) != horizon:
            raise AssertionError(f"captured {len(gate_rows)} membrane gates, expected {horizon}")
        folded = torch.stack(gate_rows, dim=-1)
        if tuple(folded.shape) != (gamma.shape[0] * 4, 256, horizon):
            raise AssertionError("captured gate does not retain the production B*4 row fold")
        gates = folded.reshape(gamma.shape[0], 4, 256, horizon)
        if not all(torch.equal(gates[:, 0], gates[:, d]) for d in range(1, 4)):
            raise AssertionError("gate-only control captured inconsistent repeated scalar gates")
        head_input = gates[..., settle:]
    elif arm in ("legacy_full", "adaptive_full"):
        head_input = components[..., settle:]
    else:
        raise ValueError(f"unknown SW0123 evaluation arm: {arm}")
    probability, _slot_embedding, _patch_embedding = assignment_head(head_input)
    primary = assignment_to_labels(probability).detach().cpu()
    qcc, _hard = production_partition(components.mean(dim=1), components, settle=settle)
    return primary, qcc.reshape(gamma.shape[0], 16, 16).detach().cpu(), components


def _serialize_score(prediction: torch.Tensor, target: torch.Tensor):
    scored = evaluate_patch_masks(prediction, target)
    per_image = {}
    means = {}
    counts = {}
    for metric in METRICS:
        values = scored["per_image"][metric].detach().cpu().to(torch.float64).numpy()
        per_image[metric] = [float(x) if math.isfinite(float(x)) else None for x in values]
        value = float(scored["mean"][metric])
        means[metric] = value if math.isfinite(value) else None
        counts[metric] = int(scored["valid_count"][metric])
    return {"metrics": means, "valid_count": counts, "per_image": per_image}


def _training_artifacts(seed, arm, checkpoint):
    checkpoint = Path(checkpoint).resolve()
    expected_checkpoint = (run.OUT / f"seed{seed}_{arm}" / "core.pt").resolve()
    if checkpoint != expected_checkpoint:
        raise AssertionError("evaluation must use the canonical immutable SW0123 arm checkpoint")
    folder = checkpoint.parent
    manifest_path = folder / "manifest.json"
    required = [checkpoint, folder / "encoder.pt", folder / "assignment_head.pt",
                folder / "rgb_decoder.pt", folder / "core_optimizer.pt",
                folder / "head_optimizer.pt", folder / "history.json",
                folder / "TRAINING_COMPLETED", manifest_path]
    if any(not p.is_file() for p in required):
        raise FileNotFoundError("SW0123 endpoint requires complete core/encoder/head/optimizer/history artifacts")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = run.implementation_fingerprint()
    history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
    preflight_path = run.ARCHIVE / f"preflight_seed{seed}_{arm}.json"
    warmup_path = run._warmup_path(seed, arm)
    if not preflight_path.is_file() or not warmup_path.is_file():
        raise FileNotFoundError("completed endpoint is missing its immutable preflight or warmup evidence")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    warmup = torch.load(warmup_path, map_location="cpu", weights_only=False)
    source, source_manifest, _source_record, ids, rows = run.source_contract(seed)
    ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    if (manifest.get("status") != "training_complete" or manifest.get("experiment") != "SW0123"
            or manifest.get("seed") != seed or manifest.get("arm") != arm
            or manifest.get("implementation_fingerprint") != fingerprint
            or manifest.get("source_core_sha256") != run.sha(source)
            or manifest.get("source_manifest_sha256") != run.sha(source_manifest)
            or manifest.get("training_ids") != ids.tolist()
            or manifest.get("training_ids_sha256") != ids_sha
            or manifest.get("gamma_rows") != rows.tolist()
            or manifest.get("matched_shuffle_seed") != 117 + seed
            or manifest.get("updates") != 256 or manifest.get("batch_size") != 16
            or manifest.get("time_steps") != 64 or manifest.get("settle") != 32
            or manifest.get("ground_truth_used_for_training") is not False
            or manifest.get("preflight_sha256") != run.sha(preflight_path)
            or manifest.get("warmup_artifact_sha256") != run.sha(warmup_path)
            or preflight.get("status") != "passed" or preflight.get("seed") != seed
            or preflight.get("arm") != arm
            or preflight.get("implementation_fingerprint") != fingerprint
            or warmup.get("status") != "passed" or warmup.get("seed") != seed
            or warmup.get("arm") != arm
            or warmup.get("implementation_fingerprint") != fingerprint
            or len(history) != 256
            or [int(row.get("update", -1)) for row in history] != list(range(1, 257))):
        raise AssertionError("training manifest/history does not match completed SW0123 recipe")
    file_keys = {"core": checkpoint, "encoder": folder / "encoder.pt",
                 "assignment_head": folder / "assignment_head.pt", "rgb_decoder": folder / "rgb_decoder.pt",
                 "core_optimizer": folder / "core_optimizer.pt",
                 "head_optimizer": folder / "head_optimizer.pt", "history": folder / "history.json"}
    hashes = {name: run.sha(path) for name, path in file_keys.items()}
    if any(manifest.get(name + "_sha256") != digest for name, digest in hashes.items()):
        raise AssertionError("training artifact hash mismatch")
    guards = manifest.get("final_training_guards", {})
    return checkpoint, folder, manifest_path, manifest, hashes, guards


def _load_eval_model(seed, arm, device, checkpoint, folder, manifest):
    source, source_manifest, _source_record, ids, _rows = run.source_contract(seed)
    if (run.sha(source) != manifest.get("source_core_sha256")
            or run.sha(source_manifest) != manifest.get("source_manifest_sha256")
            or ids.tolist() != manifest.get("training_ids")):
        raise AssertionError("trained endpoint no longer binds the registered SW0097 source/order")
    calibration_path, calibration, kappa, rms = run.load_calibration(seed)
    if (run.sha(calibration_path) != manifest.get("calibration_sha256")
            or manifest.get("kappa_sha256") != calibration.get("kappa_sha256")
            or manifest.get("spike_rms_sha256") != calibration.get("spike_rms_sha256")):
        raise AssertionError("trained adaptive buffers do not bind the passed source calibration")

    # Obtain the registered encoder architecture and preprocessing from the
    # same strict loader used at training, then load its trained endpoint.
    _source_core, encoder, patcher, mean, std, clip, *_ = run.load_training_assets(seed, device)
    encoder_path = folder / "encoder.pt"
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True), strict=True)
    encoder.eval(); patcher.eval()

    core = run.base.make_core(device, EVAL_STEPS)
    core.load_state_dict(torch.load(source, map_location=device, weights_only=True), strict=True)
    core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.shape[0])]
    if arm == "adaptive_full":
        run.configure_arm(core, arm, kappa)
    core.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.shape[0])]
    if (core.graph_generator.uses_feedback or core.kuramoto.spike_pulse_gain is not None
            or not getattr(core, "spike_per_component", False)):
        raise AssertionError("evaluation source has unsupported feedback or component layout")

    model = AdaptiveTemporalRGBModel().to(device)
    model.assignment_head.load_state_dict(
        torch.load(folder / "assignment_head.pt", map_location=device, weights_only=True), strict=True)
    model.rgb_decoder.load_state_dict(
        torch.load(folder / "rgb_decoder.pt", map_location=device, weights_only=True), strict=True)
    model.assignment_head.eval(); model.rgb_decoder.eval()
    if not torch.equal(model.assignment_head.spike_rms.cpu(), rms.cpu()):
        raise AssertionError("trained assignment head does not use the shared calibrated RMS")
    return core.eval(), encoder, patcher, mean, std, clip, model.assignment_head


def _validate_rgb_cache():
    cache, meta, cache_sha = run.sw117.validate_rgb_validation_cache()
    if cache.shape != (320, 128, 128, 3) or cache.dtype != np.uint8:
        raise AssertionError("registered validation RGB cache shape/dtype mismatch")
    return cache, meta, cache_sha


def evaluate(seed: int, arm: str, checkpoint, output, device="cuda"):
    if seed not in run.SEEDS or arm not in run.ARMS:
        raise ValueError("unregistered SW0123 evaluation seed/arm")
    checkpoint, output = Path(checkpoint), Path(output)
    sidecar = output.parent / "evaluation_manifest.json"
    predictions_path = output.parent / "frozen_predictions.pt"
    gamma_path = output.parent / "trained_encoder_gamma.pt"
    if any(path.exists() for path in (output.parent, output, sidecar, predictions_path, gamma_path)):
        raise FileExistsError("preserving existing SW0123 endpoint attempt; refusing overwrite")
    checkpoint, train_folder, train_manifest_path, manifest, hashes, final_guards = _training_artifacts(
        seed, arm, checkpoint)
    fingerprint = evaluation_fingerprint()
    rgb_cache, rgb_meta, rgb_cache_sha = _validate_rgb_cache()
    core, encoder, patcher, mean, std, clip, assignment_head = _load_eval_model(
        seed, arm, device, checkpoint, train_folder, manifest)
    output.parent.mkdir(parents=True, exist_ok=False)
    primary_batches, qcc_batches, gamma_batches = [], [], []
    for start in range(0, len(EVAL_IDS), EVAL_BATCH):
        local = np.arange(start, start + EVAL_BATCH, dtype=np.int64)
        images = run.sw117.read_rgb(rgb_cache, local, device)
        gamma = run.sw117.encode_rgb(encoder, patcher, mean, std, clip, images)
        if tuple(gamma.shape) != (EVAL_BATCH, 8, 256) or not torch.isfinite(gamma).all():
            raise AssertionError("trained encoder produced invalid fixed-320 gamma")
        out_primary, out_qcc, _components = _predict_batch(
            core, assignment_head, gamma, arm, settle=EVAL_SETTLE)
        primary_batches.append(out_primary)
        qcc_batches.append(out_qcc)
        gamma_batches.append(gamma.detach().cpu())
    primary = torch.cat(primary_batches, dim=0)
    qcc = torch.cat(qcc_batches, dim=0)
    trained_gamma = torch.cat(gamma_batches, dim=0)
    if (tuple(primary.shape) != (320, 16, 16) or tuple(qcc.shape) != (320, 16, 16)
            or tuple(trained_gamma.shape) != (320, 8, 256)):
        raise AssertionError("fixed-320 GT-free prediction arrays have unexpected shapes")
    prediction_record = {
        "experiment": "SW0123", "seed": seed, "arm": arm,
        "image_ids": [1320, 1639], "count": 320,
        "primary_readout": "argmax assignment slot; largest slot background; first-patch tie break; slots smaller than2 background",
        "secondary_readout": "actual component-spike QCC at .50/min2/largest_component",
        "primary_labels": primary, "qcc_labels": qcc,
        "checkpoint_sha256": hashes["core"], "encoder_checkpoint_sha256": hashes["encoder"],
        "assignment_head_sha256": hashes["assignment_head"], "rgb_decoder_sha256": hashes["rgb_decoder"],
        "training_manifest_sha256": run.sha(train_manifest_path),
        "trained_encoder_gamma_sha256_pending": None,
        "ground_truth_used_for_prediction": False,
    }
    _save_torch_once(predictions_path, prediction_record)
    _save_torch_once(gamma_path, {"image_ids": [1320, 1639], "gamma": trained_gamma,
                                 "encoder_sha256": hashes["encoder"],
                                 "ground_truth_used_for_gamma": False})
    prediction_sha, gamma_sha = run.sha(predictions_path), run.sha(gamma_path)

    # Do not open validation masks until all predictions and trained-encoder
    # features are on disk and their final bytes have been hashed.
    if prediction_sha != run.sha(predictions_path) or gamma_sha != run.sha(gamma_path):
        raise AssertionError("persisted fixed-320 predictions/gamma failed hash verification")
    with h5py.File(DATASET, "r") as dataset:
        if "mask" not in dataset or dataset["mask"].shape[0] != 100000:
            raise AssertionError("canonical HDF5 mask dataset is absent or has wrong physical length")
        masks = np.asarray(dataset["mask"][EVAL_IDS])
    target = clevr_mask_patch(torch.from_numpy(masks), 8)["patch_labels"].cpu()
    primary_score = _serialize_score(primary, target)
    qcc_score = _serialize_score(qcc, target)
    report = {
        "status": "complete", "experiment": "SW0123", "seed": seed, "arm": arm,
        "ids": [1320, 1639], "images": 320,
        "evaluation_contract": {"batch_size": EVAL_BATCH, "time_steps": EVAL_STEPS,
            "settle": EVAL_SETTLE, "readout_threshold": 0.50, "min_group_size": 2,
            "background": "largest_component", "ground_truth_used_for_prediction": False},
        "primary_assignment": primary_score,
        "secondary_actual_qcc": qcc_score,
        "metrics_valid": _finite_score(primary_score) and _finite_score(qcc_score),
        "prediction_sha256": prediction_sha, "prediction_file": predictions_path.name,
        "trained_encoder_gamma_sha256": gamma_sha, "trained_encoder_gamma_file": gamma_path.name,
        "validation_rgb_cache_sha256": rgb_cache_sha,
        "validation_rgb_manifest_sha256": run.sha(run.sw117.VAL_RGB_MANIFEST),
        "validation_dataset_path": str(DATASET), "validation_dataset_size_bytes": DATASET.stat().st_size,
        "training_manifest_sha256": run.sha(train_manifest_path),
        "training_artifact_sha256": hashes,
        "final_training_guards": final_guards,
        "promotion_eligible_training_guards": final_guards.get("status") == "passed",
        "evaluation_implementation_fingerprint": fingerprint,
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_training": False,
        "finished": time.time(),
    }
    _write_once(output, report)
    _write_once(sidecar, {
        "experiment": "SW0123", "seed": seed, "arm": arm,
        "evaluation_sha256": run.sha(output), "prediction_sha256": prediction_sha,
        "trained_encoder_gamma_sha256": gamma_sha,
        "training_manifest_sha256": run.sha(train_manifest_path),
        "validation_rgb_cache_sha256": rgb_cache_sha,
        "validation_rgb_manifest_sha256": run.sha(run.sw117.VAL_RGB_MANIFEST),
        "validation_dataset_size_bytes": DATASET.stat().st_size,
        "core_sha256": hashes["core"], "encoder_sha256": hashes["encoder"],
        "assignment_head_sha256": hashes["assignment_head"], "rgb_decoder_sha256": hashes["rgb_decoder"],
        "evaluation_implementation_fingerprint": fingerprint,
        "contract": report["evaluation_contract"],
    })
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=run.SEEDS, required=True)
    parser.add_argument("--arm", choices=run.ARMS, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate(args.seed, args.arm, args.checkpoint, args.output, args.device)
    print(json.dumps({"status": report["status"], "experiment": "SW0123",
                      "seed": args.seed, "arm": args.arm, "output": str(args.output),
                      "metrics_valid": report["metrics_valid"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
