"""Frozen 320-image original-spike-QCC evaluation for completed SW0133 arms."""
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
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"),
                str(ROOT / "collaborative_test")]

from collaborative_test.SW_0133_soft_partition_rgb import preflight_queue, run, train
from collaborative_test.SW_0110_xy_graph_route import run as source97
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks

EVAL_STEPS = 1024
EVAL_SETTLE = 512
EVAL_BATCH = 8
IMAGE_IDS = list(range(1320, 1640))
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")


def evaluation_fingerprint():
    paths = (
        HERE / "evaluate.py", HERE / "train.py", HERE / "run.py", HERE / "soft_partition.py",
        HERE / "protocol.json", HERE / "preflight_queue.py",
        ROOT / "collaborative_test/SW_0106_spike_partition_rgb/partition_rgb.py",
        ROOT / "collaborative_test/SW_0132_partition_relative_rgb/relative_rgb.py",
        ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
        ROOT / "snn_kuramoto_bidirectional/evaluation.py",
        ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
        ROOT / "snn_kuramoto_bidirectional/training/train_s2net_core.py",
    )
    return {path.relative_to(ROOT).as_posix(): run.sha(path) for path in paths}


def _source_eval_reference(seed):
    path = source97.SOURCE_ROOT / f"seed{seed}_positive_frozen" / "evaluation.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("images") != 320 or data.get("ids") != [1320, 1639]:
        raise ValueError("source97 reference does not use the fixed validation endpoint")
    rows = data.get("sweep")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source97 reference has no frozen score sweep")
    scored = rows[0].get("scored_targets", {}).get("our_hdf5")
    if not isinstance(scored, dict):
        raise ValueError("source97 report lacks its canonical HDF5 metric record")
    result = {}
    for metric in METRICS:
        values = scored.get("per_image", {}).get(metric)
        mean = float(scored.get("metrics", {}).get(metric, float("nan")))
        count = int(scored.get("valid_count", {}).get(metric, -1))
        if (count != 320 or not isinstance(values, list) or len(values) != 320
                or not all(math.isfinite(float(x)) for x in values)
                or not math.isfinite(mean)
                or abs(sum(map(float, values)) / 320.0 - mean) > 1e-10):
            raise ValueError(f"source97 per-image {metric} evidence is incomplete or inconsistent")
        result[metric] = {"mean": mean, "valid_count": count,
                          "per_image": [float(x) for x in values]}
    return path, run.sha(path), result


def _training_artifacts(seed, arm, training_dir, assets=None):
    training_dir = Path(training_dir)
    manifest_path = training_dir / "manifest.json"
    marker = training_dir / "TRAINING_COMPLETED"
    if not manifest_path.is_file() or not marker.is_file():
        raise FileNotFoundError("completed SW0133 manifest and marker are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "training_complete"
            or manifest.get("experiment") != "SW0133_soft_partition_rgb"
            or manifest.get("seed") != seed or manifest.get("arm") != arm
            or manifest.get("updates") != run.sw130.UPDATES
            or manifest.get("batch_size") != run.BATCH
            or manifest.get("train_time_steps") != run.STEPS
            or manifest.get("settle_steps") != run.SETTLE
            or manifest.get("ground_truth_used_for_training") is not False):
        raise ValueError("training manifest does not match the registered completed run")
    required = ("core.pt", "integration.pt", "encoder.pt", "decoder.pt", "optimizers.pt", "history.json")
    artifact_hashes = {}
    declared = manifest.get("artifact_sha256", {})
    for name in required:
        path = training_dir / name
        expected_sha = declared.get(name)
        if not path.is_file() or not expected_sha or run.sha(path) != expected_sha:
            raise ValueError(f"training artifact missing or changed: {name}")
        artifact_hashes[name] = expected_sha
    if manifest.get("implementation_fingerprint") != run.implementation_fingerprint():
        raise ValueError("preflight implementation fingerprint changed after training")
    if manifest.get("trainer_sha256") != run.sha(HERE / "train.py"):
        raise ValueError("training manifest does not bind the current trainer")
    if manifest.get("preflight_queue_sha256") != run.sha(HERE / "preflight_queue.py"):
        raise ValueError("training manifest does not bind the preflight validator")
    if assets is None:
        assets = run.sw130.validate_rgb_assets()
    if manifest.get("asset_hashes") != assets:
        raise ValueError("training manifest RGB/encoder asset provenance changed")
    preflights = train._passed_preflights(assets)
    history = json.loads((training_dir / "history.json").read_text(encoding="utf-8"))
    if len(history) != run.sw130.UPDATES:
        raise ValueError("training history does not contain all registered updates")
    for index, row in enumerate(history, start=1):
        if row.get("update") != index or row.get("assignment_live") != (arm != "phase_detached"):
            raise ValueError("training history ordering or arm objective changed")
        if any(not math.isfinite(float(row[key])) for key in
               ("total", "old_objective", "phase_primary", "positive_product_spike",
                "full_rgb_reconstruction", "joint_grad_norm_preclip", "decoder_grad_norm_preclip")):
            raise ValueError("training history contains nonfinite values")
    source_path, source_manifest_path, source_manifest, pool, ids, source_sha = run.source_contract(seed)
    ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    pool_sha = hashlib.sha256(np.asarray(pool, dtype="<i8").tobytes()).hexdigest()
    if (manifest.get("source_core_sha256") != source_sha
            or manifest.get("source_manifest_sha256") != run.sha(source_manifest_path)
            or manifest.get("source_steps") != source_manifest["steps"]
            or manifest.get("training_ids") != ids
            or manifest.get("training_ids_sha256") != ids_sha
            or manifest.get("shuffle_seed") != 117 + seed
            or manifest.get("pool_indices_sha256") != pool_sha):
        raise ValueError("training source/checkpoint/data-order provenance mismatch")
    preflight_path = run.ARCHIVE / f"preflight_seed{seed}.json"
    if (not preflight_path.is_file()
            or manifest.get("preflight_sha256") != run.sha(preflight_path)
            or manifest.get("preflight_sha256") != preflights[seed]["sha256"]
            or manifest.get("preflight_implementation_fingerprint") != run.implementation_fingerprint()):
        raise ValueError("training manifest does not bind its passed preflight")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    seed0_path = run.ARCHIVE / "preflight_seed0.json"
    if (not seed0_path.is_file() or manifest.get("seed0_lambda_record_sha256") != run.sha(seed0_path)
            or manifest.get("lambda") != float(json.loads(seed0_path.read_text(encoding="utf-8"))["lambda"])
            or preflight.get("lambda") != manifest.get("lambda")):
        raise ValueError("training manifest does not bind the frozen shared seed0 coefficient")
    if manifest.get("decoder_warmup_sha256") != run.sha(run.ARCHIVE / f"preflight_decoder_seed{seed}.pt"):
        raise ValueError("training manifest warmed decoder provenance changed")
    return manifest_path, manifest, artifact_hashes, source_sha


def _predict_all(seed, arm, device, training_dir, prediction_path, gamma_path,
                 validated_training=None, source_reference=None, assets=None):
    if validated_training is None:
        validated_training = _training_artifacts(seed, arm, training_dir, assets)
    manifest_path, manifest, artifact_hashes, source_sha = validated_training
    if source_reference is None:
        source_reference = _source_eval_reference(seed)
    source_eval_path, source_eval_sha, source_metrics = source_reference
    if prediction_path.exists() or gamma_path.exists():
        raise FileExistsError("preserve any existing SW0133 frozen prediction artifacts")
    val_cache = np.load(run.sw130.VAL_RGB, mmap_mode="r")
    wrapped, encoder, patcher, mean, std, clip, decoder, pool, ids, loaded_sha = \
        run.load_models(seed, device, arm)
    if loaded_sha != source_sha:
        raise AssertionError("evaluation loaded a different SW0097 source core")
    trained = Path(training_dir)
    wrapped.core.load_state_dict(torch.load(trained / "core.pt", map_location=device,
                                           weights_only=True), strict=True)
    integration = torch.load(trained / "integration.pt", map_location=device, weights_only=True)
    if integration.get("arm") != wrapped.arm:
        raise ValueError("trained phase/constant integration arm mismatch")
    with torch.no_grad():
        for name in ("a_d", "a_m", "b"):
            value = integration[name].to(device)
            target = getattr(wrapped, name)
            if value.shape != target.shape or not torch.isfinite(value).all():
                raise ValueError(f"trained integration parameter {name} has invalid shape/value")
            target.copy_(value)
    encoder.load_state_dict(torch.load(trained / "encoder.pt", map_location=device,
                                      weights_only=True), strict=True)
    decoder.load_state_dict(torch.load(trained / "decoder.pt", map_location=device,
                                       weights_only=True), strict=True)
    wrapped.eval(); encoder.eval(); decoder.eval()
    wrapped.core.num_time_steps = EVAL_STEPS
    criterion = run.sw130.make_criterion()
    prediction_batches, gamma_batches = [], []
    with torch.no_grad():
        for start in range(0, len(IMAGE_IDS), EVAL_BATCH):
            end = min(start + EVAL_BATCH, len(IMAGE_IDS))
            images = run.sw130.read_batch(val_cache, list(range(start, end)), device)
            gamma = run.sw130.encode(encoder, patcher, mean, std, clip, images)
            if tuple(gamma.shape) != (end - start, 8, 256) or not torch.isfinite(gamma).all():
                raise ValueError("trained-encoder validation gamma has invalid shape/value")
            result = run.sw130._forward_with_plv(wrapped, gamma, criterion,
                                                 EVAL_SETTLE, "phase", "mean")
            groups, spikes, _core_out, _plv, theta = result
            components = wrapped.core.last_component_spikes
            if components is None or tuple(components.shape) != (end-start, 4, 256, EVAL_STEPS):
                raise ValueError("full-horizon component spikes have invalid shape")
            q = run.sw130.spike_synchrony_affinity(
                components.mean(dim=1), components, settle=EVAL_SETTLE, affinity_mode="spike")
            if not torch.isfinite(q).all() or not torch.isfinite(theta).all() \
                    or not torch.isfinite(spikes).all():
                raise FloatingPointError("nonfinite SW0133 full-horizon prediction trace")
            labels, _hard, detected = run.sw130.hard_labels(spikes, components, EVAL_SETTLE)
            if len(detected) != end-start or tuple(labels.shape) != (end-start, 256):
                raise ValueError("original QCC returned malformed labels")
            prediction_batches.append(labels.detach().cpu().reshape(end-start, 16, 16))
            gamma_batches.append(gamma.detach().cpu())
    predictions = torch.cat(prediction_batches).to(torch.int64).contiguous()
    gamma_all = torch.cat(gamma_batches).contiguous()
    if tuple(predictions.shape) != (320, 16, 16) or tuple(gamma_all.shape) != (320, 8, 256):
        raise ValueError("incomplete fixed-320 evaluation predictions/gamma")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("xb") as stream:
        torch.save({"labels": predictions, "image_ids": IMAGE_IDS, "seed": seed,
                    "arm": arm, "training_manifest_sha256": run.sha(manifest_path),
                    "source_core_sha256": source_sha,
                    "ground_truth_used_for_prediction": False}, stream)
    with gamma_path.open("xb") as stream:
        torch.save({"gamma": gamma_all, "image_ids": IMAGE_IDS,
                    "encoder_sha256": artifact_hashes["encoder.pt"],
                    "ground_truth_used_for_prediction": False}, stream)
    prediction_sha, gamma_sha = run.sha(prediction_path), run.sha(gamma_path)
    persisted = torch.load(prediction_path, map_location="cpu", weights_only=True)
    persisted_gamma = torch.load(gamma_path, map_location="cpu", weights_only=True)
    if (persisted.get("image_ids") != IMAGE_IDS or persisted.get("seed") != seed
            or persisted.get("arm") != arm or persisted.get("ground_truth_used_for_prediction") is not False
            or not torch.equal(persisted.get("labels"), predictions)
            or persisted_gamma.get("image_ids") != IMAGE_IDS
            or tuple(persisted_gamma.get("gamma").shape) != (320, 8, 256)
            or run.sha(prediction_path) != prediction_sha or run.sha(gamma_path) != gamma_sha):
        raise ValueError("persisted frozen prediction/gamma hash or content verification failed")
    return {"manifest_path": str(manifest_path.resolve()), "manifest": manifest,
            "training_artifact_sha256": artifact_hashes, "source_core_sha256": source_sha,
            "source_evaluation_path": str(source_eval_path.resolve()),
            "source_evaluation_sha256": source_eval_sha, "source_metrics": source_metrics,
            "predictions": predictions, "prediction_path": prediction_path,
            "prediction_sha256": prediction_sha, "gamma_path": gamma_path,
            "gamma_sha256": gamma_sha}


def evaluate(seed, arm, device, output_dir, training_dir=None):
    if seed != train.PILOT_SEED or arm not in run.ARMS:
        raise ValueError("SW0133 evaluation is restricted to registered seed1 pilot arms")
    device, output_dir = torch.device(device), Path(output_dir)
    training_dir = Path(training_dir) if training_dir else run.ROOT / "trained_models" / \
        "SW0133_soft_partition_rgb" / f"seed{seed}_{arm}"
    if output_dir.exists():
        raise FileExistsError(f"preserving existing evaluation attempt: {output_dir}")
    assets = run.sw130.validate_rgb_assets()
    validated_training = _training_artifacts(seed, arm, training_dir, assets)
    source_reference = _source_eval_reference(seed)
    manifest_path, manifest, artifact_hashes, source_sha = validated_training
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_path, gamma_path = output_dir / "frozen_predictions.pt", output_dir / "gamma_validation.pt"
    started = time.time()
    try:
        evidence = _predict_all(seed, arm, device, training_dir, prediction_path, gamma_path,
                                validated_training, source_reference, assets)
        # No validation masks are opened until every prediction/gamma is stored and rehashed.
        import h5py
        with h5py.File(source97.DATASET, "r") as h5:
            masks = np.asarray(h5["mask"][1320:1640])
        if masks.shape not in ((320, 128, 128), (320, 128, 128, 1)):
            raise ValueError(f"canonical validation mask slice has unexpected shape {masks.shape}")
        target = clevr_mask_patch(torch.as_tensor(masks, dtype=torch.int64), 8)["patch_labels"]
        scores = evaluate_patch_masks(evidence["predictions"], target.cpu())
        metrics = {}
        for name in METRICS:
            values = scores["per_image"][name].detach().cpu().numpy().astype(np.float64)
            finite = np.isfinite(values)
            valid_count = int(finite.sum())
            mean = float(values[finite].mean()) if valid_count else None
            if valid_count != 320 or mean is None or not math.isfinite(mean):
                raise ValueError(f"fixed-320 {name} metric is incomplete/nonfinite")
            metrics[name] = {"mean": mean, "valid_count": valid_count,
                             "per_image": [float(value) for value in values]}
        report = {
            "status": "complete", "experiment": "SW0133_soft_partition_rgb",
            "seed": seed, "arm": arm, "images": 320, "ids": [1320, 1639],
            "time_steps": EVAL_STEPS, "settle": EVAL_SETTLE, "batch_size": EVAL_BATCH,
            "readout": {"affinity_mode": "spike", "threshold": 0.50,
                        "minimum_group_size": 2, "background": "largest_component"},
            "scores": {"metrics": {key: row["mean"] for key, row in metrics.items()},
                       "valid_count": {key: row["valid_count"] for key, row in metrics.items()},
                       "per_image": {key: row["per_image"] for key, row in metrics.items()}},
            "source97_reference": {"evaluation_sha256": evidence["source_evaluation_sha256"],
                                   "metrics": {key: row["mean"]
                                               for key, row in evidence["source_metrics"].items()}},
            "frozen_predictions_path": str(prediction_path.resolve()),
            "frozen_predictions_sha256": evidence["prediction_sha256"],
            "gamma_path": str(gamma_path.resolve()), "gamma_sha256": evidence["gamma_sha256"],
            "training_manifest_path": evidence["manifest_path"],
            "training_manifest_sha256": run.sha(manifest_path),
            "training_artifact_sha256": artifact_hashes,
            "source_core_sha256": source_sha,
            "ground_truth_used_for_prediction": False,
            "ground_truth_used_for_training": False,
            "ground_truth_used_for_scoring": True,
            "started": started, "completed": time.time(),
        }
        run.write_once(output_dir / "evaluation.json", report)
        sidecar = {"status": "complete", "experiment": "SW0133_soft_partition_rgb",
                   "seed": seed, "arm": arm,
                   "evaluation_sha256": run.sha(output_dir / "evaluation.json"),
                   "evaluation_fingerprint": evaluation_fingerprint(),
                   "training_manifest_sha256": run.sha(manifest_path),
                   "source_core_sha256": source_sha,
                   "frozen_predictions_sha256": evidence["prediction_sha256"],
                   "gamma_sha256": evidence["gamma_sha256"],
                   "ground_truth_used_for_prediction": False,
                   "ground_truth_used_for_scoring": True}
        run.write_once(output_dir / "evaluation_manifest.json", sidecar)
        with (output_dir / "COMPLETED").open("x", encoding="utf-8") as marker:
            marker.write("SW0133 full320 original-spike QCC evaluation complete\n")
        return report
    except BaseException as exc:
        try:
            run.write_once(output_dir / "evaluation_failure.json", {
                "status": "evaluation_failed", "seed": seed, "arm": arm,
                "error": repr(exc), "finished": time.time(),
                "training_manifest_sha256": run.sha(manifest_path)})
        except FileExistsError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(train.PILOT_SEED,), required=True)
    parser.add_argument("--arm", choices=run.ARMS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(args.seed, args.arm, torch.device(args.device), args.output,
                      training_dir=args.training_dir)
    print(json.dumps({"status": result["status"], "seed": args.seed, "arm": args.arm,
                      "output": str(args.output), "metrics": result["scores"]["metrics"]},
                     allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
