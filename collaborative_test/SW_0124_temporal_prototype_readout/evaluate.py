"""Frozen-source SW0124 stage-1 evaluation; all predictions precede GT access."""
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

from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0115_analytic_partition_rgb.loss import production_partition
from collaborative_test.SW_0122_joint_rgb_seed_replication import run as sw122
from collaborative_test.SW_0123_adaptive_temporal_assignment import run as sw123
from collaborative_test.SW_0123_adaptive_temporal_assignment.readout import assignment_to_labels
from collaborative_test.SW_0124_temporal_prototype_readout.prototype_readout import (
    temporal_prototype_assignment,
)
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity

SEEDS = (0, 1, 2)
IDS = np.arange(1320, 1640, dtype=np.int64)
BATCH, STEPS, SETTLE = 8, 1024, 512
DATASET = Path("/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5")
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
SOURCE_EVALUATION_SHAS = {
    0: "009da533e0f3e7fa86d9840a85d87640d9424acf660512f64232ad1669fe61e0",
    1: "3576ad632d1caba7d396ce65e0d094e86ff2eb9d812f8ef03d5c5e8a30a54278",
    2: "2d4eb351a530058f307c5c907aa078b9a197323c69fc4e44754d237531e9538c",
}
SLOT_REFERENCE = ROOT / "collaborative_test/SW_0122_joint_rgb_seed_replication/slot_reference.json"
SLOT_REFERENCE_SHA = "de49089f54c05e08574b654815a8fd4656cd0f306892f4ecb0c3da94ab579945"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path, value):
    path = Path(path)
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def _score(prediction, target):
    scored = evaluate_patch_masks(prediction, target)
    result = {"metrics": {}, "valid_count": {}, "per_image": {}}
    for metric in METRICS:
        values = scored["per_image"][metric].detach().cpu().double().numpy()
        mean = float(scored["mean"][metric])
        result["metrics"][metric] = mean if math.isfinite(mean) else None
        result["valid_count"][metric] = int(scored["valid_count"][metric])
        result["per_image"][metric] = [float(x) if math.isfinite(float(x)) else None
                                        for x in values]
    return result


def _check_score(score, name):
    if set(score.get("metrics", {})) != set(METRICS):
        raise AssertionError(f"{name}: unexpected metric schema")
    for metric in METRICS:
        values = np.asarray(score["per_image"][metric], dtype=np.float64)
        mean = score["metrics"][metric]
        if (values.shape != (320,) or not np.isfinite(values).all()
                or score["valid_count"].get(metric) != 320
                or mean is None or not math.isfinite(float(mean))
                or abs(float(values.mean()) - float(mean)) > 1e-12):
            raise AssertionError(f"{name}: invalid {metric} values")


def source_reference(seed):
    source, source_manifest, manifest = base.source_paths(seed)
    ids, _rows = base.train_indices(seed)
    eval_path = Path(source).parent / "evaluation.json"
    if not eval_path.is_file() or sha(eval_path) != SOURCE_EVALUATION_SHAS[seed]:
        raise AssertionError(f"seed{seed}: immutable SW0097 evaluation reference SHA mismatch")
    report = json.loads(eval_path.read_text(encoding="utf-8"))
    score = sw122._validate_source_evaluation_records(seed, source, ids, manifest, report)
    raw_score = report["sweep"][0]["scored_targets"]["our_hdf5"]
    return {"source": source, "source_manifest": source_manifest, "manifest": manifest,
            "evaluation": eval_path, "evaluation_sha256": sha(eval_path),
            "metrics": score, "score": raw_score}


def evaluation_fingerprint():
    paths = [Path(__file__), HERE / "prototype_readout.py", HERE / "protocol.json",
             ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
             ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/loss.py",
             ROOT / "collaborative_test/SW_0123_adaptive_temporal_assignment/run.py",
             ROOT / "collaborative_test/SW_0123_adaptive_temporal_assignment/calibrate.py",
             ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
             ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
             ROOT / "snn_kuramoto_bidirectional/evaluation.py"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in paths}


def _assert_reproduces_source(qcc_score, reference, tolerance=1e-10):
    _check_score(qcc_score, "reproduced source QCC")
    raw = reference["score"]
    for metric in METRICS:
        expected = np.asarray(raw["per_image"][metric], dtype=np.float64)
        observed = np.asarray(qcc_score["per_image"][metric], dtype=np.float64)
        if expected.shape != (320,) or not np.isfinite(expected).all():
            raise AssertionError(f"source reference has invalid {metric} array")
        delta = float(np.max(np.abs(expected - observed)))
        if delta > tolerance:
            raise AssertionError(f"source QCC reproduction failed for {metric}: {delta:.3g}")


def _validate_gamma_batch(gamma):
    if (gamma.ndim != 3 or tuple(gamma.shape[1:]) != (8, 256)
            or not torch.isfinite(gamma).all()):
        raise ValueError("source core input must be static finite [B,8,256] gamma")
    return gamma


@torch.inference_mode()
def _predict_seed(seed, device, max_batches=None):
    reference = source_reference(seed)
    calibration_path, calibration, _kappa, rms = sw123.load_calibration(seed)
    if calibration.get("status") != "passed" or calibration.get("seed") != seed:
        raise AssertionError("registered legacy spike RMS calibration is not valid")
    gamma, gamma_manifest = base.validate_gamma_cache(
        base.GAMMA_VAL, base.GAMMA_VAL_MANIFEST, validation=True)
    if tuple(gamma.shape) != (320, 8, 256) or gamma_manifest.get("image_ids") != [1320, 1639]:
        raise AssertionError("validation gamma is not the registered fixed320 cache")
    core = base.make_core(device, STEPS)
    core.load_state_dict(torch.load(reference["source"], map_location=device, weights_only=True), strict=True)
    core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.shape[0])]
    core.eval(); core.graph_generator.eval()
    if core.kuramoto.spike_pulse_gain is not None or core.graph_generator.uses_feedback:
        raise AssertionError("SW0097 source contains dynamics outside the registered readout contract")
    prototype_batches, qcc_batches, anchors_by_image = [], [], []
    batch_count = len(IDS) // BATCH
    if max_batches is not None:
        if max_batches < 1:
            raise ValueError("max_batches must be positive")
        batch_count = min(batch_count, int(max_batches))
    count = batch_count * BATCH
    for start in range(0, count, BATCH):
        current = _validate_gamma_batch(gamma[start:start + BATCH].to(device))
        if current.shape[0] != BATCH:
            raise AssertionError("invalid fixed validation gamma batch size")
        _groups, _mean, _out, _theta = core(current, return_core_out=True, return_theta=True)
        components = core.last_component_spikes
        if tuple(components.shape) != (BATCH, 4, 256, STEPS) or not torch.isfinite(components).all():
            raise AssertionError("source rollout did not return finite actual component spikes")
        q = spike_synchrony_affinity(components.mean(dim=1), components,
                                     settle=SETTLE, affinity_mode="spike")
        probability, anchors, _prototypes = temporal_prototype_assignment(
            components, rms.to(device), q, settle=SETTLE)
        prototype_batches.append(assignment_to_labels(probability).cpu())
        qcc, _hard = production_partition(components.mean(dim=1), components, settle=SETTLE)
        qcc_batches.append(qcc.reshape(BATCH, 16, 16).cpu())
        anchors_by_image.extend(anchors)
    prototype = torch.cat(prototype_batches)
    qcc = torch.cat(qcc_batches)
    if tuple(prototype.shape) != (count, 16, 16) or tuple(qcc.shape) != (count, 16, 16):
        raise AssertionError("source readout shape mismatch")
    return {
        "experiment": "SW0124", "seed": seed,
        "image_ids": [int(IDS[0]), int(IDS[count - 1])], "count": count,
        "prototype_labels": prototype, "qcc_labels": qcc,
        "anchors_per_image": anchors_by_image,
        "source_core_sha256": sha(reference["source"]),
        "source_manifest_sha256": sha(reference["source_manifest"]),
        "source_evaluation_sha256": reference["evaluation_sha256"],
        "calibration_sha256": sha(calibration_path),
        "spike_rms_sha256": calibration["spike_rms_sha256"],
        "gamma_sha256": sha(base.GAMMA_VAL),
        "gamma_manifest_sha256": sha(base.GAMMA_VAL_MANIFEST),
        "ground_truth_used_for_prediction": False,
    }, reference


def _serialize_prediction(path, prediction):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        torch.save(prediction, stream); stream.flush(); os.fsync(stream.fileno())


def preflight_seed(seed, output, device="cuda"):
    """Real B8/T1024 path smoke on the first validation batch, with no masks."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserving existing SW0124 preflight: {output}")
    prediction, _reference = _predict_seed(seed, device, max_batches=1)
    if (prediction["count"] != BATCH or prediction["ground_truth_used_for_prediction"] is not False
            or tuple(prediction["prototype_labels"].shape) != (BATCH, 16, 16)
            or tuple(prediction["qcc_labels"].shape) != (BATCH, 16, 16)):
        raise AssertionError("real source preflight did not produce finite fixed-shape GT-free outputs")
    if not torch.isfinite(prediction["prototype_labels"].float()).all():
        raise AssertionError("prototype preflight labels are nonfinite")
    record = {key: value for key, value in prediction.items()
              if key not in ("prototype_labels", "qcc_labels", "anchors_per_image")}
    record.update(status="passed", stage="source_prototype_preflight",
                  foreground_fraction=float((prediction["prototype_labels"] > 0).float().mean()),
                  anchor_count_per_image=[len(anchors) for anchors in prediction["anchors_per_image"]],
                  implementation_fingerprint=evaluation_fingerprint(),
                  optimizer_updates=0, ground_truth_used=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_once(output, record)
    return record


def _validate_preflights(preflight_dir, device):
    gamma, gamma_manifest = base.validate_gamma_cache(
        base.GAMMA_VAL, base.GAMMA_VAL_MANIFEST, validation=True)
    del gamma
    fingerprint = evaluation_fingerprint()
    checked = {}
    for seed in SEEDS:
        reference = source_reference(seed)
        calibration_path, calibration, _kappa, _rms = sw123.load_calibration(seed)
        path = Path(preflight_dir) / f"preflight_seed{seed}.json"
        if not path.is_file():
            raise FileNotFoundError(f"seed{seed} actual source preflight is required: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("status") != "passed" or record.get("stage") != "source_prototype_preflight"
                or record.get("seed") != seed or record.get("image_ids") != [1320, 1327]
                or record.get("count") != BATCH or record.get("ground_truth_used") is not False
                or record.get("optimizer_updates") != 0
                or record.get("source_core_sha256") != sha(reference["source"])
                or record.get("source_evaluation_sha256") != reference["evaluation_sha256"]
                or record.get("calibration_sha256") != sha(calibration_path)
                or record.get("gamma_sha256") != sha(base.GAMMA_VAL)
                or record.get("gamma_manifest_sha256") != sha(base.GAMMA_VAL_MANIFEST)
                or record.get("implementation_fingerprint") != fingerprint):
            raise AssertionError(f"seed{seed} preflight does not bind current source/assets/implementation")
        checked[seed] = {"preflight_sha256": sha(path), "record": record}
    return checked


def paired_bootstrap(candidate, source, draws=10000, seed=124):
    """Bootstrap fixed image indices after averaging paired three-seed deltas."""
    candidate = np.asarray(candidate, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    if candidate.shape != (3, 320) or source.shape != (3, 320):
        raise ValueError("bootstrap needs three paired seeds by 320 shared images")
    if not np.isfinite(candidate).all() or not np.isfinite(source).all():
        raise ValueError("bootstrap arrays must be finite")
    delta_per_image = (candidate - source).mean(axis=0)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, 320, size=(draws, 320))
    values = delta_per_image[sampled].mean(axis=1)
    lo, hi = np.quantile(values, [.025, .975])
    return {"draws": draws, "seed": seed, "lower95": float(lo), "upper95": float(hi),
            "mean_delta": float(delta_per_image.mean()),
            "common_image_indices_across_seeds": True}


def evaluate_all(output_dir, device="cuda", preflight_dir=None):
    output_dir = Path(output_dir)
    if preflight_dir is None:
        raise ValueError("all three actual-source preflight records are required")
    preflights = _validate_preflights(preflight_dir, device)
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions, references, prediction_hashes = {}, {}, {}
    # Complete and hash all three GT-free source predictions before any mask read.
    for seed in SEEDS:
        prediction, reference = _predict_seed(seed, device)
        path = output_dir / f"frozen_predictions_seed{seed}.pt"
        _serialize_prediction(path, prediction)
        prediction_hashes[seed] = sha(path)
        predictions[seed] = path
        references[seed] = reference
    for seed in SEEDS:
        if sha(predictions[seed]) != prediction_hashes[seed]:
            raise AssertionError("saved GT-free prediction changed before validation")

    if sha(SLOT_REFERENCE) != SLOT_REFERENCE_SHA:
        raise AssertionError("matched Slot reference SHA mismatch")
    slot = json.loads(SLOT_REFERENCE.read_text(encoding="utf-8"))["metrics"]
    # Open masks once. First verify historical source QCC per-image reproduction
    # for every seed; no prototype metrics are computed until all three pass.
    with h5py.File(DATASET, "r") as dataset:
        if "mask" not in dataset or dataset["mask"].shape[0] != 100000:
            raise AssertionError("canonical validation mask dataset is absent or malformed")
        masks = np.asarray(dataset["mask"][IDS])
    target = clevr_mask_patch(torch.from_numpy(masks), 8)["patch_labels"].cpu()
    scored = {}
    for seed in SEEDS:
        frozen = torch.load(predictions[seed], map_location="cpu", weights_only=False)
        qcc_score = _score(frozen["qcc_labels"], target)
        _assert_reproduces_source(qcc_score, references[seed])
        scored[seed] = {"frozen": frozen, "qcc": qcc_score}

    for seed in SEEDS:
        prototype_score = _score(scored[seed]["frozen"]["prototype_labels"], target)
        _check_score(prototype_score, f"seed{seed} temporal prototype")
        frozen = scored[seed]["frozen"]
        report = {
            "status": "complete", "experiment": "SW0124", "seed": seed,
            "ids": [1320, 1639], "images": 320,
            "evaluation_contract": {"batch_size": BATCH, "time_steps": STEPS,
                                     "settle": SETTLE, "ground_truth_used_for_prediction": False},
            "temporal_prototype": prototype_score,
            "production_qcc": scored[seed]["qcc"],
            "registered_source_qcc_metrics": references[seed]["metrics"],
            "source_qcc_baseline_reproduced": True,
            "source_qcc_max_abs_per_image_deltas": {
                metric: float(np.max(np.abs(np.asarray(scored[seed]["qcc"]["per_image"][metric])
                                           - np.asarray(references[seed]["score"]["per_image"][metric]))))
                for metric in METRICS},
            "source_core_sha256": frozen["source_core_sha256"],
            "source_manifest_sha256": frozen["source_manifest_sha256"],
            "source_evaluation_sha256": frozen["source_evaluation_sha256"],
            "calibration_sha256": frozen["calibration_sha256"],
            "gamma_sha256": frozen["gamma_sha256"],
            "prediction_sha256": prediction_hashes[seed],
            "preflight_sha256": preflights[seed]["preflight_sha256"],
            "anchor_count_per_image": [len(v) for v in frozen["anchors_per_image"]],
            "implementation_sha256": {"evaluate": sha(Path(__file__)),
                                      "readout": sha(HERE / "prototype_readout.py"),
                                      "protocol": sha(HERE / "protocol.json")},
            "ground_truth_used_for_prediction": False,
        }
        write_once(output_dir / f"evaluation_seed{seed}.json", report)

    candidate = {metric: np.stack([
        np.asarray(json.loads((output_dir / f"evaluation_seed{seed}.json").read_text())[
            "temporal_prototype"]["per_image"][metric], dtype=np.float64) for seed in SEEDS])
        for metric in METRICS}
    source = {metric: np.stack([
        np.asarray(references[seed]["score"]["per_image"][metric], dtype=np.float64)
        for seed in SEEDS]) for metric in METRICS}
    means = {metric: float(candidate[metric].mean()) for metric in METRICS}
    source_means = {metric: float(source[metric].mean()) for metric in METRICS}
    fg_gains = [float(candidate["fg_ari"][i].mean() - source["fg_ari"][i].mean())
                for i in range(3)]
    ci = paired_bootstrap(candidate["fg_ari"], source["fg_ari"])
    gates = {
        "mean_fg_ari_exceeds_source": means["fg_ari"] > source_means["fg_ari"],
        "at_least_two_seed_fg_gains": sum(value > 0 for value in fg_gains) >= 2,
        "paired_fg_ci_lower_positive": ci["lower95"] > 0,
        "foreground_iou_exceeds_slot_plus_0_05": means["foreground_iou"] > float(slot["foreground_iou"]) + .05,
        "matched_object_iou_exceeds_slot_plus_0_05": means["matched_object_iou"] > float(slot["matched_object_iou"]) + .05,
        "all_source_qcc_baselines_reproduced": True,
    }
    summary = {
        "status": "promotion_passed" if all(gates.values()) else "promotion_failed",
        "experiment": "SW0124", "seeds": list(SEEDS), "metrics": means,
        "source_metrics": source_means, "per_seed_fg_gains_vs_source": fg_gains,
        "fg_paired_bootstrap_vs_source": ci, "matched_slot_reference_sha256": SLOT_REFERENCE_SHA,
        "matched_slot_metrics": slot, "gates": gates,
        "evaluation_sha256": {str(seed): sha(output_dir / f"evaluation_seed{seed}.json") for seed in SEEDS},
        "ground_truth_used_for_prediction": False,
    }
    write_once(output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "evaluate"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-dir", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "preflight":
        if args.seed is None or args.output is None:
            parser.error("preflight requires --seed and --output")
        record = preflight_seed(args.seed, args.output, args.device)
        print(json.dumps({"status": record["status"], "seed": args.seed,
                          "output": str(args.output)}, allow_nan=False), flush=True)
    else:
        if args.output_dir is None or args.preflight_dir is None:
            parser.error("evaluate requires --output-dir and --preflight-dir")
        summary = evaluate_all(args.output_dir, args.device, args.preflight_dir)
        print(json.dumps({"status": summary["status"], "metrics": summary["metrics"],
                          "gates": summary["gates"], "output_dir": str(args.output_dir)}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
