"""Read-only verifier for a completed SW0178 fixed320 score.

Run from the repository with its canonical Python environment. The verifier
requires a completed score-queue receipt, snapshots every prediction/training
artifact before opening the fixed320 masks, calls the production validator
once, and independently recomputes metrics, paired bootstrap intervals, and
decision gates. It never runs a model, optimizer, or reserve-data read.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ["USE_PYTORCH_KERNEL_CACHE"] = "0"

import numpy as np  # Initialize NumPy/OpenMP before production imports.


def _find_root():
    for candidate in (Path.cwd(), *Path(__file__).resolve().parents):
        if (candidate / "collaborative_test" / "SW_0178_temporal_class_competition"
                / "predict_evaluate.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError("could not locate the SW0178 repository root")


ROOT = _find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from collaborative_test.SW_0178_temporal_class_competition import (
    predict_evaluate as evaluate,
    score_queue,
    train,
)
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract

COUNT = 320
ARMS = tuple(train.ARMS)
ENDPOINTS = tuple(evaluate.ENDPOINTS)
METRICS = tuple(contract.METRICS)
BOOTSTRAP_SEED = 178
BOOTSTRAP_REPLICATES = 10_000
SCORE_EXPERIMENT = "SW0178_temporal_class_competition_fixed320_score"
EXPECTED_SCORE_QUEUE_EXPERIMENT = "SW0178_score_queue_v1"
EXPECTED_SCORE_BASELINE = {
    "fg_ari": 0.7709320986336607,
    "foreground_iou": 0.6149934302630647,
    "matched_object_iou": 0.6305234820304301,
}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_vector(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (COUNT,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain 320 finite fixed-validation values")
    return array


def _json_equal(left, right, message):
    if json.dumps(left, sort_keys=True, allow_nan=False) != json.dumps(
            right, sort_keys=True, allow_nan=False):
        raise ValueError(message)


def paired_bootstrap(candidate, baseline):
    """Independently recompute the registered paired RandomState(178) interval."""
    delta = (_finite_vector(candidate, "candidate")
             - _finite_vector(baseline, "baseline"))
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    indices = rng.randint(0, COUNT, size=(BOOTSTRAP_REPLICATES, COUNT))
    samples = delta[indices].mean(axis=1)
    return {"mean": float(delta.mean()),
            "ci95": [float(x) for x in np.percentile(samples, [2.5, 97.5])],
            "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}


def _metric_snapshot(report_metric, recomputed, group, metric):
    if not isinstance(report_metric, dict) or report_metric.get("valid_count") != COUNT:
        raise ValueError(f"{group}/{metric} must report 320 valid images")
    actual = _finite_vector(report_metric.get("per_image"), f"{group}/{metric}.per_image")
    independent = _finite_vector(recomputed.get("per_image"), f"recomputed {group}/{metric}")
    if not np.array_equal(actual, independent):
        raise ValueError(f"{group}/{metric} per-image values differ from production recomputation")
    mean = float(report_metric.get("mean", np.nan))
    if not np.isfinite(mean) or abs(mean - float(independent.mean())) > 1e-12:
        raise ValueError(f"{group}/{metric} mean does not equal its per-image values")
    if abs(mean - float(recomputed["mean"])) > 1e-12:
        raise ValueError(f"{group}/{metric} mean differs from production metric output")
    return independent, mean


def _derive_gates(scores, paired, endpoint_contracts):
    baseline = scores["SW0164_foreground_induced"]
    available = tuple(arm for arm in ARMS if f"{arm}_conditional" in scores)
    gates = {"arm_eligibility": {}, "mean_rho_advantage": {},
             "endpoint_foreground_contracts": endpoint_contracts}

    for arm in available:
        endpoint_gates = {}
        eligible = True
        for endpoint in ENDPOINTS:
            key = f"{arm}_{endpoint}"
            metrics = scores[key]
            bootstrap_key = f"{endpoint}_{arm}_minus_SW0164_fg_ari"
            result = {
                "fg_mean_strictly_above_SW0164":
                    metrics["fg_ari"]["mean"] > baseline["fg_ari"]["mean"],
                "paired_fg_ci95_lower_above_zero_vs_SW0164":
                    paired[bootstrap_key]["ci95"][0] > 0,
                "foreground_iou_at_least_SW0164":
                    metrics["foreground_iou"]["mean"] >= baseline["foreground_iou"]["mean"],
                "matched_object_iou_at_least_SW0164":
                    metrics["matched_object_iou"]["mean"] >= baseline["matched_object_iou"]["mean"],
            }
            endpoint_gates[endpoint] = result
            eligible = eligible and all(result.values())
        eligible = eligible and all(endpoint_contracts[arm].values())
        gates["arm_eligibility"][arm] = {"endpoints": endpoint_gates, "eligible": bool(eligible)}

    for endpoint in ENDPOINTS:
        if not set(ARMS).issubset(available):
            gates["mean_rho_advantage"][endpoint] = {
                "status": "unavailable_training_arm_rejected"}
            continue
        rho, control = scores[f"mean_rho_ce_{endpoint}"], scores[f"product_q_ce_{endpoint}"]
        key = f"{endpoint}_mean_rho_ce_minus_product_q_ce_fg_ari"
        gates["mean_rho_advantage"][endpoint] = {
            "fg_mean_strictly_above_product_q_ce":
                rho["fg_ari"]["mean"] > control["fg_ari"]["mean"],
            "paired_fg_ci95_lower_above_zero_vs_product_q_ce": paired[key]["ci95"][0] > 0,
            "foreground_iou_at_least_product_q_ce":
                rho["foreground_iou"]["mean"] >= control["foreground_iou"]["mean"],
            "matched_object_iou_at_least_product_q_ce":
                rho["matched_object_iou"]["mean"] >= control["matched_object_iou"]["mean"],
        }
    gates["co_primary_any_arm_eligible"] = any(
        row["eligible"] for row in gates["arm_eligibility"].values())
    gates["mean_rho_mechanism_promoted"] = (
        set(ARMS).issubset(available)
        and gates["arm_eligibility"].get("mean_rho_ce", {}).get("eligible", False)
        and all(all(result.values()) for result in gates["mean_rho_advantage"].values()))
    return gates


def _prediction_and_training_snapshot(prepared):
    receipts = evaluate._prediction_receipts(prepared)
    training = {}
    for arm in prepared["manifests"]:
        artifact = Path(prepared["artifacts"][arm])
        state_path = evaluate._training_state_path(arm)
        training[arm] = {
            "state_sha256": _sha(state_path),
            "manifest_sha256": _sha(artifact / "manifest.json"),
            "checkpoint_sha256": _sha(artifact / "checkpoint.pt"),
            "optimizer_sha256": _sha(artifact / "optimizer.pt"),
        }
    for arm in prepared["rejected"]:
        state_path = evaluate._training_state_path(arm)
        rejection_path = Path(train.OUTPUT_ROOT) / "work" / f"{arm}_seed1" / "credit_rejection.json"
        training[arm] = {"state_sha256": _sha(state_path),
                         "rejection_sha256": _sha(rejection_path)}
    return receipts, training


def _assert_frozen_receipts(report, score_path, prepared, receipts, training,
                            score_queue_state, source_fingerprint):
    report_sha = _sha(score_path)
    score_state_sha = _sha(score_queue._state_path())
    manifests = prepared["manifests"]
    rejected = prepared["rejected"]
    pred_hashes = {arm: receipts[arm]["result"]["prediction_sha256"] for arm in manifests}
    protocol_hashes = {arm: receipts[arm]["result"]["protocol_sha256"] for arm in manifests}
    prediction_state_hashes = {arm: receipts[arm]["state_sha256"] for arm in manifests}
    training_hashes = {arm: training[arm]["manifest_sha256"] for arm in manifests}
    rejected_state_hashes = {arm: receipts["_rejected"][arm]["state_sha256"] for arm in rejected}

    if (report.get("experiment") != SCORE_EXPERIMENT
            or report.get("status") != "complete"
            or report.get("implementation_fingerprint") != source_fingerprint
            or report.get("ground_truth_used_for_prediction") is not False
            or report.get("ground_truth_used_for_scoring") is not True
            or report.get("mask_supervised") is not True
            or report.get("unsupervised_leaderboard_eligible") is not False
            or report.get("ground_truth_scope") != "fixed320 canonical validation modal-native32 masks"
            or report.get("bootstrap_seed") != BOOTSTRAP_SEED
            or report.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
            or report.get("seed_scope") != [1]
            or report.get("arm_scope") != list(manifests)
            or report.get("rejected_arms") != list(rejected)
            or report.get("endpoint_scope") != list(ENDPOINTS)
            or report.get("prediction_sha256") != pred_hashes
            or report.get("prediction_protocol_sha256") != protocol_hashes
            or report.get("prediction_queue_state_sha256") != prediction_state_hashes
            or report.get("rejected_training_state_sha256") != rejected_state_hashes
            or report.get("training_manifest_sha256") != training_hashes):
        raise ValueError("score report does not bind its fixed320 predictions/training/provenance")
    if (score_queue_state.get("experiment") != EXPECTED_SCORE_QUEUE_EXPERIMENT
            or score_queue_state.get("status") != "complete"
            or score_queue_state.get("score_path") != str(Path(score_path))
            or score_queue_state.get("score_sha256") != report_sha
            or score_queue_state.get("implementation_fingerprint") != source_fingerprint
            or score_queue_state.get("ground_truth_used_for_prediction") is not False
            or score_queue_state.get("ground_truth_used_for_scoring") is not True
            or score_queue_state.get("prediction_state_sha256") != prediction_state_hashes
            or score_queue_state.get("rejected_training_state_sha256") != rejected_state_hashes
            or score_queue_state.get("prediction_sha256") != pred_hashes
            or score_queue_state.get("gate") != report.get("gate")):
        raise ValueError("completed score queue receipt does not bind score report and predictions")
    return {"report_sha256": report_sha, "score_queue_state_sha256": score_state_sha,
            "prediction_sha256": pred_hashes, "protocol_sha256": protocol_hashes,
            "prediction_state_sha256": prediction_state_hashes,
            "training": training, "rejected_state_sha256": rejected_state_hashes}


def _load_fixed320_gt(prepared):
    dataset = Path(prepared["dataset_path"])
    before = dataset.stat()
    if before.st_size != prepared["dataset_size_bytes"] or before.st_mtime_ns != prepared["dataset_mtime_ns"]:
        raise ValueError("canonical dataset stat changed before independent mask read")
    start, stop = int(evaluate.IMAGE_IDS[0]), int(evaluate.IMAGE_IDS[-1]) + 1
    if (start, stop) != (1320, 1640):
        raise ValueError("verifier is restricted to registered fixed320 IDs 1320..1639")
    with evaluate.h5py.File(dataset, "r") as handle:
        if "mask" not in handle or handle["mask"].shape[0] < stop:
            raise ValueError("fixed320 canonical validation masks are unavailable")
        raw = np.asarray(handle["mask"][start:stop])
    gt = contract.modal_native32(raw)
    after = dataset.stat()
    if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
        raise ValueError("canonical dataset stat changed during independent mask read")
    return gt, {"dataset_path": str(dataset), "size_bytes": before.st_size,
                "mtime_ns": before.st_mtime_ns, "ids": [start, stop - 1]}


def verify(score_path=None):
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0178 score verifier must start before CUDA initialization")
    path = Path(score_path) if score_path is not None else Path(evaluate.SCORE_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"completed SW0178 score report is not present: {path}")
    queue_state_path = Path(score_queue._state_path())
    if not queue_state_path.is_file():
        raise FileNotFoundError(f"completed SW0178 score queue receipt is not present: {queue_state_path}")

    # CPU preparation verifies immutable training, prediction, RGB and reference
    # artifacts without reading validation masks. Capture their hashes first.
    prepared = evaluate.prepare_prediction_inputs_cpu()
    report_bytes = path.read_bytes()
    report = json.loads(report_bytes)
    score_state_bytes = queue_state_path.read_bytes()
    score_state = json.loads(score_state_bytes)
    source_fingerprint = evaluate.implementation_fingerprint()
    predictions_before, training_before = _prediction_and_training_snapshot(prepared)
    report_before = _assert_frozen_receipts(report, path, prepared, predictions_before,
        training_before, score_state, source_fingerprint)
    dataset_before = Path(prepared["dataset_path"]).stat()

    # Production validator performs the registered score recomputation once;
    # the independent arithmetic below is separate verification.
    evaluate.validate_score(path, prepared=prepared)
    if torch.cuda.is_initialized():
        raise RuntimeError("production score validator unexpectedly initialized CUDA")

    ground_truth, dataset_identity = _load_fixed320_gt(prepared)
    refs = prepared["references"]
    labels = {
        "SourceT": refs["source_T"]["labels"],
        "SW0164_foreground_induced": refs["SW0164"]["labels"],
    }
    supports = {}
    for arm in prepared["manifests"]:
        result = predictions_before[arm]["result"]
        labels[f"{arm}_conditional"] = result["labels"]["conditional"]
        labels[f"{arm}_standalone"] = result["labels"]["standalone"]
        supports[arm] = result["standalone_support"]
    expected_groups = {"SourceT", "SW0164_foreground_induced"} | {
        f"{arm}_{endpoint}" for arm in prepared["manifests"] for endpoint in ENDPOINTS}
    if set(report.get("scores", {})) != expected_groups:
        raise ValueError("score report group set differs from completed prediction arms")

    scores, summary = {}, {}
    for group in sorted(expected_groups):
        recomputed = contract.production_metrics32(labels[group], ground_truth)
        scores[group], summary[group] = {}, {}
        for metric in METRICS:
            per_image, mean = _metric_snapshot(report["scores"][group].get(metric),
                                               recomputed[metric], group, metric)
            scores[group][metric] = {"per_image": per_image, "mean": mean}
            summary[group][metric] = mean

    baseline_scores = refs["SW0164_score"]["scores"]["foreground_induced"]
    for metric, expected_mean in EXPECTED_SCORE_BASELINE.items():
        observed = scores["SW0164_foreground_induced"][metric]
        expected = baseline_scores[metric]
        if (abs(observed["mean"] - expected_mean) > 1e-12
                or expected.get("valid_count") != COUNT
                or not np.array_equal(observed["per_image"], _finite_vector(
                    expected.get("per_image"), f"pinned SW0164 {metric}"))
                or abs(float(expected.get("mean", np.nan)) - expected_mean) > 1e-12):
            raise ValueError(f"pinned SW0164 metric differs for {metric}")

    paired = {}
    baseline = scores["SW0164_foreground_induced"]
    for arm in prepared["manifests"]:
        for endpoint in ENDPOINTS:
            key = f"{endpoint}_{arm}_minus_SW0164_fg_ari"
            paired[key] = paired_bootstrap(scores[f"{arm}_{endpoint}"]["fg_ari"]["per_image"],
                                           baseline["fg_ari"]["per_image"])
    if set(ARMS).issubset(prepared["manifests"]):
        for endpoint in ENDPOINTS:
            key = f"{endpoint}_mean_rho_ce_minus_product_q_ce_fg_ari"
            paired[key] = paired_bootstrap(
                scores[f"mean_rho_ce_{endpoint}"]["fg_ari"]["per_image"],
                scores[f"product_q_ce_{endpoint}"]["fg_ari"]["per_image"])
    _json_equal(report.get("paired_bootstrap"), paired,
                "paired bootstrap does not match independent RandomState(178) recomputation")

    endpoint_contracts = {}
    source_foreground = np.asarray(labels["SourceT"]) > 0
    for arm in prepared["manifests"]:
        endpoint_contracts[arm] = {
            "conditional_exact_SourceT": bool(np.array_equal(
                np.asarray(labels[f"{arm}_conditional"]) > 0, source_foreground)),
            "standalone_exact_native_QCC": bool(np.array_equal(
                np.asarray(labels[f"{arm}_standalone"]) > 0, np.asarray(supports[arm]) > 0)),
        }
    gates = _derive_gates(scores, paired, endpoint_contracts)
    _json_equal(report.get("gate"), gates, "registered performance/support gates differ from independent derivation")

    # Verify that no prediction, protocol, training artifact, score receipt,
    # score report, source code or dataset changed during validation.
    if evaluate.implementation_fingerprint() != source_fingerprint:
        raise ValueError("SW0178 source fingerprint changed during verification")
    predictions_after, training_after = _prediction_and_training_snapshot(prepared)
    if (predictions_after["_rejected"] != predictions_before["_rejected"]
            or any(predictions_after[arm]["state_sha256"] != predictions_before[arm]["state_sha256"]
                   or predictions_after[arm]["result"]["prediction_sha256"]
                      != predictions_before[arm]["result"]["prediction_sha256"]
                   or predictions_after[arm]["result"]["protocol_sha256"]
                      != predictions_before[arm]["result"]["protocol_sha256"]
                   for arm in prepared["manifests"])
            or training_after != training_before
            or path.read_bytes() != report_bytes
            or queue_state_path.read_bytes() != score_state_bytes
            or _sha(path) != report_before["report_sha256"]
            or _sha(queue_state_path) != report_before["score_queue_state_sha256"]):
        raise ValueError("a frozen score/prediction/training receipt changed during verification")
    dataset_after = Path(prepared["dataset_path"]).stat()
    if (dataset_after.st_size != dataset_before.st_size
            or dataset_after.st_mtime_ns != dataset_before.st_mtime_ns):
        raise ValueError("canonical dataset stat changed during score verification")
    if torch.cuda.is_initialized():
        raise RuntimeError("read-only score verifier initialized CUDA")
    return {"status": "verified", "score_report_sha256": report_before["report_sha256"],
            "score_queue_state_sha256": report_before["score_queue_state_sha256"],
            "source_fingerprint_sha256": hashlib.sha256(json.dumps(
                source_fingerprint, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "training_manifest_sha256": {arm: row["manifest_sha256"]
                                         for arm, row in training_before.items()
                                         if "manifest_sha256" in row},
            "prediction_sha256": report_before["prediction_sha256"],
            "scores": summary, "gate": gates, "dataset": dataset_identity,
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "validation_scope": "fixed320 IDs 1320..1639 only; reserve unread"}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.score), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
