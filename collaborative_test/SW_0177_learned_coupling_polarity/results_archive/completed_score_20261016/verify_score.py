"""Read-only SW0177 score verifier for a completed fixed320 evaluation.

Run from the repository with the canonical server interpreter:
    python collaborative_testing/collaborative_private_runtime_20261008/
      SW0177_evaluation_recovery_20261016/verify_score.py

This calls the registered production validator once, then independently
recomputes all per-image metrics, means, paired bootstrap intervals, and gates.
It never launches a model or optimizer and never reads reserve images.
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

import numpy as np  # NumPy before Torch/legacy imports.


def _find_root():
    for candidate in (Path.cwd(), *Path(__file__).resolve().parents):
        if (candidate / "collaborative_test" / "SW_0177_learned_coupling_polarity"
                / "predict_evaluate.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError("could not locate the SW0177 repository root")


ROOT = _find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0177_learned_coupling_polarity import predict_evaluate as evaluate
from collaborative_test.SW_0177_learned_coupling_polarity import score_queue
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract


COUNT = 320
ARMS = ("signed", "nonnegative")
ENDPOINTS = ("conditional", "standalone")
GROUPS = ("SourceT", "SW0164_foreground_induced",
          "signed_conditional", "signed_standalone",
          "nonnegative_conditional", "nonnegative_standalone")
BASELINES = ("SW0164_foreground_induced",)
BOOTSTRAP_SEED = 177
BOOTSTRAP_REPLICATES = 10_000
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
PERFORMANCE_GATE_COUNT = 16
SUPPORT_GATE_COUNT = 4


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_finite_vector(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (COUNT,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite fixed320 vector")
    return array


def paired_bootstrap(candidate, baseline):
    """Independent registered paired bootstrap: RandomState(177), 10k resamples."""
    candidate = _assert_finite_vector(candidate, "candidate")
    baseline = _assert_finite_vector(baseline, "baseline")
    delta = candidate - baseline
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    indices = rng.randint(0, COUNT, size=(BOOTSTRAP_REPLICATES, COUNT))
    samples = delta[indices].mean(axis=1)
    return {"mean": float(delta.mean()),
            "ci95": [float(x) for x in np.percentile(samples, [2.5, 97.5])],
            "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}


def _compare_metrics(actual, recomputed, group, metric):
    if not isinstance(actual, dict) or actual.get("valid_count") != COUNT:
        raise ValueError(f"{group}/{metric} report lacks 320 valid observations")
    observed = _assert_finite_vector(actual.get("per_image"), f"{group}/{metric}.per_image")
    independent = _assert_finite_vector(recomputed.get("per_image"),
                                        f"recomputed {group}/{metric}.per_image")
    if not np.array_equal(observed, independent):
        raise ValueError(f"{group}/{metric} per-image values differ from independent recomputation")
    mean = float(actual.get("mean", np.nan))
    if not np.isfinite(mean) or abs(mean - float(independent.mean())) > 1e-12:
        raise ValueError(f"{group}/{metric} mean does not match its 320 per-image values")
    if abs(mean - float(recomputed["mean"])) > 1e-12:
        raise ValueError(f"{group}/{metric} mean differs from production metric recomputation")
    return independent, mean


def _derive_gates(scores, labels, supports):
    gates, paired = {}, {}
    for endpoint in ENDPOINTS:
        candidate_name = f"signed_{endpoint}"
        for baseline_name in ("SW0164_foreground_induced", f"nonnegative_{endpoint}"):
            candidate = scores[candidate_name]
            baseline = scores[baseline_name]
            pair_name = f"{endpoint}_signed_minus_{baseline_name}_fg_ari"
            paired[pair_name] = paired_bootstrap(
                candidate["fg_ari"]["per_image"], baseline["fg_ari"]["per_image"])
            prefix = f"{endpoint}_vs_{baseline_name}"
            gates[f"{prefix}_fg_mean_above"] = candidate["fg_ari"]["mean"] > baseline["fg_ari"]["mean"]
            gates[f"{prefix}_paired_fg_ci95_lower_above_zero"] = paired[pair_name]["ci95"][0] > 0
            gates[f"{prefix}_foreground_iou_not_below"] = (
                candidate["foreground_iou"]["mean"] >= baseline["foreground_iou"]["mean"])
            gates[f"{prefix}_matched_object_iou_not_below"] = (
                candidate["matched_object_iou"]["mean"] >= baseline["matched_object_iou"]["mean"])
    gates["signed_conditional_foreground_exact_SourceT"] = bool(np.array_equal(
        labels["signed_conditional"] > 0, labels["SourceT"] > 0))
    gates["nonnegative_conditional_foreground_exact_SourceT"] = bool(np.array_equal(
        labels["nonnegative_conditional"] > 0, labels["SourceT"] > 0))
    gates["signed_standalone_foreground_exact_native_QCC"] = bool(np.array_equal(
        labels["signed_standalone"] > 0, supports["signed"] > 0))
    gates["nonnegative_standalone_foreground_exact_native_QCC"] = bool(np.array_equal(
        labels["nonnegative_standalone"] > 0, supports["nonnegative"] > 0))
    if len(gates) != PERFORMANCE_GATE_COUNT + SUPPORT_GATE_COUNT:
        raise AssertionError("internal SW0177 gate schema count changed")
    return paired, gates, bool(all(gates.values()))


def _assert_prediction_and_score_receipts(prepared, report, report_path):
    if not report_path.is_file():
        raise FileNotFoundError(f"SW0177 score report is not complete: {report_path}")
    report_sha = _sha(report_path)
    score_state_path = score_queue._state_path()
    if not score_state_path.is_file():
        raise FileNotFoundError(f"SW0177 score queue receipt is not complete: {score_state_path}")
    score_state = json.loads(score_state_path.read_text(encoding="utf-8"))
    if (score_state.get("experiment") != score_queue.EXPERIMENT
            or score_state.get("status") != "complete"
            or score_state.get("score_path") != str(report_path)
            or score_state.get("score_sha256") != report_sha
            or score_state.get("implementation_fingerprint") != evaluate.implementation_fingerprint()
            or score_state.get("ground_truth_used_for_prediction") is not False
            or score_state.get("ground_truth_used_for_scoring") is not True):
        raise ValueError("SW0177 complete score-queue receipt does not bind the score report")

    if (report.get("experiment") != "SW0177_learned_coupling_fixed320_score"
            or report.get("status") != "complete"
            or report.get("implementation_fingerprint") != evaluate.implementation_fingerprint()
            or report.get("ground_truth_used_for_prediction") is not False
            or report.get("ground_truth_used_for_scoring") is not True
            or report.get("ground_truth_scope") != "fixed320 canonical validation modal-native32 masks"
            or report.get("bootstrap_seed") != BOOTSTRAP_SEED
            or report.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
            or report.get("seed_scope") != [1]
            or report.get("arm_scope") != list(ARMS)
            or report.get("endpoint_scope") != list(ENDPOINTS)):
        raise ValueError("SW0177 score report provenance/scope is invalid")
    if report.get("SW0164_score_sha256") != evaluate.SW0164_SCORE_SHA256:
        raise ValueError("SW0177 report is not bound to the pinned SW0164 score baseline")

    prediction_receipts = evaluate._prediction_receipts(prepared)
    expected_pred_sha, expected_protocol_sha, expected_state_sha = {}, {}, {}
    for arm in ARMS:
        receipt = prediction_receipts[arm]
        expected_pred_sha[arm] = receipt["result"]["prediction_sha256"]
        expected_protocol_sha[arm] = receipt["result"]["protocol_sha256"]
        expected_state_sha[arm] = receipt["state_sha256"]
    if report.get("prediction_sha256") != expected_pred_sha:
        raise ValueError("score report prediction archive hashes do not match frozen predictions")
    if report.get("prediction_protocol_sha256") != expected_protocol_sha:
        raise ValueError("score report prediction protocol hashes do not match frozen predictions")
    if report.get("prediction_queue_state_sha256") != expected_state_sha:
        raise ValueError("score report prediction queue receipt hashes changed")

    queue_prediction_sha = score_state.get("prediction_sha256")
    queue_state_sha = score_state.get("prediction_state_sha256")
    if queue_prediction_sha != expected_pred_sha or queue_state_sha != expected_state_sha:
        raise ValueError("complete score queue receipt is not bound to both arm predictions")
    expected_training = {arm: _sha(prepared["artifacts"][arm] / "manifest.json") for arm in ARMS}
    if report.get("training_manifest_sha256") != expected_training:
        raise ValueError("score report training-manifest hashes differ from actual validated artifacts")
    expected_reference_prediction = {key: prepared["references"][key]["prediction_sha256"]
                                     for key in ("source_T", "SW0164")}
    expected_reference_protocol = {key: prepared["references"][key]["protocol_sha256"]
                                   for key in ("source_T", "SW0164")}
    if report.get("reference_prediction_sha256") != expected_reference_prediction:
        raise ValueError("score report immutable reference prediction hashes changed")
    if report.get("reference_protocol_sha256") != expected_reference_protocol:
        raise ValueError("score report immutable reference protocol hashes changed")
    if expected_training.keys() != report.get("training_manifest_sha256", {}).keys():
        raise ValueError("score report has an incomplete training artifact set")
    if score_state.get("gate") != report.get("gate"):
        raise ValueError("score queue terminal gate differs from score report")
    return {"score_report_sha256": report_sha,
            "score_queue_state_sha256": _sha(score_state_path),
            "prediction_sha256": expected_pred_sha,
            "prediction_protocol_sha256": expected_protocol_sha,
            "prediction_queue_state_sha256": expected_state_sha,
            "training_manifest_sha256": expected_training,
            "implementation_fingerprint": evaluate.implementation_fingerprint(),
            "prediction_receipts": prediction_receipts,
            "score_state": score_state}


def _load_fixed320_ground_truth(prepared):
    dataset = Path(prepared["dataset_path"])
    before = dataset.stat()
    if (before.st_size != prepared["dataset_size_bytes"]
            or before.st_mtime_ns != prepared["dataset_mtime_ns"]):
        raise ValueError("canonical HDF5 stat changed before independent fixed320 scoring")
    start, stop = int(evaluate.IMAGE_IDS[0]), int(evaluate.IMAGE_IDS[-1]) + 1
    with evaluate.h5py.File(dataset, "r") as handle:
        if "mask" not in handle or handle["mask"].shape[0] <= stop - 1:
            raise ValueError("canonical HDF5 validation masks are missing")
        raw = np.asarray(handle["mask"][start:stop])
    gt = contract.modal_native32(raw)
    after = dataset.stat()
    if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
        raise ValueError("canonical HDF5 stat changed during independent fixed320 scoring")
    return gt, {"dataset_path": str(dataset), "size_bytes": before.st_size,
                "mtime_ns": before.st_mtime_ns, "id_range": [start, stop - 1]}


def verify(score_path=None):
    """Verify a completed report. Raises FileNotFoundError if it is not ready."""
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("score verifier must remain CUDA hidden")
    if torch.cuda.is_initialized():
        raise RuntimeError("score verifier must run before CUDA initialization")
    score_path = Path(score_path or evaluate.SCORE_PATH)
    if not score_path.is_file():
        raise FileNotFoundError(f"SW0177 final score report is not available: {score_path}")

    # Validate current training artifacts, fixed references and CPU checkpoint loading.
    prepared = evaluate.prepare_prediction_inputs_cpu()
    with score_path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict):
        raise ValueError("SW0177 final score report is not a JSON object")

    # Freeze every relevant producer receipt/hash before invoking a validator that opens GT.
    receipt = _assert_prediction_and_score_receipts(prepared, report, score_path)
    refs = prepared["references"]
    pre_gt_fingerprint = evaluate.implementation_fingerprint()
    pre_gt_dataset = Path(prepared["dataset_path"]).stat()
    score_receipt_before = receipt["score_queue_state_sha256"]
    prediction_snapshot = (receipt["prediction_sha256"], receipt["prediction_protocol_sha256"],
                           receipt["prediction_queue_state_sha256"])
    training_snapshot = receipt["training_manifest_sha256"]

    # Registered production path: revalidates all predictions, score metrics, gates and GT-last.
    evaluate.validate_score(score_path, prepared=prepared)
    if torch.cuda.is_initialized():
        raise RuntimeError("production score validation unexpectedly initialized CUDA")

    # Independent metric arithmetic from the pinned label arrays and fixed320 GT.
    gt, dataset_identity = _load_fixed320_ground_truth(prepared)
    predictions = receipt["prediction_receipts"]
    labels = {
        "SourceT": refs["source_T"]["labels"],
        "SW0164_foreground_induced": refs["SW0164"]["labels"],
        "signed_conditional": predictions["signed"]["result"]["labels"]["conditional"],
        "signed_standalone": predictions["signed"]["result"]["labels"]["standalone"],
        "nonnegative_conditional": predictions["nonnegative"]["result"]["labels"]["conditional"],
        "nonnegative_standalone": predictions["nonnegative"]["result"]["labels"]["standalone"],
    }
    supports = {arm: predictions[arm]["result"]["standalone_support"] for arm in ARMS}
    if set(labels) != set(GROUPS):
        raise AssertionError("independent six-group score schema changed")
    recomputed, metrics_summary = {}, {}
    for group in GROUPS:
        production = contract.production_metrics32(labels[group], gt)
        recomputed[group] = {}
        metrics_summary[group] = {}
        recorded_group = report.get("scores", {}).get(group)
        if not isinstance(recorded_group, dict) or set(recorded_group) != set(METRICS):
            raise ValueError(f"score report metric group is incomplete: {group}")
        for metric in METRICS:
            per_image, mean = _compare_metrics(recorded_group[metric], production[metric],
                                               group, metric)
            recomputed[group][metric] = {"per_image": per_image, "mean": mean}
            metrics_summary[group][metric] = mean

    expected_sw164_score = json.loads(
        Path(refs["SW0164_score_path"]).read_text(encoding="utf-8-sig"))["scores"]["foreground_induced"]
    for metric in METRICS:
        recorded = report["scores"]["SW0164_foreground_induced"][metric]
        expected = expected_sw164_score[metric]
        if (recorded.get("valid_count") != COUNT
                or not np.array_equal(_assert_finite_vector(recorded["per_image"], "SW0164 score"),
                                      _assert_finite_vector(expected["per_image"], "pinned SW0164 score"))
                or abs(float(recorded["mean"]) - float(expected["mean"])) > 1e-12):
            raise ValueError(f"SW0164 pinned production score differs for {metric}")

    paired, gates, overall = _derive_gates(recomputed, labels, supports)
    if report.get("paired_bootstrap") != paired:
        # JSON round-tripping is exact for Python finite floats; any difference indicates
        # a different RNG schedule, interval method, pairing, or candidate/baseline vector.
        raise ValueError("independent paired bootstrap differs from the frozen report")
    if report.get("gate") != {**gates, "co_primary_gate_passed": overall}:
        raise ValueError("independently derived fixed320 performance/support gates differ")

    # Close the verification against replacements during production or independent scoring.
    if evaluate.implementation_fingerprint() != pre_gt_fingerprint:
        raise ValueError("SW0177 source fingerprint changed during verification")
    current_receipt = _assert_prediction_and_score_receipts(prepared, report, score_path)
    if (current_receipt["score_queue_state_sha256"] != score_receipt_before
            or current_receipt["prediction_sha256"] != prediction_snapshot[0]
            or current_receipt["prediction_protocol_sha256"] != prediction_snapshot[1]
            or current_receipt["prediction_queue_state_sha256"] != prediction_snapshot[2]
            or current_receipt["training_manifest_sha256"] != training_snapshot):
        raise ValueError("SW0177 immutable receipt hashes changed during verification")
    now = Path(prepared["dataset_path"]).stat()
    if now.st_size != pre_gt_dataset.st_size or now.st_mtime_ns != pre_gt_dataset.st_mtime_ns:
        raise ValueError("canonical dataset stat changed during score verification")
    if torch.cuda.is_initialized():
        raise RuntimeError("read-only score verifier initialized CUDA")
    return {"status": "verified", "score_report_sha256": receipt["score_report_sha256"],
            "score_queue_state_sha256": receipt["score_queue_state_sha256"],
            "implementation_fingerprint": pre_gt_fingerprint,
            "training_manifest_sha256": training_snapshot,
            "prediction_sha256": prediction_snapshot[0],
            "prediction_protocol_sha256": prediction_snapshot[1],
            "prediction_queue_state_sha256": prediction_snapshot[2],
            "scores": metrics_summary, "gate": gates,
            "co_primary_gate_passed": overall, "dataset": dataset_identity,
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "ground_truth_scope": "fixed320 validation masks only; no reserve access"}


def self_test():
    """Pure arithmetic fixture; does not imply any actual model score was checked."""
    x = np.linspace(0.2, 0.8, COUNT, dtype=np.float64)
    baseline = np.linspace(0.1, 0.7, COUNT, dtype=np.float64)
    result = paired_bootstrap(x, baseline)
    delta = x - baseline
    rng = np.random.RandomState(177)
    expected = np.percentile(delta[rng.randint(0, COUNT,
        size=(10_000, COUNT))].mean(axis=1), [2.5, 97.5])
    if result["seed"] != 177 or result["replicates"] != 10_000 \
            or not np.array_equal(np.asarray(result["ci95"]), expected):
        raise AssertionError("paired bootstrap fixture failed")
    fake_scores = {}
    for name, offset in (("SourceT", 0.0), ("SW0164_foreground_induced", 0.0),
                         ("signed_conditional", 0.1), ("signed_standalone", 0.1),
                         ("nonnegative_conditional", 0.0),
                         ("nonnegative_standalone", 0.0)):
        fake_scores[name] = {
            "fg_ari": {"per_image": np.full(COUNT, 0.3 + offset),
                       "mean": 0.3 + offset},
            "foreground_iou": {"per_image": np.full(COUNT, 0.4), "mean": 0.4},
            "matched_object_iou": {"per_image": np.full(COUNT, 0.5), "mean": 0.5}}
    fake_labels = {name: np.ones((COUNT, 2, 2), dtype=np.int32) for name in GROUPS}
    fake_support = {arm: np.ones((COUNT, 2, 2), dtype=np.int32) for arm in ARMS}
    paired, gates, overall = _derive_gates(fake_scores, fake_labels, fake_support)
    if len(paired) != 4 or len(gates) != 20 or not overall or not all(gates.values()):
        raise AssertionError("performance/support gate fixture failed")
    return {"status": "pure_arithmetic_fixture_passed", "actual_score_tested": False}


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        result = self_test()
    else:
        result = verify(args.score)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

