"""Independent read-only verifier for a completed SW0179 score queue."""
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

import numpy as np


def _find_root():
    for candidate in (Path.cwd(), *Path(__file__).resolve().parents):
        if (candidate / "collaborative_test" / "SW_0179_qce_seed_replication" / "score.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError("could not locate the SW0179 repository root")


ROOT = _find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from collaborative_test.SW_0179_qce_seed_replication import score, score_queue
from collaborative_test.SW_0179_qce_seed_replication import predict_evaluate as evaluate
from collaborative_test.SW_0179_qce_seed_replication import train
from collaborative_test.SW_0178_temporal_class_competition import predict_evaluate as seed1_eval
from collaborative_test.SW_0165_foreground_partition_replication import (
    predict_evaluate as sw165, replication_summary as sw165_summary,
)
from collaborative_test.SW_0164_foreground_preserving_partition import predict_evaluate as sw164
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract

SEEDS = (0, 1, 2)
ENDPOINTS = ("conditional", "standalone")
METRICS = tuple(contract.METRICS)
COUNT = 320
BOOTSTRAP_SEED = 178
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_EXPERIMENT = "SW0179_qce_seed_replication_three_seed_score"
EXPECTED_QUEUE_EXPERIMENT = "SW0179_score_queue"
SW164_SEED1_SCORE_SHA256 = "978a9d97658837eb19f414a7fb7dbb35d86c6135cb1213e12e63e2837fd65df6"
SW165_SUMMARY_SHA256 = "72b17ca1a07d312400ab74f6ee7533c89f0e9d9a6c304378dd8e61c16ba379c5"
SW165_FG_MEAN = 0.8042989149474198
SLOT_FG_MEAN = 0.7816519719609135
SLOT_FG_IOU_MEAN = 0.20436012776655546
SLOT_OBJECT_IOU_MEAN = 0.2031649861884501


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite320(value, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (COUNT,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be 320 finite values")
    return array


def _assert_json_equal(actual, expected, label):
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
            expected, sort_keys=True, allow_nan=False):
        raise ValueError(f"{label} differs from independent recomputation")


def _bootstrap(candidate, baseline, indices):
    delta = _finite320(candidate, "candidate vector") - _finite320(baseline, "baseline vector")
    samples = delta[indices].mean(axis=1)
    return {"mean": float(delta.mean()),
        "ci95": [float(value) for value in np.percentile(samples, [2.5, 97.5])],
        "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
        "resampling_unit": "paired_image_after_three_seed_average"}


def _compare_metric_row(row, values, group, metric):
    if not isinstance(row, dict):
        raise ValueError(f"{group}/{metric} score row is invalid")
    reported = _finite320(row.get("per_image"), f"{group}/{metric}.per_image")
    if not np.array_equal(reported, values):
        raise ValueError(f"{group}/{metric} per-image vector mismatch")
    mean = float(values.mean())
    if not np.isfinite(row.get("mean", np.nan)) or float(row["mean"]) != mean:
        raise ValueError(f"{group}/{metric} mean mismatch")
    return mean


def _score_queue_state():
    path = Path(score_queue._state_path())
    if not path.is_file():
        raise FileNotFoundError(f"completed SW0179 score queue state is absent: {path}")
    raw = path.read_bytes()
    state = json.loads(raw)
    if state.get("experiment") != EXPECTED_QUEUE_EXPERIMENT or state.get("status") != "complete":
        raise RuntimeError(f"SW0179 score queue is not complete: {state.get('status')}")
    if (state.get("ground_truth_used_for_prediction") is not False
            or state.get("ground_truth_used_for_scoring") is not True):
        raise ValueError("score queue receipt has an invalid GT scope")
    return path, raw, state


def _snapshot_inputs(prepared):
    predictions = {}
    training = {}
    for seed in SEEDS:
        receipt = prepared["frozen_predictions"][seed]
        prediction_file = Path(receipt["result"]["directory"]) / "predictions.npz"
        protocol_file = Path(receipt["result"]["directory"]) / "protocol.json"
        predictions[str(seed)] = {
            "state_sha256": _sha(receipt["state_path"]),
            "prediction_sha256": _sha(prediction_file),
            "protocol_sha256": _sha(protocol_file),
        }
        seed_prepared = prepared["seed1_prepared"] if seed == 1 else prepared["seed_prepared"][seed]
        arm = "product_q_ce"
        artifact = Path(seed_prepared["artifacts"][arm])
        training[str(seed)] = {
            "manifest_sha256": _sha(artifact / "manifest.json"),
            "checkpoint_sha256": _sha(artifact / "checkpoint.pt"),
            "optimizer_sha256": _sha(artifact / "optimizer.pt"),
        }
    return predictions, training


def _load_baseline_vectors():
    summary_path = Path(sw165_summary.OUTPUT)
    if _sha(summary_path) != SW165_SUMMARY_SHA256:
        raise ValueError("pinned SW0165 three-seed summary changed")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("experiment") != "SW0165_foreground_partition_replication":
        raise ValueError("pinned SW0165 summary schema is invalid")
    paths = {0: Path(sw165.score_path(0)), 1: Path(sw164.SCORE_PATH), 2: Path(sw165.score_path(2))}
    summary_seed_hashes = summary.get("seed_score_sha256", {})
    if summary_seed_hashes.get("1") != SW164_SEED1_SCORE_SHA256:
        raise ValueError("SW0165 summary does not bind the pinned SW0164 seed-1 report")
    vectors = {"SW0165_foreground_induced": {}, "Slot70k": {}}
    report_hashes = {}
    for seed in SEEDS:
        path = paths[seed]
        expected_sha = SW164_SEED1_SCORE_SHA256 if seed == 1 else summary_seed_hashes.get(str(seed))
        if not expected_sha or _sha(path) != expected_sha:
            raise ValueError(f"pinned SW0165/SW0164 score seed{seed} changed")
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        if (report.get("status") != "complete" or report.get("seed_scope") != [seed]
                or report.get("ground_truth_used_for_prediction") is not False
                or report.get("ground_truth_used_for_scoring") is not True):
            raise ValueError(f"baseline seed{seed} score scope is invalid")
        report_hashes[str(seed)] = expected_sha
        for group, raw_group in (("SW0165_foreground_induced", "foreground_induced"), ("Slot70k", "Slot70k")):
            vectors[group][seed] = {}
            for metric in METRICS:
                row = report.get("scores", {}).get(raw_group, {}).get(metric, {})
                vector = _finite320(row.get("per_image"), f"baseline seed{seed}/{raw_group}/{metric}")
                if row.get("valid_count") != COUNT or abs(float(row.get("mean", np.nan)) - float(vector.mean())) > 1e-15:
                    raise ValueError(f"baseline seed{seed}/{raw_group}/{metric} summary mismatch")
                vectors[group][seed][metric] = vector
    aggregate = summary.get("three_seed_aggregate", {})
    for group, summary_prefix in (("SW0165_foreground_induced", "candidate"), ("Slot70k", "slot")):
        for metric in METRICS:
            vector = np.stack([vectors[group][seed][metric] for seed in SEEDS]).mean(axis=0)
            expected = aggregate.get(metric, {}).get(f"{summary_prefix}_mean_across_three_seeds")
            if expected is None or abs(float(vector.mean()) - float(expected)) > 1e-15:
                raise ValueError(f"baseline summary aggregation mismatch for {group}/{metric}")
            if group == "SW0165_foreground_induced" and metric == "fg_ari" and abs(float(vector.mean()) - SW165_FG_MEAN) > 1e-15:
                raise ValueError("SW0165 three-seed FG-ARI baseline mismatch")
            if group == "Slot70k":
                expected_slot = {"fg_ari": SLOT_FG_MEAN,
                    "foreground_iou": SLOT_FG_IOU_MEAN,
                    "matched_object_iou": SLOT_OBJECT_IOU_MEAN}.get(metric)
                if expected_slot is not None and abs(float(vector.mean()) - expected_slot) > 1e-15:
                    raise ValueError(f"Slot three-seed {metric} baseline mismatch")
    return vectors, report_hashes, _sha(summary_path)


def _derive_gate(scores, paired):
    candidate = scores["candidate_standalone"]
    sw = scores["SW0165_foreground_induced_standalone"]
    slot = scores["Slot70k_standalone"]
    checks = {
        "fg_strictly_above_sw0165_three_seed_mean": candidate["fg_ari"]["mean"] > sw["fg_ari"]["mean"],
        "fg_paired_ci95_lower_above_zero_vs_sw0165": paired["standalone_candidate_minus_SW0165_foreground_induced"]["fg_ari"]["ci95"][0] > 0,
        "fg_strictly_above_slot_three_seed_mean": candidate["fg_ari"]["mean"] > slot["fg_ari"]["mean"],
        "fg_paired_ci95_lower_above_zero_vs_slot": paired["standalone_candidate_minus_Slot70k"]["fg_ari"]["ci95"][0] > 0,
        "foreground_iou_strictly_above_slot_three_seed_mean": candidate["foreground_iou"]["mean"] > slot["foreground_iou"]["mean"],
        "matched_object_iou_strictly_above_slot_three_seed_mean": candidate["matched_object_iou"]["mean"] > slot["matched_object_iou"]["mean"],
    }
    return {"primary_endpoint": "standalone", **checks,
        "fg_ari_at_least_0_87_goal": candidate["fg_ari"]["mean"] >= 0.87,
        "primary_gate_passed": all(checks.values()),
        "mask_supervised": True, "unsupervised_leaderboard_eligible": False}


def verify(score_path=None):
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0179 verifier must run before CUDA initialization")
    queue_path, queue_bytes, queue_state = _score_queue_state()
    report_path = Path(score_path) if score_path is not None else Path(score.SCORE_PATH)
    if not report_path.is_file():
        raise FileNotFoundError(f"completed SW0179 score report is absent: {report_path}")
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)

    # This producer validates completed predictions/references on CPU and never opens GT.
    prepared = score.prepare_score_inputs_cpu()
    if prepared.get("ground_truth_opened") is not False:
        raise ValueError("CPU input admission unexpectedly opened validation GT")
    source_fingerprint = score.implementation_fingerprint()
    if report.get("implementation_fingerprint") != source_fingerprint:
        raise ValueError("score report implementation fingerprint is stale")
    predictions_before, training_before = _snapshot_inputs(prepared)
    for seed in SEEDS:
        expected = prepared["frozen_predictions"][seed]["result"]
        observed = predictions_before[str(seed)]
        if (observed["prediction_sha256"] != expected["prediction_sha256"]
                or observed["protocol_sha256"] != expected["protocol_sha256"]):
            raise ValueError(f"seed{seed} prediction bytes changed after CPU admission")
    baseline_vectors, baseline_hashes, summary_sha = _load_baseline_vectors()
    if (report.get("experiment") != EXPECTED_EXPERIMENT or report.get("status") != "complete"
            or report.get("seed_scope") != list(SEEDS) or report.get("endpoint_scope") != list(ENDPOINTS)
            or report.get("metric_scope") != list(METRICS)
            or report.get("ground_truth_used_for_prediction") is not False
            or report.get("ground_truth_used_for_scoring") is not True
            or report.get("mask_supervised") is not True
            or report.get("unsupervised_leaderboard_eligible") is not False
            or report.get("ground_truth_scope") != "canonical fixed320 validation modal-native32 masks only"
            or report.get("bootstrap") != {"rng": "numpy RandomState", "seed": BOOTSTRAP_SEED,
                "replicates": BOOTSTRAP_REPLICATES,
                "resampling_unit": "320 paired images after per-image mean across candidate seeds"}
            or report.get("baseline_summary_sha256") != summary_sha
            or report.get("baseline_score_sha256") != baseline_hashes):
        raise ValueError("SW0179 score report scope or baseline provenance is invalid")

    expected_pred = {seed: prepared["frozen_predictions"][seed]["result"]["prediction_sha256"] for seed in SEEDS}
    expected_proto = {seed: prepared["frozen_predictions"][seed]["result"]["protocol_sha256"] for seed in SEEDS}
    expected_states = {seed: prepared["frozen_predictions"][seed]["state_sha256"] for seed in SEEDS}
    hashes = report.get("candidate_prediction_hashes", {})
    if set(hashes) != {str(seed) for seed in SEEDS}:
        raise ValueError("score report prediction seed set mismatch")
    for seed in SEEDS:
        row = hashes[str(seed)]
        if (row.get("prediction_sha256") != expected_pred[seed]
                or row.get("protocol_sha256") != expected_proto[seed]
                or row.get("state_sha256") != expected_states[seed]):
            raise ValueError(f"score report seed{seed} prediction receipt mismatch")
    expected_training = {str(seed): training_before[str(seed)]["manifest_sha256"] for seed in SEEDS}
    if report.get("training_manifest_sha256") != expected_training:
        raise ValueError("score report training manifest hash mismatch")
    if (queue_state.get("score_path") != str(report_path)
            or queue_state.get("score_sha256") != _sha(report_path)
            or queue_state.get("gate") != report.get("gate")):
        raise ValueError("score queue receipt does not bind final score bytes/gate")
    queue_fp = queue_state.get("queue_fingerprint", {})
    if (queue_fp.get("experiment") != EXPECTED_QUEUE_EXPERIMENT
            or queue_fp.get("score_implementation_fingerprint") != source_fingerprint
            or queue_fp.get("score_queue_sha256") != _sha(score_queue.__file__)
            or queue_fp.get("baseline_summary_sha256") != summary_sha):
        raise ValueError("score queue implementation/baseline fingerprint mismatch")
    for seed in SEEDS:
        qrow = queue_fp.get("frozen_candidate_predictions", {}).get(str(seed), {})
        if (qrow.get("prediction_sha256") != expected_pred[seed]
                or qrow.get("protocol_sha256") != expected_proto[seed]
                or qrow.get("state_sha256") != expected_states[seed]):
            raise ValueError(f"score queue seed{seed} frozen prediction binding mismatch")

    dataset = Path(prepared["dataset_path"])
    dataset_before = dataset.stat()
    if (dataset_before.st_size != prepared["dataset_size_bytes"]
            or dataset_before.st_mtime_ns != prepared["dataset_mtime_ns"]):
        raise ValueError("canonical dataset changed before validation mask read")
    if tuple((int(evaluate.IMAGE_IDS[0]), int(evaluate.IMAGE_IDS[-1]) + 1)) != (1320, 1640):
        raise ValueError("verifier is restricted to validation IDs 1320..1639")
    with evaluate.h5py.File(dataset, "r") as hdf:
        if "mask" not in hdf or hdf["mask"].shape[0] < 1640:
            raise ValueError("canonical fixed320 mask dataset is missing")
        gt = contract.modal_native32(np.asarray(hdf["mask"][1320:1640]))

    # Recompute every candidate seed/endpoint separately with the production metric contract.
    scores = {}
    candidate_vectors = {"conditional": {seed: {} for seed in SEEDS},
                         "standalone": {seed: {} for seed in SEEDS}}
    for endpoint in ENDPOINTS:
        by_seed = {seed: contract.production_metrics32(
            prepared["frozen_predictions"][seed]["result"]["labels"][endpoint], gt) for seed in SEEDS}
        group = f"candidate_{endpoint}"
        scores[group] = {}
        for metric in METRICS:
            matrix = np.stack([_finite320(by_seed[seed][metric]["per_image"], f"seed{seed}/{endpoint}/{metric}")
                               for seed in SEEDS])
            if not np.isfinite(matrix).all(): raise ValueError("candidate metric matrix contains non-finite values")
            per_seed = {str(seed): matrix[i].tolist() for i, seed in enumerate(SEEDS)}
            row = report.get("scores", {}).get(group, {}).get(metric, {})
            if set(row.get("per_seed_per_image", {})) != set(per_seed):
                raise ValueError(f"{group}/{metric} per-seed vector set mismatch")
            for seed in SEEDS:
                reported_seed = _finite320(row["per_seed_per_image"][str(seed)], f"{group}/{metric}/seed{seed}")
                if not np.array_equal(reported_seed, matrix[seed]):
                    raise ValueError(f"{group}/{metric} seed{seed} differs from production recomputation")
                candidate_vectors[endpoint][seed][metric] = matrix[seed]
            mean_vector = matrix.mean(axis=0)
            mean = _compare_metric_row(row, mean_vector, group, metric)
            seed_means = {str(seed): float(by_seed[seed][metric]["mean"]) for seed in SEEDS}
            if any(abs(float(row.get("per_seed_mean", {}).get(seed, np.nan)) - value) > 1e-15
                   for seed, value in seed_means.items()):
                raise ValueError(f"{group}/{metric} per-seed mean map mismatch")
            scores[group][metric] = {"per_image": mean_vector, "mean": mean}

    # Independently reconstruct pinned baseline aggregate vectors and compare the full report rows.
    for baseline, vectors in baseline_vectors.items():
        for endpoint in ENDPOINTS:
            group = f"{baseline}_{endpoint}"
            scores[group] = {}
            for metric in METRICS:
                matrix = np.stack([vectors[seed][metric] for seed in SEEDS])
                per_image = matrix.mean(axis=0)
                row = report.get("scores", {}).get(group, {}).get(metric, {})
                if set(row.get("per_seed_per_image", {})) != {str(seed) for seed in SEEDS}:
                    raise ValueError(f"{group}/{metric} baseline seed vectors missing")
                for i, seed in enumerate(SEEDS):
                    if not np.array_equal(_finite320(row["per_seed_per_image"][str(seed)], "baseline report vector"), matrix[i]):
                        raise ValueError(f"{group}/{metric} baseline seed{seed} vector mismatch")
                mean = _compare_metric_row(row, per_image, group, metric)
                expected_seed_means = {str(seed): float(matrix[i].mean()) for i, seed in enumerate(SEEDS)}
                if any(abs(float(row.get("per_seed_mean", {}).get(seed, np.nan)) - value) > 1e-15
                       for seed, value in expected_seed_means.items()):
                    raise ValueError(f"{group}/{metric} baseline seed means mismatch")
                scores[group][metric] = {"per_image": per_image, "mean": mean}

    # Seed-1 output must exactly reproduce the already-pinned SW0178 seed-1 score.
    seed1_report_path = Path(seed1_eval.SCORE_PATH)
    if _sha(seed1_report_path) != score.SEED1_SCORE_SHA256:
        raise ValueError("pinned SW0178 seed-1 score changed")
    seed1_report = json.loads(seed1_report_path.read_text(encoding="utf-8"))
    for metric in METRICS:
        expected = _finite320(seed1_report["scores"]["product_q_ce_standalone"][metric]["per_image"],
                              f"SW0178 seed1 {metric}")
        actual = candidate_vectors["standalone"][1][metric]
        if not np.array_equal(actual, expected):
            raise ValueError(f"SW0179 seed-1 prediction no longer exactly matches SW0178 {metric}")

    rng = np.random.RandomState(BOOTSTRAP_SEED)
    indices = rng.randint(0, COUNT, size=(BOOTSTRAP_REPLICATES, COUNT))
    paired = {}
    for endpoint in ENDPOINTS:
        for baseline in ("SW0165_foreground_induced", "Slot70k"):
            key = f"{endpoint}_candidate_minus_{baseline}"
            paired[key] = {metric: _bootstrap(scores[f"candidate_{endpoint}"][metric]["per_image"],
                scores[f"{baseline}_{endpoint}"][metric]["per_image"], indices) for metric in METRICS}
    _assert_json_equal(report.get("paired_bootstrap"), paired, "paired bootstrap report")
    standalone = scores["candidate_standalone"]
    sw = scores["SW0165_foreground_induced_standalone"]
    slot = scores["Slot70k_standalone"]
    gate_checks = {
        "primary_endpoint": "standalone",
        "fg_strictly_above_sw0165_three_seed_mean": standalone["fg_ari"]["mean"] > sw["fg_ari"]["mean"],
        "fg_paired_ci95_lower_above_zero_vs_sw0165": paired["standalone_candidate_minus_SW0165_foreground_induced"]["fg_ari"]["ci95"][0] > 0,
        "fg_strictly_above_slot_three_seed_mean": standalone["fg_ari"]["mean"] > slot["fg_ari"]["mean"],
        "fg_paired_ci95_lower_above_zero_vs_slot": paired["standalone_candidate_minus_Slot70k"]["fg_ari"]["ci95"][0] > 0,
        "foreground_iou_strictly_above_slot_three_seed_mean": standalone["foreground_iou"]["mean"] > slot["foreground_iou"]["mean"],
        "matched_object_iou_strictly_above_slot_three_seed_mean": standalone["matched_object_iou"]["mean"] > slot["matched_object_iou"]["mean"],
    }
    expected_gate = {**gate_checks, "fg_ari_at_least_0_87_goal": standalone["fg_ari"]["mean"] >= 0.87,
        "primary_gate_passed": all(gate_checks.values()), "mask_supervised": True,
        "unsupervised_leaderboard_eligible": False}
    _assert_json_equal(report.get("gate"), expected_gate, "registered gates")

    # Recheck all bytes and dataset metadata after GT use; no reserve index is touched.
    if score.implementation_fingerprint() != source_fingerprint:
        raise ValueError("SW0179 score source fingerprint changed during audit")
    predictions_after, training_after = _snapshot_inputs(prepared)
    if predictions_after != predictions_before or training_after != training_before:
        raise ValueError("prediction/training artifacts changed during score audit")
    if _load_baseline_vectors()[1:] != (baseline_hashes, summary_sha):
        raise ValueError("pinned baseline artifacts changed during score audit")
    if report_path.read_bytes() != report_bytes or queue_path.read_bytes() != queue_bytes:
        raise ValueError("score report or completed queue receipt changed during audit")
    dataset_after = dataset.stat()
    if dataset_after.st_size != dataset_before.st_size or dataset_after.st_mtime_ns != dataset_before.st_mtime_ns:
        raise ValueError("canonical dataset changed during score audit")
    if torch.cuda.is_initialized(): raise RuntimeError("read-only verifier initialized CUDA")
    return {"status": "verified", "score_sha256": _sha(report_path),
        "queue_state_sha256": _sha(queue_path), "prediction_sha256": expected_pred,
        "protocol_sha256": expected_proto, "training_manifest_sha256": expected_training,
        "scores": {group: {metric: row["mean"] for metric, row in values.items()} for group, values in scores.items()},
        "gate": expected_gate, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "validation_scope": "canonical fixed320 IDs 1320..1639 only; reserve unread"}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.score), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
