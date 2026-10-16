"""GT-last three-seed SW0179 score aggregation for the fixed 320-image set."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

if "--worker" not in __import__("sys").argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ["USE_PYTORCH_KERNEL_CACHE"] = "0"

import numpy as np
import h5py

from collaborative_test.SW_0179_qce_seed_replication import predict_evaluate as pe
from collaborative_test.SW_0179_qce_seed_replication import train
from collaborative_test.SW_0178_temporal_class_competition import predict_evaluate as seed1_eval
from collaborative_test.SW_0165_foreground_partition_replication import (
    predict_evaluate as sw165, replication_summary as sw165_summary,
)
from collaborative_test.SW_0164_foreground_preserving_partition import predict_evaluate as sw164
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0130_phase_state_integration import run as sw130

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (0, 1, 2)
ENDPOINTS = ("conditional", "standalone")
METRICS = tuple(contract.METRICS)
COUNT = int(contract.COUNT)
IMAGE_IDS = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES = 178, 10_000
SCORE_PATH = HERE / "results_archive" / "evaluation_seedmean_20261016.json"
SEED1_SCORE_SHA256 = "f7c404615edef11c57ac6e1d71dd450ededab7f2e9307b54996d6365a51106df"
SW164_SEED1_SCORE_SHA256 = "978a9d97658837eb19f414a7fb7dbb35d86c6135cb1213e12e63e2837fd65df6"
SW165_SUMMARY_SHA256 = "72b17ca1a07d312400ab74f6ee7533c89f0e9d9a6c304378dd8e61c16ba379c5"
SW165_FG_BASELINE = 0.8042989149474198
SLOT_FG_BASELINE = 0.7816519719609135
SLOT_FOREGROUND_IOU_BASELINE = 0.20436012776655546
SLOT_MATCHED_IOU_BASELINE = 0.2031649861884501


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def implementation_fingerprint():
    relatives = (
        "collaborative_test/SW_0179_qce_seed_replication/DESIGN_KO.md",
        "collaborative_test/SW_0179_qce_seed_replication/train.py",
        "collaborative_test/SW_0179_qce_seed_replication/training_queue.py",
        "collaborative_test/SW_0179_qce_seed_replication/predict_evaluate.py",
        "collaborative_test/SW_0179_qce_seed_replication/prediction_queue.py",
        "collaborative_test/SW_0179_qce_seed_replication/score.py",
        "collaborative_test/SW_0178_temporal_class_competition/predict_evaluate.py",
        "collaborative_test/SW_0165_foreground_partition_replication/predict_evaluate.py",
        "collaborative_test/SW_0165_foreground_partition_replication/replication_summary.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/predict_evaluate.py",
        "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
    )
    paths = {relative: ROOT / relative for relative in relatives}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing: raise FileNotFoundError(f"SW0179 score dependencies missing: {missing}")
    return {relative: sha256_file(path) for relative, path in paths.items()}


def _prediction_queue_state(seed):
    path = pe.HERE / "results_archive" / "prediction_queue_20261016" / f"product_q_ce_seed{seed}_state.json"
    if not path.is_file(): raise FileNotFoundError(f"SW0179 prediction queue state missing for seed {seed}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _validate_new_prediction(seed, prepared):
    path, state = _prediction_queue_state(seed)
    result = pe.validate_prediction(train.ARMS[0], seed=seed, prepared=prepared)
    if (state.get("experiment") != "SW0179_prediction_queue" or state.get("seed") != seed
            or state.get("arm") != train.ARMS[0] or state.get("status") != "complete"
            or state.get("prediction_sha256") != result["prediction_sha256"]
            or state.get("protocol_sha256") != result["protocol_sha256"]
            or state.get("ground_truth_used_for_prediction") is not False):
        raise ValueError(f"SW0179 seed{seed} prediction queue receipt mismatch")
    return {"result": result, "state": state, "state_path": path,
        "state_sha256": sha256_file(path)}


def _validate_seed1_prediction(prepared):
    arm = "product_q_ce"
    pred = seed1_eval.validate_prediction(arm, prepared=prepared)
    score_path = Path(seed1_eval.SCORE_PATH)
    if sha256_file(score_path) != SEED1_SCORE_SHA256:
        raise ValueError("immutable SW0178 seed-1 score changed")
    score_report = json.loads(score_path.read_text(encoding="utf-8"))
    if score_report.get("prediction_sha256", {}).get(arm) != pred.get("prediction_sha256"):
        raise ValueError("immutable SW0178 seed-1 score does not bind its prediction")
    state_path = (seed1_eval.HERE / "results_archive" / "prediction_queue_v1_20261016"
                  / f"{arm}_state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (state.get("experiment") != "SW0178_prediction_queue_v1" or state.get("arm") != arm
            or state.get("status") != "complete" or state.get("prediction_sha256") != pred["prediction_sha256"]
            or state.get("protocol_sha256") != pred["protocol_sha256"]
            or state.get("ground_truth_used_for_prediction") is not False):
        raise ValueError("immutable SW0178 seed-1 prediction receipt mismatch")
    return {"result": pred, "state": state, "state_path": state_path,
        "state_sha256": sha256_file(state_path), "score_report": score_report,
        "score_path": score_path, "score_sha256": SEED1_SCORE_SHA256}


def _baseline_reports():
    summary_path = Path(sw165_summary.OUTPUT)
    if sha256_file(summary_path) != SW165_SUMMARY_SHA256:
        raise ValueError("pinned SW0165 three-seed score summary changed")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("experiment") != "SW0165_foreground_partition_replication" or summary.get("status") != "complete":
        raise ValueError("SW0165 three-seed reference summary is invalid")
    reports = {}
    paths = {
        0: Path(sw165.score_path(0)),
        1: Path(sw164.SCORE_PATH),
        2: Path(sw165.score_path(2)),
    }
    hashes = summary.get("seed_score_sha256", {})
    if hashes.get("1") != SW164_SEED1_SCORE_SHA256:
        raise ValueError("SW0165 summary seed-1 score does not match its pinned SW0164 source")
    for seed in SEEDS:
        path = paths[seed]
        expected_sha = SW164_SEED1_SCORE_SHA256 if seed == 1 else hashes.get(str(seed))
        if not isinstance(expected_sha, str) or sha256_file(path) != expected_sha:
            raise ValueError(f"SW0165 incumbent score seed{seed} changed")
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        if (report.get("status") != "complete" or report.get("seed_scope") != [seed]
                or report.get("ground_truth_used_for_prediction") is not False
                or report.get("ground_truth_used_for_scoring") is not True):
            raise ValueError(f"baseline score seed{seed} scope invalid")
        for group in ("foreground_induced", "Slot70k"):
            for metric in METRICS:
                row = report.get("scores", {}).get(group, {}).get(metric, {})
                values = np.asarray(row.get("per_image"), dtype=np.float64)
                if values.shape != (COUNT,) or not np.isfinite(values).all():
                    raise ValueError(f"baseline score seed{seed} {group}/{metric} invalid")
                if not np.isclose(float(values.mean()), float(row.get("mean", np.nan)), rtol=0, atol=1e-15):
                    raise ValueError(f"baseline score seed{seed} {group}/{metric} mean mismatch")
        labels = {"SW0165_foreground_induced": {}, "Slot70k": {}}
        raw_group = {"SW0165_foreground_induced": "foreground_induced", "Slot70k": "Slot70k"}
        for group, report_group in raw_group.items():
            for metric in METRICS:
                labels[group][metric] = np.asarray(report["scores"][report_group][metric]["per_image"], dtype=np.float64)
        reports[seed] = {"path": path, "sha256": expected_sha, "report": report, "metric_vectors": labels}
    aggregate = summary.get("three_seed_aggregate", {})
    for metric in METRICS:
        candidate = _average_seed_metric_vectors({seed: reports[seed]["metric_vectors"]["SW0165_foreground_induced"][metric]
                                                  for seed in SEEDS})
        slot = _average_seed_metric_vectors({seed: reports[seed]["metric_vectors"]["Slot70k"][metric]
                                             for seed in SEEDS})
        if (not np.isclose(float(candidate.mean()), aggregate.get(metric, {}).get("candidate_mean_across_three_seeds", np.nan), rtol=0, atol=1e-15)
                or not np.isclose(float(slot.mean()), aggregate.get(metric, {}).get("slot_mean_across_three_seeds", np.nan), rtol=0, atol=1e-15)):
            raise ValueError(f"SW0165 summary/per-seed metric mismatch for {metric}")
    if (not np.isclose(aggregate["fg_ari"]["candidate_mean_across_three_seeds"], SW165_FG_BASELINE, rtol=0, atol=1e-15)
            or not np.isclose(aggregate["fg_ari"]["slot_mean_across_three_seeds"], SLOT_FG_BASELINE, rtol=0, atol=1e-15)
            or not np.isclose(aggregate["foreground_iou"]["slot_mean_across_three_seeds"], SLOT_FOREGROUND_IOU_BASELINE, rtol=0, atol=1e-15)
            or not np.isclose(aggregate["matched_object_iou"]["slot_mean_across_three_seeds"], SLOT_MATCHED_IOU_BASELINE, rtol=0, atol=1e-15)):
        raise ValueError("SW0165/Slot three-seed means differ from registered design baselines")
    return {"summary_path": summary_path, "summary_sha256": SW165_SUMMARY_SHA256,
        "summary": summary, "reports": reports}


def prepare_score_inputs_cpu():
    """Validate all three frozen prediction sets and baselines without opening GT."""
    if pe.torch.cuda.is_initialized():
        raise RuntimeError("SW0179 score preparation must precede CUDA initialization")
    new_prepared, frozen = {}, {}
    for seed in (0, 2):
        prepared = pe.prepare_prediction_inputs_cpu(seed)
        receipt = _validate_new_prediction(seed, prepared)
        new_prepared[seed] = prepared
        frozen[seed] = receipt
    seed1_prepared = seed1_eval.prepare_prediction_inputs_cpu()
    frozen[1] = _validate_seed1_prediction(seed1_prepared)
    source_t = {0: new_prepared[0]["references"]["source_T"]["labels"],
                2: new_prepared[2]["references"]["source_T"]["labels"],
                1: seed1_prepared["references"]["source_T"]["labels"]}
    for seed in SEEDS:
        result = frozen[seed]["result"]
        labels = result["labels"]
        support = result.get("standalone_support")
        if (set(labels) != set(ENDPOINTS) or np.asarray(labels["conditional"]).shape != (COUNT, 32, 32)
                or np.asarray(labels["standalone"]).shape != (COUNT, 32, 32)
                or support is None or np.asarray(support).shape != (COUNT, 32, 32)):
            raise ValueError(f"SW0179 seed{seed} fixed320 prediction schema invalid")
        if not np.array_equal(np.asarray(labels["conditional"]) > 0, np.asarray(source_t[seed]) > 0):
            raise ValueError(f"SW0179 seed{seed} conditional foreground differs from SourceT")
        if not np.array_equal(np.asarray(labels["standalone"]) > 0, np.asarray(support) > 0):
            raise ValueError(f"SW0179 seed{seed} standalone foreground differs from native QCC")
    baselines = _baseline_reports()
    dataset = Path(sw165.DATASET)
    stat = dataset.stat()
    return {"seed_prepared": new_prepared, "seed1_prepared": seed1_prepared,
        "frozen_predictions": frozen, "source_T_labels": source_t, "baseline": baselines,
        "baseline_metric_vectors": {group: {seed: baselines["reports"][seed]["metric_vectors"][group]
            for seed in SEEDS} for group in ("SW0165_foreground_induced", "Slot70k")},
        "dataset": dataset, "dataset_path": str(dataset.resolve()),
        "dataset_size_bytes": stat.st_size, "dataset_mtime_ns": stat.st_mtime_ns,
        "implementation_fingerprint": implementation_fingerprint(),
        "ground_truth_opened": False}


def _metrics_by_group(predictions, gt):
    rows = {}
    for group, seed_labels in predictions.items():
        by_seed = {seed: contract.production_metrics32(labels, gt) for seed, labels in seed_labels.items()}
        aggregate = {}
        for metric in METRICS:
            matrix = np.stack([np.asarray(by_seed[seed][metric]["per_image"], dtype=np.float64)
                               for seed in SEEDS])
            if matrix.shape != (3, COUNT) or not np.isfinite(matrix).all():
                raise ValueError("SW0179 per-seed metric matrix is invalid")
            per_image = matrix.mean(axis=0)
            aggregate[metric] = {"per_image": per_image, "mean": float(per_image.mean()),
                "per_seed_mean": {str(seed): float(by_seed[seed][metric]["mean"]) for seed in SEEDS},
                "per_seed_per_image": {str(seed): np.asarray(by_seed[seed][metric]["per_image"], dtype=np.float64)
                    for seed in SEEDS}}
        rows[group] = aggregate
    return rows


def _average_seed_metric_vectors(vectors):
    if not isinstance(vectors, dict) or set(vectors) != set(SEEDS):
        raise ValueError("SW0179 requires three finite metric vectors over the shared fixed320 images")
    matrix = np.stack([np.asarray(vectors[seed], dtype=np.float64) for seed in SEEDS])
    if matrix.shape != (3, COUNT) or not np.isfinite(matrix).all():
        raise ValueError("SW0179 requires three finite metric vectors over the shared fixed320 images")
    return matrix.mean(axis=0)


def _bootstrap_rows(candidate, baseline, indices):
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if delta.shape != (COUNT,) or not np.isfinite(delta).all():
        raise ValueError("SW0179 bootstrap requires paired finite fixed320 image means")
    sampled = delta[indices].mean(axis=1)
    return {"mean": float(delta.mean()),
        "ci95": [float(value) for value in np.percentile(sampled, [2.5, 97.5])],
        "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
        "resampling_unit": "paired_image_after_three_seed_average"}


def _gates(scores, paired):
    standalone = scores["candidate_standalone"]
    fg = standalone["fg_ari"]["mean"]
    fg_baseline = scores["SW0165_foreground_induced_standalone"]["fg_ari"]["mean"]
    slot_baseline = scores["Slot70k_standalone"]["fg_ari"]["mean"]
    gate = {
        "primary_endpoint": "standalone",
        "fg_strictly_above_sw0165_three_seed_mean": fg > fg_baseline,
        "fg_paired_ci95_lower_above_zero_vs_sw0165": paired["standalone_candidate_minus_SW0165_fg_ari"]["ci95"][0] > 0,
        "fg_strictly_above_slot_three_seed_mean": fg > slot_baseline,
        "fg_paired_ci95_lower_above_zero_vs_slot": paired["standalone_candidate_minus_Slot70k_fg_ari"]["ci95"][0] > 0,
        "foreground_iou_strictly_above_slot_three_seed_mean": standalone["foreground_iou"]["mean"] > scores["Slot70k_standalone"]["foreground_iou"]["mean"],
        "matched_object_iou_strictly_above_slot_three_seed_mean": standalone["matched_object_iou"]["mean"] > scores["Slot70k_standalone"]["matched_object_iou"]["mean"],
        "fg_ari_at_least_0_87_goal": fg >= 0.87,
    }
    gate["primary_gate_passed"] = all(value for key, value in gate.items()
        if key not in {"primary_endpoint", "fg_ari_at_least_0_87_goal", "primary_gate_passed"})
    gate["mask_supervised"] = True
    gate["unsupervised_leaderboard_eligible"] = False
    return gate


def score_all(*, output=SCORE_PATH, prepared=None):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    p = prepared or prepare_score_inputs_cpu()
    frozen = p["frozen_predictions"]
    # Re-read prediction validators and queue receipts immediately before the first GT access.
    for seed in SEEDS:
        receipt = frozen[seed]
        if sha256_file(receipt["state_path"]) != receipt["state_sha256"]:
            raise ValueError(f"SW0179 seed{seed} prediction state changed before GT")
        result = receipt["result"]
        if seed == 1:
            fresh = seed1_eval.validate_prediction("product_q_ce", prepared=p["seed1_prepared"])
        else:
            fresh = pe.validate_prediction(train.ARMS[0], seed=seed, prepared=p["seed_prepared"][seed])
        if (fresh["prediction_sha256"] != result["prediction_sha256"]
                or fresh["protocol_sha256"] != result["protocol_sha256"]):
            raise ValueError(f"SW0179 seed{seed} prediction changed before GT")
    baseline_files = {str(row["path"]): row["sha256"] for row in p["baseline"]["reports"].values()}
    summary_path = p["baseline"]["summary_path"]
    if sha256_file(summary_path) != p["baseline"]["summary_sha256"]:
        raise ValueError("SW0165 score summary changed before GT")
    dataset = p["dataset"]
    before = dataset.stat()
    if (before.st_size != p["dataset_size_bytes"] or before.st_mtime_ns != p["dataset_mtime_ns"]
            or Path(dataset).resolve() != Path(sw165.DATASET).resolve()):
        raise ValueError("canonical HDF5 identity changed before SW0179 score")

    # First validation-mask read occurs only after all three candidate prediction protocols and hashes passed.
    with h5py.File(dataset, "r") as hdf:
        if "mask" not in hdf or hdf["mask"].shape[0] <= int(IMAGE_IDS[-1]):
            raise ValueError("canonical fixed320 validation masks are missing")
        gt = contract.modal_native32(np.asarray(hdf["mask"][int(IMAGE_IDS[0]):int(IMAGE_IDS[-1]) + 1]))

    candidate = {endpoint: {seed: np.asarray(frozen[seed]["result"]["labels"][endpoint]) for seed in SEEDS}
                 for endpoint in ENDPOINTS}
    predictions = {"candidate_conditional": candidate["conditional"],
        "candidate_standalone": candidate["standalone"]}
    scores = _metrics_by_group(predictions, gt)
    # Baseline reports were themselves pinned by the frozen SW0165 replication summary.
    for group in ("SW0165_foreground_induced", "Slot70k"):
        for endpoint in ENDPOINTS:
            rows = {}
            for metric in METRICS:
                vectors = {seed: p["baseline_metric_vectors"][group][seed][metric] for seed in SEEDS}
                matrix = np.stack([vectors[seed] for seed in SEEDS])
                averaged = _average_seed_metric_vectors(vectors)
                rows[metric] = {"per_image": averaged, "mean": float(averaged.mean()),
                    "per_seed_mean": {str(seed): float(matrix[i].mean()) for i, seed in enumerate(SEEDS)},
                    "per_seed_per_image": {str(seed): matrix[i] for i, seed in enumerate(SEEDS)}}
            scores[f"{group}_{endpoint}"] = rows

    seed1_baseline = p["frozen_predictions"][1]["score_report"]
    seed1_actual = contract.production_metrics32(candidate["standalone"][1], gt)
    for metric in METRICS:
        actual = seed1_actual[metric]
        expected = seed1_baseline["scores"]["product_q_ce_standalone"][metric]
        if (actual["mean"] != expected["mean"] or
                not np.array_equal(np.asarray(actual["per_image"]), np.asarray(expected["per_image"]))):
            raise ValueError(f"SW0178 seed-1 candidate fails production metric recomputation: {metric}")

    rng = np.random.RandomState(BOOTSTRAP_SEED)
    indices = rng.randint(0, COUNT, size=(BOOTSTRAP_REPLICATES, COUNT))
    paired = {}
    for endpoint in ENDPOINTS:
        cand = scores[f"candidate_{endpoint}"]
        for baseline_name in ("SW0165_foreground_induced", "Slot70k"):
            base = scores[f"{baseline_name}_{endpoint}"]
            key = f"{endpoint}_candidate_minus_{baseline_name}"
            paired[key] = {metric: _bootstrap_rows(cand[metric]["per_image"], base[metric]["per_image"], indices)
                           for metric in METRICS}
    gate = _gates(scores, {"standalone_candidate_minus_SW0165_fg_ari": paired["standalone_candidate_minus_SW0165_foreground_induced"]["fg_ari"],
                           "standalone_candidate_minus_Slot70k_fg_ari": paired["standalone_candidate_minus_Slot70k"]["fg_ari"]})

    for seed in SEEDS:
        receipt = frozen[seed]
        if sha256_file(receipt["state_path"]) != receipt["state_sha256"]:
            raise ValueError(f"SW0179 seed{seed} prediction state changed during GT scoring")
        expected_pred = receipt["result"]["prediction_sha256"]
        expected_proto = receipt["result"]["protocol_sha256"]
        if seed == 1:
            current = seed1_eval.validate_prediction("product_q_ce", prepared=p["seed1_prepared"])
        else:
            current = pe.validate_prediction(train.ARMS[0], seed=seed, prepared=p["seed_prepared"][seed])
        if current["prediction_sha256"] != expected_pred or current["protocol_sha256"] != expected_proto:
            raise ValueError(f"SW0179 seed{seed} prediction bytes changed during GT scoring")
    if sha256_file(summary_path) != SW165_SUMMARY_SHA256:
        raise ValueError("SW0165 summary changed during GT scoring")
    for path, expected in baseline_files.items():
        if sha256_file(path) != expected: raise ValueError("baseline score file changed during GT scoring")
    after = dataset.stat()
    if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
        raise ValueError("canonical HDF5 changed during SW0179 scoring")

    json_scores = {group: {metric: {**values, "per_image": values["per_image"].tolist(),
        "per_seed_per_image": {seed: vector.tolist() for seed, vector in values["per_seed_per_image"].items()}}
        for metric, values in group_scores.items()} for group, group_scores in scores.items()}
    hashes = {str(seed): {"prediction_sha256": frozen[seed]["result"]["prediction_sha256"],
                          "protocol_sha256": frozen[seed]["result"]["protocol_sha256"],
                          "state_sha256": frozen[seed]["state_sha256"]} for seed in SEEDS}
    manifest_hashes = {
        "0": sha256_file(p["seed_prepared"][0]["artifacts"][train.ARMS[0]] / "manifest.json"),
        "2": sha256_file(p["seed_prepared"][2]["artifacts"][train.ARMS[0]] / "manifest.json"),
        "1": sha256_file(p["seed1_prepared"]["artifacts"]["product_q_ce"] / "manifest.json"),
    }
    report = {"experiment": "SW0179_qce_seed_replication_three_seed_score", "status": "complete",
        "seed_scope": list(SEEDS), "endpoint_scope": list(ENDPOINTS), "metric_scope": list(METRICS),
        "candidate_prediction_hashes": hashes, "training_manifest_sha256": manifest_hashes,
        "baseline_summary_sha256": SW165_SUMMARY_SHA256,
        "baseline_score_sha256": {str(seed): p["baseline"]["reports"][seed]["sha256"] for seed in SEEDS},
        "scores": json_scores, "paired_bootstrap": paired, "gate": gate,
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
        "mask_supervised": True, "unsupervised_leaderboard_eligible": False,
        "ground_truth_scope": "canonical fixed320 validation modal-native32 masks only",
        "dataset": {"path": p["dataset_path"], "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns},
        "bootstrap": {"rng": "numpy RandomState", "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "resampling_unit": "320 paired images after per-image mean across candidate seeds"},
        "implementation_fingerprint": implementation_fingerprint()}
    _atomic_json(output, report)
    report["score_sha256"] = sha256_file(output)
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=SCORE_PATH)
    args = parser.parse_args(argv)
    report = score_all(output=args.output)
    print(json.dumps({"status": report["status"], "gate": report["gate"],
        "score_sha256": report["score_sha256"], "output": str(args.output)}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
