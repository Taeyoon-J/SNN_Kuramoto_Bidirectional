"""Read-only, fail-closed SW0123 nine-trajectory summary and pilot gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0123_adaptive_temporal_assignment import coordinator, evaluate, run
from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0122_joint_rgb_seed_replication import run as sw122
from collaborative_test.SW_0122_joint_rgb_seed_replication import summarize as sw122_summary

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
SEEDS = (0, 1, 2)
ARMS = ("legacy_full", "adaptive_full", "gate_only_control")
DRAWS = 10_000
BOOTSTRAP_SEED = 123
SLOT_REFERENCE = sw122_summary.SLOT_REFERENCE


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _score_payload(score, *, identity):
    if not isinstance(score, dict) or set(score.get("metrics", {})) != set(METRICS):
        raise ValueError(f"{identity}: metric keys do not match the registered three-metric contract")
    if set(score.get("valid_count", {})) != set(METRICS) or set(score.get("per_image", {})) != set(METRICS):
        raise ValueError(f"{identity}: missing valid-count or per-image metric arrays")
    means, per_image = {}, {}
    for metric in METRICS:
        values = np.asarray(score["per_image"][metric], dtype=np.float64)
        mean = float(score["metrics"][metric])
        if (values.shape != (320,) or int(score["valid_count"][metric]) != 320
                or not np.isfinite(values).all() or not math.isfinite(mean)
                or abs(float(values.mean()) - mean) > 1e-12):
            raise ValueError(f"{identity}: invalid {metric} count, values, or mean")
        means[metric] = mean
        per_image[metric] = values
    return {"metrics": means, "per_image": per_image}


def _endpoint_score(path: Path, *, seed: int, arm: str):
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("status") != "complete" or report.get("experiment") != "SW0123"
            or report.get("seed") != seed or report.get("arm") != arm
            or report.get("ids") != [1320, 1639] or report.get("images") != 320
            or report.get("ground_truth_used_for_prediction") is not False):
        raise ValueError(f"seed{seed}/{arm}: endpoint identity/GT contract mismatch")
    score = _score_payload(report.get("primary_assignment"), identity=f"seed{seed}/{arm}/primary")
    secondary = _score_payload(report.get("secondary_actual_qcc"),
                               identity=f"seed{seed}/{arm}/secondary_qcc")
    score.update(evaluation_sha256=_sha(path), evaluation_manifest_sha256=_sha(path.parent / "evaluation_manifest.json"))
    score["secondary_actual_qcc_metrics"] = secondary["metrics"]
    return score, report


def _source_score(seed):
    source, manifest_path, manifest, ids, _rows = run.source_contract(seed)
    path = source.parent / "evaluation.json"
    marker = source.parent / "COMPLETED"
    if not marker.is_file():
        raise FileNotFoundError(f"seed{seed}: completed SW0097 source marker is missing")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or manifest.get("ground_truth_used_for_training") is not False
            or report.get("checkpoint") != str(source) or report.get("ids") != [1320, 1639]
            or report.get("images") != 320 or report.get("ground_truth_used_for_prediction") is not False):
        raise ValueError(f"seed{seed}: source evaluation identity/GT contract mismatch")
    score = sw122._validate_source_evaluation_records(seed, source, ids, manifest, report)
    nested = report["sweep"][0]["scored_targets"]["our_hdf5"]
    normalized = _score_payload(nested, identity=f"seed{seed}/source")
    for metric in METRICS:
        if abs(normalized["metrics"][metric] - float(score[metric])) > 1e-12:
            raise ValueError(f"seed{seed}: source validator mean disagreement for {metric}")
    evaluation_sha = _sha(path)
    if seed == 0:
        historical = sw122.validate_historical_seed0()
        historical_summary = json.loads(sw122.HISTORICAL_SUMMARY.read_text(encoding="utf-8"))
        if historical_summary["arms"]["source"].get("evaluation_sha256") != evaluation_sha:
            raise ValueError("seed0 source evaluation differs from frozen SW0117 historical evidence")
        source_binding = {"binding": "frozen_SW0117_seed0_summary", **historical}
    else:
        bound_manifests = {}
        for arm in ("control", "analytic_candidate"):
            train_manifest_path = sw122.OUT / f"seed{seed}_{arm}" / "manifest.json"
            train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
            if (train_manifest.get("experiment") != "SW0122" or train_manifest.get("seed") != seed
                    or train_manifest.get("status") != "training_complete"
                    or train_manifest.get("source_core_sha256") != _sha(source)
                    or train_manifest.get("source_manifest_sha256") != _sha(manifest_path)
                    or train_manifest.get("source_evaluation_sha256") != evaluation_sha
                    or train_manifest.get("training_ids_sha256") != hashlib.sha256(
                        np.asarray(ids, dtype="<i8").tobytes()).hexdigest()):
                raise ValueError(f"seed{seed}: source evaluation is not bound by SW0122 {arm} training")
            bound_manifests[arm] = _sha(train_manifest_path)
        source_binding = {"binding": "both_completed_SW0122_paired_train_manifests",
                          "training_manifest_sha256": bound_manifests}
    normalized.update(evaluation_sha256=_sha(path), source_core_sha256=_sha(source),
                      source_manifest_sha256=_sha(manifest_path),
                      completed_marker_sha256=_sha(marker), source_binding=source_binding,
                      training_ids_sha256=hashlib.sha256(
                          np.asarray(ids, dtype="<i8").tobytes()).hexdigest())
    return normalized


def paired_bootstrap(candidate, reference, *, draws=DRAWS, seed=BOOTSTRAP_SEED):
    """Paired image resampling; each sampled image index is shared across seeds."""
    if draws < 1 or tuple(sorted(candidate)) != SEEDS or tuple(sorted(reference)) != SEEDS:
        raise ValueError("bootstrap requires all three ordered seed rows and positive draws")
    delta = np.stack([candidate[s]["per_image"]["fg_ari"]
                      - reference[s]["per_image"]["fg_ari"] for s in SEEDS])
    image_mean_delta = delta.mean(axis=0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 320, size=(draws, 320))
    samples = image_mean_delta[indices].mean(axis=1)
    return {"mean_delta": float(image_mean_delta.mean()),
            "ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
            "draws": draws, "seed": seed,
            "unit": "paired validation image; the same sampled indices apply to all three seeds"}


def promotion_gates(scores, slot_means, adaptive_guards):
    means = {arm: {metric: float(np.mean([scores[arm][s]["metrics"][metric] for s in SEEDS]))
                   for metric in METRICS} for arm in (*ARMS, "source")}
    candidate = scores["adaptive_full"]
    ci = {reference: paired_bootstrap(candidate, scores[reference])
          for reference in ("legacy_full", "gate_only_control", "source")}
    source_gains = {str(seed): candidate[seed]["metrics"]["fg_ari"]
                    - scores["source"][seed]["metrics"]["fg_ari"] for seed in SEEDS}
    gates = {
        "adaptive_mean_fg_exceeds_legacy_full": means["adaptive_full"]["fg_ari"] > means["legacy_full"]["fg_ari"],
        "adaptive_mean_fg_exceeds_gate_only_control": means["adaptive_full"]["fg_ari"] > means["gate_only_control"]["fg_ari"],
        "adaptive_mean_fg_exceeds_source": means["adaptive_full"]["fg_ari"] > means["source"]["fg_ari"],
        "paired_ci_lower_positive_vs_legacy_full": ci["legacy_full"]["ci95"][0] > 0,
        "paired_ci_lower_positive_vs_gate_only_control": ci["gate_only_control"]["ci95"][0] > 0,
        "paired_ci_lower_positive_vs_source": ci["source"]["ci95"][0] > 0,
        "at_least_two_seed_fg_gains_vs_source": sum(value > 0 for value in source_gains.values()) >= 2,
        "foreground_iou_exceeds_matched_slot_plus_0_05": (
            means["adaptive_full"]["foreground_iou"] > slot_means["foreground_iou"] + 0.05),
        "matched_object_iou_exceeds_matched_slot_plus_0_05": (
            means["adaptive_full"]["matched_object_iou"] > slot_means["matched_object_iou"] + 0.05),
        "all_adaptive_posttraining_guards_passed": (
            set(adaptive_guards) == {str(seed) for seed in SEEDS}
            and all(seed_guards.get("training_guard_status") is True
                    and seed_guards.get("activity_guard") is True
                    and seed_guards.get("assignment_credit_guard") is True
                    for seed_guards in adaptive_guards.values())),
    }
    return means, ci, source_gains, gates


def _task_status():
    tasks = coordinator.task_plan()
    if len(tasks) != 27 or len({task["task_id"] for task in tasks}) != 27:
        raise AssertionError("registered SW0123 plan must contain 27 unique stages")
    pending, invalid = [], []
    for task in tasks:
        path = coordinator.artifact_path(task)
        if not path.is_file():
            pending.append(task["task_id"])
        elif not coordinator.valid_result(task):
            invalid.append(task["task_id"])
    return tasks, pending, invalid


def _slot_means():
    expected_sha = sw122_summary.EXPECTED_SLOT_REFERENCE_SHA256
    if _sha(SLOT_REFERENCE) != expected_sha:
        raise AssertionError("matched own-data Slot reference SHA mismatch")
    record = json.loads(SLOT_REFERENCE.read_text(encoding="utf-8"))
    if (record.get("source_experiment") != "SW0095_full70k_aligned_loss"
            or record.get("training_pool", {}).get("unique_images") != 70000
            or record.get("evaluation", {}).get("ids_inclusive") != [1320, 1639]
            or record.get("evaluation", {}).get("count") != 320):
        raise AssertionError("matched Slot reference contract mismatch")
    means = {metric: float(record["metrics"][metric]) for metric in METRICS}
    if not all(math.isfinite(x) for x in means.values()):
        raise AssertionError("nonfinite matched Slot metrics")
    return means, _sha(SLOT_REFERENCE)


def summarize():
    tasks, pending, invalid = _task_status()
    if pending or invalid:
        return {"experiment": "SW0123", "status": "pending" if pending else "invalid_artifacts",
                "registered_task_count": len(tasks), "validated_task_count": len(tasks) - len(pending) - len(invalid),
                "pending_task_ids": pending, "invalid_task_ids": invalid,
                "promotion_gate_passed": False}

    scores = {arm: {} for arm in (*ARMS, "source")}
    artifacts = {}
    adaptive_guards = {}
    for seed in SEEDS:
        scores["source"][seed] = _source_score(seed)
        for arm in ARMS:
            folder = run.OUT / f"seed{seed}_{arm}"
            manifest_path = folder / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("status") != "training_complete" or manifest.get("experiment") != "SW0123"
                    or manifest.get("seed") != seed or manifest.get("arm") != arm
                    or manifest.get("updates") != 256 or manifest.get("batch_size") != 16
                    or manifest.get("time_steps") != 64 or manifest.get("settle") != 32
                    or manifest.get("ground_truth_used_for_training") is not False
                    or manifest.get("implementation_fingerprint") != run.implementation_fingerprint()):
                raise ValueError(f"seed{seed}/{arm}: training manifest contract mismatch")
            source, source_manifest, _source_record, ids, rows = run.source_contract(seed)
            ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
            if (manifest.get("source_core_sha256") != _sha(source)
                    or manifest.get("source_manifest_sha256") != _sha(source_manifest)
                    or manifest.get("training_ids_sha256") != ids_sha
                    or manifest.get("training_ids") != ids.tolist()
                    or manifest.get("gamma_rows") != rows.tolist()
                    or manifest.get("matched_shuffle_seed") != 117 + seed):
                raise ValueError(f"seed{seed}/{arm}: source/order provenance mismatch")
            file_names = ("core", "encoder", "assignment_head", "rgb_decoder", "core_optimizer",
                          "head_optimizer", "history")
            file_hashes = {name: _sha(folder / f"{name}.pt") if name != "history"
                           else _sha(folder / "history.json") for name in file_names}
            if any(manifest.get(f"{name}_sha256") != value for name, value in file_hashes.items()):
                raise ValueError(f"seed{seed}/{arm}: checkpoint/history hash mismatch")
            eval_path = coordinator.artifact_path(next(t for t in tasks if t["stage"] == "evaluate"
                                                       and t["seed"] == seed and t["arm"] == arm))
            score, report = _endpoint_score(eval_path, seed=seed, arm=arm)
            if (report.get("training_manifest_sha256") != _sha(manifest_path)
                    or report.get("training_artifact_sha256") != file_hashes):
                raise ValueError(f"seed{seed}/{arm}: evaluation is not bound to completed training artifacts")
            scores[arm][seed] = score
            guards = manifest.get("final_training_guards", {})
            if arm == "adaptive_full":
                guard_result = {
                    "training_guard_status": guards.get("status") == "passed",
                    "activity_guard": guards.get("activity_guard") == "passed",
                    "assignment_credit_guard": guards.get("assignment_credit_guard") == "passed",
                }
                adaptive_guards[str(seed)] = guard_result
            artifacts[f"seed{seed}_{arm}"] = {
                "training_manifest_sha256": _sha(manifest_path),
                "training_files_sha256": file_hashes,
                "preflight_sha256": manifest.get("preflight_sha256"),
                "calibration_sha256": manifest.get("calibration_sha256"),
                "evaluation_sha256": score["evaluation_sha256"],
                "evaluation_manifest_sha256": score["evaluation_manifest_sha256"],
                "final_training_guards": guards,
            }
    slot, slot_sha = _slot_means()
    means, bootstrap, gains, gates = promotion_gates(scores, slot, adaptive_guards)
    code_hashes = {"summary": _sha(Path(__file__)), "runner": _sha(run.HERE / "run.py"),
                   "evaluator": _sha(evaluate.HERE / "evaluate.py"),
                   "coordinator": _sha(coordinator.HERE / "coordinator.py"),
                   "dispatcher": _sha(HERE / "dispatcher.py"),
                   "protocol": _sha(HERE / "protocol.json"), "matched_slot_reference": slot_sha}
    return {
        "experiment": "SW0123", "status": "complete",
        "promotion_gate_passed": all(gates.values()),
        "task_validation": {"registered_task_count": len(tasks),
                            "validated_task_ids": [task["task_id"] for task in tasks]},
        "per_seed_metrics": {label: {str(seed): scores[label][seed]["metrics"] for seed in SEEDS}
                             for label in (*ARMS, "source")},
        "per_seed_secondary_actual_qcc_metrics": {
            arm: {str(seed): scores[arm][seed]["secondary_actual_qcc_metrics"] for seed in SEEDS}
            for arm in ARMS},
        "three_seed_means": means,
        "adaptive_minus_source_fg_ari_by_seed": gains,
        "paired_bootstrap_adaptive_minus_reference_fg_ari": bootstrap,
        "matched_slot_metrics": slot, "matched_slot_reference_sha256": slot_sha,
        "adaptive_posttraining_guards": adaptive_guards,
        "gate_checks": gates,
        "artifact_provenance": {
            "source_evaluation_sha256": {str(s): scores["source"][s]["evaluation_sha256"] for s in SEEDS},
            "source_core_sha256": {str(s): scores["source"][s]["source_core_sha256"] for s in SEEDS},
            "source_manifest_sha256": {str(s): scores["source"][s]["source_manifest_sha256"] for s in SEEDS},
            "source_evaluation_binding": {str(s): scores["source"][s]["source_binding"] for s in SEEDS},
            "endpoint_evaluation_sha256": {label: {str(s): scores[label][s]["evaluation_sha256"]
                                                   for s in SEEDS} for label in ARMS},
            "training": artifacts,
            "implementation_sha256": code_hashes,
        },
        "interpretation": "All nine SW0123 trajectories are reported. Promotion is allowed only if every preregistered pilot gate passes; this summary does not alter prior experiment decisions.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=HERE / "results_archive" / "pilot_summary_20261009.json")
    args = parser.parse_args(argv)
    result = summarize()
    if result["status"] != "complete":
        print(json.dumps({"status": result["status"], "registered_task_count": result["registered_task_count"],
                          "validated_task_count": result["validated_task_count"],
                          "pending_task_ids": result["pending_task_ids"],
                          "invalid_task_ids": result["invalid_task_ids"]}, allow_nan=False))
        raise SystemExit(2)
    if args.output.exists():
        raise FileExistsError(f"preserving existing SW0123 summary: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"],
                      "promotion_gate_passed": result["promotion_gate_passed"],
                      "three_seed_means": result["three_seed_means"],
                      "gate_checks": result["gate_checks"], "output": str(args.output)}, allow_nan=False))


if __name__ == "__main__":
    main()
