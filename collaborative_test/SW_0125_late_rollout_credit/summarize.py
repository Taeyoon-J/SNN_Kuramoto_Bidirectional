"""Read-only, fail-closed SW0125 three-seed evaluation summary."""
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

from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0117_joint_analytic_rgb import run as sw117
from collaborative_test.SW_0117_joint_analytic_rgb import coordinator as sw117_coordinator
from collaborative_test.SW_0122_joint_rgb_seed_replication import run as sw122
from collaborative_test.SW_0122_joint_rgb_seed_replication import coordinator as sw122_coordinator
from collaborative_test.SW_0122_joint_rgb_seed_replication import summarize as sw122_summary
from collaborative_test.SW_0125_late_rollout_credit import coordinator, run

SEEDS = (0, 1, 2)
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
DRAWS = 10_000
BOOTSTRAP_SEED = 125
SLOT_REFERENCE = sw122_summary.SLOT_REFERENCE
EXPECTED_SLOT_SHA256 = sw122_summary.EXPECTED_SLOT_REFERENCE_SHA256
SLOT_MARGINS = {"foreground_iou": 0.20358913115224261,
                "matched_object_iou": 0.20693697915214362}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _score_payload(payload, identity):
    if (not isinstance(payload, dict)
            or set(payload.get("metrics", {})) != set(METRICS)
            or set(payload.get("valid_count", {})) != set(METRICS)
            or set(payload.get("per_image", {})) != set(METRICS)):
        raise ValueError(f"{identity}: three-metric report schema mismatch")
    means, arrays = {}, {}
    for metric in METRICS:
        values = np.asarray(payload["per_image"][metric], dtype=np.float64)
        mean = float(payload["metrics"][metric])
        if (values.shape != (320,) or int(payload["valid_count"][metric]) != 320
                or not np.isfinite(values).all() or not math.isfinite(mean)
                or abs(float(values.mean()) - mean) > 1e-12):
            raise ValueError(f"{identity}: invalid {metric} values/count/mean")
        means[metric] = mean
        arrays[metric] = values
    return {"metrics": means, "per_image": arrays}


def _report_score(path, *, identity, expected_sha=None, expected_experiment=None,
                  expected_seed=None, expected_arm=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = _sha(path)
    if expected_sha is not None and actual_sha != expected_sha:
        raise ValueError(f"{identity}: immutable evaluation SHA mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("ids") != [1320, 1639] or report.get("images") != 320
            or report.get("ground_truth_used_for_prediction") is not False):
        raise ValueError(f"{identity}: endpoint ID/count/GT contract mismatch")
    if expected_experiment is not None and report.get("experiment") != expected_experiment:
        raise ValueError(f"{identity}: experiment identity mismatch")
    if expected_seed is not None and report.get("seed") != expected_seed:
        raise ValueError(f"{identity}: seed identity mismatch")
    if expected_arm is not None and report.get("arm") != expected_arm:
        raise ValueError(f"{identity}: arm identity mismatch")
    try:
        payload = report["sweep"][0]["scored_targets"]["our_hdf5"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"{identity}: registered SW0040 QCC score path is missing") from exc
    result = _score_payload(payload, identity)
    result["evaluation_sha256"] = actual_sha
    result["path"] = str(path)
    return result, report


def _tasks_valid():
    tasks = coordinator.task_plan()
    if len(tasks) != 9 or len({row["task_id"] for row in tasks}) != 9:
        raise ValueError("SW0125 registration must contain nine unique stage tasks")
    invalid = [row["task_id"] for row in tasks if not coordinator.valid_result(row)]
    if invalid:
        raise RuntimeError("SW0125 tasks are pending or invalid: " + ", ".join(invalid))
    return tasks


def _stage_task(tasks, stage, seed, arm=None):
    matches = [row for row in tasks if row["stage"] == stage and row["seed"] == seed
               and (arm is None or row["arm"] == arm)]
    if len(matches) != 1:
        raise ValueError(f"expected one registered {stage} task for seed {seed}")
    return matches[0]


def _native_source_evaluation_path(source_checkpoint):
    """Native SW0097 score path; endpoint-reproduction reports are separate."""
    return Path(source_checkpoint).parent / "evaluation.json"


def _seed0_source_hashes(historical_summary, endpoint_report_sha256):
    """Return native SW0097 score SHA and distinct SW0116 training-binding SHA."""
    native_sha = historical_summary["arms"]["source"]["evaluation_sha256"]
    if len(native_sha) != 64 or len(endpoint_report_sha256) != 64:
        raise ValueError("seed0 source provenance must contain SHA256 digests")
    return native_sha, endpoint_report_sha256


def _source_score(seed):
    source, manifest_path, manifest, ids, _rows = run.source_contract(seed)
    if seed == 0:
        # The source checkpoint's native SW0097 evaluation is distinct from
        # the SW0116 live-encoder endpoint report used to bind SW0117 training.
        history = json.loads(sw122.HISTORICAL_SUMMARY.read_text(encoding="utf-8"))
        historical_proof = sw122.validate_historical_seed0()
        if historical_proof["summary_sha256"] != _sha(sw122.HISTORICAL_SUMMARY):
            raise ValueError("seed0 historical summary SHA mismatch")
        endpoint = sw117.validate_source_endpoint_review()
        expected_sha, binding_sha = _seed0_source_hashes(history, endpoint["report_sha256"])
        endpoint_path = sw117.SOURCE_ENDPOINT_REPORT
        endpoint_report = json.loads(endpoint_path.read_text(encoding="utf-8"))
    else:
        reference = sw122.validate_source_reference(seed)
        if reference["source_core_sha256"] != _sha(source):
            raise ValueError(f"seed{seed}: source reference checkpoint SHA mismatch")
        if reference["source_manifest_sha256"] != _sha(manifest_path):
            raise ValueError(f"seed{seed}: source reference manifest SHA mismatch")
        expected_sha = reference["source_evaluation_sha256"]
        binding_sha = expected_sha
    if not (source.parent / "COMPLETED").is_file():
        raise FileNotFoundError(f"seed{seed}: immutable SW0097 completion marker missing")
    path = _native_source_evaluation_path(source)
    score, report = _report_score(path, identity=f"seed{seed}/source97", expected_sha=expected_sha)
    if (report.get("checkpoint") != str(source)
            or manifest.get("status") != "complete"
            or manifest.get("source_model_seed") != seed
            or manifest.get("ground_truth_used_for_training") is not False):
        raise ValueError(f"seed{seed}: source report is not for the registered source core")
    if seed == 0:
        cached_source = endpoint_report["scores"]["source16_vs_modal8"]
        for metric in METRICS:
            cached_values = np.asarray(cached_source["per_image"][metric], dtype=np.float64)
            if (cached_source["valid_count"].get(metric) != 320
                    or cached_values.shape != (320,)
                    or not np.isfinite(cached_values).all()
                    or not np.array_equal(cached_values, score["per_image"][metric])):
                raise ValueError(f"seed0 native source evaluation differs from SW0116 proof: {metric}")
    return score, {"checkpoint_sha256": _sha(source),
                   "manifest_sha256": _sha(manifest_path),
                   "evaluation_sha256": expected_sha,
                   "training_binding_evaluation_sha256": binding_sha,
                   "completion_marker_sha256": _sha(source.parent / "COMPLETED"),
                   "training_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()}


def _short_candidate(seed, tasks, source_binding):
    if seed == 0:
        old_coordinator = sw117_coordinator
    else:
        old_coordinator = sw122_coordinator
    for stage in ("preflight", "train", "evaluate"):
        matches = [task for task in old_coordinator.task_plan()
                   if task.get("stage") == stage and task.get("seed") == seed
                   and task.get("arm") == "analytic_candidate"]
        if len(matches) != 1 or not old_coordinator.valid_result(matches[0]):
            raise RuntimeError(f"matched short candidate seed{seed}/{stage} is invalid")
    task = next(task for task in old_coordinator.task_plan()
                if task.get("stage") == "evaluate" and task.get("seed") == seed
                and task.get("arm") == "analytic_candidate")
    path = old_coordinator.artifact_path(task)
    score, report = _report_score(path, identity=f"seed{seed}/matched_short_candidate",
                                   expected_experiment=f"SW0{117 if seed == 0 else 122}",
                                   expected_seed=seed, expected_arm="analytic_candidate")
    manifest = path.parent / "manifest.json"
    train_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    source_eval_key = "source_endpoint_report_sha256" if seed == 0 else "source_evaluation_sha256"
    if (train_manifest.get("status") != "training_complete"
            or train_manifest.get("updates") != 256 or train_manifest.get("batch_size") != 16
            or train_manifest.get("time_steps") != 64 or train_manifest.get("settle") != 32
            or train_manifest.get("matched_shuffle_seed") != 117 + seed
            or train_manifest.get("core_graph_lr") != run.TRAIN_CORE_LR
            or train_manifest.get("encoder_lr") != run.ENCODER_LR
            or train_manifest.get("clip_norm") != run.CLIP_NORM
            or train_manifest.get("lambda_joint") != run.LAMBDA
            or train_manifest.get("ground_truth_used_for_training") is not False):
        raise ValueError(f"seed{seed}: matched short candidate recipe is not the registered recipe")
    source, source_manifest, _meta, ids, _rows = run.source_contract(seed)
    expected_ids_sha = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    if (train_manifest.get("source_core_sha256") != _sha(source)
            or train_manifest.get("source_manifest_sha256") != _sha(source_manifest)
            or train_manifest.get("training_ids_sha256") != expected_ids_sha
            or train_manifest.get(source_eval_key) != source_binding[
                "training_binding_evaluation_sha256"]
            or train_manifest.get("encoder_source_sha256") != _sha(sw117.ENCODER_PATH)
            or train_manifest.get("preprocessing_sha256") != _sha(sw117.STATS_PATH)
            or train_manifest.get("core_sha256") != _sha(path.parent / "core.pt")
            or report.get("training_manifest_sha256") != _sha(manifest)
            or report.get("checkpoint_sha256") != _sha(path.parent / "core.pt")):
        raise ValueError(f"seed{seed}: matched short candidate is not bound to the same source/order")
    return score, {"evaluation_sha256": score["evaluation_sha256"],
                   "training_manifest_sha256": _sha(manifest),
                   "checkpoint_sha256": train_manifest.get("core_sha256"),
                   "implementation_fingerprint": train_manifest.get("implementation_fingerprint")}


def paired_bootstrap(candidate, reference, draws=DRAWS, seed=BOOTSTRAP_SEED):
    """Three-seed paired-image bootstrap, sharing each sampled image across seeds."""
    if draws < 1 or tuple(sorted(candidate)) != SEEDS or tuple(sorted(reference)) != SEEDS:
        raise ValueError("paired bootstrap requires all three seeds and positive draws")
    deltas = np.stack([candidate[s]["per_image"]["fg_ari"]
                       - reference[s]["per_image"]["fg_ari"] for s in SEEDS])
    mean_by_image = deltas.mean(axis=0)
    indices = np.random.default_rng(seed).integers(0, 320, size=(draws, 320))
    draws_values = mean_by_image[indices].mean(axis=1)
    return {"mean_delta": float(mean_by_image.mean()),
            "ci95": [float(value) for value in np.quantile(draws_values, [0.025, 0.975])],
            "draws": draws, "seed": seed,
            "unit": "paired image; identical sampled indices applied to all three seeds"}


def promotion_gates(candidate, short, source, slot_means):
    means = {label: {metric: float(np.mean([values[s]["metrics"][metric] for s in SEEDS]))
                     for metric in METRICS}
             for label, values in (("late", candidate), ("matched_short", short), ("source97", source))}
    ci_short = paired_bootstrap(candidate, short)
    ci_source = paired_bootstrap(candidate, source)
    per_seed_source_gain = {str(seed): candidate[seed]["metrics"]["fg_ari"]
                            - source[seed]["metrics"]["fg_ari"] for seed in SEEDS}
    gates = {
        "late_mean_fg_exceeds_matched_short": means["late"]["fg_ari"] > means["matched_short"]["fg_ari"],
        "late_mean_fg_exceeds_source97": means["late"]["fg_ari"] > means["source97"]["fg_ari"],
        "at_least_two_seed_fg_gains_vs_source97": sum(v > 0 for v in per_seed_source_gain.values()) >= 2,
        "paired_fg_ci_lower_positive_vs_matched_short": ci_short["ci95"][0] > 0,
        "paired_fg_ci_lower_positive_vs_source97": ci_source["ci95"][0] > 0,
        "foreground_iou_exceeds_slot_plus_0_05": (
            means["late"]["foreground_iou"] > slot_means["foreground_iou"] + 0.05),
        "matched_object_iou_exceeds_slot_plus_0_05": (
            means["late"]["matched_object_iou"] > slot_means["matched_object_iou"] + 0.05),
    }
    return means, ci_short, ci_source, per_seed_source_gain, gates


def summarize():
    tasks = _tasks_valid()
    late, short, source = {}, {}, {}
    artifacts = {}
    for seed in SEEDS:
        source[seed], source_binding = _source_score(seed)
        short[seed], short_binding = _short_candidate(seed, tasks, source_binding)
        eval_task = _stage_task(tasks, "evaluate", seed, run.ARM)
        path = coordinator.artifact_path(eval_task)
        late[seed], report = _report_score(path, identity=f"seed{seed}/late_candidate",
            expected_experiment="SW0125", expected_seed=seed, expected_arm=run.ARM)
        manifest_path = path.parent.parent / "manifest.json"
        sidecar_path = path.parent / "evaluation_manifest.json"
        train_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if (report.get("status") != "complete"
                or report.get("training_manifest_sha256") != _sha(manifest_path)
                or report.get("checkpoint_sha256") != _sha(path.parent.parent / "core.pt")
                or report.get("encoder_checkpoint_sha256") != _sha(path.parent.parent / "encoder.pt")
                or sidecar.get("evaluation_sha256") != _sha(path)
                or sidecar.get("training_manifest_sha256") != _sha(manifest_path)
                or train_manifest.get("status") != "training_complete"
                or train_manifest.get("implementation_fingerprint") != run.implementation_fingerprint()
                or train_manifest.get("source_evaluation_sha256") != source_binding[
                    "training_binding_evaluation_sha256"]
                or train_manifest.get("source_core_sha256") != source_binding["checkpoint_sha256"]
                or train_manifest.get("shuffle_seed") != 117 + seed
                or train_manifest.get("updates") != 256 or train_manifest.get("batch_size") != 16
                or train_manifest.get("time_steps") != 1024 or train_manifest.get("settle") != 512
                or train_manifest.get("tail_steps") != 64
                or train_manifest.get("lambda_joint") != run.LAMBDA
                or train_manifest.get("ground_truth_used_for_training") is not False):
            raise ValueError(f"seed{seed}: SW0125 endpoint is not bound to completed fixed recipe")
        artifacts[str(seed)] = {"source97": source_binding, "matched_short_candidate": short_binding,
            "late_evaluation_sha256": _sha(path),
            "late_evaluation_manifest_sha256": _sha(sidecar_path),
            "late_training_manifest_sha256": _sha(manifest_path),
            "late_core_sha256": _sha(path.parent.parent / "core.pt"),
            "late_encoder_sha256": _sha(path.parent.parent / "encoder.pt"),
            "late_history_sha256": _sha(path.parent.parent / "history.json")}
    if _sha(SLOT_REFERENCE) != EXPECTED_SLOT_SHA256:
        raise ValueError("matched 70k Slot reference SHA mismatch")
    slot_doc = json.loads(SLOT_REFERENCE.read_text(encoding="utf-8"))
    if (slot_doc.get("evaluation", {}).get("ids_inclusive") != [1320, 1639]
            or slot_doc.get("evaluation", {}).get("count") != 320
            or slot_doc.get("training_pool", {}).get("unique_images") != 70000):
        raise ValueError("matched Slot reference evaluation/source contract mismatch")
    slot_means = {metric: float(slot_doc["metrics"][metric]) for metric in METRICS}
    if not all(math.isfinite(value) for value in slot_means.values()):
        raise ValueError("matched Slot reference contains nonfinite metric means")
    means, ci_short, ci_source, source_gains, gates = promotion_gates(late, short, source, slot_means)
    return {"experiment": "SW0125", "status": "complete",
            "promotion_gate_passed": all(gates.values()),
            "task_count": len(tasks), "validated_task_ids": [task["task_id"] for task in tasks],
            "per_seed_metrics": {
                "source97": {str(s): source[s]["metrics"] for s in SEEDS},
                "matched_short_analytic_candidate": {str(s): short[s]["metrics"] for s in SEEDS},
                "late_candidate": {str(s): late[s]["metrics"] for s in SEEDS}},
            "three_seed_means": means,
            "late_minus_source97_fg_by_seed": source_gains,
            "paired_bootstrap_late_minus_short_fg": ci_short,
            "paired_bootstrap_late_minus_source97_fg": ci_source,
            "matched_slot_means": slot_means,
            "matched_slot_reference_sha256": _sha(SLOT_REFERENCE),
            "promotion_gates": gates, "artifacts": artifacts,
            "interpretation": "This evaluates the registered full-1024 forward with a 64-step truncated-BPTT tail. It does not establish gating contribution or data-scaling effects.",
            "ground_truth_used_for_prediction": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=run.ARCHIVE / "summary_late_three_seed_20261009.json")
    args = parser.parse_args()
    result = summarize()
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output.exists():
        raise FileExistsError(f"preserving existing summary: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
    print(json.dumps({"status": result["status"],
                      "promotion_gate_passed": result["promotion_gate_passed"],
                      "three_seed_means": result["three_seed_means"],
                      "promotion_gates": result["promotion_gates"],
                      "output": str(args.output)}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
