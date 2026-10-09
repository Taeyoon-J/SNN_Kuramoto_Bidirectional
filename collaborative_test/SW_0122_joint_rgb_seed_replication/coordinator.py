"""Owner-aware SW0113 task adapter for the SW0122 seed-1/2 paired replication."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "results_archive"
OUT = ROOT / "trained_models/SW0122_joint_analytic_rgb_replication"
RUNNER = HERE / "run.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0122_joint_rgb_seed_replication import run


def task_plan():
    tasks = [{"experiment": "SW0122", "task_id": "sw0122_validate_seed0_references",
              "stage": "validate-baselines", "seed": None, "arm": None,
              "depends_on": [], "priority": 0}]
    for seed in (1, 2):
        for arm in run.ARMS:
            pre = {"experiment": "SW0122", "task_id": f"sw0122_preflight_s{seed}_{arm}",
                   "stage": "preflight", "seed": seed, "arm": arm,
                   "depends_on": ["sw0122_validate_seed0_references"], "priority": 1}
            train = {"experiment": "SW0122", "task_id": f"sw0122_train_s{seed}_{arm}",
                     "stage": "train", "seed": seed, "arm": arm,
                     "depends_on": [pre["task_id"]], "priority": 2}
            evaluate = {"experiment": "SW0122", "task_id": f"sw0122_eval_s{seed}_{arm}",
                        "stage": "evaluate", "seed": seed, "arm": arm,
                        "depends_on": [train["task_id"]], "priority": 3}
            tasks.extend((pre, train, evaluate))
    return tasks


def artifact_path(task):
    if task["stage"] == "validate-baselines":
        return ARCHIVE / "historical_seed0_validated.json"
    if task["stage"] == "preflight":
        return ARCHIVE / f"preflight_seed{task['seed']}_{task['arm']}.json"
    folder = OUT / f"seed{task['seed']}_{task['arm']}"
    return folder / ("manifest.json" if task["stage"] == "train" else "evaluation.json")


def command(task, device="cuda"):
    args = [sys.executable, str(RUNNER), task["stage"], "--device", device]
    if task["seed"] is not None:
        args += ["--seed", str(task["seed"]), "--arm", task["arm"]]
    if task["stage"] == "train":
        args += ["--output", str(artifact_path(task).parent)]
    elif task["stage"] == "evaluate":
        folder = artifact_path(task).parent
        args += ["--checkpoint", str(folder / "core.pt"), "--output", str(artifact_path(task))]
    else:
        args += ["--output", str(artifact_path(task))]
    return args


def _finite_score(score):
    names = {"fg_ari", "foreground_iou", "matched_object_iou"}
    if (set(score.get("metrics", {})) != names or set(score.get("valid_count", {})) != names
            or set(score.get("per_image", {})) != names):
        return False
    try:
        return all(score["valid_count"][name] == 320
                   and len(score["per_image"][name]) == 320
                   and all(math.isfinite(float(x)) for x in score["per_image"][name])
                   and math.isfinite(float(score["metrics"][name]))
                   and abs(sum(map(float, score["per_image"][name])) / 320.0
                           - float(score["metrics"][name])) <= 1e-12
                   for name in names)
    except (KeyError, TypeError, ValueError):
        return False


def valid_result(task):
    path = artifact_path(task)
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        stage, seed, arm = task["stage"], task["seed"], task["arm"]
        if stage == "validate-baselines":
            expected = run.validate_historical_seed0()
            return (record.get("status") == "passed" and record.get("experiment") == "SW0122"
                    and record.get("historical_sw0117_seed0") == expected)
        if stage == "preflight":
            if not (record.get("status") == "passed" and record.get("experiment") == "SW0122"
                    and record.get("seed") == seed and record.get("arm") == arm
                    and record.get("implementation_fingerprint") == run.implementation_fingerprint()
                    and record.get("ground_truth_used") is False
                    and len(record.get("batches", [])) == 4
                    and record.get("throwaway_b16_adam_update") is True
                    and record.get("throwaway_parameter_groups_changed") == {
                        "encoder": True, "graph": True, "core": True}
                    and len(record.get("training_ids", [])) == 4096
                    and record.get("matched_shuffle_seed") == 117 + seed
                    and record.get("lambda_joint") == (run.LAMBDA if arm == "analytic_candidate" else 0.0)
                    and math.isfinite(float(record.get("fixed_rows_scramble_rgb_loss_excess", float("nan"))))
                    and float(record["fixed_rows_scramble_rgb_loss_excess"]) > 0):
                return False
            for row in record["batches"]:
                parity = row.get("cache_live_parity", {})
                diffs = parity.get("cache_vs_live_trace_max_abs_diff", {})
                if (len(row.get("global_ids", [])) != 16
                        or not math.isfinite(float(parity.get("max_gamma_cache_abs_diff", float("nan"))))
                        or float(parity["max_gamma_cache_abs_diff"]) > 2e-5
                        or not math.isfinite(float(parity.get("cached_baseline_old_loss_abs_diff", float("nan"))))
                        or float(parity["cached_baseline_old_loss_abs_diff"]) > 2e-5
                        or parity.get("cached_baseline_labels_H_exact") is not True
                        or parity.get("same_input_reference_loss_exact") is not True
                        or parity.get("same_input_reference_traces_exact") is not True
                        or parity.get("same_input_reference_labels_H_exact") is not True
                        or set(diffs) != {"spikes", "core_out", "theta", "component_spikes"}
                        or not all(math.isfinite(float(v)) for v in diffs.values())):
                    return False
                for family in ("encoder", "graph", "core", "joint"):
                    old = float(row.get("old_gradient_norms", {}).get(family, float("nan")))
                    if not math.isfinite(old) or old <= 0:
                        return False
                    if arm == "analytic_candidate":
                        rgb = float(row.get("rgb_gradient_norms", {}).get(family, float("nan")))
                        if not math.isfinite(rgb) or rgb <= 0:
                            return False
                if arm == "analytic_candidate":
                    qnorm = float(row.get("rgb_to_Q_gradient_norm", float("nan")))
                    if not math.isfinite(qnorm) or qnorm <= 0:
                        return False
            return True
        folder = path.parent
        manifest_path = folder / "manifest.json"
        if stage == "train":
            core, encoder = folder / "core.pt", folder / "encoder.pt"
            optimizer, history = folder / "optimizer.pt", folder / "history.json"
            rows = json.loads(history.read_text(encoding="utf-8"))
            pf_path = ARCHIVE / f"preflight_seed{seed}_{arm}.json"
            pf = json.loads(pf_path.read_text(encoding="utf-8"))
            return (record.get("status") == "training_complete" and record.get("experiment") == "SW0122"
                    and record.get("seed") == seed and record.get("arm") == arm
                    and record.get("updates") == 256 and record.get("batch_size") == 16
                    and record.get("matched_shuffle_seed") == 117 + seed
                    and record.get("ground_truth_used_for_training") is False
                    and record.get("implementation_fingerprint") == run.implementation_fingerprint()
                    and record.get("preflight_sha256") == run.sha(pf_path)
                    and record.get("source_core_sha256") == pf.get("source_core_sha256")
                    and record.get("source_manifest_sha256") == pf.get("source_manifest_sha256")
                    and record.get("source_evaluation_sha256") == pf.get("source_evaluation_sha256")
                    and record.get("encoder_source_sha256") == pf.get("encoder_sha256")
                    and record.get("preprocessing_sha256") == pf.get("preprocessing_sha256")
                    and record.get("gamma_train_sha256") == pf.get("gamma_train_sha256")
                    and record.get("gamma_train_manifest_sha256") == pf.get("gamma_train_manifest_sha256")
                    and record.get("rgb_cache_sha256") == pf.get("rgb_cache_sha256")
                    and record.get("rgb_cache_manifest_sha256") == pf.get("rgb_cache_manifest_sha256")
                    and record.get("training_ids_sha256") == pf.get("training_ids_sha256")
                    and record.get("matched_shuffle_seed") == 117 + seed
                    and record.get("lambda_joint") == pf.get("lambda_joint")
                    and record.get("lambda_artifact_sha256") == pf.get("lambda_artifact_sha256")
                    and record.get("core_sha256") == run.sha(core)
                    and record.get("encoder_sha256") == run.sha(encoder)
                    and record.get("optimizer_sha256") == run.sha(optimizer)
                    and record.get("history_sha256") == run.sha(history)
                    and (folder / "TRAINING_COMPLETED").is_file()
                    and len(rows) == 256 and [r.get("update") for r in rows] == list(range(1, 257))
                    and all(math.isfinite(float(r[k])) for r in rows
                            for k in ("total", "old", "primary", "positive_actual_spike_product",
                                      "rgb", "gradient_norm_preclip")))
        if stage == "evaluate":
            training = json.loads(manifest_path.read_text(encoding="utf-8"))
            sidecar_path = folder / "evaluation_manifest.json"
            if not sidecar_path.is_file():
                return False
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            return (record.get("experiment") == "SW0122" and record.get("seed") == seed
                    and record.get("arm") == arm and record.get("ground_truth_used_for_prediction") is False
                    and record.get("training_manifest_sha256") == run.sha(manifest_path)
                    and record.get("checkpoint_sha256") == run.sha(folder / "core.pt")
                    and record.get("encoder_checkpoint_sha256") == run.sha(folder / "encoder.pt")
                    and record.get("evaluation_runner_sha256") == run.sha(RUNNER)
                    and sidecar.get("experiment") == "SW0122" and sidecar.get("seed") == seed
                    and sidecar.get("arm") == arm and sidecar.get("evaluation_sha256") == run.sha(path)
                    and sidecar.get("training_manifest_sha256") == run.sha(manifest_path)
                    and sidecar.get("checkpoint_sha256") == run.sha(folder / "core.pt")
                    and sidecar.get("evaluation_contract", {}).get("ids") == [1320, 1639]
                    and training.get("status") == "training_complete"
                    and _finite_score(record["sweep"][0]["scored_targets"]["our_hdf5"]))
        return False
    except (OSError, KeyError, IndexError, TypeError, ValueError, AssertionError):
        return False
