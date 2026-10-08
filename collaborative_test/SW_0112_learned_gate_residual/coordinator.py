"""SW0112 task adapter for the central SW0113 dispatcher."""
import json
import math
import sys
from pathlib import Path

from run import ARCHIVE, ARMS, OUT, RUNNER, SEEDS, sha


def task_plan():
    result = []
    for seed in SEEDS:
        for arm in ARMS:
            pre = {"experiment": "SW0112", "task_id": f"sw0112_preflight_s{seed}_{arm}",
                   "stage": "preflight", "seed": seed, "arm": arm,
                   "depends_on": [], "priority": 0}
            train = {"experiment": "SW0112", "task_id": f"sw0112_train_s{seed}_{arm}",
                     "stage": "train", "seed": seed, "arm": arm,
                     "depends_on": [pre["task_id"]], "priority": 1}
            evaluate = {"experiment": "SW0112", "task_id": f"sw0112_eval_s{seed}_{arm}",
                        "stage": "evaluate", "seed": seed, "arm": arm,
                        "depends_on": [train["task_id"]], "priority": 1}
            result.extend((pre, train, evaluate))
    return result


def artifact_path(task):
    if task["stage"] == "preflight":
        return ARCHIVE / f"preflight_seed{task['seed']}_{task['arm']}.json"
    folder = OUT / f"seed{task['seed']}_{task['arm']}"
    return folder / ("manifest.json" if task["stage"] == "train" else "evaluation.json")


def command(task, device="cuda"):
    stage = task["stage"]
    cmd = [sys.executable, str(RUNNER), stage, "--seed", str(task["seed"]),
           "--arm", task["arm"], "--device", device]
    if stage in ("preflight", "train"):
        output = (artifact_path(task) if stage == "preflight"
                  else OUT / f"seed{task['seed']}_{task['arm']}")
        return cmd + ["--output", str(output)]
    folder = OUT / f"seed{task['seed']}_{task['arm']}"
    return cmd + ["--checkpoint", str(folder / "core.pt"),
                  "--output", str(folder / "evaluation.json")]


def valid_result(task):
    path = artifact_path(task)
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
        if task["stage"] == "preflight":
            rows = data.get("batches", [])
            good = (data.get("status") == "passed" and data.get("experiment") == "SW0112"
                    and data.get("seed") == task["seed"] and data.get("arm") == task["arm"]
                    and data.get("ground_truth_used") is False and len(rows) == 4
                    and data.get("zero_residual_legacy_equivalence") is True
                    and data.get("graph_frozen") is True
                    and data.get("bounded_gate_formula_finite") is True
                    and data.get("membrane_gate_fold_order_exact") is True
                    and data.get("strict_checkpoint_roundtrip_bitwise") is True
                    and data.get("throwaway_b16_adam_update") is True)
            if task["arm"] == "gate_candidate":
                good = good and all(math.isfinite(float(row["gate_residual_gradient_norm"]))
                                    and float(row["gate_residual_gradient_norm"]) > 0
                                    for row in rows)
            return bool(good)
        if task["stage"] == "train":
            history = json.loads((path.parent / "history.json").read_text())
            return (data.get("status") == "training_complete" and data.get("experiment") == "SW0112"
                    and data.get("seed") == task["seed"] and data.get("arm") == task["arm"]
                    and data.get("updates") == 256 and data.get("batch_size") == 16
                    and data.get("phase_delay_steps") == 2
                    and data.get("ground_truth_used_for_training") is False
                    and len(history) == 256 and (path.parent / "TRAINING_COMPLETED").is_file()
                    and (path.parent / "core.pt").is_file()
                    and data.get("core_sha256") == sha(path.parent / "core.pt")
                    and all(all(math.isfinite(float(row[key])) for key in
                                ("total", "phase_primary", "positive_actual_spike_product", "preclip_norm"))
                            for row in history)
                    and data.get("source_core_sha256") is not None
                    and len(data.get("training_ids", [])) == 4096)
        sidecar = path.parent / "evaluation_manifest.json"
        manifest = json.loads(sidecar.read_text())
        score = data["sweep"][0]["scored_targets"]["our_hdf5"]
        checkpoint = OUT / f"seed{task['seed']}_{task['arm']}" / "core.pt"
        return (manifest.get("status") == "complete" and manifest.get("seed") == task["seed"]
                and manifest.get("arm") == task["arm"]
                and manifest.get("evaluation_sha256") == sha(path)
                and Path(manifest.get("checkpoint", "")).resolve() == checkpoint.resolve()
                and manifest.get("checkpoint_sha256") == sha(checkpoint)
                and manifest.get("batch_size") == 8 and manifest.get("steps") == 1024
                and manifest.get("settle") == 512 and manifest.get("membrane_vth") == 0.06
                and manifest.get("synchrony_threshold") == 0.5
                and manifest.get("ground_truth_used_for_prediction") is False
                and data.get("ids") == [1320, 1639] and data.get("images") == 320
                and data.get("ground_truth_used_for_prediction") is False
                and data["sweep"][0].get("synchrony_threshold") == 0.5
                and all(score.get("valid_count", {}).get(metric) == 320
                        and len(score["per_image"][metric]) == 320
                        and all(math.isfinite(float(v)) for v in score["per_image"][metric])
                        and math.isfinite(float(score["metrics"][metric]))
                        and abs(sum(float(v) for v in score["per_image"][metric]) / 320
                                  - float(score["metrics"][metric])) <= 1e-12
                        for metric in ("fg_ari", "foreground_iou", "matched_object_iou")))
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        return False
