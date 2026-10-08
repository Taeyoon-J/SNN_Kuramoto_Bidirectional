"""SW0110 task contracts for the single SW0113 master dispatcher."""
import json
import math
import sys
from pathlib import Path

from run import ARMS, ARCHIVE, OUT, RUNNER, SEEDS


def task_plan():
    jobs = []
    for seed in SEEDS:
        for arm in ARMS:
            pre = {"experiment": "SW0110", "task_id": f"sw0110_preflight_s{seed}_{arm}",
                   "stage": "preflight", "seed": seed, "arm": arm,
                   "depends_on": [], "priority": 0}
            train = {"experiment": "SW0110", "task_id": f"sw0110_train_s{seed}_{arm}",
                     "stage": "train", "seed": seed, "arm": arm,
                     "depends_on": [pre["task_id"]], "priority": 1}
            evaluate = {"experiment": "SW0110", "task_id": f"sw0110_eval_s{seed}_{arm}",
                        "stage": "evaluate", "seed": seed, "arm": arm,
                        "depends_on": [train["task_id"]], "priority": 1}
            jobs.extend((pre, train, evaluate))
    return jobs


def artifact_path(task):
    if task["stage"] == "preflight":
        return ARCHIVE / f"preflight_seed{task['seed']}_{task['arm']}.json"
    folder = OUT / f"seed{task['seed']}_{task['arm']}"
    return folder / ("manifest.json" if task["stage"] == "train" else "evaluation.json")


def command(task, device="cuda"):
    cmd = [sys.executable, str(RUNNER), task["stage"], "--seed", str(task["seed"]),
           "--arm", task["arm"], "--device", device]
    if task["stage"] == "preflight":
        return cmd + ["--output", str(artifact_path(task))]
    if task["stage"] == "train":
        return cmd + ["--output", str(OUT / f"seed{task['seed']}_{task['arm']}")]
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
            passed = (data.get("status") == "passed" and data.get("seed") == task["seed"]
                      and data.get("arm") == task["arm"] and len(data.get("batches", [])) == 4
                      and data.get("zero_p_legacy_equivalence") is True
                      and data.get("native_drive_bitwise_unchanged") is True
                      and data.get("strict_checkpoint_roundtrip_bitwise") is True
                      and data.get("throwaway_b16_adam_update") is True)
            if task["arm"] == "xy_candidate":
                passed = passed and all(math.isfinite(float(b["xy_projection_gradient_norm"]))
                                        and float(b["xy_projection_gradient_norm"]) > 0
                                        for b in data["batches"])
            return passed
        if task["stage"] == "train":
            history = json.loads((path.parent / "history.json").read_text())
            return (data.get("status") == "training_complete" and data.get("seed") == task["seed"]
                    and data.get("arm") == task["arm"] and data.get("updates") == 256
                    and data.get("batch_size") == 16 and data.get("ground_truth_used_for_training") is False
                    and len(history) == 256 and (path.parent / "core.pt").is_file())
        row = data["sweep"][0]
        score = row["scored_targets"]["our_hdf5"]
        metrics = ("fg_ari", "foreground_iou", "matched_object_iou")
        return (data.get("ids") == [1320, 1639] and data.get("images") == 320
                and data.get("ground_truth_used_for_prediction") is False
                and row.get("synchrony_threshold") == 0.5
                and all(score.get("valid_count", {}).get(m) == 320
                        and len(score["per_image"][m]) == 320
                        and all(math.isfinite(float(v)) for v in score["per_image"][m])
                        for m in metrics))
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        return False


def main():
    raise SystemExit("SW0110 tasks are dispatched only by the reviewed SW0113 master queue.")


if __name__ == "__main__":
    main()
