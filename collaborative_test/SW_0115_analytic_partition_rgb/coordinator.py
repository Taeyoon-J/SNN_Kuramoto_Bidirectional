"""Seed-0 task adapter for the central SW0113 dispatcher."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0115_analytic_partition_rgb.run import (
    ARMS, ARCHIVE, OUT, RUNNER, SEEDS, implementation_fingerprint, sha,
)


def task_plan():
    tasks = []
    for arm in ARMS:
        preflight = {"experiment": "SW0115", "task_id": f"sw0115_preflight_s0_{arm}",
                     "stage": "preflight", "seed": 0, "arm": arm,
                     "depends_on": [], "priority": 0}
        train = {"experiment": "SW0115", "task_id": f"sw0115_train_s0_{arm}",
                 "stage": "train", "seed": 0, "arm": arm,
                 "depends_on": [preflight["task_id"]], "priority": 1}
        evaluate = {"experiment": "SW0115", "task_id": f"sw0115_eval_s0_{arm}",
                    "stage": "evaluate", "seed": 0, "arm": arm,
                    "depends_on": [train["task_id"]], "priority": 1}
        tasks.extend((preflight, train, evaluate))
    return tasks


def artifact_path(task):
    if task["stage"] == "preflight":
        return ARCHIVE / f"preflight_seed{task['seed']}_{task['arm']}.json"
    folder = OUT / f"seed{task['seed']}_{task['arm']}"
    return folder / ("manifest.json" if task["stage"] == "train" else "evaluation.json")


def command(task, device="cuda"):
    cmd = [sys.executable, str(RUNNER), task["stage"], "--seed", str(task["seed"]),
           "--arm", task["arm"], "--device", device]
    if task["stage"] in ("preflight", "train"):
        output = artifact_path(task)
        if task["stage"] == "train":
            output = output.parent
        return cmd + ["--output", str(output)]
    folder = OUT / f"seed{task['seed']}_{task['arm']}"
    return cmd + ["--checkpoint", str(folder / "core.pt"),
                  "--output", str(artifact_path(task))]


def _finite_metrics(score):
    expected = ("fg_ari", "foreground_iou", "matched_object_iou")
    if set(score.get("valid_count", {})) != set(expected):
        return False
    if set(score.get("per_image", {})) != set(expected):
        return False
    for metric in expected:
        values = score["per_image"][metric]
        try:
            if (score["valid_count"][metric] != 320 or len(values) != 320
                    or not all(math.isfinite(float(x)) for x in values)
                    or not math.isfinite(float(score["metrics"][metric]))
                    or abs(sum(float(x) for x in values) / 320.0
                           - float(score["metrics"][metric])) > 1e-12):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def training_preflight_matches(task, record):
    preflight = ARCHIVE / f"preflight_seed{task['seed']}_{task['arm']}.json"
    return preflight.is_file() and record.get("preflight_sha256") == sha(preflight)


def valid_result(task):
    path = artifact_path(task)
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text())
        if task["stage"] == "preflight":
            ok = (record.get("status") == "passed" and record.get("experiment") == "SW0115"
                  and record.get("seed") == task["seed"] and record.get("arm") == task["arm"]
                  and record.get("implementation_fingerprint") == implementation_fingerprint()
                  and record.get("ground_truth_used") is False
                  and record.get("graph_frozen") is True and len(record.get("batches", [])) == 4
                  and record.get("throwaway_b16_adam_update") is True
                  and len(record.get("training_ids", [])) == 4096
                  and record.get("matched_shuffle_seed") == 117 + task["seed"]
                  and bool(record.get("rgb_cache_sha256"))
                  and bool(record.get("gamma_train_sha256"))
                  and bool(record.get("source_core_sha256")))
            batches = record.get("batches", [])
            ok = ok and all(math.isfinite(float(row.get("rgb_to_q_gradient_norm", 0)))
                            and float(row["rgb_to_q_gradient_norm"]) > 0
                            and math.isfinite(float(row.get("rgb_eligible_core_gradient_norm", 0)))
                            and float(row["rgb_eligible_core_gradient_norm"]) > 0
                            for row in batches)
            if task["arm"] == "analytic_candidate":
                ok = ok and math.isfinite(float(record.get("lambda_rgb", float("nan"))))
                ok = ok and float(record["scrambled_minus_real_rgb_loss"]) > 0
            return bool(ok)
        if task["stage"] == "train":
            history = json.loads((path.parent / "history.json").read_text())
            core = path.parent / "core.pt"
            history_path = path.parent / "history.json"
            return (record.get("status") == "training_complete"
                    and record.get("experiment") == "SW0115"
                    and record.get("seed") == task["seed"] and record.get("arm") == task["arm"]
                    and record.get("updates") == 256 and record.get("batch_size") == 16
                    and record.get("ground_truth_used_for_training") is False
                    and record.get("implementation_fingerprint") == implementation_fingerprint()
                    and len(record.get("training_ids", [])) == 4096
                    and training_preflight_matches(task, record)
                    and len(history) == 256 and (path.parent / "TRAINING_COMPLETED").is_file()
                    and core.is_file() and record.get("core_sha256") == sha(core)
                    and record.get("history_sha256") == sha(history_path)
                    and all(math.isfinite(float(row[k])) for row in history
                            for k in ("total", "old", "primary", "positive_actual_spike_product",
                                      "rgb", "gradient_norm_preclip")))
        if (record.get("experiment") != "SW0115" or record.get("seed") != task["seed"]
                or record.get("arm") != task["arm"] or record.get("ids") != [1320, 1639]
                or record.get("images") != 320
                or record.get("ground_truth_used_for_prediction") is not False):
            return False
        train_manifest_path = OUT / f"seed{task['seed']}_{task['arm']}" / "manifest.json"
        train_manifest = json.loads(train_manifest_path.read_text())
        checkpoint = train_manifest_path.parent / "core.pt"
        if (train_manifest.get("status") != "training_complete"
                or train_manifest.get("arm") != task["arm"]
                or train_manifest.get("seed") != task["seed"]
                or train_manifest.get("core_sha256") != record.get("checkpoint_sha256")
                or record.get("training_manifest_sha256") != sha(train_manifest_path)
                or not checkpoint.is_file() or sha(checkpoint) != record.get("checkpoint_sha256")):
            return False
        sidecar_path = path.parent / "evaluation_manifest.json"
        if not sidecar_path.is_file():
            return False
        sidecar = json.loads(sidecar_path.read_text())
        if (sidecar.get("experiment") != "SW0115"
                or sidecar.get("seed") != task["seed"]
                or sidecar.get("arm") != task["arm"]
                or sidecar.get("evaluation_file") != path.name
                or sidecar.get("evaluation_sha256") != sha(path)
                or sidecar.get("training_manifest_sha256") != sha(train_manifest_path)
                or sidecar.get("checkpoint_sha256") != sha(checkpoint)
                or sidecar.get("evaluation_runner_sha256") != record.get("evaluation_runner_sha256")
                or sidecar.get("shared_evaluator_sha256") != record.get("shared_evaluator_sha256")
                or sidecar.get("evaluation_contract") != {
                    "ids": [1320, 1639], "images": 320, "batch_size": 8,
                    "time_steps": 1024, "settle": 512, "membrane_vth": 0.06,
                    "readout_threshold": 0.50, "min_group_size": 2,
                    "background": "largest_component", "ground_truth_used_for_prediction": False,
                }):
            return False
        return _finite_metrics(record["sweep"][0]["scored_targets"]["our_hdf5"])
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        return False


def main():
    raise SystemExit("SW0115 is dispatched only by the reviewed central SW0113 queue.")


if __name__ == "__main__":
    main()
