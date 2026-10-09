"""Seed-0 SW0120 sequential task adapter for the reviewed central dispatcher."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "results_archive"
OUT = ROOT / "trained_models/SW0120_degenerate_trace_gradient_guard"
RUNNER = HERE / "run.py"
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0117_joint_analytic_rgb import run as sw117
from collaborative_test.SW_0120_degenerate_trace_gradient_guard import run as sw120


def task_plan():
    control = {"experiment": "SW0120", "task_id": "sw0120_validate_sw117_control_s0",
               "stage": "validate-control", "seed": 0, "arm": "analytic_candidate",
               "depends_on": [], "priority": 0}
    pf = {"experiment": "SW0120", "task_id": "sw0120_preflight_s0_guarded_candidate",
          "stage": "preflight", "seed": 0, "arm": "analytic_candidate",
          "depends_on": [control["task_id"]], "priority": 0}
    train = {"experiment": "SW0120", "task_id": "sw0120_train_s0_guarded_candidate",
             "stage": "train", "seed": 0, "arm": "analytic_candidate",
             "depends_on": [pf["task_id"]], "priority": 1}
    evaluate = {"experiment": "SW0120", "task_id": "sw0120_eval_s0_guarded_candidate",
                "stage": "evaluate", "seed": 0, "arm": "analytic_candidate",
                "depends_on": [train["task_id"]], "priority": 1}
    return [control, pf, train, evaluate]


def artifact_path(task):
    stage = task["stage"]
    if stage == "validate-control":
        return ARCHIVE / "sw0117_control_validation.json"
    if stage == "preflight":
        return ARCHIVE / "preflight_seed0_guarded_candidate.json"
    if stage == "train":
        return OUT / "seed0_guarded_candidate/manifest.json"
    if stage == "evaluate":
        return OUT / "seed0_guarded_candidate/evaluation.json"
    raise ValueError(f"unknown SW0120 stage {stage}")


def command(task, device="cuda"):
    import sys
    output = artifact_path(task)
    if task["stage"] == "train":
        output = output.parent
    args = [sys.executable, str(RUNNER), task["stage"], "--output", str(output),
            "--device", device]
    if task["stage"] == "evaluate":
        args += ["--checkpoint", str(OUT / "seed0_guarded_candidate/core.pt")]
    return args


def valid_result(task):
    path = artifact_path(task)
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if task["stage"] == "validate-control":
            control, manifest = sw117_control()
            return (record.get("status") == "passed" and record.get("experiment") == "SW0120"
                    and record.get("control_manifest_sha256") == sw117.sha(control / "manifest.json")
                    and record.get("control_core_sha256") == sw117.sha(control / "core.pt")
                    and record.get("lambda_joint") == 7.865416617457706
                    and record.get("source_core_sha256") == manifest.get("source_core_sha256"))
        if task["stage"] == "preflight":
            proof_sha = sw117.sha(HERE / "update3_guard_proof_20261009.json")
            return (record.get("status") == "passed" and record.get("experiment") == "SW0120"
                    and record.get("seed") == 0 and record.get("arm") == "analytic_candidate"
                    and record.get("update3_proof_sha256") == proof_sha
                    and record.get("implementation_fingerprint") == sw120.implementation_fingerprint()
                    and record.get("ground_truth_used") is False
                    and len(record.get("same_input_reference_parity", [])) == 4
                    and record.get("throwaway_parameter_groups_changed") == {
                        "encoder": True, "graph": True, "core": True}
                    and float(record.get("fixed_h_rows_scramble_rgb_loss_excess", float("nan"))) > 0)
        if task["stage"] == "train":
            folder = path.parent
            core, encoder = folder / "core.pt", folder / "encoder.pt"
            optimizer, history = folder / "optimizer.pt", folder / "history.json"
            rows = json.loads(history.read_text(encoding="utf-8"))
            preflight = ARCHIVE / "preflight_seed0_guarded_candidate.json"
            return (record.get("status") == "training_complete"
                    and record.get("experiment") == "SW0120"
                    and record.get("seed") == 0 and record.get("arm") == "analytic_candidate"
                    and record.get("updates") == 256 and record.get("batch_size") == 16
                    and record.get("matched_shuffle_seed") == 117
                    and record.get("ground_truth_used_for_training") is False
                    and record.get("preflight_sha256") == sw117.sha(preflight)
                    and record.get("implementation_fingerprint") == sw120.implementation_fingerprint()
                    and record.get("core_sha256") == sw117.sha(core)
                    and record.get("encoder_sha256") == sw117.sha(encoder)
                    and record.get("optimizer_sha256") == sw117.sha(optimizer)
                    and record.get("history_sha256") == sw117.sha(history)
                    and len(rows) == 256 and (folder / "TRAINING_COMPLETED").is_file())
        if task["stage"] == "evaluate":
            training = OUT / "seed0_guarded_candidate/manifest.json"
            sidecar = path.parent / "evaluation_manifest.json"
            score = record["sweep"][0]["scored_targets"]["our_hdf5"]
            names = {"fg_ari", "foreground_iou", "matched_object_iou"}
            return (record.get("experiment") == "SW0120"
                    and record.get("ground_truth_used_for_prediction") is False
                    and record.get("training_manifest_sha256") == sw117.sha(training)
                    and sidecar.is_file()
                    and json.loads(sidecar.read_text(encoding="utf-8")).get("evaluation_sha256") == sw117.sha(path)
                    and set(score.get("metrics", {})) == names
                    and set(score.get("valid_count", {})) == names
                    and set(score.get("per_image", {})) == names
                    and all(score["valid_count"][k] == 320
                            and len(score["per_image"][k]) == 320
                            and all(math.isfinite(float(v)) for v in score["per_image"][k])
                            and math.isfinite(float(score["metrics"][k]))
                            and abs(sum(float(v) for v in score["per_image"][k]) / 320.0
                                       - float(score["metrics"][k])) <= 1e-12
                            for k in names))
        return False
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def sw117_control():
    matches = []
    for p in sw117.OUT.rglob("manifest.json"):
        m = json.loads(p.read_text(encoding="utf-8"))
        if m.get("experiment") == "SW0117" and m.get("seed") == 0 and m.get("arm") == "analytic_candidate":
            matches.append((p.parent, m))
    if len(matches) != 1:
        raise ValueError("SW0117 completed candidate control is not unique")
    return matches[0]
