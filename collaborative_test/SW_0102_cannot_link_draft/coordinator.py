"""SW0102 matched three-seed queue with free-GPU-only scheduling."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "trained_models/SW0095_full70k_aligned_loss"
CONTROL = ROOT / "trained_models/SW0097_graph_adaptation"
OUT = ROOT / "trained_models/SW0102_cannot_link_draft"
RUNNER = HERE / "run.py"
CALIBRATION = OUT / "calibration/calibration.json"
LAMBDA = 3.692937560668443
DEADLINE = 1791369469  # 2026-10-07 10:37:49 UTC; do not start near cutoff


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_ids(seed):
    indices = torch.randperm(70000,
        generator=torch.Generator().manual_seed(117 + seed))[:4096]
    return [int(index) if index < 1000 else int(index) + 640
            for index in indices.tolist()]


def validate_contracts():
    calibration = read_json(CALIBRATION)
    if calibration.get("status") != "calibration_complete":
        raise AssertionError("no complete real-data calibration")
    if calibration.get("shared_lambda") != LAMBDA:
        raise AssertionError("calibrated lambda changed")
    for seed in range(3):
        row = calibration["seeds"][str(seed)]
        if not row.get("seed_has_real_hard_negative") or not row.get("seed_has_valid_gradient_calibration"):
            raise AssertionError(f"seed{seed} did not meet hard-negative/gradient calibration guards")
        if not row.get("source_state_unchanged_after_no_update_calibration"):
            raise AssertionError(f"seed{seed} calibration modified its source")
        source_dir = SOURCE / f"seed{seed}"
        control_dir = CONTROL / f"seed{seed}_positive_frozen"
        if not (source_dir / "COMPLETED").is_file() or not (control_dir / "COMPLETED").is_file():
            raise FileNotFoundError(f"completed seed{seed} source/control missing")
        source_manifest = read_json(source_dir / "manifest.json")
        control_manifest = read_json(control_dir / "manifest.json")
        source_sha = sha(source_dir / "core.pt")
        if (source_manifest.get("status") != "complete"
                or source_manifest.get("source_model_seed") != seed):
            raise AssertionError(f"SW0095 seed{seed} manifest mismatch")
        required = {"status": "complete", "source_model_seed": seed,
                    "seed": 117 + seed, "steps": 256, "batch": 16,
                    "train_steps": 64, "train_settle": 32,
                    "core_lr": 3e-5, "ground_truth_used_for_training": False}
        for key, value in required.items():
            if control_manifest.get(key) != value:
                raise AssertionError(f"SW0097 seed{seed} {key} mismatch")
        if control_manifest.get("source_sha256") != source_sha:
            raise AssertionError(f"SW0095 source differs from matched SW0097 seed{seed}")
        if control_manifest.get("training_ids") != expected_ids(seed):
            raise AssertionError(f"SW0097 seed{seed} training order differs from matched permutation")
        folder = OUT / f"seed{seed}"
        if folder.exists():
            raise FileExistsError(f"candidate output exists; inspect and do not overwrite: {folder}")
    preflight = OUT / "preflight_seed0"
    if not (preflight / "PREFLIGHT_COMPLETED").is_file():
        raise AssertionError("full B16 one-update preflight did not complete")
    preflight_manifest = read_json(preflight / "manifest.json")
    preflight_history = read_json(preflight / "history.json")
    if (preflight_manifest.get("source_sha256") != sha(SOURCE / "seed0/core.pt")
            or preflight_manifest.get("steps") != 1
            or preflight_manifest.get("batch") != 16
            or preflight_manifest.get("cannot_link_lambda") != LAMBDA
            or preflight_manifest.get("status") != "training_complete"):
        raise AssertionError("throwaway preflight contract mismatch")
    row = preflight_history[0]
    if (not all(map(lambda key: __import__("math").isfinite(float(row[key])),
                    ("loss", "primary", "spike", "cannot_link_cut_loss")))
            or row["gradient_norm"]["core"] <= 0
            or row["gradient_norm"]["graph"] != 0
            or row["gradient_norm"]["encoder"] != 0
            or not preflight_manifest.get("initial_encoder_gamma_max_diff", 1.) < 2e-5):
        raise AssertionError("throwaway full-batch preflight finite-gradient/frozen-contract check failed")
    return calibration


def free_gpus():
    available = []
    owner_rows = []
    for gpu in (0, 1, 2, 3):
        process_query = subprocess.run(
            ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True, check=True)
        memory_query = subprocess.run(
            ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        pids = [line.strip() for line in process_query.stdout.splitlines() if line.strip()]
        memory_mb = int(memory_query.stdout.strip().splitlines()[0])
        owner_rows.append({"gpu": gpu, "compute_processes": pids,
                           "memory_used_mib": memory_mb})
        if not pids and memory_mb <= 128:
            available.append(gpu)
    return available, owner_rows


def worker_command(seed):
    return [sys.executable, str(RUNNER), "--arm", "positive_frozen",
            "--source-seed", str(seed), "--source-checkpoint",
            str(SOURCE / f"seed{seed}/core.pt"), "--shuffle-seed", str(117 + seed),
            "--output", str(OUT / f"seed{seed}"), "--steps", "256", "--batch", "16",
            "--validation-count", "320", "--train-time-steps", "64", "--train-settle", "32",
            "--gate-mode", "raw", "--cannot-link-lambda", str(LAMBDA), "--device", "cuda:0"]


def write_state(value):
    temp = OUT / "queue_state.tmp"
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(OUT / "queue_state.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", required=True)
    args = parser.parse_args()
    if time.time() + 600 >= DEADLINE:
        raise RuntimeError("hard deadline is too close to safely begin a full seed")
    validate_contracts()
    OUT.mkdir(parents=True, exist_ok=True)
    launch_path = OUT / "queue_launch.json"
    if launch_path.exists():
        raise FileExistsError("queue_launch exists; inspect recorded workers and do not relaunch")
    gpu_pool, first_owners = free_gpus()
    if len(gpu_pool) < 1:
        raise RuntimeError("no genuinely free GPU available; no jobs launched")
    launch = {"status": "running", "lambda": LAMBDA, "seed_order": [0, 1, 2],
              "steps": 256, "batch": 16, "training_ids_per_seed": 4096,
              "train_steps": 64, "train_settle": 32, "lr": 3e-5,
              "evaluation_images": 320, "evaluation_batch": 8,
              "evaluation_steps": 1024, "evaluation_settle": 512,
              "ground_truth_used_for_training": False,
              "deadline_unix": DEADLINE, "owners_before_launch": first_owners,
              "workers": [], "started_unix": time.time()}
    temp_launch = launch_path.with_suffix(".json.tmp")
    temp_launch.write_text(json.dumps(launch, indent=2) + "\n")
    temp_launch.replace(launch_path)
    active = {}
    pending = [0, 1, 2]
    state = {"status": "launching", "workers": [], "pending_seeds": pending,
             "updated_unix": time.time()}

    while pending or active:
        now = time.time()
        if now + 600 >= DEADLINE and pending:
            state.update(status="stopped_at_deadline_buffer", pending_seeds=pending)
            write_state(state)
            break
        free, owners = free_gpus()
        free = [gpu for gpu in free if gpu not in active]
        for gpu in free:
            if not pending:
                break
            seed = pending.pop(0)
            log_path = HERE / "preflight_attempts" / f"seed{seed}_gpu{gpu}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2")
            process = subprocess.Popen(worker_command(seed), stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True, env=env)
            log.close()
            worker = {"seed": seed, "gpu": gpu, "pid": process.pid,
                      "log": str(log_path), "output": str(OUT / f"seed{seed}"),
                      "source_sha256": sha(SOURCE / f"seed{seed}/core.pt"),
                      "control_source_sha256": read_json(CONTROL / f"seed{seed}_positive_frozen/manifest.json")["source_sha256"],
                      "shuffle_seed": 117 + seed, "started_unix": time.time(),
                      "owners_at_worker_launch": owners}
            launch["workers"].append(worker)
            active[gpu] = (seed, process, worker)
            temp_launch.write_text(json.dumps(launch, indent=2, allow_nan=False) + "\n")
            temp_launch.replace(launch_path)
            state["workers"] = launch["workers"]
            state["pending_seeds"] = pending
            state.update(status="running", updated_unix=time.time())
            write_state(state)
        completed_gpus = []
        for gpu, (seed, process, worker) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            completed_gpus.append(gpu)
            worker["completed_unix"] = time.time()
            worker["exit_code"] = code
            folder = OUT / f"seed{seed}"
            if code != 0 or not (folder / "COMPLETED").is_file():
                worker["status"] = "failed"
                state.update(status="failed", failed_seed=seed, exit_code=code,
                             updated_unix=time.time())
                write_state(state)
                raise RuntimeError(f"seed{seed} candidate failed; preserve artifacts in {folder}")
            manifest = read_json(folder / "manifest.json")
            evaluation_path = folder / "evaluation.json"
            if not evaluation_path.is_file():
                raise FileNotFoundError(f"seed{seed} full320 evaluation missing")
            if (manifest.get("status") != "complete"
                    or manifest.get("source_sha256") != worker["source_sha256"]
                    or manifest.get("steps") != 256 or manifest.get("batch") != 16
                    or manifest.get("unique_images_seen") != 4096
                    or manifest.get("train_steps") != 64
                    or manifest.get("train_settle") != 32
                    or manifest.get("cannot_link_lambda") != LAMBDA
                    or manifest.get("training_ids") != expected_ids(seed)
                    or manifest.get("gate_mode") != "raw"):
                raise AssertionError(f"seed{seed} final manifest fails matched-training contract")
            if any(key.startswith("graph_generator.") for key in manifest.get("changed_core_keys", [])):
                raise AssertionError(f"seed{seed} frozen graph changed")
            evaluation = read_json(evaluation_path)
            if evaluation.get("images") != 320 or evaluation.get("ids") != [1320, 1639]:
                raise AssertionError(f"seed{seed} evaluation did not use all 320 registered IDs")
            scored = evaluation["sweep"][0]["scored_targets"]["our_hdf5"]
            metrics = scored["metrics"]
            counts = scored["valid_count"]
            for metric in ("fg_ari", "foreground_iou", "matched_object_iou"):
                value = float(metrics[metric])
                if not math.isfinite(value) or counts[metric] != 320:
                    raise AssertionError(f"seed{seed} invalid full320 {metric}: {value}/{counts[metric]}")
            worker["status"] = "complete"
            worker["evaluation_path"] = str(evaluation_path)
            worker["evaluation_sha256"] = sha(evaluation_path)
            worker["metrics"] = metrics
            worker["training_id_order_matches_control"] = True
            del active[gpu]
            state["workers"] = launch["workers"]
            state["pending_seeds"] = pending
            state.update(status="running", updated_unix=time.time())
            write_state(state)
        if completed_gpus:
            launch["workers"] = state["workers"]
            temp_launch.write_text(json.dumps(launch, indent=2, allow_nan=False) + "\n")
            temp_launch.replace(launch_path)
        if pending or active:
            time.sleep(10)

    if not pending and not active:
        launch["status"] = "complete"
        launch["completed_unix"] = time.time()
        temp_launch.write_text(json.dumps(launch, indent=2, allow_nan=False) + "\n")
        temp_launch.replace(launch_path)
        state.update(status="complete", pending_seeds=[], updated_unix=time.time())
        write_state(state)


if __name__ == "__main__":
    main()
