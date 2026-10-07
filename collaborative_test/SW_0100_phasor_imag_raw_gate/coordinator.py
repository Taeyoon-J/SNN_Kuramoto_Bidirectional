"""Matched frozen-core phasor-imaginary gate pilot; preflight/run are explicit."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "trained_models/SW0100_phasor_imag_raw_gate"
RUNNER = ROOT / "collaborative_test/SW_0094_aligned_joint_pilot/run.py"
CONTROL = ROOT / "trained_models/SW0097_graph_adaptation"
SOURCE = ROOT / "trained_models/SW0095_full70k_aligned_loss"
MODE = "phasor_imag_raw"


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compare_contract(seed):
    source = SOURCE / f"seed{seed}"
    control = CONTROL / f"seed{seed}_positive_frozen"
    if not (source / "COMPLETED").is_file() or not (control / "COMPLETED").is_file():
        raise FileNotFoundError(f"completed source/control missing for seed {seed}")
    source_manifest = read_json(source / "manifest.json")
    control_manifest = read_json(control / "manifest.json")
    if source_manifest.get("status") != "complete" or source_manifest.get("source_model_seed") != seed:
        raise AssertionError(f"unexpected SW0095 source manifest for seed {seed}")
    required = {"status": "complete", "source_model_seed": seed, "seed": 117 + seed,
                "steps": 256, "batch": 16, "train_steps": 64, "train_settle": 32,
                "core_lr": 3e-5, "ground_truth_used_for_training": False}
    for key, value in required.items():
        if control_manifest.get(key) != value:
            raise AssertionError(f"SW0097 seed{seed} control {key} mismatch")
    if control_manifest.get("source_sha256") != sha256(source / "core.pt"):
        raise AssertionError(f"SW0095 source SHA differs from matched SW0097 seed{seed}")
    if len(control_manifest.get("training_ids", [])) != 4096:
        raise AssertionError(f"SW0097 seed{seed} control does not contain 4096 IDs")
    return source, control, source_manifest, control_manifest


def command(seed, folder, preflight=False):
    source, _, _, _ = compare_contract(seed)
    args = [sys.executable, str(RUNNER), "--arm", "positive_frozen",
            "--source-seed", str(seed), "--source-checkpoint", str(source / "core.pt"),
            "--shuffle-seed", str(117 + seed), "--output", str(folder),
            "--steps", "1" if preflight else "256",
            "--batch", "2" if preflight else "16", "--validation-count", "320",
            "--train-time-steps", "64", "--train-settle", "32",
            "--gate-mode", MODE]
    if preflight:
        args += ["--device", "cpu", "--preflight"]
    return args


def free_gpus():
    available = []
    for gpu in (0, 1, 2, 3):
        result = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid",
                                 "--format=csv,noheader"], capture_output=True,
                                text=True, check=True)
        if not any(value.isdigit() for value in result.stdout.split()):
            available.append(gpu)
    return available


def free_gpu():
    available = free_gpus()
    return available[0] if available else None


def verify_candidate(seed, folder):
    _, _, source_manifest, control_manifest = compare_contract(seed)
    manifest = read_json(folder / "manifest.json")
    expected = {"status": "complete", "gate_mode": MODE, "source_model_seed": seed,
                "seed": 117 + seed, "steps": 256, "batch": 16,
                "train_steps": 64, "train_settle": 32, "core_lr": 3e-5,
                "unique_images_seen": 4096}
    for key, value in expected.items():
        if manifest.get(key) != value:
            # run.py records training_complete before evaluation; require its
            # final status below and validate invariant training metadata here.
            if key != "status" or manifest.get(key) != "training_complete":
                raise AssertionError(f"SW0100 seed{seed} manifest {key} mismatch")
    if manifest.get("source_sha256") != sha256(SOURCE / f"seed{seed}/core.pt"):
        raise AssertionError(f"SW0100 seed{seed} source checkpoint changed")
    if manifest.get("training_ids") != control_manifest.get("training_ids"):
        raise AssertionError(f"SW0100 seed{seed} training IDs/order differ from SW0097")
    changed_keys = manifest.get("changed_core_keys", [])
    if any(key.startswith("graph_generator.") for key in changed_keys):
        raise AssertionError(f"SW0100 seed{seed} changed frozen graph tensors")
    if not any(not key.startswith("graph_generator.") for key in changed_keys):
        raise AssertionError(f"SW0100 seed{seed} changed no trainable core tensors")
    if not (folder / "evaluation.json").is_file():
        raise FileNotFoundError(f"SW0100 seed{seed} full evaluation is missing")
    evaluation = read_json(folder / "evaluation.json")
    if evaluation.get("images") != 320 or evaluation.get("ids") != [1320, 1639]:
        raise AssertionError(f"SW0100 seed{seed} did not score 320 valid images")


def run_swaps():
    script = ROOT / "collaborative_test/SW_0100_phasor_imag_raw_gate/swaps.py"
    for seed in range(3):
        source, _, _, _ = compare_contract(seed)
        tasks = [
            (source / "core.pt", "raw", MODE, "source_zero_update"),
            (OUT / f"seed{seed}/core.pt", MODE, "raw", "trained_candidate_zero_update"),
        ]
        for checkpoint, source_mode, destination_mode, condition in tasks:
            output = OUT / "swaps" / f"seed{seed}_{condition}.json"
            if output.exists():
                raise FileExistsError(f"inspect existing swap output; no rerun: {output}")
            gpu = free_gpu()
            while gpu is None:
                print(json.dumps({"status": "waiting_for_free_gpu_swap", "seed": seed}), flush=True)
                time.sleep(55)
                gpu = free_gpu()
            command = [sys.executable, str(script), "--checkpoint", str(checkpoint),
                       "--source-mode", source_mode, "--destination-mode", destination_mode,
                       "--seed", str(seed), "--condition", condition,
                       "--output", str(output), "--device", "cuda"]
            subprocess.run(command, check=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2"))


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for seed in range(3):
        compare_contract(seed)

    if args.preflight:
        invariance = OUT / "actual_asset_invariance.json"
        if not invariance.is_file():
            script = ROOT / "collaborative_test/SW_0100_phasor_imag_raw_gate/actual_asset_invariance.py"
            with (OUT / "actual_asset_invariance.log").open("w") as log:
                subprocess.run([sys.executable, str(script)], stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            result = read_json(invariance)
            if result.get("status") != "complete" or result.get("ground_truth_read") is not False:
                raise AssertionError("actual-source gate invariance audit did not pass")
            for row in result.get("per_seed", []):
                if any(value != 0.0 for value in row["graph_theta_carrier_max_abs_delta"].values()):
                    raise AssertionError("gate mode changed graph/theta/carrier under disabled feedback")
        for seed in range(3):
            folder = OUT / f"preflight_seed{seed}"
            if (folder / "PREFLIGHT_COMPLETED").is_file():
                continue
            if folder.exists():
                raise FileExistsError(f"inspect partial preflight; no automatic restart: {folder}")
            with (OUT / f"preflight_seed{seed}.log").open("w") as log:
                subprocess.run(command(seed, folder, True), stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            if not (folder / "PREFLIGHT_COMPLETED").is_file():
                raise RuntimeError(f"real-asset preflight did not complete: seed{seed}")
        (OUT / "PREFLIGHT_COMPLETED").write_text(
            "three source-seed CPU forward/backward/update checks passed\n")
        print("SW0100_THREE_CPU_ASSET_PREFLIGHTS_PASSED", flush=True)
        return

    if not (OUT / "PREFLIGHT_COMPLETED").is_file():
        raise RuntimeError("run all three actual-source CPU preflights first")
    for seed in (1, 2, 0):
        folder = OUT / f"seed{seed}"
        if folder.exists():
            raise FileExistsError(f"inspect existing output; no automatic restart: {folder}")
    pending = [1, 2, 0]
    active = {}
    while pending or active:
        for gpu in free_gpus():
            if not pending:
                break
            seed = pending.pop(0)
            folder = OUT / f"seed{seed}"
            if folder.exists():
                raise FileExistsError(f"inspect existing output; no automatic restart: {folder}")
            log = (OUT / f"seed{seed}.log").open("w")
            process = subprocess.Popen(command(seed, folder), stdout=log,
                                       stderr=subprocess.STDOUT,
                                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                                                OMP_NUM_THREADS="2"))
            active[seed] = (process, log, gpu)
            (OUT / "state.json").write_text(json.dumps(
                {"status": "running", "active": {
                    str(s): {"gpu": g, "pid": p.pid} for s, (p, _, g) in active.items()},
                 "pending": pending, "gate_mode": MODE}, indent=2) + "\n")
            print(f"SW0100_WORKER_PID={process.pid} SEED={seed} GPU={gpu}", flush=True)
        for seed, (process, log, gpu) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            if code != 0:
                raise RuntimeError(f"SW0100 seed{seed} failed; preserve output and inspect")
            verify_candidate(seed, OUT / f"seed{seed}")
            del active[seed]
            print(f"SW0100_SEED_COMPLETE={seed} GPU={gpu}", flush=True)
        if pending or active:
            time.sleep(5 if active else 55)
    run_swaps()
    (OUT / "state.json").write_text(json.dumps(
        {"status": "complete", "gate_mode": MODE, "seeds": [0, 1, 2]}, indent=2) + "\n")
    print("SW0100_THREE_SEED_TRAINING_AND_FULL320_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
