"""Reuse exact frozen controls; change only temporal training window."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "trained_models/SW0098_long_window"
spec = importlib.util.spec_from_file_location("graph_pilot", ROOT / "collaborative_test/SW_0097_graph_adaptation/coordinator.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def write(value):
    temp = OUT / "state.tmp"
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(OUT / "state.json")


def command(seed, folder, preflight=False):
    return base.command(seed, "positive_frozen", folder, preflight) + ["--train-time-steps", "256", "--train-settle", "128"]


def free_gpu():
    for gpu in (0, 1):
        r = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, check=True)
        if not any(c.isdigit() for c in r.stdout):
            return gpu
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--memory-check", action="store_true")
    p.add_argument("--daemon", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in range(3):
        folder = base.OUT / f"seed{seed}_positive_frozen"
        m = json.loads((folder / "manifest.json").read_text())
        assert (folder / "COMPLETED").exists() and m["seed"] == 117 + seed
        assert m["train_steps"] == 64 and m["train_settle"] == 32 and m["steps"] == 256 and m["batch"] == 16
    if a.daemon:
        assert (OUT / "PREFLIGHT_COMPLETED").exists() and (OUT / "memory_check/PREFLIGHT_COMPLETED").exists()
        if (OUT / "queue.pid").exists():
            raise FileExistsError("inspect existing queue; no duplicate launch")
        with (OUT / "queue.log").open("w") as log:
            process = subprocess.Popen([sys.executable, __file__], stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        (OUT / "queue.pid").write_text(str(process.pid) + "\n")
        print(f"QUEUE_PID={process.pid}")
        return
    lock = (OUT / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if a.preflight:
        for seed in range(3):
            folder = OUT / f"preflight_seed{seed}"
            if (folder / "PREFLIGHT_COMPLETED").exists():
                continue
            with (OUT / f"preflight_seed{seed}.log").open("w") as log:
                subprocess.run(command(seed, folder, True), stdout=log, stderr=subprocess.STDOUT, check=True)
        (OUT / "PREFLIGHT_COMPLETED").write_text("three actual256/128 backward checks passed\n")
        print("THREE_LONG_WINDOW_PREFLIGHTS_PASSED")
        return
    if a.memory_check:
        assert (OUT / "PREFLIGHT_COMPLETED").exists()
        gpu = free_gpu()
        if gpu is None:
            raise RuntimeError("memory check requires a free GPU0/1")
        args = command(2, OUT / "memory_check", True)
        args[args.index("--batch") + 1] = "16"
        args[args.index("--device") + 1] = "cuda"
        with (OUT / "memory_check.log").open("w") as log:
            subprocess.run(args, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2"), stdout=log, stderr=subprocess.STDOUT, check=True)
        print("FULL_BATCH_GPU_MEMORY_CHECK_PASSED")
        return
    assert (OUT / "PREFLIGHT_COMPLETED").exists() and (OUT / "memory_check/PREFLIGHT_COMPLETED").exists()
    try:
        for seed in (1, 2, 0):
            folder = OUT / f"seed{seed}"
            if folder.exists():
                raise FileExistsError(f"inspect existing output; no rerun: {folder}")
            gpu = free_gpu()
            while gpu is None:
                write({"status": "waiting_for_free_gpu", "seed": seed})
                time.sleep(55)
                gpu = free_gpu()
            with (OUT / f"seed{seed}.log").open("w") as log:
                process = subprocess.Popen(command(seed, folder), env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2"), stdout=log, stderr=subprocess.STDOUT)
                write({"status": "running", "seed": seed, "gpu": gpu, "pid": process.pid})
                if process.wait() or not (folder / "COMPLETED").exists():
                    raise RuntimeError(f"long-window training/evaluation failed: {folder}")
        write({"status": "complete", "completed": time.time()})
    except Exception as error:
        write({"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
