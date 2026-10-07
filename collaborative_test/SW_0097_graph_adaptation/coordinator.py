"""Matched graph/frozen training, limited to GPUs0/1."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "trained_models/SW0097_graph_adaptation"
RUNNER = ROOT / "collaborative_test/SW_0094_aligned_joint_pilot/run.py"
ARMS = ("positive_frozen", "positive_graph")


def write(name, value):
    temp = OUT / f"{name}.tmp"
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(OUT / f"{name}.json")


def command(seed, arm, output, preflight=False):
    source = ROOT / f"trained_models/SW0095_full70k_aligned_loss/seed{seed}"
    m = json.loads((source / "manifest.json").read_text())
    assert (source / "COMPLETED").exists() and m["source_model_seed"] == seed and m["status"] == "complete"
    args = [sys.executable, str(RUNNER), "--arm", arm, "--source-seed", str(seed),
            "--source-checkpoint", str(source / "core.pt"), "--shuffle-seed", str(117 + seed),
            "--output", str(output), "--steps", "1" if preflight else "256",
            "--batch", "2" if preflight else "16", "--validation-count", "320"]
    if preflight:
        args += ["--device", "cpu", "--preflight"]
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.daemon:
        assert (OUT / "PREFLIGHT_COMPLETED").exists()
        if (OUT / "queue.pid").exists():
            raise FileExistsError("inspect existing queue; no duplicate launch")
        with (OUT / "queue.log").open("w") as log:
            p = subprocess.Popen([sys.executable, __file__], stdout=log, stderr=log,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        (OUT / "queue.pid").write_text(str(p.pid) + "\n")
        print(f"QUEUE_PID={p.pid}")
        return
    lock = (OUT / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.preflight:
        for seed in range(3):
            for arm in ARMS:
                folder = OUT / f"preflight_seed{seed}_{arm}"
                if (folder / "PREFLIGHT_COMPLETED").exists():
                    continue
                with (OUT / f"preflight_seed{seed}_{arm}.log").open("w") as log:
                    subprocess.run(command(seed, arm, folder, True), stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
        (OUT / "PREFLIGHT_COMPLETED").write_text("all six real-asset backward/update checks passed\n")
        print("ALL_SIX_PREFLIGHTS_PASSED")
        return
    assert (OUT / "PREFLIGHT_COMPLETED").exists()
    try:
        for seed in (1, 2, 0):
            for arm in ARMS:
                folder = OUT / f"seed{seed}_{arm}"
                if folder.exists():
                    raise FileExistsError(f"inspect existing output; no rerun: {folder}")
                gpu = None
                while gpu is None:
                    for candidate in (0, 1):
                        result = subprocess.run(["nvidia-smi", f"--id={candidate}", "--query-compute-apps=pid",
                                                 "--format=csv,noheader"], capture_output=True, text=True, check=True)
                        if not any(c.isdigit() for c in result.stdout):
                            gpu = candidate
                            break
                    if gpu is None:
                        write("state", {"status": "waiting_for_free_gpu", "seed": seed, "arm": arm})
                        time.sleep(55)
                with (OUT / f"seed{seed}_{arm}.log").open("w") as log:
                    process = subprocess.Popen(command(seed, arm, folder), stdout=log, stderr=subprocess.STDOUT,
                                               env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2"))
                    write("state", {"status": "running", "gpu": gpu, "seed": seed, "arm": arm, "pid": process.pid})
                    if process.wait() or not (folder / "COMPLETED").exists():
                        raise RuntimeError(f"training/evaluation failed: {folder}")
        write("state", {"status": "complete", "completed": time.time()})
    except Exception as error:
        write("state", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
