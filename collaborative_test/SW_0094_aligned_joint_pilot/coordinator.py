"""Server-side queue: no token-consuming polling, and no GPU sharing."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "trained_models/SW0094_aligned_joint_pilot"
ARMS = ("absolute_frozen", "positive_frozen", "positive_graph", "positive_joint")


def state(value):
    temp = OUT / "state.tmp"
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(OUT / "state.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        for arm in ARMS:
            destination = OUT / f"preflight_{arm}"
            if (destination / "PREFLIGHT_COMPLETED").exists():
                continue
            if destination.exists():
                raise RuntimeError(f"inspect failed preflight before retrying: {destination}")
            with (OUT / f"preflight_{arm}.log").open("w") as log:
                subprocess.run([sys.executable, str(HERE / "run.py"), "--arm", arm,
                                "--output", str(destination), "--device", "cpu",
                                "--steps", "1", "--batch", "2", "--preflight"],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        print("PREFLIGHT_ALL_FOUR_PASSED", flush=True)
        return
    if args.daemon:
        with (OUT / "queue.log").open("a") as log:
            child = subprocess.Popen([sys.executable, str(HERE / "coordinator.py")],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True)
        (OUT / "queue.pid").write_text(str(child.pid) + "\n")
        print(f"QUEUE_PID={child.pid}")
        return
    lock = (OUT / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for arm in ARMS:
        if not (OUT / f"preflight_{arm}/PREFLIGHT_COMPLETED").exists():
            raise RuntimeError(f"missing real-data preflight: {arm}")
    pending = [arm for arm in ARMS if not (OUT / arm / "COMPLETED").exists()]
    for arm in pending:
        if (OUT / arm).exists():
            raise RuntimeError(f"existing incomplete arm needs diagnosis: {arm}")
    active = {}
    failed = []
    while pending or active:
        for gpu, (arm, process, log, gpu_lock) in list(active.items()):
            code = process.poll()
            if code is not None:
                log.close()
                gpu_lock.close()
                del active[gpu]
                if code or not (OUT / arm / "COMPLETED").exists():
                    failed.append({"arm": arm, "exit_code": code})
        if failed:
            # Let already-running jobs finish, but never start more after failure.
            pending.clear()
        for gpu in (0, 1):
            if not pending or gpu in active:
                continue
            probe = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid",
                                    "--format=csv,noheader"], capture_output=True, text=True)
            if probe.returncode or any(c.isdigit() for c in probe.stdout):
                continue
            gpu_lock = (OUT / f"gpu{gpu}.lock").open("a")
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                gpu_lock.close()
                continue
            arm = pending.pop(0)
            log = (OUT / f"{arm}.log").open("w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2")
            process = subprocess.Popen([sys.executable, str(HERE / "run.py"), "--arm", arm,
                                        "--output", str(OUT / arm)], env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (arm, process, log, gpu_lock)
        state({"status": "failed_waiting_for_live_jobs" if failed else
                         "running" if active else "waiting_for_free_gpu",
               "queue_pid": os.getpid(), "updated": time.time(), "pending": pending,
               "active": [{"gpu": gpu, "arm": arm, "pid": proc.pid}
                          for gpu, (arm, proc, _, _) in active.items()],
               "failed": failed, "allowed_gpus": [0, 1]})
        if pending or active:
            time.sleep(60)
    state({"status": "failed" if failed else "complete", "failed": failed,
           "queue_pid": os.getpid(), "completed": time.time()})
    if not failed:
        subprocess.run([sys.executable, str(HERE / "summarize.py"), "--results", str(OUT)], check=True)


if __name__ == "__main__":
    main()
