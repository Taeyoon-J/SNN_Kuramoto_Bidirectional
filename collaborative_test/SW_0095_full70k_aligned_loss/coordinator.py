"""Expand the aligned-loss candidate to a complete70k pass for each model seed."""
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
OUT = ROOT / "trained_models/SW0095_full70k_aligned_loss"
PILOT = ROOT / "trained_models/SW0094_aligned_joint_pilot"
RUNNER = ROOT / "collaborative_test/SW_0094_aligned_joint_pilot/run.py"


def state(value):
    temp = OUT / "state.tmp"
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(OUT / "state.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--allow-own-sharing", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.daemon:
        with (OUT / "queue.log").open("a") as log:
            command = [sys.executable, __file__]
            if args.allow_own_sharing:
                command.append("--allow-own-sharing")
            proc = subprocess.Popen(command, stdout=log, stderr=log,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        (OUT / "queue.pid").write_text(str(proc.pid) + "\n")
        print(f"QUEUE_PID={proc.pid}")
        return
    lock = (OUT / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for seed in (1, 2):
        if not (ROOT / f"trained_models/SW0095_seed{seed}_preflight/PREFLIGHT_COMPLETED").exists():
            raise RuntimeError(f"seed{seed} real-asset preflight missing")
    pending = [seed for seed in (0, 1, 2) if not (OUT / f"seed{seed}/COMPLETED").exists()]
    for seed in pending:
        if (OUT / f"seed{seed}").exists():
            raise RuntimeError(f"inspect incomplete seed{seed} before resuming")
    # The winning frozen arm is independent of the unfinished encoder arm.
    while True:
        pilot_state = json.loads((PILOT / "state.json").read_text())
        full = PILOT / "positive_frozen/evaluation_full320.json"
        if not (PILOT / "positive_frozen/COMPLETED").exists():
            state({"status": "dependency_failed", "dependency": str(PILOT)})
            raise RuntimeError("pilot queue failed")
        if full.exists():
            try:
                d = json.loads(full.read_text())
            except json.JSONDecodeError:
                time.sleep(60)
                continue
            assert d["images"] == 320 and d["ids"] == [1320, 1639]
            assert not d["ground_truth_used_for_prediction"]
            break
        state({"status": "waiting_for_full_validation", "queue_pid": os.getpid(),
               "updated": time.time(), "pending_seeds": pending})
        time.sleep(60)
    active = {}
    failed = []
    while pending or active:
        for gpu, (seed, proc, log) in list(active.items()):
            code = proc.poll()
            if code is not None:
                log.close()
                del active[gpu]
                if code or not (OUT / f"seed{seed}/COMPLETED").exists():
                    failed.append({"seed": seed, "exit_code": code})
        if failed:
            pending.clear()
        for gpu in (0, 1, 2):
            if not pending or gpu in active:
                continue
            probe = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid",
                                    "--format=csv,noheader"], capture_output=True, text=True)
            occupied = any(c.isdigit() for c in probe.stdout)
            owned_with_room = False
            if probe.returncode == 0 and occupied and args.allow_own_sharing:
                pids = [int(line.strip()) for line in probe.stdout.splitlines() if line.strip().isdigit()]
                try:
                    all_owned = bool(pids) and all(Path(f"/proc/{pid}").stat().st_uid == os.getuid() for pid in pids)
                except FileNotFoundError:
                    all_owned = False
                free = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free",
                                       "--format=csv,noheader,nounits"], capture_output=True, text=True)
                owned_with_room = all_owned and free.returncode == 0 and int(free.stdout.strip()) >= 8192
            if probe.returncode or (occupied and not owned_with_room):
                continue
            seed = pending.pop(0)
            log = (OUT / f"seed{seed}.log").open("w")
            command = [sys.executable, str(RUNNER), "--arm", "positive_frozen",
                       "--source-seed", str(seed), "--steps", "4375", "--batch", "16",
                       "--validation-count", "320", "--output", str(OUT / f"seed{seed}")]
            proc = subprocess.Popen(command, stdout=log, stderr=log,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2"))
            active[gpu] = (seed, proc, log)
        state({"status": "failed_waiting_for_live_jobs" if failed else
                         "running" if active else "waiting_for_free_gpu",
               "queue_pid": os.getpid(), "updated": time.time(), "pending_seeds": pending,
               "active": [{"gpu": gpu, "seed": seed, "pid": proc.pid}
                          for gpu, (seed, proc, _) in active.items()], "failed": failed})
        if pending or active:
            time.sleep(60)
    state({"status": "failed" if failed else "complete", "failed": failed,
           "queue_pid": os.getpid(), "completed": time.time()})


if __name__ == "__main__":
    main()
