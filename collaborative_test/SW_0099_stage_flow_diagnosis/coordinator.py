"""Sequential 9-checkpoint queue for the registered validation-only diagnosis."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "trained_models/SW0099_stage_flow_diagnosis"
TAG = "corrected_dendrite_order"
STATE = OUT / f"state_{TAG}.json"
RUNNER = ROOT / "collaborative_test/SW_0099_stage_flow_diagnosis/diagnose.py"
CHECKPOINTS = [
    (condition, seed, ROOT / f"trained_models/{template.format(seed=seed)}")
    for condition, template in (
        ("SW0095", "SW0095_full70k_aligned_loss/seed{seed}/core.pt"),
        ("SW0097", "SW0097_graph_adaptation/seed{seed}_positive_frozen/core.pt"),
        ("SW0098", "SW0098_long_window/seed{seed}/core.pt"),
    )
    for seed in range(3)
]


def free_gpu():
    for gpu in (0, 1, 2, 3):
        p = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid",
                            "--format=csv,noheader"], capture_output=True, text=True, check=True)
        if not any(line.strip().isdigit() for line in p.stdout.splitlines()):
            return gpu
    return None


def write(value):
    temp = OUT / f"state_{TAG}.tmp"
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(STATE)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if STATE.exists():
        raise FileExistsError("inspect existing SW0099 state before starting; no automatic restart")
    results = OUT / f"results_{TAG}"
    results.mkdir(exist_ok=True)
    for condition, seed, checkpoint in CHECKPOINTS:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        output = results / f"{condition.lower()}_seed{seed}.json"
        if output.exists():
            raise FileExistsError(f"inspect existing output before launch: {output}")
    write({"status": "starting", "queue": [[c, s] for c, s, _ in CHECKPOINTS]})
    try:
        for condition, seed, checkpoint in CHECKPOINTS:
            gpu = free_gpu()
            while gpu is None:
                write({"status": "waiting_for_free_gpu", "condition": condition, "seed": seed})
                time.sleep(55)
                gpu = free_gpu()
            output = results / f"{condition.lower()}_seed{seed}.json"
            log_path = OUT / f"{condition.lower()}_seed{seed}_{TAG}.log"
            command = [sys.executable, str(RUNNER), "--checkpoint", str(checkpoint),
                       "--condition", condition, "--seed", str(seed), "--ids", "1320", "1336",
                       "--output", str(output)]
            with log_path.open("w") as log:
                p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                                              OMP_NUM_THREADS="2"),
                                     stdin=subprocess.DEVNULL)
                write({"status": "running", "condition": condition, "seed": seed,
                       "gpu": gpu, "pid": p.pid, "output": str(output)})
                if p.wait() or not output.exists():
                    raise RuntimeError(f"diagnostic failed: {condition} seed{seed}")
                record = json.loads(output.read_text())
                if record.get("status") != "complete":
                    raise RuntimeError(f"diagnostic did not complete: {output}")
        write({"status": "complete", "tag": TAG, "completed_unix": time.time(),
               "completed": [f"{c}_seed{s}" for c, s, _ in CHECKPOINTS]})
    except Exception as e:
        write({"status": "failed", "tag": TAG, "error": repr(e)})
        raise


if __name__ == "__main__":
    main()
