"""Expand a completed pilot's unchanged readout to all320 validation images."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--gpu", type=int, choices=[0, 1, 2], required=True)
    p.add_argument("--wait", action="store_true")
    p.add_argument("--daemon", action="store_true")
    p.add_argument("--allow-own-sharing", action="store_true")
    args = p.parse_args()
    folder = ROOT / "trained_models/SW0094_aligned_joint_pilot" / args.arm
    if not (folder / "COMPLETED").exists():
        raise RuntimeError("pilot training/evaluation must complete first")
    if args.daemon:
        if (folder / "full320_queue.json").exists():
            raise FileExistsError("inspect existing full validation queue")
        with (folder / "full320_queue.log").open("w") as log:
            queued = [sys.executable, __file__, "--arm", args.arm,
                      "--gpu", str(args.gpu), "--wait"]
            if args.allow_own_sharing:
                queued.append("--allow-own-sharing")
            child = subprocess.Popen(queued, stdout=log,
                                     stderr=log, stdin=subprocess.DEVNULL,
                                     start_new_session=True)
        record = {"queue_pid": child.pid, "gpu": args.gpu, "arm": args.arm}
        (folder / "full320_queue.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record))
        return
    result = json.loads((folder / "evaluation.json").read_text())
    if (folder / "evaluation_full320.json").exists() or (folder / "full320_launch.json").exists():
        raise FileExistsError("inspect existing evaluation before relaunching")
    while True:
        probe = subprocess.run(["nvidia-smi", f"--id={args.gpu}", "--query-compute-apps=pid",
                                "--format=csv,noheader"], capture_output=True, text=True, check=True)
        if not any(c.isdigit() for c in probe.stdout):
            break
        if args.allow_own_sharing:
            pids = [int(line.strip()) for line in probe.stdout.splitlines() if line.strip().isdigit()]
            try:
                all_owned = bool(pids) and all(Path(f"/proc/{pid}").stat().st_uid == os.getuid() for pid in pids)
            except FileNotFoundError:
                all_owned = False
            free = subprocess.run(["nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.free",
                                   "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
            if all_owned and int(free.stdout.strip()) >= 8192:
                break
        if not args.wait:
            raise RuntimeError("GPU is occupied")
        time.sleep(60)
    source = result["gamma_source"]
    command = [sys.executable, str(ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py"),
               "--checkpoint", result["checkpoint"], "--gamma-path", source["path"],
               "--gamma-global-start", str(source["global_start"]),
               "--gamma-manifest", source["manifest_path"],
               "--dataset-path", result["target_sources"]["our_hdf5"]["path"],
               "--output-path", str(folder / "evaluation_full320.json"),
               "--start", "1320", "--count", "320", "--thresholds", ".50",
               "--background", "largest_component", "--device", "cuda"]
    if args.allow_own_sharing:
        command.extend(["--batch-size", "1"])
    for key in ("steps", "settle", "membrane_vth", "min_group_size", "dendritic_projection",
                "graph_spatial_decay", "geodesic_steps", "geodesic_radius", "geodesic_contrast",
                "geodesic_temperature", "geodesic_cap", "kuramoto_backend", "gate_mode"):
        command.extend(["--" + key.replace("_", "-"), str(result["inference"][key])])
    with (folder / "full320.log").open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu)),
                                 start_new_session=True)
    record = {"pid": child.pid, "gpu": args.gpu, "arm": args.arm, "command": command}
    (folder / "full320_launch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
