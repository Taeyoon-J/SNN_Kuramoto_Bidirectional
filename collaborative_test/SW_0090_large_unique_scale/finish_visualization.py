"""Select the strongest aggregate checkpoint and render fixed qualitative IDs."""
import argparse
import json
import os
import subprocess
from pathlib import Path

METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, default=Path("/Data0/kevinswk/patch_v2_sw"))
a = p.parse_args()
results = a.root / "trained_models/SW0090_full320_long"
summary = json.loads((results / "summary.json").read_text())
baseline = summary["baseline_sw0072_2500x10_mean"]

def score(row):
    return sum(float(row[key]) / float(baseline[key]) for key in METRICS)

epoch = max(summary["epochs"], key=lambda item: score(summary["epochs"][item]["mean"]))
seeds = summary["epochs"][epoch]["seeds"]
seed = max(seeds, key=lambda item: score(seeds[item]))
checkpoint = a.root / f"trained_models/SW0090_unique70000_s{seed}_e10/checkpoints/epoch_{int(epoch):02d}.pt"
output = results / "visualizations"
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "1"
cmd = [
    "/Data0/kevinswk/envs/snn/bin/python",
    str(a.root / "collaborative_test/SW_0090_large_unique_scale/visualize_examples.py"),
    "--checkpoint", str(checkpoint),
    "--gamma", str(a.root / "data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"),
    "--dataset", "/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5",
    "--output", str(output), "--ids", "1320", "1321", "1322", "--device", "cuda",
]
subprocess.run(cmd, check=True, env=env)
selection = {
    "selection_rule": "maximize sum of each three-seed mean metric divided by SW0072 baseline; then same rule for qualitative seed",
    "selected_epoch": int(epoch), "selected_seed": int(seed),
    "selected_metrics": seeds[seed], "checkpoint": str(checkpoint),
    "visualization_ids_pre_registered": [1320, 1321, 1322],
}
(results / "visualization_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
print(json.dumps(selection, indent=2))

