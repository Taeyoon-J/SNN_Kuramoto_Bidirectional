"""CPU-only signed-affinity/time-window diagnosis on unchanged checkpoints."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import torch
from scipy.stats import rankdata

from run import ROOT, DATASET, VAL_GAMMA, hparams, aligned_affinity
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.training.train_s2net_core import _component_spike_synchrony

OUT = ROOT / "trained_models/SW0094_aligned_joint_pilot"


def measure(matrix, labels):
    mask = (labels[:, None] > 0) & (labels[None, :] > 0)
    mask &= torch.triu(torch.ones_like(mask), diagonal=1).bool()
    same = labels[:, None] == labels[None, :]
    values, target = matrix[mask].numpy(), same[mask].numpy()
    if len(values) == 0 or len(set(target)) < 2:
        return None
    positive_count = int(target.sum())
    negative_count = len(target) - positive_count
    ranks = rankdata(values, method="average")
    auc = (ranks[target].sum() - positive_count * (positive_count + 1) / 2.) / (positive_count * negative_count)
    return {"fg_pair_auc": float(auc),
            "same_object_mean": float(values[target].mean()),
            "different_object_mean": float(values[~target].mean()),
            "same_object_edge_recall_at_0p5": float((values[target] >= .5).mean()),
            "different_object_edge_rate_at_0p5": float((values[~target] >= .5).mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--daemon", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.daemon:
        with (OUT / "window_diagnosis.log").open("a") as log:
            child = subprocess.Popen([sys.executable, __file__], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=log, start_new_session=True,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        (OUT / "window_diagnosis.pid").write_text(str(child.pid) + "\n")
        print(f"DIAGNOSIS_PID={child.pid}")
        return
    torch.set_num_threads(2)
    ids = list(range(1320, 1324))
    gamma = torch.load(VAL_GAMMA, map_location="cpu", weights_only=True)[:4]
    with h5py.File(DATASET, "r") as dataset:
        labels = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"].reshape(4, -1)
    result = {"status": "running", "pid": os.getpid(), "device": "cpu",
              "diagnostic_only": True, "ids": ids, "seeds": [0, 2],
              "gt_used_only_for_post_forward_pair_diagnosis": True, "rows": []}
    path = OUT / "window_diagnosis.json"
    if path.exists():
        raise FileExistsError("inspect existing diagnosis before rerunning")
    for seed in (0, 2):
        for epoch in (1, 10):
            checkpoint = ROOT / f"trained_models/SW0090_unique70000_s{seed}_e10/checkpoints/epoch_{epoch:02d}.pt"
            core = S2NetCore(hparams(), device="cpu").eval()
            core.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
            for steps, settle in ((64, 32), (1024, 512)):
                with torch.no_grad():
                    core(gamma, return_core_out=True, num_time_steps=steps)
                    absolute = _component_spike_synchrony(core, settle)
                    positive = aligned_affinity(core, settle)
                mask = ~torch.eye(256, dtype=torch.bool)
                for i, image_id in enumerate(ids):
                    result["rows"].append({"seed": seed, "epoch": epoch,
                        "image_id": image_id, "steps": steps, "settle": settle,
                        "absolute": measure(absolute[i], labels[i]),
                        "positive": measure(positive[i], labels[i]),
                        "offdiagonal_edge_disagreement": float(((absolute[i] >= .5) != (positive[i] >= .5))[mask].float().mean())})
                path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    result["status"] = "complete"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("WINDOW_DIAGNOSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
