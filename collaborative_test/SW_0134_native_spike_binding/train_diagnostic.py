"""No-GT first-TRAIN-batch readout diagnostics after the fixed-budget pilot."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from collaborative_test.SW_0134_native_spike_binding import run, evaluate
from collaborative_test.SW_0134_native_spike_binding.binder import render_slot_rgb
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout

ROOT, HERE = run.ROOT, run.HERE
ARMS = tuple(run.ARMS)
BATCH = run.BATCH


def _arm_diagnostics(seed, arm, device, rows, cache, assets):
    wrapped, encoder, patcher, mean, std, clip, binder, decoder, manifest, state, checkpoint = \
        evaluate._load_training(seed, arm, device, ROOT / "trained_models/SW0134_native_spike_binding",
                                assets=assets)
    images = run.sw130.read_batch(cache, rows, device)
    with torch.no_grad():
        gamma = run.sw130.encode(encoder, patcher, mean, std, clip, images)
        trace = late_rollout(wrapped, gamma, total_steps=1024, live_tail_steps=0)
        spikes = trace["component_spikes"]
        events = spikes[..., -512:]
        head_input = (trace["component_gates"][..., -512:]
                      if arm == "gate_joint" else events)
        assignments, slots, _features = binder(head_input)
        prediction = render_slot_rgb(assignments, slots, decoder,
                                     chunk_size=1024, checkpoint_chunks=False)
        # This intervention permutes P columns while holding slot latents/decoder
        # weights fixed; P-derived centroids and relative coordinates are recomputed.
        # It deliberately does not permute P and slots together (which is invariant).
        scrambled = torch.roll(assignments, shifts=1, dims=-1)
        scrambled_prediction = render_slot_rgb(scrambled, slots, decoder,
                                                chunk_size=1024, checkpoint_chunks=False)
        target = images.permute(0, 2, 3, 1).contiguous() / 255.0
        mse = (prediction - target).square().mean(dim=(1, 2, 3))
        scrambled_mse = (scrambled_prediction - target).square().mean(dim=(1, 2, 3))
        probs = assignments.float()
        occupancy = probs.mean(dim=1)
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(-1)
        hard_counts = torch.nn.functional.one_hot(probs.argmax(dim=-1), num_classes=probs.shape[-1]).sum(dim=1)
        record = {
            "arm": arm, "checkpoint_sha256": run.sha(checkpoint),
            "source_core_sha256": manifest["source_core_sha256"],
            "input_mode": "gate" if arm == "gate_joint" else "actual_emitted_spikes",
            "mean_event_activity": float(events.mean().cpu()),
            "mean_native_gate_activity": float(trace["component_gates"].mean().cpu()),
            "mean_patch_assignment_entropy": float(entropy.mean().cpu()),
            "mean_slot_probability_occupancy": [float(x) for x in occupancy.mean(0).cpu()],
            "per_image_nonempty_hard_slots": [int((row > 0).sum()) for row in hard_counts.cpu()],
            "per_image_rgb_mse": [float(x) for x in mse.cpu()],
            "per_image_slot_column_scramble_rgb_mse": [float(x) for x in scrambled_mse.cpu()],
            "mean_rgb_mse": float(mse.mean().cpu()),
            "mean_slot_column_scramble_rgb_mse": float(scrambled_mse.mean().cpu()),
            "slot_column_scramble": ("roll P columns by +1 while holding slot latents and decoder weights fixed; "
                                      "recompute P-derived centroids and relative coordinates"),
            "ground_truth_used": False,
            "optimizer_updates": 0,
        }
    if (not all(math.isfinite(v) for v in (record["mean_event_activity"],
                                            record["mean_native_gate_activity"],
                                            record["mean_rgb_mse"],
                                            record["mean_slot_column_scramble_rgb_mse"]))
            or len(record["per_image_rgb_mse"]) != BATCH):
        raise FloatingPointError("nonfinite/partial SW0134 TRAIN diagnostic")
    return record


def diagnose(seed, device, output):
    if seed != 1:
        raise ValueError("SW0134 registered head-diagnostic batch is seed1 only")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing TRAIN diagnostic: {output}")
    assets = run.sw130.validate_rgb_assets()
    pool = run.source_contract(seed)[3]
    ids = run.source_contract(seed)[4]
    rows = np.asarray(pool[:BATCH], dtype=np.int64)
    train_ids = [int(x) for x in np.asarray(ids[:BATCH], dtype=np.int64)]
    cache = np.load(run.sw130.TRAIN_RGB, mmap_mode="r")
    diagnostics = {arm: _arm_diagnostics(seed, arm, torch.device(device), rows, cache, assets)
                   for arm in ARMS}
    report = {"experiment": "SW0134_native_spike_binding", "status": "complete",
              "seed": seed, "stage": "post_training_train_only_readout_diagnostic",
              "image_ids": train_ids, "train_cache_sha256": assets["train_cache_sha256"],
              "implementation": {str((HERE / name).relative_to(ROOT).as_posix()):
                                 run.sha(HERE / name)
                                 for name in ("run.py", "binder.py", "rollout.py",
                                              "train.py", "train_diagnostic.py")},
              "diagnostics": diagnostics, "ground_truth_used": False,
              "optimizer_updates": 0,
              "interpretation": "Descriptive TRAIN-only endpoint readout diagnostic; not an eligibility or promotion gate."}
    run.write_once(output, report)
    if run.sha(output) == "":
        raise AssertionError("diagnostic artifact hash missing")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = diagnose(args.seed, args.device, args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "arms": list(report["diagnostics"])}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
