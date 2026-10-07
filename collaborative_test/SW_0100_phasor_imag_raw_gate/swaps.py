"""Fixed 16-image zero-update raw/mode swaps for SW0100 provenance only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"),
                str(ROOT / "collaborative_test")]
from SW_0099_stage_flow_diagnosis.diagnose import (
    DATASET, VAL_GAMMA, auc_stats, load_checkpoint, read_labels,
)
from snn_kuramoto_bidirectional.evaluation import evaluate_patch_masks, spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.sinusoidal_gating import sinusoidal_gating
from snn_kuramoto_bidirectional.spike_classifier import (
    spike_synchrony_affinity, spike_synchrony_components,
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def evaluate(checkpoint, source_mode, destination_mode, seed, condition, device, output):
    if output.exists():
        raise FileExistsError(f"inspect existing swap output; no overwrite: {output}")
    gamma = torch.load(VAL_GAMMA, map_location="cpu", weights_only=True)
    rows = []
    with h5py.File(DATASET, "r") as dataset:
        for start in (1320, 1328):
            ids = list(range(start, start + 8))
            core = load_checkpoint(checkpoint, device)
            core.gate_mode = destination_mode
            x = gamma[start - 1320:start - 1320 + 8].to(device)
            captured_gates, captured_drives = [], []
            import snn_kuramoto_bidirectional.s2net_cls as s2net_module
            original_gating = s2net_module.sinusoidal_gating

            def capture_gating(history, t, delay, gate_mode="sigmoid"):
                drive_t, gate_t = original_gating(history, t, delay, gate_mode)
                if t >= 512:
                    captured_gates.append(gate_t.detach())
                    captured_drives.append(drive_t.detach())
                return drive_t, gate_t

            s2net_module.sinusoidal_gating = capture_gating
            try:
                with torch.inference_mode():
                    _, spikes, _, theta = core(x, return_core_out=True, return_theta=True,
                                               num_time_steps=1024)
            finally:
                s2net_module.sinusoidal_gating = original_gating
            if not torch.isfinite(spikes).all():
                raise FloatingPointError("nonfinite spikes in zero-update mode swap")
            if not captured_gates or not captured_drives:
                raise RuntimeError("actual gating function did not produce trace values")
            gate_hist = torch.stack(captured_gates, dim=1)
            drive_hist = torch.stack(captured_drives, dim=1)
            if not torch.isfinite(gate_hist).all() or not torch.isfinite(drive_hist).all():
                raise FloatingPointError("nonfinite gate or drive in zero-update mode swap")
            delayed_theta = theta[:, 510:-2]
            sine = delayed_theta.sin()
            aligned_signal = sine.mean(dim=-1).abs().mean()
            component_scale = sine.abs().mean(dim=-1).mean()
            cancellation_index = 1. - aligned_signal / component_scale.clamp_min(1e-12)
            spike_affinity = spike_synchrony_affinity(
                core.last_component_spikes.mean(dim=1), core.last_component_spikes,
                settle=512)
            predicted = spike_synchrony_components(
                spikes.cpu(), synchrony_threshold=.5, min_group_size=2, settle=512,
                components=core.last_component_spikes.cpu(), background="largest_component")
            labels = read_labels(dataset, ids)
            pred_labels = spatial_components_to_patch_labels(predicted, 16)
            scores = evaluate_patch_masks(pred_labels, labels.reshape(8, 16, 16))["per_image"]
            for i, image_id in enumerate(ids):
                stage = auc_stats(spike_affinity[i], labels[i])
                rows.append({"image_id": image_id,
                             "fg_ari": float(scores["fg_ari"][i]),
                             "foreground_iou": float(scores["foreground_iou"][i]),
                             "matched_object_iou": float(scores["matched_object_iou"][i]),
                             "predicted_groups": len(predicted[i]),
                             "target_groups": int(torch.unique(labels[i][labels[i] > 0]).numel()),
                             "spike_pair_auc_near": stage["distance_bins_euclidean_patches"]["0-2"]["auc"],
                             "spike_pair_auc_mid": stage["distance_bins_euclidean_patches"]["2-5"]["auc"],
                             "spike_pair_separation_near": stage["distance_bins_euclidean_patches"]["0-2"]["same_minus_different"],
                             "spike_pair_separation_mid": stage["distance_bins_euclidean_patches"]["2-5"]["same_minus_different"]})
            gate_stats = {"settle_gate_mean": float(gate_hist.mean()),
                          "settle_gate_variance": float(gate_hist.var(unbiased=False)),
                          "settle_gated_drive_variance": float(drive_hist.var(unbiased=False)),
                          "component_sine_cancellation_index": float(cancellation_index)}
            del core, spikes, theta, predicted, labels, gate_hist, drive_hist
            if device == "cuda":
                torch.cuda.empty_cache()
    record = {"status": "complete", "diagnostic_only": True,
              "seed": seed, "condition": condition,
              "source_gate_mode": source_mode, "evaluated_gate_mode": destination_mode,
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
              "ids": list(range(1320, 1336)), "images": 16, "batch_size": 8,
              "steps": 1024, "settle": 512, "threshold": .50,
              "ground_truth_used_only_after_forward": True,
              "gate_and_cancellation_stats": gate_stats,
              "rows": rows,
              "interpretation_limit": "fixed representation check only; not used to select mode or tune"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(str(output), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-mode", choices=["raw", "phasor_imag_raw"], required=True)
    parser.add_argument("--destination-mode", choices=["raw", "phasor_imag_raw"], required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--condition", choices=["source_zero_update", "trained_candidate_zero_update"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.source_mode, args.destination_mode, args.seed,
             args.condition, args.device, args.output)


if __name__ == "__main__":
    main()
