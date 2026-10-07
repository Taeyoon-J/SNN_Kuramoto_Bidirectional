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
    DATASET, VAL_GAMMA, auc_stats, load_checkpoint, product_affinity, read_labels,
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


def evaluate(checkpoint, source_mode, destination_mode, seed, condition, device, output, smoke_first_batch=False):
    if output.exists():
        raise FileExistsError(f"inspect existing swap output; no overwrite: {output}")
    gamma = torch.load(VAL_GAMMA, map_location="cpu", weights_only=True)
    rows = []
    all_gates, all_drives = [], []
    sine_mean_abs_sum, sine_abs_sum = 0.0, 0.0
    sine_mean_count, sine_component_count = 0, 0
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
            # Use the actual gate-function outputs captured during this forward,
            # with SW0099's product-Pearson rule for the gated-drive pair metric.
            if drive_hist.shape[1] != 512:
                raise AssertionError(f"captured drive history has {drive_hist.shape[1]} frames; expected 512 settle frames")
            gated_drive_affinity = product_affinity(drive_hist.permute(0, 3, 2, 1), settle=0)
            if gated_drive_affinity.shape != (len(ids), 256, 256) or not torch.isfinite(gated_drive_affinity).all():
                raise FloatingPointError("gated-drive affinity must have a finite, nonempty 512-frame trace")
            predicted = spike_synchrony_components(
                spikes.cpu(), synchrony_threshold=.5, min_group_size=2, settle=512,
                components=core.last_component_spikes.cpu(), background="largest_component")
            labels = read_labels(dataset, ids)
            pred_labels = spatial_components_to_patch_labels(predicted, 16)
            scores = evaluate_patch_masks(pred_labels, labels.reshape(8, 16, 16))["per_image"]
            for i, image_id in enumerate(ids):
                stage = auc_stats(spike_affinity[i], labels[i])
                drive_stage = auc_stats(gated_drive_affinity[i], labels[i])
                spike_near = stage["distance_bins_euclidean_patches"]["0-2"]
                spike_mid = stage["distance_bins_euclidean_patches"]["2-5"]
                drive_near = drive_stage["distance_bins_euclidean_patches"]["0-2"]
                drive_mid = drive_stage["distance_bins_euclidean_patches"]["2-5"]
                rows.append({"image_id": image_id,
                             "fg_ari": float(scores["fg_ari"][i]),
                             "foreground_iou": float(scores["foreground_iou"][i]),
                             "matched_object_iou": float(scores["matched_object_iou"][i]),
                             "predicted_groups": len(predicted[i]),
                             "target_groups": int(torch.unique(labels[i][labels[i] > 0]).numel()),
                             "spike_pair_auc_near": spike_near.get("auc"),
                             "spike_pair_auc_mid": spike_mid.get("auc"),
                             "spike_pair_valid_pairs_near": spike_near.get("pairs", 0),
                             "spike_pair_valid_pairs_mid": spike_mid.get("pairs", 0),
                             "spike_pair_separation_near": spike_near.get("same_minus_different"),
                             "spike_pair_separation_mid": spike_mid.get("same_minus_different"),
                             "gated_drive_pair_auc_near": drive_near.get("auc"),
                             "gated_drive_pair_auc_mid": drive_mid.get("auc"),
                             "gated_drive_pair_valid_pairs_near": drive_near.get("pairs", 0),
                             "gated_drive_pair_valid_pairs_mid": drive_mid.get("pairs", 0),
                             "gated_drive_pair_separation_near": drive_near.get("same_minus_different"),
                             "gated_drive_pair_separation_mid": drive_mid.get("same_minus_different")})
            all_gates.append(gate_hist.detach().cpu())
            all_drives.append(drive_hist.detach().cpu())
            sine_mean_abs_sum += float(sine.mean(dim=-1).abs().sum())
            sine_abs_sum += float(sine.abs().sum())
            sine_mean_count += sine.mean(dim=-1).numel()
            sine_component_count += sine.numel()
            del core, spikes, theta, predicted, labels, gate_hist, drive_hist, gated_drive_affinity
            if device == "cuda":
                torch.cuda.empty_cache()
            if smoke_first_batch:
                break
    gate_values = torch.cat(all_gates, dim=0)
    drive_values = torch.cat(all_drives, dim=0)
    gate_stats = {"images": len(rows), "settle_steps": 512,
                  "pooled_gate_mean": float(gate_values.mean()),
                  "pooled_gate_variance": float(gate_values.var(unbiased=False)),
                  "pooled_gated_drive_variance": float(drive_values.var(unbiased=False)),
                  "variance_scope": "pooled over all recorded images, 512 settle steps, and 256 patches (drive includes four oscillator components)",
                  "component_sine_cancellation_index": 1.0 -
                      (sine_mean_abs_sum / max(1, sine_mean_count)) /
                      max(1e-12, sine_abs_sum / max(1, sine_component_count))}

    def bin_summary(field):
        values = [row[field] for row in rows if row[field] is not None]
        return {"mean": sum(values) / len(values) if values else None,
                "valid_images": len(values), "images": len(rows)}

    record = {"status": "complete", "diagnostic_only": True,
              "seed": seed, "condition": condition,
              "source_gate_mode": source_mode, "evaluated_gate_mode": destination_mode,
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
              "ids": [row["image_id"] for row in rows], "images": len(rows), "batch_size": 8,
              "steps": 1024, "settle": 512, "threshold": .50,
              "ground_truth_used_only_after_forward": True,
              "first_batch_postprocessing_smoke": bool(smoke_first_batch),
              "gate_and_cancellation_stats": gate_stats,
              "distance_bin_pair_auc": {
                  "spike_near_0_2": bin_summary("spike_pair_auc_near"),
                  "spike_mid_2_5": bin_summary("spike_pair_auc_mid"),
                  "gated_drive_near_0_2": bin_summary("gated_drive_pair_auc_near"),
                  "gated_drive_mid_2_5": bin_summary("gated_drive_pair_auc_mid")},
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
    parser.add_argument("--smoke-first-batch", action="store_true", help="run only the first fixed batch to verify the 512-frame gated-drive postprocessing")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.source_mode, args.destination_mode, args.seed,
             args.condition, args.device, args.output, smoke_first_batch=args.smoke_first_batch)


if __name__ == "__main__":
    main()
