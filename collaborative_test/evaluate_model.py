"""Score a trained core under the evaluation contract.

The final number comes from masks a classifier builds out of spikes, not from
clustering theta. Phase readouts are computed too, clearly labelled, because the
contract keeps them as diagnostics.

Nothing here uses the true object count, the true masks, or anything else from
the ground truth to choose a threshold or a group count: the classifier's two
thresholds are arguments, chosen on validation.
"""
import argparse, json, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "snn_kuramoto_bidirectional"))
from evaluation import evaluate_patch_masks, spatial_components_to_patch_labels
from spike_classifier import spike_synchrony_components, spike_spatial_components
from hyperparameter import S2NetHyperparameters
from s2net_cls import S2NetCore
from loss_function import phase_locking_value


def build_core(args, device):
    torch.manual_seed(0)
    hp = S2NetHyperparameters(
        num_feature_maps=8, num_regions=args.num_regions, sc=None,
        gamma_drive_mode="static", num_time_steps=args.num_time_steps,
        theta_init="gamma", gamma_phase_mode="standardize_tanh", osc_dim=args.osc_dim,
        freq_gain=args.freq_gain, graph_mode="learned", graph_top_k=args.graph_top_k,
        k=float(args.num_regions), graph_spatial_decay=args.graph_spatial_decay,
        low_n=-4., high_n=0., membrane_vth=0.06, membrane_low_m=-4., membrane_high_m=0.,
        gate_mode=args.gate_mode, spike_per_component=args.spike_per_component,
        spike_pulse_gain=args.spike_pulse_gain, center_pulse=not args.no_center_pulse,
        spike_classify_method="spatial_components",
        spike_spatial_grid_size=(args.grid, args.grid),
    ).validate()
    core = S2NetCore(hp, device=device).to(device)
    core.load_state_dict(torch.load(args.checkpoint, map_location=device))
    core.eval()
    return core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gamma-seq-path", required=True)
    ap.add_argument("--targets", required=True, help="targets_v1.pt from make_targets.py")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--limit", type=int, default=None, help="score the first N of the split")
    ap.add_argument("--num-regions", type=int, default=256)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--osc-dim", type=int, default=4)
    ap.add_argument("--graph-top-k", type=int, default=32)
    ap.add_argument("--graph-spatial-decay", type=float, default=0.55)
    ap.add_argument("--freq-gain", type=float, default=2.0)
    ap.add_argument("--gate-mode", default="raw")
    ap.add_argument("--spike-per-component", action="store_true")
    ap.add_argument("--spike-pulse-gain", type=float, default=0.0)
    ap.add_argument("--no-center-pulse", action="store_true")
    ap.add_argument("--num-time-steps", type=int, default=256)
    ap.add_argument("--settle", type=int, default=64)
    ap.add_argument("--foreground-threshold", type=float, default=0.15)
    ap.add_argument("--synchrony-threshold", type=float, default=0.5)
    ap.add_argument("--min-group-size", type=int, default=2)
    ap.add_argument("--membrane-threshold", type=float, default=0.06,
                    help="used when --use-model-spikes is off; the core's own v_th")
    ap.add_argument("--background", default="largest_component",
                    choices=["largest_component", "activity"])
    ap.add_argument("--membrane-spikes", action="store_true",
                    help=(
                        "Binarise the membrane at --membrane-threshold instead of "
                        "reading the layer's own spikes. Off by default: an earlier "
                        "version thresholded at the median, which makes every unit "
                        "fire exactly half the time and left foreground IoU at "
                        "chance."
                    ))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    start, end = manifest["ranges"][args.split]
    if args.limit is not None:
        end = min(end, start + args.limit)
    blob = torch.load(args.targets)
    if blob.get("contract_version") != 1:
        raise ValueError("targets were not built under contract version 1.")
    target = blob["patch_labels"][start:end]
    names = blob["names"][start:end]
    if names != manifest["splits"][args.split][: len(names)]:
        raise ValueError("target order does not match the split manifest.")
    gamma = torch.load(args.gamma_seq_path, map_location="cpu").float()[start:end]
    if gamma.size(0) != target.size(0):
        raise ValueError(f"gamma has {gamma.size(0)} rows for {target.size(0)} targets.")
    print(f"{args.split} split: {gamma.size(0)} images, ids {start}..{end - 1}", flush=True)

    core = build_core(args, args.device)
    spikes_all, comp_all, plv_all, rates = [], [], [], []
    with torch.no_grad():
        for s in range(0, gamma.size(0), 25):
            _, model_spikes, membrane, theta = core(gamma[s:s + 25].to(args.device),
                                                    return_core_out=True, return_theta=True)
            # The model's own spikes, from its membrane threshold. Re-binarising
            # the membrane at its median destroyed the foreground signal: it makes
            # every unit fire exactly half the time by construction, so every patch
            # looks active and foreground IoU collapses to the fraction of the grid
            # that is genuinely foreground.
            spike = ((membrane > args.membrane_threshold).float()
                     if args.membrane_spikes else model_spikes.float())
            spikes_all.append(spike.cpu())
            rates.append(float(spike[:, :, args.settle:].mean()))
            comp_all.append(core.last_component_spikes.cpu()
                            if core.last_component_spikes is not None else None)
            plv_all.append(phase_locking_value(theta, settle=args.settle, combine="product").cpu())
    spikes = torch.cat(spikes_all)
    components = torch.cat([c for c in comp_all]) if comp_all[0] is not None else None
    plv = torch.cat(plv_all)

    groups = spike_synchrony_components(
        spikes, foreground_threshold=args.foreground_threshold,
        synchrony_threshold=args.synchrony_threshold,
        min_group_size=args.min_group_size, settle=args.settle, components=components,
        background=args.background,
    )
    prediction = spatial_components_to_patch_labels(groups, args.grid)
    scored = evaluate_patch_masks(prediction, target)

    counts = [len(g) for g in groups]
    report = {
        "split": args.split, "images": int(gamma.size(0)),
        "contract_version": 1, "manifest_version": manifest["version"],
        "checkpoint": str(args.checkpoint),
        "classifier": "spike_synchrony_components",
        "background": args.background,
        "foreground_threshold": args.foreground_threshold,
        "synchrony_threshold": args.synchrony_threshold,
        "min_group_size": args.min_group_size,
        "metrics": {k: float(v) for k, v in scored["mean"].items()},
        "valid_count": {k: int(v) for k, v in scored["valid_count"].items()},
        "diagnostics": {
            "spike_rate": sum(rates) / len(rates),
            "groups_per_image_mean": sum(counts) / len(counts),
            "images_with_no_group": sum(1 for c in counts if c == 0),
            "plv_mean": float(plv.mean()),
            "predicted_foreground_fraction": float((prediction != 0).float().mean()),
            "target_foreground_fraction": float((target != 0).float().mean()),
        },
    }
    print()
    print("%-26s %s" % ("SCORED FROM SPIKE MASKS", "mean"))
    print("-" * 40)
    for k, v in report["metrics"].items():
        print("%-26s %.4f   (n=%d)" % (k, v, report["valid_count"][k]))
    print()
    for k, v in report["diagnostics"].items():
        print("  %-34s %.4f" % (k, v))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.json_out, "w"), indent=1)
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
