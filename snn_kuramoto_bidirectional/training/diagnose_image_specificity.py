"""
Measure whether S2NetCore trajectories actually depend on the input image.

The `unique_binary_masks = 1` / `mean_pairwise_mask_iou = 1.0` failure has two
very different causes, and they need opposite fixes:

  A. The drive term is too weak relative to omega/coupling, so theta follows a
     nearly image-independent trajectory from its shared zero initial state.
     Losses cannot fix this; the dynamics have to change.
  B. Trajectories do diverge, but thresholding collapses them into one mask.
     This is a readout/regularizer problem.

This script separates the two by tracking, at every recurrent step, how far
apart different images are in phase space and in activity space, relative to
how far apart their gamma inputs are.

Read `specificity_ratio` in the output:
  ~0      -> cause A: the core is ignoring the image
  ~1+     -> cause B: image information survives; look at the readout instead

Example:
    python -m training.diagnose_image_specificity \
        --gamma-seq-path runs/gamma_seq.pt \
        --sc-path runs/sc.pt \
        --output-dir runs/diagnostics \
        --num-samples 32
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
    from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
except ModuleNotFoundError:
    from hyperparameter import S2NetHyperparameters
    from s2net_cls import S2NetCore


def mean_pairwise_cosine_distance(values):
    """Mean off-diagonal cosine distance between flattened samples."""
    flat = values.reshape(values.size(0), -1).float()
    normalized = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    similarity = normalized @ normalized.transpose(0, 1)
    return float(_off_diagonal_mean(1.0 - similarity))


def mean_pairwise_phase_distance(theta):
    """
    Mean circular distance between samples, in [0, 1].

    theta: [B, N, D]. Uses (1 - cos(delta)) / 2 so that wrapping is handled
    correctly; plain Euclidean distance on raw phases is misleading once theta
    has drifted past 2*pi.
    """
    diff = theta.unsqueeze(1) - theta.unsqueeze(0)  # [B, B, N, D]
    distance = (1.0 - torch.cos(diff)).mean(dim=(-2, -1)) / 2.0
    return float(_off_diagonal_mean(distance))


def mask_statistics(activity, threshold):
    """Mask diversity across samples for one aggregated activity map."""
    masks = (activity >= float(threshold)).reshape(activity.size(0), -1)
    unique = torch.unique(masks, dim=0).size(0)

    intersection = (masks.unsqueeze(1) & masks.unsqueeze(0)).sum(dim=-1).float()
    union = (masks.unsqueeze(1) | masks.unsqueeze(0)).sum(dim=-1).float()
    iou = torch.where(union > 0, intersection / union.clamp_min(1.0), torch.ones_like(union))

    return {
        "unique_binary_masks": int(unique),
        "mean_pairwise_mask_iou": float(_off_diagonal_mean(iou)),
        "mean_mask_density": float(masks.float().mean()),
    }


def _off_diagonal_mean(matrix):
    size = matrix.size(0)
    if size < 2:
        return torch.zeros((), device=matrix.device)
    mask = ~torch.eye(size, dtype=torch.bool, device=matrix.device)
    return matrix[mask].mean()


@torch.no_grad()
def diagnose(core, gamma_seq, threshold=0.5, num_time_steps=None):
    core.eval()
    forward_kwargs = {"return_core_out": True, "return_theta": True}
    if num_time_steps is not None:
        forward_kwargs["num_time_steps"] = int(num_time_steps)

    _, spikes, core_out, theta = core(gamma_seq, **forward_kwargs)
    activity = torch.sigmoid(core_out)  # [B, N, T]

    # Baseline: how different are the inputs themselves?
    gamma_distance = mean_pairwise_cosine_distance(gamma_seq)

    per_step = []
    for t in range(theta.size(1)):
        step_activity = activity[:, :, t]
        per_step.append(
            {
                "step": int(t),
                "phase_distance": mean_pairwise_phase_distance(theta[:, t]),
                "activity_distance": mean_pairwise_cosine_distance(step_activity),
                "mean_activity": float(step_activity.mean()),
                "active_units": int((step_activity >= float(threshold)).sum(dim=1).float().mean().item()),
                **mask_statistics(step_activity, threshold),
            }
        )

    aggregated = activity.mean(dim=2)
    final_activity_distance = per_step[-1]["activity_distance"]

    return {
        "num_samples": int(gamma_seq.size(0)),
        "num_time_steps": int(theta.size(1)),
        "num_oscillators": int(theta.size(2)),
        "osc_dim": int(theta.size(3)),
        "gamma_input_distance": gamma_distance,
        "specificity_ratio": (
            final_activity_distance / gamma_distance if gamma_distance > 1e-8 else 0.0
        ),
        "phase_distance_first_step": per_step[0]["phase_distance"],
        "phase_distance_last_step": per_step[-1]["phase_distance"],
        "activity_distance_first_step": per_step[0]["activity_distance"],
        "activity_distance_last_step": final_activity_distance,
        "spike_rate": float(spikes.float().mean()),
        "active_patches_by_time": [item["active_units"] for item in per_step],
        "time_aggregated": mask_statistics(aggregated, threshold),
        "per_step": per_step,
    }


def save_figure(report, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [item["step"] for item in report["per_step"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(steps, [item["phase_distance"] for item in report["per_step"]], marker="o")
    axes[0].set_title("pairwise phase distance")
    axes[0].set_xlabel("recurrent step")
    axes[0].set_ylabel("(1 - cos) / 2")
    axes[0].set_ylim(bottom=0.0)

    axes[1].plot(steps, [item["activity_distance"] for item in report["per_step"]], marker="o", label="activity")
    axes[1].axhline(report["gamma_input_distance"], linestyle="--", color="gray", label="gamma input")
    axes[1].set_title("pairwise activity distance")
    axes[1].set_xlabel("recurrent step")
    axes[1].legend()
    axes[1].set_ylim(bottom=0.0)

    axes[2].plot(steps, [item["active_units"] for item in report["per_step"]], marker="o", color="tab:red")
    axes[2].set_title("active units per step")
    axes[2].set_xlabel("recurrent step")
    axes[2].set_ylim(bottom=0.0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Check whether S2NetCore trajectories depend on the input image."
    )
    parser.add_argument("--gamma-seq-path", required=True)
    parser.add_argument("--sc-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional trained core. Without it the untrained initialization is measured.",
    )
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Seed for core initialization. Required for comparing configurations: an "
            "untrained core's omega/kappa/tau draw moves these metrics as much as the "
            "setting under test, so sweeps must hold the seed fixed and ideally average "
            "over several."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--low-n", type=float, default=0.0)
    parser.add_argument("--high-n", type=float, default=4.0)
    parser.add_argument("--branch", type=int, default=4)
    parser.add_argument("--osc-dim", type=int, default=4)
    parser.add_argument("--gamma-drive-mode", default="sequence", choices=["sequence", "static"])
    parser.add_argument("--num-time-steps", type=int, default=None)
    parser.add_argument("--gamma-phase-mode", default="none", choices=["none", "tanh", "standardize_tanh"])
    parser.add_argument("--theta-init", default="zeros", choices=["zeros", "gamma", "gamma_noise"])
    parser.add_argument("--theta-init-noise", type=float, default=0.0)
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device(args.device)
    gamma_seq = torch.load(args.gamma_seq_path, map_location="cpu").float()
    if gamma_seq.dim() != 3:
        raise ValueError(f"gamma_seq must have shape [B, T, N], but got {tuple(gamma_seq.shape)}.")
    sc = torch.load(args.sc_path, map_location="cpu").float()

    num_samples = min(int(args.num_samples), gamma_seq.size(0))
    if num_samples < 2:
        raise ValueError("At least two samples are required to measure pairwise divergence.")
    gamma_seq = gamma_seq[:num_samples].to(device)

    num_feature_maps = gamma_seq.size(1)
    hparams = S2NetHyperparameters(
        num_feature_maps=num_feature_maps,
        num_regions=gamma_seq.size(2),
        sc=sc,
        k=args.k,
        dt=args.dt,
        low_n=args.low_n,
        high_n=args.high_n,
        branch=args.branch,
        osc_dim=args.osc_dim,
        gamma_drive_mode=args.gamma_drive_mode,
        num_time_steps=(
            num_feature_maps if args.num_time_steps is None else int(args.num_time_steps)
        ),
        gamma_phase_mode=args.gamma_phase_mode,
        theta_init=args.theta_init,
        theta_init_noise=args.theta_init_noise,
        spike_classify_method="spatial_components",
        spike_spatial_grid_size=_infer_grid(gamma_seq.size(2)),
    )
    hparams.validate()

    core = S2NetCore(hparams, device=device).to(device)
    if args.checkpoint_path is not None:
        core.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

    report = diagnose(
        core,
        gamma_seq,
        threshold=args.threshold,
        num_time_steps=(
            int(args.num_time_steps)
            if args.num_time_steps is not None and args.gamma_drive_mode == "static"
            else None
        ),
    )
    report["config"] = {
        "gamma_drive_mode": args.gamma_drive_mode,
        "gamma_phase_mode": args.gamma_phase_mode,
        "theta_init": args.theta_init,
        "osc_dim": args.osc_dim,
        "checkpoint_path": args.checkpoint_path,
        "threshold": args.threshold,
        "seed": int(args.seed),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "image_specificity.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    if not args.skip_figure:
        save_figure(report, output_dir / "image_specificity.png")

    ratio = report["specificity_ratio"]
    print(f"gamma input distance      : {report['gamma_input_distance']:.6f}")
    print(f"phase distance   step 0   : {report['phase_distance_first_step']:.6f}")
    print(f"phase distance   step {report['num_time_steps'] - 1:<3}: {report['phase_distance_last_step']:.6f}")
    print(f"activity distance step 0  : {report['activity_distance_first_step']:.6f}")
    print(f"activity distance final   : {report['activity_distance_last_step']:.6f}")
    print(f"specificity ratio         : {ratio:.6f}")
    print(f"time-aggregated masks     : {report['time_aggregated']}")
    if ratio < 0.05:
        print(
            "\nVERDICT: the core is nearly image-independent. The drive term is too weak "
            "relative to omega/coupling, or theta starts from a shared state. Try "
            "--theta-init gamma, --gamma-phase-mode standardize_tanh, or a larger kappa/K ratio. "
            "Tuning loss weights will not fix this."
        )
    else:
        print(
            "\nVERDICT: image information reaches the activity. If masks are still identical, "
            "the collapse happens at thresholding/readout, not in the dynamics."
        )
    print(f"\nsaved: {report_path}")


def _infer_grid(num_regions):
    side = int(num_regions ** 0.5)
    return (side, side) if side * side == num_regions else (1, num_regions)


if __name__ == "__main__":
    main()
