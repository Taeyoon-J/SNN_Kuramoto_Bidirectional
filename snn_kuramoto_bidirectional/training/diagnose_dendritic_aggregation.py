"""Compare dendritic branch aggregation modes with one frozen checkpoint."""

import argparse
import csv
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
    from snn_kuramoto_bidirectional.s2net_cls import S2NetClassifier
    from snn_kuramoto_bidirectional.training.train_input_layer_generator import (
        load_training_images,
    )
except ModuleNotFoundError:
    from hyperparameter import S2NetHyperparameters
    from s2net_cls import S2NetClassifier
    from training.train_input_layer_generator import load_training_images


AGGREGATION_MODES = ("sum", "relu_sum", "abs_sum")


def _statistics(values):
    values = values.detach().float()
    absolute = values.abs()
    return {
        "shape": list(values.shape),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "std": float(values.std(unbiased=False)),
        "abs_mean": float(absolute.mean()),
        "abs_max": float(absolute.max()),
    }


def _evaluate_mode(args, images, mode, device):
    grid_size = (
        int(args.patch_grid_size[0])
        if len(args.patch_grid_size) == 1
        else tuple(int(value) for value in args.patch_grid_size)
    )
    grid_pair = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
    hparams = S2NetHyperparameters(
        num_feature_maps=args.num_feature_maps,
        num_regions=grid_pair[0] * grid_pair[1],
        kernel_size=args.kernel_size,
        gamma_patch_grid_size=grid_size,
        sc_sigma_color=args.sc_sigma_color,
        sc_m_min=args.sc_m_min,
        sc_self_connectivity=args.sc_self_connectivity,
        k=args.k,
        dt=args.dt,
        low_n=args.low_n,
        high_n=args.high_n,
        branch=args.branch,
        dendritic_aggregation=mode,
        spike_spatial_grid_size=grid_size,
    )
    model = S2NetClassifier(hparams, device=device).to(device)
    model.load_checkpoints(
        input_layer_path=args.input_encoder_path,
        core_path=args.checkpoint_path,
        map_location=device,
        eval_mode=True,
    )

    h_wave_steps = []

    def capture_h_wave(module, inputs, output):
        h_wave_steps.append(output.detach().cpu().clone())

    handle = model.core.dendric_layer.register_forward_hook(capture_h_wave)
    try:
        with torch.inference_mode():
            output = model(images, return_details=True, classify=False)
    finally:
        handle.remove()

    h_wave = torch.stack(h_wave_steps, dim=1)
    membrane = output.core_out.detach().cpu()
    spikes = output.spikes.detach().cpu()
    result = {
        "aggregation_mode": mode,
        "h_wave": _statistics(h_wave),
        "membrane": _statistics(membrane),
        "spikes": {
            "shape": list(spikes.shape),
            "nonzero_count": int(torch.count_nonzero(spikes)),
            "actual_spike_rate": float((spikes != 0).float().mean()),
        },
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "state_dict_keys": sorted(model.core.state_dict().keys()),
    }
    return result, {"h_wave": h_wave, "membrane": membrane, "spikes": spikes}


def _print_result(result):
    print(f"Dendritic aggregation mode: {result['aggregation_mode']}")
    for name in ("h_wave", "membrane"):
        values = result[name]
        print(f"{name}:")
        for statistic in ("min", "mean", "max", "std", "abs_mean", "abs_max"):
            print(f"  {statistic}: {values[statistic]:.10f}")
    print("spikes:")
    print(f"  nonzero count: {result['spikes']['nonzero_count']}")
    print(f"  actual spike rate: {result['spikes']['actual_spike_rate']:.10f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare branch aggregation modes using identical images and one "
            "unchanged S2Net checkpoint."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-dir")
    source.add_argument("--dataset-path")
    parser.add_argument("--hdf5-key", default="image")
    parser.add_argument("--input-encoder-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--num-feature-maps", type=int, default=8)
    parser.add_argument("--patch-grid-size", type=int, nargs="+", default=[16])
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--low-n", type=float, default=0.0)
    parser.add_argument("--high-n", type=float, default=4.0)
    parser.add_argument("--branch", type=int, default=4)
    parser.add_argument("--sc-sigma-color", type=float, default=0.25)
    parser.add_argument("--sc-m-min", type=float, default=0.5)
    parser.add_argument("--sc-self-connectivity", type=float, default=0.0)
    parser.add_argument(
        "--dendritic-aggregation",
        action="append",
        choices=AGGREGATION_MODES,
        dest="aggregation_modes",
        help=(
            "Mode to evaluate. Repeat this option to select multiple modes. "
            "If omitted, all three modes are evaluated."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if len(args.patch_grid_size) not in (1, 2) or any(
        value <= 0 for value in args.patch_grid_size
    ):
        parser.error("--patch-grid-size requires one or two positive integers.")
    modes = args.aggregation_modes or list(AGGREGATION_MODES)
    device = torch.device(args.device)
    images = load_training_images(
        image_dir=args.image_dir,
        dataset_path=args.dataset_path,
        hdf5_key=args.hdf5_key,
        image_size=args.image_size,
        max_images=args.num_samples,
    ).to(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    reference_parameter_count = None
    reference_state_dict_keys = None
    for mode in modes:
        result, tensors = _evaluate_mode(args, images, mode, device)
        if reference_parameter_count is None:
            reference_parameter_count = result["trainable_parameter_count"]
            reference_state_dict_keys = result["state_dict_keys"]
        else:
            if result["trainable_parameter_count"] != reference_parameter_count:
                raise RuntimeError("Aggregation modes changed the parameter count.")
            if result["state_dict_keys"] != reference_state_dict_keys:
                raise RuntimeError("Aggregation modes changed checkpoint keys.")
        torch.save(tensors, output_dir / f"{mode}_outputs.pt")
        _print_result(result)
        results.append(result)

    serializable_results = []
    for result in results:
        result = dict(result)
        result.pop("state_dict_keys")
        serializable_results.append(result)
    with (output_dir / "aggregation_comparison.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(serializable_results, file, indent=2)

    columns = [
        "aggregation_mode",
        "h_wave_abs_mean",
        "h_wave_max",
        "membrane_abs_mean",
        "membrane_max",
        "spike_nonzero_count",
        "actual_spike_rate",
    ]
    with (output_dir / "aggregation_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "aggregation_mode": result["aggregation_mode"],
                "h_wave_abs_mean": result["h_wave"]["abs_mean"],
                "h_wave_max": result["h_wave"]["max"],
                "membrane_abs_mean": result["membrane"]["abs_mean"],
                "membrane_max": result["membrane"]["max"],
                "spike_nonzero_count": result["spikes"]["nonzero_count"],
                "actual_spike_rate": result["spikes"]["actual_spike_rate"],
            })

    print("Comparison")
    print(
        f"{'mode':<12} {'h abs mean':>12} {'h max':>12} "
        f"{'mem abs mean':>14} {'mem max':>12} {'spike rate':>12}"
    )
    for result in results:
        print(
            f"{result['aggregation_mode']:<12} "
            f"{result['h_wave']['abs_mean']:>12.6f} "
            f"{result['h_wave']['max']:>12.6f} "
            f"{result['membrane']['abs_mean']:>14.6f} "
            f"{result['membrane']['max']:>12.6f} "
            f"{result['spikes']['actual_spike_rate']:>12.6f}"
        )
    print(f"Saved comparison: {output_dir}")


if __name__ == "__main__":
    main()
