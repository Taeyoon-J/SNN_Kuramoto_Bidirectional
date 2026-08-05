"""Train the patch S2Net core with image-conditioned online SC."""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
    from snn_kuramoto_bidirectional.loss_function import UnsupervisedS2NetLoss
    from snn_kuramoto_bidirectional.s2net_cls import S2NetClassifier
    from snn_kuramoto_bidirectional.training.train_input_layer_generator import load_training_images
except ModuleNotFoundError:
    from hyperparameter import S2NetHyperparameters
    from loss_function import UnsupervisedS2NetLoss
    from s2net_cls import S2NetClassifier
    from training.train_input_layer_generator import load_training_images


def train_s2net_core(
    model,
    dataloader,
    epochs=100,
    lr=1e-3,
    criterion=None,
    optimizer=None,
    device=None,
    save_path=None,
    loss_signal="sigmoid_membrane",
    grad_clip_norm=1.0,
    verbose=False,
):
    """Train S2NetCore while generating gamma and SC from each image batch."""
    device = _resolve_device(device, model)
    model = model.to(device)
    criterion = criterion if criterion is not None else UnsupervisedS2NetLoss(
        spike_v_th=model.core.membrane_layer.vth,
    )
    optimizer = optimizer if optimizer is not None else torch.optim.Adam(
        model.core.parameters(), lr=lr
    )
    loss_history = []

    model.train()
    model.gamma_generator.input_layer.eval()
    for epoch in range(1, int(epochs) + 1):
        epoch_loss = 0.0
        sample_count = 0
        epoch_parts = {}
        for batch in dataloader:
            images = _unpack_image_batch(batch).to(device)
            output = model(images, return_details=True, classify=False)
            loss_values = _select_loss_signal(
                spikes=output.spikes,
                core_out=output.core_out,
                loss_signal=loss_signal,
            )
            loss, parts = criterion(
                spikes=loss_values,
                object_groups=None,
                sc=output.sc,
                core_out=output.core_out,
                images=images,
            )

            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.core.parameters(), float(grad_clip_norm)
                )
            optimizer.step()

            batch_size = images.size(0)
            epoch_loss += loss.item() * batch_size
            sample_count += batch_size
            for name, value in parts.items():
                epoch_parts[name] = epoch_parts.get(name, 0.0) + value.item() * batch_size

        mean_loss = epoch_loss / sample_count
        loss_history.append(mean_loss)
        if verbose:
            parts_text = " ".join(
                f"{name}={value / sample_count:.6f}"
                for name, value in epoch_parts.items()
            )
            print(
                f"Epoch {epoch:04d}/{int(epochs):04d} | "
                f"loss={mean_loss:.8f} | {parts_text}",
                flush=True,
            )

    if save_path is not None:
        save_s2net_core(model.core, save_path)
    return model, loss_history


@torch.no_grad()
def evaluate_s2net_core(model, dataloader, criterion=None, device=None):
    """Evaluate the image-driven model using freshly generated batch SC."""
    device = _resolve_device(device, model)
    model = model.to(device)
    criterion = criterion if criterion is not None else UnsupervisedS2NetLoss(
        spike_v_th=model.core.membrane_layer.vth,
    )
    model.eval()
    total_loss = 0.0
    total_count = 0
    last_parts = None
    for batch in dataloader:
        images = _unpack_image_batch(batch).to(device)
        output = model(images, return_details=True)
        loss, parts = criterion(
            spikes=torch.sigmoid(output.core_out),
            object_groups=output.object_groups,
            sc=output.sc,
            core_out=output.core_out,
            images=images,
        )
        total_loss += loss.item() * images.size(0)
        total_count += images.size(0)
        last_parts = {name: value.item() for name, value in parts.items()}
    return {"loss": total_loss / total_count, "parts": last_parts}


def _select_loss_signal(spikes, core_out, loss_signal):
    if loss_signal == "spikes":
        return spikes
    if loss_signal == "membrane":
        return core_out
    if loss_signal == "sigmoid_membrane":
        return torch.sigmoid(core_out)
    raise ValueError('loss_signal must be "spikes", "membrane", or "sigmoid_membrane".')


def _parse_pair_arg(value, name):
    if value is None:
        return None
    if len(value) == 1 and value[0] > 0:
        return int(value[0])
    if len(value) == 2 and value[0] > 0 and value[1] > 0:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"{name} must receive one positive int or two positive ints.")


def _unpack_image_batch(batch):
    return batch if torch.is_tensor(batch) else batch[0]


def _resolve_device(device, module):
    if device is not None:
        return torch.device(device)
    return next(module.parameters()).device


def save_s2net_core(core, save_path):
    """Save a trained S2NetCore state_dict."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(core.state_dict(), save_path)


def load_s2net_core(core, checkpoint_path, device=None):
    """Load a saved state_dict into an already constructed S2NetCore."""
    device = _resolve_device(device, core)
    core = core.to(device)
    core.load_state_dict(torch.load(checkpoint_path, map_location=device))
    core.eval()
    return core


def main():
    import argparse

    defaults = S2NetHyperparameters()
    parser = argparse.ArgumentParser(
        description="Train patch S2NetCore from images with online batch-specific SC."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-dir")
    source.add_argument("--dataset-path")
    parser.add_argument("--hdf5-key", default="image")
    parser.add_argument("--input-encoder-path", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--num-feature-maps", type=int, default=defaults.num_feature_maps)
    parser.add_argument("--patch-grid-size", type=int, nargs="+", default=[defaults.gamma_patch_grid_size])
    parser.add_argument("--patch-reduction", choices=["mean", "max"], default=defaults.gamma_patch_reduction)
    parser.add_argument("--kernel-size", type=int, default=defaults.kernel_size)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k", type=float, default=defaults.k)
    parser.add_argument("--dt", type=float, default=defaults.dt)
    parser.add_argument("--low-n", type=float, default=defaults.low_n)
    parser.add_argument("--high-n", type=float, default=defaults.high_n)
    parser.add_argument("--branch", type=int, default=defaults.branch)
    parser.add_argument("--sc-sigma-color", type=float, default=defaults.sc_sigma_color)
    parser.add_argument("--sc-m-min", type=float, default=defaults.sc_m_min)
    parser.add_argument("--sc-self-connectivity", type=float, default=defaults.sc_self_connectivity)
    parser.add_argument("--spike-classify-method", default=defaults.spike_classify_method, choices=["spike_rhythm", "spike_interval", "spatial_components"])
    parser.add_argument("--spike-spatial-threshold", type=float, default=defaults.spike_spatial_threshold)
    parser.add_argument("--spike-rate-weight", type=float, default=defaults.spike_rate_weight)
    parser.add_argument("--spike-smooth-weight", type=float, default=defaults.spike_smooth_weight)
    parser.add_argument("--spike-diversity-weight", type=float, default=defaults.spike_diversity_weight)
    parser.add_argument("--structural-weight", type=float, default=defaults.structural_weight)
    parser.add_argument("--object-overlap-weight", type=float, default=defaults.object_overlap_weight)
    parser.add_argument("--sample-diversity-weight", type=float, default=defaults.sample_diversity_weight)
    parser.add_argument("--spatial-compactness-weight", type=float, default=defaults.spatial_compactness_weight)
    parser.add_argument("--temporal-balance-weight", type=float, default=defaults.temporal_balance_weight)
    parser.add_argument("--edge-membrane-weight", type=float, default=defaults.edge_membrane_weight)
    parser.add_argument("--edge-membrane-margin", type=float, default=defaults.edge_membrane_margin)
    parser.add_argument("--loss-signal", default="sigmoid_membrane", choices=["spikes", "membrane", "sigmoid_membrane"])
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    grid_size = _parse_pair_arg(args.patch_grid_size, "patch-grid-size")
    grid_pair = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
    num_regions = grid_pair[0] * grid_pair[1]
    images = load_training_images(
        image_dir=args.image_dir,
        dataset_path=args.dataset_path,
        hdf5_key=args.hdf5_key,
        image_size=args.image_size,
        max_images=args.max_images,
    )

    hparams = S2NetHyperparameters(
        num_feature_maps=args.num_feature_maps,
        num_regions=num_regions,
        kernel_size=args.kernel_size,
        gamma_patch_grid_size=grid_size,
        gamma_patch_reduction=args.patch_reduction,
        sc_sigma_color=args.sc_sigma_color,
        sc_m_min=args.sc_m_min,
        sc_self_connectivity=args.sc_self_connectivity,
        k=args.k,
        dt=args.dt,
        low_n=args.low_n,
        high_n=args.high_n,
        branch=args.branch,
        spike_classify_method=args.spike_classify_method,
        spike_spatial_grid_size=grid_size,
        spike_spatial_threshold=args.spike_spatial_threshold,
    )
    model = S2NetClassifier(hparams, device=args.device)
    model.load_input_layer(args.input_encoder_path, map_location=args.device)
    for parameter in model.gamma_generator.input_layer.parameters():
        parameter.requires_grad_(False)

    loader = DataLoader(TensorDataset(images), batch_size=args.batch_size, shuffle=True)
    criterion = UnsupervisedS2NetLoss(
        spike_rate_weight=args.spike_rate_weight,
        spike_smooth_weight=args.spike_smooth_weight,
        spike_diversity_weight=args.spike_diversity_weight,
        structural_weight=args.structural_weight,
        object_overlap_weight=args.object_overlap_weight,
        sample_diversity_weight=args.sample_diversity_weight,
        spatial_compactness_weight=args.spatial_compactness_weight,
        temporal_balance_weight=args.temporal_balance_weight,
        edge_membrane_weight=args.edge_membrane_weight,
        edge_membrane_margin=args.edge_membrane_margin,
        spike_v_th=model.core.membrane_layer.vth,
        patch_grid_size=grid_size,
    )
    _, losses = train_s2net_core(
        model,
        loader,
        epochs=args.epochs,
        lr=args.lr,
        criterion=criterion,
        device=args.device,
        save_path=args.save_path,
        loss_signal=args.loss_signal,
        grad_clip_norm=args.grad_clip_norm,
        verbose=args.verbose,
    )
    print(f"trained S2NetCore: {args.save_path}")
    print(f"loss: {losses[0]:.6f} -> {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
