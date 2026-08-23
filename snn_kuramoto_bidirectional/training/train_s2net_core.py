"""Train the patch S2Net core with image-conditioned online SC."""

import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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
    gradient_diagnostic_epochs=None,
    gradient_diagnostic_images=None,
    gradient_diagnostic_output_dir=None,
    epoch_metrics_path=None,
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
    diagnostic_epochs = {
        int(epoch) for epoch in (gradient_diagnostic_epochs or [])
    }
    diagnostic_rows = []
    epoch_metrics = []
    if diagnostic_epochs:
        if gradient_diagnostic_images is None:
            raise ValueError(
                "gradient_diagnostic_images is required when diagnostic epochs are set."
            )
        if gradient_diagnostic_output_dir is None:
            raise ValueError(
                "gradient_diagnostic_output_dir is required when diagnostic epochs are set."
            )

    model.train()
    model.gamma_generator.input_layer.eval()
    for epoch in range(1, int(epochs) + 1):
        epoch_loss = 0.0
        sample_count = 0
        epoch_parts = {}
        epoch_spike_nonzero = 0
        epoch_spike_elements = 0
        epoch_active_oscillators = 0
        epoch_dense_sum = 0.0
        epoch_dense_abs_sum = 0.0
        epoch_dense_elements = 0
        for batch in dataloader:
            images = _unpack_image_batch(batch).to(device)
            output = model(images, return_details=True, classify=False)
            detached_spikes = output.spikes.detach()
            detached_dense = output.dense_i.detach()
            epoch_spike_nonzero += int(torch.count_nonzero(detached_spikes))
            epoch_spike_elements += detached_spikes.numel()
            epoch_active_oscillators += int(
                (detached_spikes != 0).any(dim=2).sum()
            )
            epoch_dense_sum += float(detached_dense.sum())
            epoch_dense_abs_sum += float(detached_dense.abs().sum())
            epoch_dense_elements += detached_dense.numel()
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
                dense_i=output.dense_i,
                dendritic_h=output.dendritic_h,
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
        actual_spike_rate = (
            epoch_spike_nonzero / epoch_spike_elements
            if epoch_spike_elements > 0
            else 0.0
        )
        total_spike_count_per_image = epoch_spike_nonzero / sample_count
        active_oscillator_count_per_image = (
            epoch_active_oscillators / sample_count
        )
        mean_dense = epoch_dense_sum / epoch_dense_elements
        absolute_mean_dense = epoch_dense_abs_sum / epoch_dense_elements
        loss_history.append(mean_loss)
        metric_row = {
            "epoch": int(epoch),
            "total_loss": float(mean_loss),
            "actual_spike_rate": float(actual_spike_rate),
            "spike_nonzero_count": int(epoch_spike_nonzero),
            "spike_element_count": int(epoch_spike_elements),
            "total_spike_count_per_image": float(total_spike_count_per_image),
            "active_oscillator_count_per_image": float(
                active_oscillator_count_per_image
            ),
            "mean_dense": float(mean_dense),
            "absolute_mean_dense": float(absolute_mean_dense),
        }
        weights = _loss_weights(criterion)
        for name, value in epoch_parts.items():
            if name == "total":
                continue
            raw_value = value / sample_count
            metric_row[f"raw_{name}"] = float(raw_value)
            metric_row[f"weighted_{name}"] = float(weights[name] * raw_value)
        epoch_metrics.append(metric_row)
        if epoch_metrics_path is not None:
            _save_epoch_metrics(epoch_metrics, epoch_metrics_path)
        if verbose:
            parts_text = " ".join(
                f"{name}={value / sample_count:.6f}"
                for name, value in epoch_parts.items()
            )
            print(
                f"Epoch {epoch:04d}/{int(epochs):04d} | "
                f"loss={mean_loss:.8f} | "
                f"actual_spike_rate={actual_spike_rate:.8f} | "
                f"spikes_per_image={total_spike_count_per_image:.4f} | "
                f"active_oscillators_per_image="
                f"{active_oscillator_count_per_image:.4f} | "
                f"mean_dense={mean_dense:.8f} | "
                f"absolute_mean_dense={absolute_mean_dense:.8f} | "
                f"{parts_text}",
                flush=True,
            )

        if epoch in diagnostic_epochs:
            rows = _measure_loss_gradients(
                model=model,
                images=gradient_diagnostic_images.to(device),
                criterion=criterion,
                loss_signal=loss_signal,
                epoch=epoch,
            )
            diagnostic_rows.extend(rows)
            _save_gradient_diagnostics(
                diagnostic_rows,
                gradient_diagnostic_output_dir,
            )
            print(
                f"Gradient diagnostics saved for epoch {epoch}: "
                f"{gradient_diagnostic_output_dir}",
                flush=True,
            )

    if save_path is not None:
        save_s2net_core(model.core, save_path)
    return model, loss_history


def _measure_loss_gradients(model, images, criterion, loss_signal, epoch):
    """Measure per-loss gradients on one fixed batch without updating weights."""
    output = model(images, return_details=True, classify=False)
    loss_values = _select_loss_signal(
        spikes=output.spikes,
        core_out=output.core_out,
        loss_signal=loss_signal,
    )
    total_loss, parts = criterion(
        spikes=loss_values,
        object_groups=None,
        sc=output.sc,
        core_out=output.core_out,
        images=images,
        dense_i=output.dense_i,
        dendritic_h=output.dendritic_h,
    )
    weights = _loss_weights(criterion)
    parameter = model.core.dendric_layer.oscillator_dense.weight

    def gradient(value):
        if not value.requires_grad:
            return torch.zeros_like(parameter)
        result = torch.autograd.grad(
            value,
            parameter,
            retain_graph=True,
            allow_unused=True,
        )[0]
        return torch.zeros_like(parameter) if result is None else result.detach()

    raw_gradients = {}
    weighted_gradients = {}
    for name, value in parts.items():
        if name == "total":
            continue
        raw_gradients[name] = gradient(value)
        weighted_gradients[name] = gradient(weights[name] * value)

    dense_gradient = weighted_gradients.get(
        "dense_magnitude_loss",
        torch.zeros_like(parameter),
    )
    dense_norm = dense_gradient.norm()
    actual_spike_rate = float((output.spikes.detach() != 0).float().mean())
    dense_abs_mean = float(output.dense_i.detach().abs().mean())

    def cosine_with_dense(value):
        value_norm = value.norm()
        if float(dense_norm) == 0.0 or float(value_norm) == 0.0:
            return None
        return float(
            F.cosine_similarity(
                value.reshape(1, -1),
                dense_gradient.reshape(1, -1),
                dim=1,
                eps=1e-12,
            )[0]
        )

    rows = []
    for name, value in parts.items():
        if name == "total":
            continue
        raw_gradient = raw_gradients[name]
        weighted_gradient = weighted_gradients[name]
        rows.append({
            "epoch": int(epoch),
            "loss": name,
            "raw_loss": float(value.detach()),
            "weight": float(weights[name]),
            "weighted_loss": float((weights[name] * value).detach()),
            "raw_gradient_norm": float(raw_gradient.norm()),
            "weighted_gradient_norm": float(weighted_gradient.norm()),
            "cosine_with_dense_gradient": cosine_with_dense(weighted_gradient),
            "actual_spike_rate": actual_spike_rate,
            "dense_abs_mean": dense_abs_mean,
        })

    total_gradient = gradient(total_loss)
    rows.append({
        "epoch": int(epoch),
        "loss": "total",
        "raw_loss": float(total_loss.detach()),
        "weight": 1.0,
        "weighted_loss": float(total_loss.detach()),
        "raw_gradient_norm": float(total_gradient.norm()),
        "weighted_gradient_norm": float(total_gradient.norm()),
        "cosine_with_dense_gradient": cosine_with_dense(total_gradient),
        "actual_spike_rate": actual_spike_rate,
        "dense_abs_mean": dense_abs_mean,
    })
    return rows


def _loss_weights(criterion):
    return {
        "spike_rate": criterion.spike_rate_weight,
        "spike_smooth": criterion.spike_smooth_weight,
        "spike_diversity": criterion.spike_diversity_weight,
        "structural": criterion.structural_weight,
        "object_overlap": criterion.object_overlap_weight,
        "sample_diversity": criterion.sample_diversity_weight,
        "spatial_compactness": criterion.spatial_compactness_weight,
        "temporal_balance": criterion.temporal_balance_weight,
        "edge_membrane": criterion.edge_membrane_weight,
        "dense_magnitude_loss": criterion.dense_magnitude_weight,
        "dendritic_cancellation_loss": criterion.dendritic_cancellation_weight,
    }


def _save_gradient_diagnostics(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "epoch",
        "loss",
        "raw_loss",
        "weight",
        "weighted_loss",
        "raw_gradient_norm",
        "weighted_gradient_norm",
        "cosine_with_dense_gradient",
        "actual_spike_rate",
        "dense_abs_mean",
    ]
    with (output_dir / "loss_gradient_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "loss_gradient_diagnostics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(rows, file, indent=2)


def _save_epoch_metrics(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for name in row:
            if name not in columns:
                columns.append(name)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


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
            dense_i=output.dense_i,
            dendritic_h=output.dendritic_h,
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
    parser.add_argument("--dense-magnitude-weight", type=float, default=defaults.dense_magnitude_weight)
    parser.add_argument("--dendritic-cancellation-weight", type=float, default=defaults.dendritic_cancellation_weight)
    parser.add_argument("--loss-signal", default="sigmoid_membrane", choices=["spikes", "membrane", "sigmoid_membrane"])
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--gradient-diagnostic-epochs", type=int, nargs="+", default=None)
    parser.add_argument("--gradient-diagnostic-output-dir", default=None)
    parser.add_argument("--epoch-metrics-path", default=None)
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
    diagnostic_images = images[:min(args.batch_size, len(images))]
    diagnostic_output_dir = args.gradient_diagnostic_output_dir
    if args.gradient_diagnostic_epochs and diagnostic_output_dir is None:
        diagnostic_output_dir = str(Path(args.save_path).parent / "gradient_diagnostics")
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
        dense_magnitude_weight=args.dense_magnitude_weight,
        dendritic_cancellation_weight=args.dendritic_cancellation_weight,
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
        gradient_diagnostic_epochs=args.gradient_diagnostic_epochs,
        gradient_diagnostic_images=diagnostic_images,
        gradient_diagnostic_output_dir=diagnostic_output_dir,
        epoch_metrics_path=args.epoch_metrics_path,
    )
    print(f"trained S2NetCore: {args.save_path}")
    print(f"loss: {losses[0]:.6f} -> {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
