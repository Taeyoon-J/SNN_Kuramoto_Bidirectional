import torch
import sys
from pathlib import Path
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
    from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
    from snn_kuramoto_bidirectional.sc_generator import pearson_cor_sc
except ModuleNotFoundError:
    from hyperparameter import S2NetHyperparameters
    from loss_function import UnsupervisedS2NetLoss
    from s2net_cls import S2NetCore
    from sc_generator import pearson_cor_sc


def train_s2net_core(
    core,
    dataloader,
    epochs=100,
    lr=1e-3,
    criterion=None,
    optimizer=None,
    device=None,
    save_path=None,
    verbose=False,
):
    """
    Train only S2NetCore from precomputed gamma sequences with an unsupervised
    spike/object-group loss.

    Each dataloader batch must contain gamma_seq. The fixed SC is read from
    core.sc.

    Expected shapes:
        gamma_seq: [B, T, num_regions]

    Returns:
        core, loss_history
    """
    device = _resolve_device(device, core)
    core = core.to(device)
    criterion = criterion if criterion is not None else UnsupervisedS2NetLoss()
    optimizer = optimizer if optimizer is not None else torch.optim.Adam(core.parameters(), lr=lr)
    loss_history = []

    core.train()
    for epoch in range(1, int(epochs) + 1):
        epoch_loss = 0.0
        sample_count = 0
        for batch in dataloader:
            gamma_seq = _unpack_gamma_batch(batch)
            gamma_seq = gamma_seq.to(device)

            object_groups, spikes = core(gamma_seq)
            loss, _ = criterion(
                spikes=spikes,
                object_groups=object_groups,
                sc=core.sc,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = gamma_seq.size(0)
            epoch_loss += loss.item() * batch_size
            sample_count += batch_size
        mean_loss = epoch_loss / sample_count
        loss_history.append(mean_loss)
        if verbose:
            print(
                f"Epoch {epoch:04d}/{int(epochs):04d} | loss={mean_loss:.8f}",
                flush=True,
            )

    if save_path is not None:
        save_s2net_core(core, save_path)

    return core, loss_history


@torch.no_grad()
def evaluate_s2net_core(core, dataloader, criterion=None, device=None):
    """Evaluate S2NetCore using its fixed SC and precomputed gamma sequences."""
    device = _resolve_device(device, core)
    core = core.to(device)
    criterion = criterion if criterion is not None else UnsupervisedS2NetLoss()

    core.eval()
    total_loss = 0.0
    total_count = 0
    last_parts = None
    for batch in dataloader:
        gamma_seq = _unpack_gamma_batch(batch)
        gamma_seq = gamma_seq.to(device)

        object_groups, spikes = core(gamma_seq)
        loss, parts = criterion(
            spikes=spikes,
            object_groups=object_groups,
            sc=core.sc,
        )

        batch_size = gamma_seq.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        last_parts = {name: value.item() for name, value in parts.items()}

    return {
        "loss": total_loss / total_count,
        "parts": last_parts,
    }


def _unpack_gamma_batch(batch):
    if torch.is_tensor(batch):
        return batch
    if len(batch) < 1:
        raise ValueError("Each batch must contain gamma_seq.")
    return batch[0]


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
    state_dict = torch.load(checkpoint_path, map_location=device)
    core.load_state_dict(state_dict)
    core.eval()
    return core


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train S2NetCore from precomputed gamma sequences.")
    parser.add_argument("--gamma-seq-path", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--sc-path", default=None)
    parser.add_argument("--sc-save-path", default=None)
    parser.add_argument("--num-feature-maps", type=int, default=None)
    parser.add_argument("--num-regions", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--low-n", type=float, default=0.0)
    parser.add_argument("--high-n", type=float, default=4.0)
    parser.add_argument("--branch", type=int, default=4)
    parser.add_argument("--spike-classify-method", default="spike_interval", choices=["spike_rhythm", "spike_interval"])
    parser.add_argument("--spike-rhythm-threshold", type=float, default=0.8)
    parser.add_argument("--spike-rhythm-min-group-size", type=int, default=2)
    parser.add_argument("--spike-rhythm-return-all-groups", action="store_true")
    parser.add_argument("--spike-interval-size", type=int, default=1)
    parser.add_argument("--spike-interval-threshold", type=float, default=0.5)
    parser.add_argument("--spike-interval-min-group-size", type=int, default=1)
    parser.add_argument("--no-spike-interval-include-partial", action="store_true")
    parser.add_argument("--spike-rate-weight", type=float, default=1.0)
    parser.add_argument("--spike-smooth-weight", type=float, default=0.1)
    parser.add_argument("--spike-diversity-weight", type=float, default=0.1)
    parser.add_argument("--structural-weight", type=float, default=0.1)
    parser.add_argument("--object-overlap-weight", type=float, default=1.0)
    parser.add_argument("--spike-target-rate", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    gamma_seq = torch.load(args.gamma_seq_path, map_location="cpu").float()
    if gamma_seq.dim() != 3:
        raise ValueError(f"gamma_seq must have shape [B, T, N], but got {tuple(gamma_seq.shape)}.")

    num_feature_maps = gamma_seq.size(1) if args.num_feature_maps is None else int(args.num_feature_maps)
    num_regions = gamma_seq.size(2) if args.num_regions is None else int(args.num_regions)
    if gamma_seq.size(1) != num_feature_maps:
        raise ValueError(f"gamma_seq T={gamma_seq.size(1)} does not match num_feature_maps={num_feature_maps}.")
    if gamma_seq.size(2) != num_regions:
        raise ValueError(f"gamma_seq N={gamma_seq.size(2)} does not match num_regions={num_regions}.")

    if args.sc_path is None:
        sc = pearson_cor_sc(gamma_seq.reshape(-1, num_regions))
        if args.sc_save_path is not None:
            sc_path = Path(args.sc_save_path)
            sc_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sc, sc_path)
            print(f"saved generated sc {tuple(sc.shape)} to {sc_path}", flush=True)
    else:
        sc = torch.load(args.sc_path, map_location="cpu").float()

    if tuple(sc.shape) != (num_regions, num_regions):
        raise ValueError(f"sc must have shape {(num_regions, num_regions)}, but got {tuple(sc.shape)}.")

    hparams = S2NetHyperparameters(
        num_feature_maps=num_feature_maps,
        num_regions=num_regions,
        kernel_size=args.kernel_size,
        sc=sc,
        k=args.k,
        dt=args.dt,
        low_n=args.low_n,
        high_n=args.high_n,
        branch=args.branch,
        spike_classify_method=args.spike_classify_method,
        spike_rhythm_threshold=args.spike_rhythm_threshold,
        spike_rhythm_min_group_size=args.spike_rhythm_min_group_size,
        spike_rhythm_return_all_groups=args.spike_rhythm_return_all_groups,
        spike_interval_size=args.spike_interval_size,
        spike_interval_threshold=args.spike_interval_threshold,
        spike_interval_min_group_size=args.spike_interval_min_group_size,
        spike_interval_include_partial=not args.no_spike_interval_include_partial,
    )
    hparams.validate()

    core = S2NetCore(hparams, device=args.device)
    loader = DataLoader(
        TensorDataset(gamma_seq),
        batch_size=int(args.batch_size),
        shuffle=True,
    )
    criterion = UnsupervisedS2NetLoss(
        spike_rate_weight=args.spike_rate_weight,
        spike_smooth_weight=args.spike_smooth_weight,
        spike_diversity_weight=args.spike_diversity_weight,
        structural_weight=args.structural_weight,
        object_overlap_weight=args.object_overlap_weight,
        spike_target_rate=args.spike_target_rate,
    )

    _, losses = train_s2net_core(
        core,
        loader,
        epochs=args.epochs,
        lr=args.lr,
        criterion=criterion,
        device=args.device,
        save_path=args.save_path,
        verbose=args.verbose,
    )
    print(f"trained S2NetCore: {args.save_path}")
    print(f"loss: {losses[0]:.6f} -> {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
