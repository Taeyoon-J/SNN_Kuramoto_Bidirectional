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
    from snn_kuramoto_bidirectional.loss_function import (
        UnsupervisedS2NetLoss,
        phase_locking_value,
        phase_alignment,
        signal_synchrony,
        graph_teacher_synchrony_loss,
        edge_membrane_separation_loss,
    )
    from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
    from snn_kuramoto_bidirectional.sc_generator import pearson_cor_sc
except ModuleNotFoundError:
    from hyperparameter import S2NetHyperparameters
    from loss_function import (
        UnsupervisedS2NetLoss,
        phase_locking_value,
        phase_alignment,
        signal_synchrony,
        graph_teacher_synchrony_loss,
        edge_membrane_separation_loss,
    )
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
    loss_signal="sigmoid_membrane",
    grad_clip_norm=1.0,
    verbose=False,
    plv_settle=0,
    plv_source="phase",
    plv_combine="mean",
    spike_plv_weight=0.0,
    graph_teacher_weight=0.0,
    graph_teacher_temperature=0.1,
    graph_teacher_signal="membrane",
    edge_membrane_weight=0.0,
    edge_membrane_margin=0.3,
    edge_membrane_grid_size=(16, 16),
    checkpoint_dir=None,
    checkpoint_epochs=(),
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
    parts_history = []

    core.train()
    for epoch in range(1, int(epochs) + 1):
        epoch_loss = 0.0
        sample_count = 0
        epoch_parts = {}
        for batch in dataloader:
            gamma_seq = _unpack_gamma_batch(batch)
            gamma_seq = gamma_seq.to(device)

            object_groups, spikes, core_out, plv, theta = _forward_with_plv(
                core,
                gamma_seq,
                criterion,
                plv_settle,
                plv_source,
                plv_combine,
            )
            loss_values = _select_loss_signal(
                spikes=spikes,
                core_out=core_out,
                loss_signal=loss_signal,
            )
            loss, parts = criterion(
                spikes=loss_values,
                object_groups=object_groups,
                sc=core.sc,
                plv=plv,
                theta=theta,
                plv_settle=int(plv_settle),
            )

            if float(spike_plv_weight) != 0.0:
                spike_plv = _component_spike_synchrony(core, plv_settle)
                if spike_plv is None:
                    raise ValueError("spike_plv_weight requires --spike-per-component")
                spike_loss, spike_parts = criterion(
                    plv=spike_plv,
                    plv_settle=int(plv_settle),
                )
                loss = loss + float(spike_plv_weight) * spike_loss
                parts.update({f"spike_{name}": value for name, value in spike_parts.items()})

            if float(graph_teacher_weight) != 0.0:
                with torch.no_grad():
                    graph = (
                        core.graph_generator(gamma_seq)
                        if core.graph_generator is not None
                        else core.sc.to(gamma_seq.device).unsqueeze(0).expand(gamma_seq.size(0), -1, -1)
                    )
                if graph_teacher_signal not in ("membrane", "spikes"):
                    raise ValueError("graph_teacher_signal must be membrane or spikes")
                teacher_signal = core_out if graph_teacher_signal == "membrane" else spikes
                teacher_loss = graph_teacher_synchrony_loss(
                    teacher_signal,
                    graph,
                    settle=int(plv_settle),
                    temperature=float(graph_teacher_temperature),
                )
                loss = loss + float(graph_teacher_weight) * teacher_loss
                parts["graph_teacher_synchrony"] = teacher_loss.detach()
                parts["total"] = loss.detach()

            if float(edge_membrane_weight) != 0.0:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise ValueError("Edge membrane loss requires paired gamma and RGB batches")
                images = batch[1].to(device=device, dtype=core_out.dtype) / 255.0
                edge_loss = edge_membrane_separation_loss(
                    core_out, images, edge_membrane_grid_size,
                    margin=float(edge_membrane_margin),
                )
                loss = loss + float(edge_membrane_weight) * edge_loss
                parts["edge_membrane_separation"] = edge_loss.detach()
                parts["total"] = loss.detach()

            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(core.parameters(), float(grad_clip_norm))
            optimizer.step()

            batch_size = gamma_seq.size(0)
            epoch_loss += loss.item() * batch_size
            sample_count += batch_size
            for name, value in parts.items():
                epoch_parts[name] = epoch_parts.get(name, 0.0) + value.item() * batch_size
        mean_loss = epoch_loss / sample_count
        mean_parts = {
            name: value / sample_count
            for name, value in epoch_parts.items()
        }
        loss_history.append(mean_loss)
        parts_history.append(mean_parts)
        if checkpoint_dir is not None and epoch in set(map(int, checkpoint_epochs)):
            save_s2net_core(core, Path(checkpoint_dir) / f"epoch_{epoch:02d}.pt")
        if verbose:
            parts_text = " ".join(
                f"{name}={value:.6f}"
                for name, value in mean_parts.items()
            )
            print(
                f"Epoch {epoch:04d}/{int(epochs):04d} | loss={mean_loss:.8f} | {parts_text}",
                flush=True,
            )

    if save_path is not None:
        save_s2net_core(core, save_path)

    return core, loss_history


@torch.no_grad()
def evaluate_s2net_core(core, dataloader, criterion=None, device=None, plv_settle=0):
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

        object_groups, spikes, core_out, plv, theta = _forward_with_plv(
            core, gamma_seq, criterion, plv_settle
        )
        loss_values = _select_loss_signal(
            spikes=spikes,
            core_out=core_out,
            loss_signal="sigmoid_membrane",
        )
        loss, parts = criterion(
            spikes=loss_values,
            object_groups=object_groups,
            sc=core.sc,
            plv=plv,
            theta=theta,
            plv_settle=int(plv_settle),
        )

        batch_size = gamma_seq.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        last_parts = {name: value.item() for name, value in parts.items()}

    return {
        "loss": total_loss / total_count,
        "parts": last_parts,
    }


def _uses_theta(criterion):
    return any(
        float(getattr(criterion, name, 0.0)) != 0.0
        for name in ("phase_quantization_weight", "phase_spread_weight")
    )


def _uses_plv(criterion):
    return any(
        float(getattr(criterion, name, 0.0)) != 0.0
        for name in (
            "plv_bimodality_weight",
            "plv_balance_weight",
            "plv_coherence_weight",
            "plv_group_count_weight",
            "plv_collapse_weight",
        )
    )


def _forward_with_plv(
    core,
    gamma_seq,
    criterion,
    plv_settle,
    plv_source="phase",
    plv_combine="mean",
):
    """
    Run the core, returning the synchrony matrix only when the loss needs it.

    plv_source "phase" reads the Kuramoto phases, which bypasses the dendritic
    and membrane layers. "membrane" and "spikes" read the spiking side instead,
    putting the SNN inside the trained path.
    """
    if not _uses_plv(criterion) and not _uses_theta(criterion):
        groups, spikes, core_out = core(gamma_seq, return_core_out=True)
        return groups, spikes, core_out, None, None
    groups, spikes, core_out, theta = core(
        gamma_seq, return_core_out=True, return_theta=True
    )
    if not _uses_plv(criterion):
        return groups, spikes, core_out, None, theta
    if plv_source == "phase":  # readout mode: README 4.3 training
        plv = phase_locking_value(
            theta,
            settle=int(plv_settle),
            combine=plv_combine,
        )
    elif plv_source == "alignment":
        plv = phase_alignment(theta, settle=int(plv_settle))
    elif plv_source == "membrane":
        plv = signal_synchrony(core_out, settle=int(plv_settle))
    elif plv_source == "spikes":
        plv = signal_synchrony(spikes, settle=int(plv_settle))
    else:
        raise ValueError(
            'plv_source must be "phase", "alignment", "membrane", or "spikes".'
        )
    return groups, spikes, core_out, plv, theta


def _select_loss_signal(spikes, core_out, loss_signal):
    if loss_signal == "spikes":
        return spikes
    if loss_signal == "membrane":
        return core_out
    if loss_signal == "sigmoid_membrane":  # readout mode: legacy activity-loss input, not PLV source
        return torch.sigmoid(core_out)
    raise ValueError('loss_signal must be "spikes", "membrane", or "sigmoid_membrane".')


def _component_spike_synchrony(core, settle):
    """Product of per-component spike synchrony matrices, shaped [B,N,N]."""
    components = getattr(core, "last_component_spikes", None)
    if components is None:
        return None
    synchrony = [
        signal_synchrony(components[:, component], settle=int(settle))
        for component in range(components.size(1))
    ]
    return torch.stack(synchrony).prod(dim=0)


def _parse_pair_arg(value, name):
    if value is None:
        return None
    if len(value) == 1:
        if value[0] <= 0:
            raise ValueError(f"{name} must be positive.")
        return int(value[0])
    if len(value) == 2:
        first, second = int(value[0]), int(value[1])
        if first <= 0 or second <= 0:
            raise ValueError(f"{name} values must be positive.")
        return (first, second)
    raise ValueError(f"{name} must receive one int or two ints.")


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
    parser.add_argument("--num-feature-maps", type=int, default=8)
    parser.add_argument("--num-regions", type=int, default=256)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k", type=float, default=256.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--gamma-drive-mode",
        default="static",
        choices=["sequence", "static"],
        help=(
            "sequence: one Kuramoto step per visual channel (legacy, T = channels). "
            "static: constant sensory drive with --num-time-steps recurrent steps, "
            "decoupled from the channel count."
        ),
    )
    parser.add_argument(
        "--num-time-steps",
        type=int,
        default=64,
        help="Recurrent length for static drive mode.",
    )
    parser.add_argument("--osc-dim", type=int, default=4)
    parser.add_argument(
        "--spike-per-component",
        action="store_true",
        help=(
            "Run the shared dendritic and membrane layers separately for each "
            "oscillator component, preserving component-wise spike histories."
        ),
    )
    parser.add_argument(
        "--gamma-phase-mode",
        default="standardize_tanh",
        choices=["none", "tanh", "standardize_tanh"],
        help="Map raw gamma onto a phase range before sin(gamma - theta).",
    )
    parser.add_argument(
        "--theta-init",
        default="gamma",
        choices=["zeros", "gamma", "gamma_noise"],
        help="zeros makes every image start from an identical oscillator state.",
    )
    parser.add_argument("--theta-init-noise", type=float, default=0.0)
    parser.add_argument(
        "--freq-gain",
        type=float,
        default=2.0,
        help=(
            "Let the image set oscillator frequencies. At 0 the input only pins "
            "phases and the network settles into global synchrony, which erases "
            "every group distinction. Measured optimum for binding was ~2.0."
        ),
    )
    parser.add_argument(
        "--spike-pulse-gain",
        type=float,
        default=0.0,
        help=(
            "Let spikes act back on the Kuramoto phases through the coupling "
            "graph. At 0 the flow is one-way and the dendritic and membrane "
            "layers get no gradient, so they never learn anything."
        ),
    )
    parser.add_argument(
        "--graph-mode",
        default="learned",
        choices=["static", "learned"],
        help="static uses the fixed sc; learned builds a sparse graph per image.",
    )
    parser.add_argument("--graph-top-k", type=int, default=32)
    parser.add_argument("--graph-hidden-dim", type=int, default=16)
    parser.add_argument("--graph-coupling-gain", type=float, default=8.0)
    parser.add_argument("--graph-temperature", type=float, default=0.1)
    parser.add_argument("--kuramoto-backend", choices=["pairwise", "factorized"],
                        default="pairwise",
                        help="Pairwise legacy interaction or cached factorized coupling kernels.")
    parser.add_argument("--geodesic-steps", type=int, default=0)
    parser.add_argument("--geodesic-radius", type=float, default=1.5)
    parser.add_argument("--geodesic-contrast", type=float, default=2.0)
    parser.add_argument("--geodesic-temperature", type=float, default=0.5)
    parser.add_argument("--geodesic-cap", type=float, default=16.0)
    parser.add_argument(
        "--graph-spatial-decay",
        type=float,
        default=0.55,
        help=(
            "Distance prior on the learned graph, weight ~ decay ** grid_distance. "
            "Objects are connected regions and the features do not know that."
        ),
    )
    parser.add_argument(
        "--graph-feedback-strength",
        type=float,
        default=0.0,
        help=(
            "Let the graph track the synchrony it produces, so oscillators that "
            "stay in phase couple more strongly. A graph held fixed for the whole "
            "rollout cannot sharpen a group once it starts to form."
        ),
    )
    parser.add_argument("--graph-feedback-momentum", type=float, default=0.9)
    parser.add_argument("--plv-bimodality-weight", type=float, default=1.0)
    parser.add_argument("--plv-balance-weight", type=float, default=10.0)
    parser.add_argument("--plv-coherence-weight", type=float, default=0.5)
    parser.add_argument(
        "--plv-group-count-weight",
        type=float,
        default=0.0,
        help=(
            "Target the effective number of synchronised groups instead of a mean "
            "synchrony level. Prefer this: with CLEVR ground truth the ideal mean "
            "PLV is 0.867 when background counts as a group and 0.003 when it does "
            "not, so any single density target sits between two valid solutions."
        ),
    )
    parser.add_argument("--plv-target-groups", type=float, default=7.0)
    parser.add_argument(
        "--phase-quantization-weight",
        type=float,
        default=0.0,
        help=(
            "Snap oscillator phases onto evenly spaced slots, stating the "
            "one-object-per-phase-window hypothesis directly. plv_bimodality "
            "cannot express it: more than two groups on a circle cannot all be "
            "mutually antiphase."
        ),
    )
    parser.add_argument(
        "--phase-spread-weight",
        type=float,
        default=0.0,
        help="Penalise every unit collapsing onto one phase slot.",
    )
    parser.add_argument("--phase-num-slots", type=float, default=7.0)
    parser.add_argument(
        "--plv-collapse-weight",
        type=float,
        default=1.0,
        help=(
            "Barrier against a uniform synchrony matrix. plv_bimodality is zero at "
            "global synchrony as well as at a real partition, and that is the minimum "
            "the optimiser reaches unless this is on."
        ),
    )
    parser.add_argument("--plv-target-density", type=float, default=0.867)
    parser.add_argument(
        "--plv-settle",
        type=int,
        default=32,
        help="Recurrent steps to discard as transient before measuring synchrony.",
    )
    parser.add_argument(
        "--plv-source",
        default="phase",
        choices=["phase", "alignment", "membrane", "spikes"],
        help=(
            "Which structure the loss shapes. 'phase' uses PLV, which scores two "
            "units locked a quarter cycle apart as fully synchronous. 'alignment' "
            "requires the phase difference to be near zero, which is what a code "
            "where an object is the set of units active at the same moment needs. "
            "'membrane' and 'spikes' read the SNN side instead."
        ),
    )
    parser.add_argument(
        "--plv-combine",
        choices=["mean", "product", "min", "component_mean"],
        default="mean",
        help="How oscillator components are combined when computing phase PLV.",
    )
    parser.add_argument(
        "--spike-plv-weight",
        type=float,
        default=0.0,
        help=(
            "Add the same PLV-family objective on product-combined per-component "
            "spike synchrony while retaining the selected primary PLV source. "
            "Requires --spike-per-component."
        ),
    )
    parser.add_argument(
        "--membrane-vth",
        type=float,
        default=0.06,
        help=(
            "Spike threshold. The default was never matched to the signal: the "
            "membrane oscillates with a standard deviation near 0.1, so 0.5 sits "
            "about five times the amplitude and the network cannot fire."
        ),
    )
    parser.add_argument(
        "--membrane-low-m",
        type=float,
        default=-4.0,
        help=(
            "Lower bound of the membrane tau init. The default U(0, 4) gives a "
            "leak near 0.85, which low-passes away the oscillation carrying the "
            "information; U(-4, 0) lets the membrane track the rhythm instead."
        ),
    )
    parser.add_argument("--membrane-high-m", type=float, default=0.0)
    parser.add_argument(
        "--gate-mode",
        default="raw",
        choices=["sigmoid", "raw", "phase_mean"],
        help=(
            "'sigmoid' keeps the original gate, compressed to [0.5, 0.731]. "
            "'phase_mean' reduces the osc_dim axis before the sine; the same "
            "information reduced the other way round measured 0.216 vs 0.067."
        ),
    )
    parser.add_argument("--low-n", type=float, default=-4.0)
    parser.add_argument("--high-n", type=float, default=0.0)
    parser.add_argument("--branch", type=int, default=4)
    parser.add_argument("--dendritic-projection", default="shared",
                        choices=["shared", "per_region"],
                        help="Share one dendritic projection or learn one per region.")
    parser.add_argument("--spike-classify-method", default="spike_interval", choices=["spike_rhythm", "spike_interval", "spatial_components"])
    parser.add_argument("--spike-rhythm-threshold", type=float, default=0.8)
    parser.add_argument("--spike-rhythm-min-group-size", type=int, default=2)
    parser.add_argument("--spike-rhythm-return-all-groups", action="store_true")
    parser.add_argument("--spike-interval-size", type=int, default=1)
    parser.add_argument("--spike-interval-threshold", type=float, default=0.5)
    parser.add_argument("--spike-interval-min-group-size", type=int, default=1)
    parser.add_argument("--no-spike-interval-include-partial", action="store_true")
    parser.add_argument("--spike-spatial-grid-size", type=int, nargs="+", default=[16])
    parser.add_argument("--spike-spatial-threshold", type=float, default=0.5)
    parser.add_argument("--spike-spatial-min-group-size", type=int, default=2)
    parser.add_argument("--spike-spatial-activity-source", default="sigmoid_membrane", choices=["spikes", "membrane", "sigmoid_membrane"])
    parser.add_argument("--spike-spatial-time-aggregate", default="mean", choices=["max", "mean"])
    parser.add_argument("--spike-rate-weight", type=float, default=0.0)
    parser.add_argument("--spike-smooth-weight", type=float, default=0.0)
    parser.add_argument("--spike-diversity-weight", type=float, default=0.0)
    parser.add_argument("--structural-weight", type=float, default=0.0)
    parser.add_argument("--graph-teacher-weight", type=float, default=0.0)
    parser.add_argument("--graph-teacher-temperature", type=float, default=0.1)
    parser.add_argument("--graph-teacher-signal", choices=["membrane", "spikes"],
                        default="membrane")
    parser.add_argument("--edge-image-hdf5", default=None,
                        help="Aligned HDF5 RGB images for optional edge membrane loss.")
    parser.add_argument("--edge-membrane-weight", type=float, default=0.0)
    parser.add_argument("--edge-membrane-margin", type=float, default=0.3)
    parser.add_argument("--object-overlap-weight", type=float, default=0.0)
    parser.add_argument(
        "--sample-diversity-weight", "--sample-activity-diversity-weight",
        dest="sample_diversity_weight", type=float, default=0.0,
        help=("Weight sample_activity_diversity_loss on the selected --loss-signal "
              "(SW0050 uses sigmoid_membrane)."),
    )
    parser.add_argument("--spatial-compactness-weight", type=float, default=0.0)
    parser.add_argument("--temporal-balance-weight", type=float, default=0.0)
    parser.add_argument("--activity-confidence-weight", type=float, default=0.0)
    parser.add_argument("--activity-area-weight", type=float, default=0.0)
    parser.add_argument("--activity-contrast-weight", type=float, default=0.0)
    parser.add_argument("--activity-min-area", type=float, default=0.05)
    parser.add_argument("--activity-max-area", type=float, default=0.35)
    parser.add_argument("--activity-target-std", type=float, default=0.15)
    parser.add_argument("--loss-patch-grid-size", type=int, nargs="+", default=None)
    parser.add_argument("--spike-target-rate", type=float, default=0.1)
    parser.add_argument("--loss-signal", default="sigmoid_membrane", choices=["spikes", "membrane", "sigmoid_membrane"])
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Seed for initialization and data order. Runs that differ only in a "
            "setting the loss cannot see still spanned 0.244 to 0.382 in ARI, so "
            "without a fixed seed any difference below about 0.07 is unreadable."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Optional directory for selected intermediate epoch state_dicts.")
    parser.add_argument("--checkpoint-epochs", type=int, nargs="*", default=[],
                        help="Epochs to checkpoint; requires --checkpoint-dir.")
    args = parser.parse_args()
    if args.checkpoint_epochs and args.checkpoint_dir is None:
        parser.error("--checkpoint-epochs requires --checkpoint-dir")
    if any(epoch <= 0 or epoch > args.epochs for epoch in args.checkpoint_epochs):
        parser.error("checkpoint epochs must lie in [1, --epochs]")

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

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
        osc_dim=args.osc_dim,
        gamma_drive_mode=args.gamma_drive_mode,
        num_time_steps=(
            num_feature_maps if args.num_time_steps is None else int(args.num_time_steps)
        ),
        gamma_phase_mode=args.gamma_phase_mode,
        theta_init=args.theta_init,
        theta_init_noise=args.theta_init_noise,
        freq_gain=args.freq_gain,
        spike_pulse_gain=args.spike_pulse_gain,
        graph_mode=args.graph_mode,
        graph_top_k=args.graph_top_k,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_coupling_gain=args.graph_coupling_gain,
        graph_temperature=args.graph_temperature,
        kuramoto_backend=args.kuramoto_backend,
        graph_spatial_decay=args.graph_spatial_decay,
        geodesic_steps=args.geodesic_steps,
        geodesic_radius=args.geodesic_radius,
        geodesic_contrast=args.geodesic_contrast,
        geodesic_temperature=args.geodesic_temperature,
        geodesic_cap=args.geodesic_cap,
        graph_feedback_strength=args.graph_feedback_strength,
        graph_feedback_momentum=args.graph_feedback_momentum,
        membrane_vth=args.membrane_vth,
        membrane_low_m=args.membrane_low_m,
        membrane_high_m=args.membrane_high_m,
        gate_mode=args.gate_mode,
        spike_per_component=args.spike_per_component,
        low_n=args.low_n,
        high_n=args.high_n,
        dendritic_projection=args.dendritic_projection,
        branch=args.branch,
        spike_classify_method=args.spike_classify_method,
        spike_rhythm_threshold=args.spike_rhythm_threshold,
        spike_rhythm_min_group_size=args.spike_rhythm_min_group_size,
        spike_rhythm_return_all_groups=args.spike_rhythm_return_all_groups,
        spike_interval_size=args.spike_interval_size,
        spike_interval_threshold=args.spike_interval_threshold,
        spike_interval_min_group_size=args.spike_interval_min_group_size,
        spike_interval_include_partial=not args.no_spike_interval_include_partial,
        spike_spatial_grid_size=_parse_pair_arg(args.spike_spatial_grid_size, "spike-spatial-grid-size"),
        spike_spatial_threshold=args.spike_spatial_threshold,
        spike_spatial_min_group_size=args.spike_spatial_min_group_size,
        spike_spatial_activity_source=args.spike_spatial_activity_source,
        spike_spatial_time_aggregate=args.spike_spatial_time_aggregate,
    )
    hparams.validate()

    core = S2NetCore(hparams, device=args.device)
    if float(args.edge_membrane_weight) != 0.0:
        if args.edge_image_hdf5 is None:
            raise ValueError("--edge-image-hdf5 is required when --edge-membrane-weight is nonzero")
        import h5py
        with h5py.File(args.edge_image_hdf5, "r") as dataset:
            if len(dataset["image"]) < len(gamma_seq):
                raise ValueError("HDF5 has fewer RGB images than gamma examples")
            images = torch.from_numpy(dataset["image"][:len(gamma_seq)]).permute(0, 3, 1, 2)
        if images.dtype != torch.uint8 or images.size(1) != 3:
            raise ValueError("Edge RGB images must have uint8 shape [B,3,H,W]")
        training_data = TensorDataset(gamma_seq, images)
    else:
        training_data = TensorDataset(gamma_seq)
    loader = DataLoader(
        training_data,
        batch_size=int(args.batch_size),
        shuffle=True,
    )
    criterion = UnsupervisedS2NetLoss(
        spike_rate_weight=args.spike_rate_weight,
        spike_smooth_weight=args.spike_smooth_weight,
        spike_diversity_weight=args.spike_diversity_weight,
        structural_weight=args.structural_weight,
        object_overlap_weight=args.object_overlap_weight,
        sample_diversity_weight=args.sample_diversity_weight,
        spatial_compactness_weight=args.spatial_compactness_weight,
        temporal_balance_weight=args.temporal_balance_weight,
        activity_confidence_weight=args.activity_confidence_weight,
        activity_area_weight=args.activity_area_weight,
        activity_contrast_weight=args.activity_contrast_weight,
        spike_target_rate=args.spike_target_rate,
        patch_grid_size=(
            _parse_pair_arg(args.loss_patch_grid_size, "loss-patch-grid-size")
            if args.loss_patch_grid_size is not None
            else _parse_pair_arg(args.spike_spatial_grid_size, "spike-spatial-grid-size")
        ),
        activity_min_area=args.activity_min_area,
        activity_max_area=args.activity_max_area,
        activity_target_std=args.activity_target_std,
        plv_bimodality_weight=args.plv_bimodality_weight,
        plv_balance_weight=args.plv_balance_weight,
        plv_coherence_weight=args.plv_coherence_weight,
        plv_group_count_weight=args.plv_group_count_weight,
        plv_collapse_weight=args.plv_collapse_weight,
        phase_quantization_weight=args.phase_quantization_weight,
        phase_spread_weight=args.phase_spread_weight,
        phase_num_slots=args.phase_num_slots,
        plv_target_density=args.plv_target_density,
        plv_target_groups=args.plv_target_groups,
    )

    _, losses = train_s2net_core(
        core,
        loader,
        epochs=args.epochs,
        lr=args.lr,
        criterion=criterion,
        device=args.device,
        save_path=args.save_path,
        loss_signal=args.loss_signal,
        grad_clip_norm=args.grad_clip_norm,
        verbose=args.verbose,
        plv_settle=args.plv_settle,
        plv_source=args.plv_source,
        plv_combine=args.plv_combine,
        spike_plv_weight=args.spike_plv_weight,
        graph_teacher_weight=args.graph_teacher_weight,
        graph_teacher_temperature=args.graph_teacher_temperature,
        graph_teacher_signal=args.graph_teacher_signal,
        edge_membrane_weight=args.edge_membrane_weight,
        edge_membrane_margin=args.edge_membrane_margin,
        edge_membrane_grid_size=_parse_pair_arg(args.spike_spatial_grid_size, "spike-spatial-grid-size"),
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_epochs=args.checkpoint_epochs,
    )
    print(f"trained S2NetCore: {args.save_path}")
    print(f"loss: {losses[0]:.6f} -> {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
