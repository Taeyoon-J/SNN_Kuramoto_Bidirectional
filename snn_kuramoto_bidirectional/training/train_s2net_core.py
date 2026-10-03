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
        patch_pool_rgb,
        phase_locking_value,
        phase_alignment,
        signal_synchrony,
    )
    from snn_kuramoto_bidirectional.s2net_cls import GammaGenerator, S2NetCore
    from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_image_folder
    from snn_kuramoto_bidirectional.sc_generator import pearson_cor_sc
except ModuleNotFoundError:
    from hyperparameter import S2NetHyperparameters
    from loss_function import (
        UnsupervisedS2NetLoss,
        patch_pool_rgb,
        phase_locking_value,
        phase_alignment,
        signal_synchrony,
    )
    from s2net_cls import GammaGenerator, S2NetCore
    from train_gamma_initializer import load_image_folder
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
    gamma_generator=None,
    encoder_lr=None,
    recon_grid=None,
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
    if optimizer is None:
        # End to end: the feature encoder learns alongside the core. It was
        # trained as a reconstruction autoencoder and never asked to separate
        # objects; per-channel IoU against ground truth averages 0.079, with two
        # channels at exactly 0.000, so the features cap everything downstream.
        groups = [{"params": list(core.parameters()), "lr": lr}]
        if gamma_generator is not None:
            gamma_generator = gamma_generator.to(device)
            # A pretrained encoder is easy to destroy at the core's learning
            # rate: at lr 1e-3 for 50 epochs the phase readout fell 23% on both
            # seeds tried. Its own rate is separate so it can be moved gently.
            groups.append({
                "params": list(gamma_generator.parameters()),
                "lr": lr if encoder_lr is None else float(encoder_lr),
            })
        optimizer = torch.optim.Adam(groups)
    loss_history = []
    parts_history = []

    core.train()
    if gamma_generator is not None:
        gamma_generator.train()
    for epoch in range(1, int(epochs) + 1):
        epoch_loss = 0.0
        sample_count = 0
        epoch_parts = {}
        for batch in dataloader:
            raw_batch = _unpack_gamma_batch(batch).to(device)
            recon_target = None
            if gamma_generator is not None:
                # the reconstruction target is pooled RGB, external to the model,
                # so it cannot be gamed by changing the features
                if float(getattr(criterion, "slot_reconstruction_weight", 0.0)) != 0.0:
                    recon_target = patch_pool_rgb(raw_batch, recon_grid)
                gamma_seq = gamma_generator(raw_batch)      # images -> gamma
            else:
                gamma_seq = raw_batch

            object_groups, spikes, core_out, plv, theta = _forward_with_plv(
                core, gamma_seq, criterion, plv_settle, plv_source, plv_combine
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
                recon_target=recon_target,
                assignment=_cluster_assignment(core, theta, plv_settle, loss_values),
            )
            if float(spike_plv_weight) != 0.0:
                # The phase objective is kept and the spiking one added beside
                # it. Replacing it was the confound the last two attempts at this
                # shared: both switched plv_source to the spiking side, which
                # also removed every phase term, so a loss could not be told from
                # the absence of a gain.
                spike_plv = _spike_component_plv(core, plv_settle)
                if spike_plv is not None:
                    aux, aux_parts = criterion(plv=spike_plv, plv_settle=int(plv_settle))
                    loss = loss + float(spike_plv_weight) * aux
                    parts.update({"spike_" + k: v for k, v in aux_parts.items()})

            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                clipped = list(core.parameters())
                if gamma_generator is not None:
                    clipped += list(gamma_generator.parameters())
                torch.nn.utils.clip_grad_norm_(clipped, float(grad_clip_norm))
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
        if gamma_generator is not None:
            enc = Path(save_path).with_name(Path(save_path).stem + "_encoder.pt")
            torch.save(gamma_generator.state_dict(), enc)

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
            assignment=_cluster_assignment(core, theta, plv_settle, loss_values),
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
        for name in ("phase_quantization_weight", "phase_spread_weight",
                     "slot_reconstruction_weight", "mincut_weight")
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
            "mincut_weight",
        )
    )


def _forward_with_plv(core, gamma_seq, criterion, plv_settle, plv_source="phase",
                      plv_combine="mean"):
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
    if plv_source == "phase":
        plv = phase_locking_value(theta, settle=int(plv_settle), combine=plv_combine)
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


def _spike_component_plv(core, plv_settle):
    """
    Synchrony between per-component spike trains, combined by product. [B, N, N]

    The same shape of readout the phases use, applied to the spiking side. It
    exists so the loss can reach the spiking layers at all: the dendritic and
    membrane time constants are drawn from sigmoid(U(-4, 0)) and have never been
    trained towards preserving synchrony, because the objective reads the phases.

    Measured, the spiking path loses 0.165 foreground ARI and loses it in roughly
    equal bites -- gate 0.02, dendrite 0.045, membrane 0.046, thresholding 0.028 --
    so there is no single stage to repair. Letting the layers see the objective is
    what is left.
    """
    comp = getattr(core, "last_component_spikes", None)
    if comp is None:
        return None
    per = [
        signal_synchrony(comp[:, d], settle=int(plv_settle)).clamp_min(0)
        for d in range(comp.size(1))
    ]
    return torch.stack(per).prod(dim=0)


def _cluster_assignment(core, theta, plv_settle, spikes=None):
    """
    Soft partition from the readout, or None when the core has no readout.

    In signal mode the head reads the spike trains, which is the readout the
    architecture specifies: units that spike together are one object. That also
    keeps the spiking layers on the gradient path, which reading the phases does
    not -- a head on theta bypasses the SNN entirely.
    """
    readout = getattr(core, "cluster_readout", None)
    if readout is None:
        return None
    if readout.feature_source == "signal":
        if spikes is None:
            return None
        return readout(signal=spikes, settle=int(plv_settle))
    if theta is None:
        return None
    return readout(theta=theta, settle=int(plv_settle))


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
    missing, unexpected = core.load_state_dict(state_dict, strict=False)
    stray = [k for k in missing if not k.startswith("cluster_readout.")]
    if stray or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the core: missing {stray}, unexpected {list(unexpected)}"
        )
    if missing:
        # the checkpoint predates the readout, so it starts from its own init
        print(f"[load] readout not in checkpoint, initialising {len(missing)} tensors")
    core.eval()
    return core


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train S2NetCore from precomputed gamma sequences.")
    parser.add_argument(
        "--gamma-seq-path",
        default=None,
        help="Precomputed gamma. Mutually exclusive with --image-dir.",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help=(
            "Train end to end from images instead of frozen gamma, so the CNN "
            "feature encoder learns alongside the core. Requires --num-regions "
            "and --gamma-patch-grid-size."
        ),
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--gamma-patch-grid-size",
        type=int,
        default=None,
        help="Patch grid for end-to-end mode, e.g. 16 for a 16x16 grid.",
    )
    parser.add_argument(
        "--input-encoder-path",
        default=None,
        help="Optional pretrained CNNFeatureEncoder to start the encoder from.",
    )
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
    parser.add_argument(
        "--gamma-drive-mode",
        default="sequence",
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
        default=None,
        help="Recurrent length for static drive mode. Defaults to the gamma channel count.",
    )
    parser.add_argument("--osc-dim", type=int, default=4)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help=(
            "Train on the first N samples only. Everything in this repository has "
            "been trained on all 1000 images with no split; this is how a larger "
            "set is used while leaving later images genuinely unseen."
        ),
    )
    parser.add_argument(
        "--spike-plv-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on the PLV family applied to per-component spike synchrony, "
            "added alongside the phase objective rather than replacing it. "
            "Requires --spike-per-component. The spiking layers receive no "
            "gradient at all without this."
        ),
    )
    parser.add_argument(
        "--spike-per-component",
        action="store_true",
        help=(
            "Run the spiking layers once per oscillator component instead of "
            "mixing them in the dendrite. Collapsing the components is worth "
            "0.202 foreground ARI, and the dendrite does it in its first layer."
        ),
    )
    parser.add_argument(
        "--gamma-time-phases",
        type=int,
        default=0,
        help=(
            "Give each patch this many drive vectors, cycled over the rollout, "
            "instead of one held constant. 0 is the current constant drive. The "
            "cycle length is independent of the channel count on purpose: tying "
            "them together is what made the legacy sequence mode fail."
        ),
    )
    parser.add_argument(
        "--gamma-phase-mode",
        default="none",
        choices=["none", "tanh", "standardize_tanh"],
        help="Map raw gamma onto a phase range before sin(gamma - theta).",
    )
    parser.add_argument(
        "--theta-init",
        default="zeros",
        choices=["zeros", "gamma", "gamma_noise"],
        help="zeros makes every image start from an identical oscillator state.",
    )
    parser.add_argument("--theta-init-noise", type=float, default=0.0)
    parser.add_argument(
        "--freq-gain",
        type=float,
        default=0.0,
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
        default="static",
        choices=["static", "learned"],
        help="static uses the fixed sc; learned builds a sparse graph per image.",
    )
    parser.add_argument("--graph-top-k", type=int, default=8)
    parser.add_argument("--graph-hidden-dim", type=int, default=16)
    parser.add_argument("--graph-coupling-gain", type=float, default=8.0)
    parser.add_argument("--graph-temperature", type=float, default=0.1)
    parser.add_argument(
        "--graph-spatial-decay",
        type=float,
        default=None,
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
    parser.add_argument("--plv-bimodality-weight", type=float, default=0.0)
    parser.add_argument("--plv-balance-weight", type=float, default=0.0)
    parser.add_argument("--plv-coherence-weight", type=float, default=0.0)
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
    parser.add_argument(
        "--plv-combine",
        choices=["mean", "product", "min"],
        default="mean",
        help=(
            "How the osc_dim components enter the synchrony matrix. \"mean\" averages "
            "the phases first, which compresses which group a patch belongs to into "
            "one number; measured at readout time, combining per-component synchrony "
            "instead raised foreground ARI from 0.430 to 0.501 on the same checkpoint."
        ),
    )
    parser.add_argument("--phase-num-slots", type=float, default=7.0)
    parser.add_argument(
        "--readout-slots",
        type=int,
        default=0,
        help=(
            "Size of the differentiable partition. 0 keeps the old setup, where "
            "spectral clustering runs after training and the loss never sees the "
            "grouping it is judged on. Set it near the typical object count."
        ),
    )
    parser.add_argument(
        "--readout-source",
        choices=["phase", "signal"],
        default="phase",
        help=(
            "What the readout head reads. \"signal\" reads the spike trains, which "
            "is the readout the architecture specifies and the only one that keeps "
            "the SNN on the gradient path."
        ),
    )
    parser.add_argument("--readout-embed-dim", type=int, default=16)
    parser.add_argument("--readout-iters", type=int, default=3)
    parser.add_argument(
        "--readout-temperature",
        type=float,
        default=0.5,
        help=(
            "Softmax temperature of the slot assignment. Measured on 192-step spike "
            "trains, 0.5 leaves the assignment exactly uniform -- a stationary point "
            "training never escapes -- and 0.05 was the first value that committed."
        ),
    )
    parser.add_argument("--mincut-ortho-weight", type=float, default=0.0)
    parser.add_argument("--mincut-floor", type=float, default=0.01)
    parser.add_argument("--mincut-floor-weight", type=float, default=0.5)
    parser.add_argument("--mincut-entropy-weight", type=float, default=0.5)
    parser.add_argument(
        "--mincut-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on the relaxed normalized cut over the readout's assignment. "
            "This is the only term that scores the partition itself; every other "
            "objective here scores a summary of the synchrony matrix instead."
        ),
    )
    parser.add_argument(
        "--plv-collapse-weight",
        type=float,
        default=0.0,
        help=(
            "Barrier against a uniform synchrony matrix. plv_bimodality is zero at "
            "global synchrony as well as at a real partition, and that is the minimum "
            "the optimiser reaches unless this is on."
        ),
    )
    parser.add_argument("--plv-target-density", type=float, default=0.25)
    parser.add_argument(
        "--plv-settle",
        type=int,
        default=0,
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
        "--membrane-vth",
        type=float,
        default=0.5,
        help=(
            "Spike threshold. The default was never matched to the signal: the "
            "membrane oscillates with a standard deviation near 0.1, so 0.5 sits "
            "about five times the amplitude and the network cannot fire."
        ),
    )
    parser.add_argument(
        "--membrane-low-m",
        type=float,
        default=0.0,
        help=(
            "Lower bound of the membrane tau init. The default U(0, 4) gives a "
            "leak near 0.85, which low-passes away the oscillation carrying the "
            "information; U(-4, 0) lets the membrane track the rhythm instead."
        ),
    )
    parser.add_argument("--membrane-high-m", type=float, default=4.0)
    parser.add_argument(
        "--gate-mode",
        default="sigmoid",
        choices=["sigmoid", "raw", "phase_mean"],
        help=(
            "'sigmoid' keeps the original gate, compressed to [0.5, 0.731]. "
            "'phase_mean' reduces the osc_dim axis before the sine; the same "
            "information reduced the other way round measured 0.216 vs 0.067."
        ),
    )
    parser.add_argument("--low-n", type=float, default=0.0)
    parser.add_argument("--high-n", type=float, default=4.0)
    parser.add_argument("--branch", type=int, default=4)
    parser.add_argument("--spike-classify-method", default="spike_interval", choices=["spike_rhythm", "spike_interval", "spatial_components"])
    parser.add_argument("--spike-rhythm-threshold", type=float, default=0.8)
    parser.add_argument("--spike-rhythm-min-group-size", type=int, default=2)
    parser.add_argument("--spike-rhythm-return-all-groups", action="store_true")
    parser.add_argument("--spike-interval-size", type=int, default=1)
    parser.add_argument("--spike-interval-threshold", type=float, default=0.5)
    parser.add_argument("--spike-interval-min-group-size", type=int, default=1)
    parser.add_argument("--no-spike-interval-include-partial", action="store_true")
    parser.add_argument("--spike-spatial-grid-size", type=int, nargs="+", default=None)
    parser.add_argument("--spike-spatial-threshold", type=float, default=0.5)
    parser.add_argument("--spike-spatial-min-group-size", type=int, default=2)
    parser.add_argument("--spike-spatial-activity-source", default="sigmoid_membrane", choices=["spikes", "membrane", "sigmoid_membrane"])
    parser.add_argument("--spike-spatial-time-aggregate", default="mean", choices=["max", "mean"])
    parser.add_argument("--spike-rate-weight", type=float, default=1.0)
    parser.add_argument("--spike-smooth-weight", type=float, default=0.1)
    parser.add_argument("--spike-diversity-weight", type=float, default=0.1)
    parser.add_argument("--structural-weight", type=float, default=0.1)
    parser.add_argument("--object-overlap-weight", type=float, default=0.0)
    parser.add_argument("--sample-diversity-weight", type=float, default=0.0)
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
    parser.add_argument(
        "--slot-reconstruction-weight",
        type=float,
        default=0.0,
        help=(
            "Require the phase grouping to explain the image: patches are rebuilt "
            "from their slot's mean pooled RGB. Every other loss is a generic "
            "structure prior that knows nothing about objects, which is how "
            "targets like a group count get optimised perfectly while the task "
            "score falls to chance. Needs --image-dir."
        ),
    )
    parser.add_argument("--slot-num-slots", type=int, default=7)
    parser.add_argument("--slot-temperature", type=float, default=0.3)
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=None,
        help=(
            "Separate learning rate for the feature encoder in end-to-end mode. "
            "Defaults to --lr, which measured 23%% worse on the phase readout for "
            "a pretrained encoder."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    if (args.gamma_seq_path is None) == (args.image_dir is None):
        raise ValueError("Pass exactly one of --gamma-seq-path or --image-dir.")

    gamma_generator = None
    if args.image_dir is not None:
        if args.gamma_patch_grid_size is None:
            raise ValueError("--gamma-patch-grid-size is required with --image-dir.")
        images = load_image_folder(
            args.image_dir, image_size=args.image_size, max_images=args.max_images
        )
        grid = int(args.gamma_patch_grid_size)
        train_tensor = images
        num_feature_maps = int(args.num_feature_maps or 8)
        num_regions = grid * grid
        print(f"end-to-end: {images.size(0)} images, {grid}x{grid} grid, "
              f"{num_regions} oscillators", flush=True)
        # gamma is produced on the fly, so SC has to come from a first pass
        gamma_seq = None
    else:
        gamma_seq = torch.load(args.gamma_seq_path, map_location="cpu").float()
        if gamma_seq.dim() != 3:
            raise ValueError(f"gamma_seq must have shape [B, T, N], but got {tuple(gamma_seq.shape)}.")
        train_tensor = gamma_seq
        if args.train_limit is not None:
            if int(args.train_limit) < 1 or int(args.train_limit) > train_tensor.size(0):
                raise ValueError(
                    f"--train-limit must lie in [1, {train_tensor.size(0)}]."
                )
            train_tensor = train_tensor[: int(args.train_limit)]
            print(f"training on the first {train_tensor.size(0)} of "
                  f"{gamma_seq.size(0)} samples", flush=True)

    if gamma_seq is not None:
        num_feature_maps = gamma_seq.size(1) if args.num_feature_maps is None else int(args.num_feature_maps)
        num_regions = gamma_seq.size(2) if args.num_regions is None else int(args.num_regions)
        if gamma_seq.size(1) != num_feature_maps:
            raise ValueError(f"gamma_seq T={gamma_seq.size(1)} does not match num_feature_maps={num_feature_maps}.")
        if gamma_seq.size(2) != num_regions:
            raise ValueError(f"gamma_seq N={gamma_seq.size(2)} does not match num_regions={num_regions}.")

    if args.graph_mode == "learned":
        # the graph is built per image, so the fixed SC is never read
        sc = None
    elif args.sc_path is None:
        if gamma_seq is None:
            raise ValueError("--sc-path is required in end-to-end mode with a static graph.")
        sc = pearson_cor_sc(gamma_seq.reshape(-1, num_regions))
        if args.sc_save_path is not None:
            sc_path = Path(args.sc_save_path)
            sc_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sc, sc_path)
            print(f"saved generated sc {tuple(sc.shape)} to {sc_path}", flush=True)
    else:
        sc = torch.load(args.sc_path, map_location="cpu").float()

    if sc is not None and tuple(sc.shape) != (num_regions, num_regions):
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
        graph_spatial_decay=args.graph_spatial_decay,
        graph_feedback_strength=args.graph_feedback_strength,
        graph_feedback_momentum=args.graph_feedback_momentum,
        membrane_vth=args.membrane_vth,
        membrane_low_m=args.membrane_low_m,
        membrane_high_m=args.membrane_high_m,
        gate_mode=args.gate_mode,
        gamma_time_phases=args.gamma_time_phases,
        spike_per_component=args.spike_per_component,
        readout_slots=args.readout_slots,
        readout_source=args.readout_source,
        readout_signal_dim=(
            max(int(args.num_time_steps) - int(args.plv_settle), 1)
            if args.readout_source == "signal" else None
        ),
        readout_embed_dim=args.readout_embed_dim,
        readout_iters=args.readout_iters,
        readout_temperature=args.readout_temperature,
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
        spike_spatial_grid_size=_parse_pair_arg(args.spike_spatial_grid_size, "spike-spatial-grid-size"),
        spike_spatial_threshold=args.spike_spatial_threshold,
        spike_spatial_min_group_size=args.spike_spatial_min_group_size,
        spike_spatial_activity_source=args.spike_spatial_activity_source,
        spike_spatial_time_aggregate=args.spike_spatial_time_aggregate,
    )
    hparams.validate()

    core = S2NetCore(hparams, device=args.device)
    if args.image_dir is not None:
        hparams.gamma_mode = "patch"
        hparams.gamma_patch_grid_size = int(args.gamma_patch_grid_size)
        gamma_generator = GammaGenerator(hparams, device=args.device).to(args.device)
        if args.input_encoder_path is not None:
            gamma_generator.input_layer.load_state_dict(
                torch.load(args.input_encoder_path, map_location=args.device)
            )
            print(f"loaded pretrained encoder: {args.input_encoder_path}", flush=True)
    loader = DataLoader(
        TensorDataset(train_tensor),
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
        mincut_weight=args.mincut_weight,
        mincut_ortho_weight=args.mincut_ortho_weight,
        mincut_floor=args.mincut_floor,
        mincut_floor_weight=args.mincut_floor_weight,
        mincut_entropy_weight=args.mincut_entropy_weight,
        phase_quantization_weight=args.phase_quantization_weight,
        phase_spread_weight=args.phase_spread_weight,
        phase_num_slots=args.phase_num_slots,
        slot_reconstruction_weight=args.slot_reconstruction_weight,
        slot_num_slots=args.slot_num_slots,
        slot_temperature=args.slot_temperature,
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
        gamma_generator=gamma_generator,
        encoder_lr=args.encoder_lr,
        recon_grid=args.gamma_patch_grid_size,
    )
    print(f"trained S2NetCore: {args.save_path}")
    print(f"loss: {losses[0]:.6f} -> {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
