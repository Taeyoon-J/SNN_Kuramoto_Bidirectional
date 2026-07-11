import torch
from pathlib import Path

from loss_function import UnsupervisedS2NetLoss


def train_s2net_core(
    core,
    dataloader,
    epochs=100,
    lr=1e-3,
    criterion=None,
    optimizer=None,
    device=None,
    save_path=None,
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
    for _ in range(int(epochs)):
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
        loss_history.append(epoch_loss / sample_count)

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
