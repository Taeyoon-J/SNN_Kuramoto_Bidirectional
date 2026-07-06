import torch
import torch.nn as nn
from pathlib import Path


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
    Train only S2NetCore from precomputed gamma sequences.

    Each dataloader batch must be:
        gamma_seq, sc, labels

    Expected shapes:
        gamma_seq: [B, T, num_regions]
        sc:        [num_regions, num_regions] or [B, num_regions, num_regions]
        labels:    [B]

    Returns:
        core, loss_history
    """
    device = _resolve_device(device, core)
    core = core.to(device)
    criterion = criterion if criterion is not None else nn.NLLLoss()
    optimizer = optimizer if optimizer is not None else torch.optim.Adam(core.parameters(), lr=lr)
    loss_history = []

    core.train()
    for _ in range(int(epochs)):
        epoch_loss = 0.0
        sample_count = 0
        for gamma_seq, sc, labels in dataloader:
            gamma_seq = gamma_seq.to(device)
            sc = sc.to(device)
            labels = labels.to(device).long()

            log_probs, _ = core(gamma_seq, sc)
            loss = criterion(log_probs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            sample_count += batch_size
        loss_history.append(epoch_loss / sample_count)

    if save_path is not None:
        save_s2net_core(core, save_path)

    return core, loss_history


@torch.no_grad()
def evaluate_s2net_core(core, dataloader, criterion=None, device=None):
    """Evaluate S2NetCore on batches of gamma_seq, sc, labels."""
    device = _resolve_device(device, core)
    core = core.to(device)
    criterion = criterion if criterion is not None else nn.NLLLoss()

    core.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for gamma_seq, sc, labels in dataloader:
        gamma_seq = gamma_seq.to(device)
        sc = sc.to(device)
        labels = labels.to(device).long()

        log_probs, _ = core(gamma_seq, sc)
        loss = criterion(log_probs, labels)
        predictions = log_probs.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_count += batch_size

    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
    }


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
