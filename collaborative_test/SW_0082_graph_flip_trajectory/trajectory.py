"""Epoch checkpoint helpers, isolated so their contract is unit testable."""
from pathlib import Path

import torch


def save_epoch_checkpoint(core, checkpoint_dir, epoch):
    if not isinstance(epoch, int) or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"epoch_{epoch:02d}_core.pt"
    if path.exists():
        raise FileExistsError(f"refusing epoch checkpoint overwrite: {path}")
    torch.save(core.state_dict(), path)
    return path


def assert_checkpoint_equal(final_path, epoch_path):
    final = torch.load(final_path, map_location="cpu", weights_only=True)
    epoch = torch.load(epoch_path, map_location="cpu", weights_only=True)
    if final.keys() != epoch.keys() or any(
            not torch.equal(final[key].cpu(), epoch[key].cpu()) for key in final):
        raise AssertionError("final core differs from last epoch checkpoint")
    return True
