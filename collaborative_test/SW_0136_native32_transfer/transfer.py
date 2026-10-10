"""Strict loader for completed SW0134 seed1 models at native32 inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding import train as sw134_train
from collaborative_test.SW_0135_native32_spike_binding import resolution
from collaborative_test.SW_0135_native32_spike_binding.binder import NativeSpikeSlotBinder, RelativeSlotRGBDecoder
from collaborative_test.SW_0135_native32_spike_binding.foundation import (
    load_native32_foundation, sha256_file,
)

ARMS = ("actual_joint", "actual_frozen")
EXPECTED_CHECKPOINTS = {
    "actual_joint": "7bb7efcaf3640c6de3ca4ac5b8f8f265f16f6e75a2ffb24b0515862930e49beb",
    "actual_frozen": "361b426179a4eb3f2d39169398557e0b898c995afab62c68cdcb025677b783ed",
}
INTEGRATION_KEYS = {"a_d", "a_m", "b"}


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def state_dict_sha256(modules):
    """Stable hash of named module state, including buffers such as initial_noise."""
    h = hashlib.sha256()
    for module_name, module in modules:
        for key, value in sorted(module.state_dict().items()):
            if not torch.is_tensor(value) or not torch.isfinite(value).all():
                raise ValueError(f"invalid transferred state tensor {module_name}.{key}")
            tensor = value.detach().cpu().contiguous()
            h.update(module_name.encode() + b"\0" + key.encode() + b"\0")
            h.update(str(tensor.dtype).encode() + b"\0")
            h.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
            h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def split_wrapped_state(wrapped_state):
    """Strip only registered `core.` names and preserve the three learned scalars."""
    if not isinstance(wrapped_state, dict):
        raise TypeError("wrapped_state_dict must be a state dictionary")
    core = {}
    integration = {}
    extras = set()
    for key, value in wrapped_state.items():
        if key.startswith("core."):
            core[key[len("core."):]] = value
        elif key in INTEGRATION_KEYS:
            integration[key] = value
        else:
            extras.add(key)
    if extras or not core or set(integration) != INTEGRATION_KEYS:
        raise ValueError(f"unsupported/missing wrapped state keys: extras={sorted(extras)}, integration={sorted(integration)}")
    for name, tensor in integration.items():
        if not torch.is_tensor(tensor) or tuple(tensor.shape) != (4,) or not torch.isfinite(tensor).all():
            raise ValueError(f"transferred integration parameter {name} must be finite four-component state")
    if any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in core.values()):
        raise ValueError("wrapped native core state contains nonfinite or non-tensor values")
    return core, integration


def compose_native_wrapper_state(mapped_core_state, integration_state, wrapper_template):
    """Require a complete target wrapper key set, retaining mapped core and learned scalars."""
    if set(integration_state) != INTEGRATION_KEYS:
        raise ValueError("native transfer must include all three trained integration parameters")
    expected = {f"core.{key}" for key in mapped_core_state} | INTEGRATION_KEYS
    if set(wrapper_template) != expected:
        raise ValueError("mapped native/integration transfer does not cover exact wrapper state schema")
    result = {f"core.{key}": value for key, value in mapped_core_state.items()}
    result.update(integration_state)
    return result


def strict_load_heads(binder, decoder, payload):
    """Strictly restore the complete learned binder (including noise) and decoder."""
    for key in ("binder_state_dict", "decoder_state_dict"):
        if not isinstance(payload.get(key), dict) or not payload[key]:
            raise ValueError(f"completed checkpoint lacks {key}")
    binder.load_state_dict(payload["binder_state_dict"], strict=True)
    decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
    expected_noise = payload["binder_state_dict"].get("initial_noise")
    if not torch.is_tensor(expected_noise) or not torch.equal(
            binder.initial_noise.detach().cpu(), expected_noise.detach().cpu()):
        raise AssertionError("trained binder persistent initial_noise buffer was not transferred")


def _validated_arm(arm, *, assets=None):
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    folder = ROOT / "trained_models/SW0134_native_spike_binding" / f"{arm}_seed1"
    manifest, checkpoint, checkpoint_path = sw134_train.validate_completed_training(
        1, arm, folder, assets=assets)
    checkpoint_sha = sha256_file(checkpoint_path)
    if (checkpoint_sha != EXPECTED_CHECKPOINTS[arm]
            or manifest.get("checkpoint_sha256") != checkpoint_sha
            or manifest.get("updates") != 4096 or manifest.get("passes") != 16
            or manifest.get("total_image_exposures") != 65536
            or manifest.get("ground_truth_used_for_training") is not False):
        raise ValueError(f"SW0134 {arm} completed checkpoint does not match registered terminal artifact")
    marker_path = folder / "TRAINING_COMPLETED.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    return {"folder": folder, "manifest": manifest, "manifest_path": folder / "manifest.json",
            "marker_path": marker_path, "marker": marker, "checkpoint": checkpoint,
            "checkpoint_path": checkpoint_path, "checkpoint_sha256": checkpoint_sha}


def load_transferred_model(arm, device="cuda:0", *, foundation=None):
    """Validate completion then strict-load every trained SW0134 tensor into native32 models."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if foundation is None:
        foundation = load_native32_foundation(1, device, verify_rgb_assets=True)
    assets = foundation.provenance["rgb_asset_validation"]
    verified = _validated_arm(arm, assets=assets)
    payload = verified["checkpoint"]
    required_top = {"wrapped_state_dict", "encoder_state_dict", "binder_state_dict",
                    "decoder_state_dict", "source_core_sha256", "seed", "arm", "lambda"}
    if set(payload) != required_top or payload.get("seed") != 1 or payload.get("arm") != arm:
        raise ValueError("SW0134 checkpoint schema/identity does not match its completed arm")
    if payload.get("source_core_sha256") != foundation.provenance["source_core_sha256"]:
        raise ValueError("SW0134 and native32 source97 checkpoint lineage do not match")

    native_state, integration = split_wrapped_state(payload["wrapped_state_dict"])
    mapped = resolution.convert_state_dict(native_state, foundation.wrapped.core.state_dict())
    foundation.wrapped.core.load_state_dict(mapped, strict=True)
    # Keep the exact learned 12 integration values; never initialize to zero on transfer.
    wrapper_state = compose_native_wrapper_state(mapped, integration,
                                                  foundation.wrapped.state_dict())
    foundation.wrapped.load_state_dict(wrapper_state, strict=True)

    missing = foundation.encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    if missing is not None and (missing.missing_keys or missing.unexpected_keys):
        raise AssertionError("trained encoder strict load failed")
    binder = NativeSpikeSlotBinder(seed=135).to(device)
    decoder = RelativeSlotRGBDecoder(seed=106).to(device)
    strict_load_heads(binder, decoder, payload)
    for module in (foundation.wrapped, foundation.encoder, binder, decoder):
        module.eval()
    transferred_sha = state_dict_sha256((
        ("wrapped", foundation.wrapped), ("encoder", foundation.encoder),
        ("binder", binder), ("decoder", decoder)))
    record = {
        "arm": arm, "seed": 1,
        "source_core_sha256": foundation.provenance["source_core_sha256"],
        "source_manifest_sha256": foundation.provenance["source_manifest_sha256"],
        "checkpoint_path": str(verified["checkpoint_path"]),
        "checkpoint_sha256": verified["checkpoint_sha256"],
        "manifest_sha256": sha256_file(verified["manifest_path"]),
        "completion_marker_sha256": sha256_file(verified["marker_path"]),
        "updates": verified["manifest"]["updates"],
        "passes": verified["manifest"]["passes"],
        "total_image_exposures": verified["manifest"]["total_image_exposures"],
        "integration_values": {name: payload["wrapped_state_dict"][name].detach().cpu().tolist()
                                for name in sorted(INTEGRATION_KEYS)},
        "binder_initial_noise_sha256": _hash_bytes(
            binder.initial_noise.detach().cpu().contiguous().numpy().tobytes()),
        "converted_full_state_sha256": transferred_sha,
        "strict_transfer": True,
        "ground_truth_used": False,
        "optimizer_updates": 0,
    }
    return foundation, binder, decoder, record
