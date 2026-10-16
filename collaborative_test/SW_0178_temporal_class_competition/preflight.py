"""Read-only CPU admission for the SW0161 settled TRAIN cache boundary."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _import_hidden(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import importlib
        return importlib.import_module(name)
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def flatten_modal_targets(modal: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Explicitly convert `[B,32,32]` full modal IDs to `[B,1024]` labels."""
    if isinstance(modal, np.ndarray):
        if modal.dtype.kind not in "iu":
            raise TypeError("modal target grid must have integer dtype")
        value = torch.from_numpy(np.ascontiguousarray(modal))
    elif torch.is_tensor(modal):
        value = modal
        if value.dtype == torch.bool or value.dtype.is_floating_point or value.dtype.is_complex:
            raise TypeError("modal target grid must have integer dtype")
    else:
        raise TypeError("modal target must be a numpy array or torch tensor")
    if value.ndim == 2 and tuple(value.shape) == (32, 32):
        value = value.unsqueeze(0)
    if value.ndim != 3 or tuple(value.shape[1:]) != (32, 32):
        raise ValueError("modal target must have shape [B,32,32]")
    if bool((value < 0).any()):
        raise ValueError("modal target IDs must be nonnegative")
    return value.reshape(value.shape[0], 1024).to(dtype=torch.int64)


def prepare_cache_cpu(*, cache_root=None, selected_rows: int = 8) -> dict:
    """Validate complete SW0161 cache and load its first ordered TRAIN rows.

    This entrypoint never creates/adopts cache rows and verifies source cache
    metadata bytes remain unchanged. The upstream validator is only invoked
    after verifying all 512 ledger rows are already committed.
    """
    if selected_rows != 8:
        raise ValueError("SW0178 source admission is fixed to the first eight TRAIN rows")
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0178 cache preparation must happen before CUDA initialization")
    cache_module = _import_hidden("collaborative_test.SW_0161_supervised_spike_partition.cache")
    train_module = _import_hidden("collaborative_test.SW_0161_supervised_spike_partition.train")
    root = Path(cache_root) if cache_root is not None else Path(cache_module.CACHE_ROOT)
    manifest_path, ledger_path = root / "manifest.json", root / "ledger.json"
    if not manifest_path.is_file() or not ledger_path.is_file():
        raise FileNotFoundError("SW0161 complete source cache is unavailable")
    manifest_sha_before, ledger_sha_before = _sha_file(manifest_path), _sha_file(ledger_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or manifest.get("count") != 512
            or len(ledger.get("rows", [])) != 512):
        raise ValueError("SW0178 requires an already complete 512-row cache; refusing adoption/write")
    validated = train_module.validate_cache(root)
    if validated != manifest:
        raise ValueError("SW0161 validated manifest differs from the immutable bytes read at entry")
    ids = manifest["bindings"]["image_ids"]
    rgb_hashes = manifest["bindings"]["image_rgb_sha256"]
    store = cache_module._AtomicTraceCache(root, manifest["bindings"], ids, rgb_hashes)
    if len(store.records) != 512:
        raise ValueError("SW0161 row store is not fully committed")
    rows = [store.load_row(index) for index in range(8)]
    traces = np.stack([row[0] for row in rows])
    targets = np.stack([row[1] for row in rows])
    if traces.dtype != np.float32 or traces.shape != (8, 4, 1024, 512):
        raise ValueError("cached settled traces must be float32 [8,4,1024,512]")
    if targets.dtype != np.int64 or targets.shape != (8, 1024):
        raise ValueError("cached modal labels must be int64 [8,1024]")
    row_records = store.records[:8]
    if ([int(record["image_id"]) for record in row_records] != [int(v) for v in ids[:8]]
            or [record["row_file_sha256"] for record in row_records] != manifest["rows_sha256"][:8]):
        raise ValueError("first eight cache rows do not match admitted TRAIN ID/order bindings")
    if (_sha_file(manifest_path) != manifest_sha_before
            or _sha_file(ledger_path) != ledger_sha_before):
        raise RuntimeError("cache validation modified source manifest or ledger")
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0178 CPU cache preparation initialized CUDA")
    return {
        "cache_root": str(root.resolve()),
        "manifest_sha256": manifest_sha_before,
        "ledger_sha256": ledger_sha_before,
        "bindings_sha256": manifest["bindings_sha256"],
        "cache_implementation_fingerprint": manifest["bindings"]["cache_implementation_fingerprint"],
        "selected_rows": 8,
        "selected_image_ids": [int(v) for v in ids[:8]],
        "selected_row_file_sha256": [record["row_file_sha256"] for record in row_records],
        "component_spikes": torch.from_numpy(traces),
        "modal_targets": torch.from_numpy(targets),
        "source_cache_status": "complete_read_only_cpu",
        "cuda_initialized": False,
    }


def settled_cache_rho_q(component_spikes: torch.Tensor):
    """Compute the production rho/Q from already-settled cached component traces."""
    if not torch.is_tensor(component_spikes) or component_spikes.ndim != 4:
        raise ValueError("settled cache rows must have shape [B,4,N,512]")
    if (component_spikes.shape[1] != 4 or component_spikes.shape[2] != 1024
            or component_spikes.shape[3] != 512 or component_spikes.dtype != torch.float32):
        raise ValueError("settled cache rows must be float32 [B,4,1024,512]")
    spikes = component_spikes.mean(dim=1)
    from collaborative_test.SW_0178_temporal_class_competition.class_competition import native_component_rho_q
    return native_component_rho_q(spikes, component_spikes, settle=0)
