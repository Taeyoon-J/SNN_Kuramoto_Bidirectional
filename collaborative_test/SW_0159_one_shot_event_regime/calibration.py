"""Pinned source-T calibration and one-shot native threshold initialization."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "collaborative_test/SW_0157_train_gate_credit/results_archive/completed_20261014/0_train_gate_credit_seed1.json"
REPORT_SHA256 = "83e968cade400d54dae56395c42f1bb54443231903cb5f2642fc4f34e5e56b0e"
SOURCE_MARGIN_RMS = (2.7085165787417393, 2.598093824488581,
                     2.623449164156643, 2.2200754439435535)
THRESHOLD_FRACTION = 0.5
BASE_THRESHOLD = 0.06


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_margin_rms(report_path=REPORT):
    path = Path(report_path)
    if sha256_file(path) != REPORT_SHA256:
        raise ValueError("SW0157 calibration report SHA mismatch")
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("experiment") != "SW0157_train_gate_credit_diagnostic"
            or record.get("status") != "complete" or record.get("seed") != 1
            or record.get("optimizer_updates") != 0):
        raise ValueError("calibration is not the completed no-update SW0157 report")
    source = record.get("models", {}).get("source97_T", {})
    if source.get("model_state_unchanged") is not True or source.get("optimizer_updates") != 0:
        raise ValueError("calibration source-T model was not frozen")
    batches = source.get("batches", [])
    expected = [[0, 1, 2, 3], [2048, 2049, 2050, 2051]]
    if [row.get("source_positions") for row in batches] != expected:
        raise ValueError("calibration TRAIN positions differ from registration")
    per_batch = []
    for row in batches:
        values = row.get("event_dynamics", {}).get("per_component", [])
        if len(values) != 4:
            raise ValueError("calibration must contain four component margins")
        vals = [float(item["margin_rms"]) for item in values]
        if not all(math.isfinite(v) and v > 0 for v in vals):
            raise ValueError("source margin RMS must be finite and positive")
        per_batch.append(vals)
    pooled = tuple(math.sqrt(sum(batch[i] ** 2 for batch in per_batch) / len(per_batch))
                   for i in range(4))
    if any(abs(x-y) > 1e-12 for x, y in zip(pooled, SOURCE_MARGIN_RMS)):
        raise ValueError("pinned source-T RMS constants disagree with report")
    return pooled


def one_shot_bias(margin_rms, *, fraction=THRESHOLD_FRACTION):
    """Return b where native threshold .06*exp(b) equals fraction*rms."""
    values = tuple(float(x) for x in margin_rms)
    if len(values) != 4 or not 0 < fraction < 1:
        raise ValueError("expected four RMS values and a registered fraction in (0,1)")
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("margin RMS values must be finite and positive")
    thresholds = tuple(fraction * value for value in values)
    biases = tuple(math.log(value / BASE_THRESHOLD) for value in thresholds)
    if not all(math.isfinite(x) for x in biases):
        raise ValueError("one-shot threshold bias is nonfinite")
    return {"margin_rms": values, "thresholds": thresholds, "bias": biases,
            "fraction": float(fraction), "base_threshold": BASE_THRESHOLD}


def install_bias(wrapped, bias):
    """Copy the four registered initial biases into existing native b parameters."""
    import torch
    value = torch.as_tensor(bias, dtype=wrapped.b.dtype, device=wrapped.b.device)
    if tuple(value.shape) != (4,) or not torch.isfinite(value).all():
        raise ValueError("SW0159 bias must be four finite values")
    with torch.no_grad():
        wrapped.b.copy_(value)
    return wrapped.b
