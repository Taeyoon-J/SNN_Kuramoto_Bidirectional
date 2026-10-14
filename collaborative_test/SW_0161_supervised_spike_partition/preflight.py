"""Disposable two-update admission check against the sealed SW161 TRAIN cache."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np  # NumPy precedes torch on the local Windows runtime.
import torch

from collaborative_test.SW_0135_native32_spike_binding.binder import NativeSpikeSlotBinder
from collaborative_test.SW_0152_source_preserving_partition_equivariance import training as source_training
from collaborative_test.SW_0161_supervised_spike_partition import cache, train

EXPERIMENT = train.EXPERIMENT
PREFLIGHT_OUTPUT = HERE / "results_archive" / "preflight_20261014" / "preflight.json"
TOLERANCE = 1.0e-5


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    rels = (
        "collaborative_test/SW_0161_supervised_spike_partition/protocol.json",
        "collaborative_test/SW_0161_supervised_spike_partition/objective.py",
        "collaborative_test/SW_0161_supervised_spike_partition/cache.py",
        "collaborative_test/SW_0161_supervised_spike_partition/train.py",
        "collaborative_test/SW_0161_supervised_spike_partition/preflight.py",
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/binder.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py",
        "collaborative_test/SW_0152_source_preserving_partition_equivariance/training.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
    )
    paths = [ROOT / rel for rel in rels]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW161 preflight dependency missing: {missing[0]}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _json_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_upstream(cache_root=train.CACHE_ROOT):
    """Rebuild all expensive/cache lineage before a GPU lease is acquired."""
    manifest = train.validate_cache(cache_root)
    manifest_path = Path(cache_root) / "manifest.json"
    ledger_path = Path(cache_root) / "ledger.json"
    bindings_path = Path(cache_root) / "bindings.json"
    return {
        "experiment": EXPERIMENT,
        "cache_manifest_sha256": sha256_file(manifest_path),
        "cache_ledger_sha256": sha256_file(ledger_path),
        "cache_bindings_file_sha256": sha256_file(bindings_path),
        "cache_bindings_sha256": manifest["bindings_sha256"],
        "cache_implementation_fingerprint": cache.implementation_fingerprint(),
        "train_implementation_fingerprint": train.implementation_fingerprint(),
        "preflight_implementation_fingerprint": implementation_fingerprint(),
        "cache_count": manifest["count"],
        "cache_indices": list(range(train.BATCH)),
        "image_ids": manifest["bindings"]["image_ids"][:train.BATCH],
        "source_model_state_sha256": manifest["source_model_state_before_sha256"],
        "source_model_unchanged": manifest["source_model_unchanged"],
    }


def _arm_passes_admission(arm):
    if not isinstance(arm, dict) or arm.get("updates_run") != 2:
        return False
    if (not isinstance(arm.get("initial_binder_state_sha256"), str)
            or len(arm["initial_binder_state_sha256"]) != 64):
        return False
    if arm.get("finite_probabilities") is not True:
        return False
    if type(arm.get("max_probability_row_sum_error")) not in (int, float):
        return False
    if not math.isfinite(arm["max_probability_row_sum_error"]):
        return False
    if arm["max_probability_row_sum_error"] > TOLERANCE:
        return False
    updates = arm.get("updates")
    if not isinstance(updates, list) or len(updates) != 2:
        return False
    for index, update in enumerate(updates, 1):
        if (not isinstance(update, dict) or update.get("update") != index
                or update.get("finite_loss") is not True
                or update.get("finite_probabilities") is not True
                or type(update.get("max_probability_row_sum_error")) not in (int, float)
                or not math.isfinite(update["max_probability_row_sum_error"])
                or update["max_probability_row_sum_error"] > TOLERANCE):
            return False
        gradients, changes = update.get("gradient_norms"), update.get("parameter_changed")
        if (not isinstance(gradients, dict) or set(gradients) != {"input_projection", "gru"}
                or not isinstance(changes, dict) or set(changes) != {"input_projection", "gru"}):
            return False
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in gradients.values()):
            return False
        if any(value is not True for value in changes.values()):
            return False
    return True


def _arm_record_wellformed(arm):
    """Integrity checks permit a finite scientific rejection to be recorded."""
    if not isinstance(arm, dict) or arm.get("updates_run") != 2:
        return False
    initial = arm.get("initial_binder_state_sha256")
    if not isinstance(initial, str) or len(initial) != 64:
        return False
    if arm.get("finite_probabilities") is not True:
        return False
    error = arm.get("max_probability_row_sum_error")
    if type(error) not in (int, float) or not math.isfinite(error) or error > TOLERANCE:
        return False
    updates = arm.get("updates")
    if not isinstance(updates, list) or len(updates) != 2:
        return False
    for index, update in enumerate(updates, 1):
        if (not isinstance(update, dict) or update.get("update") != index
                or update.get("finite_loss") is not True
                or update.get("finite_probabilities") is not True
                or update.get("gradient_finite") is not True):
            return False
        for key in ("mean_loss", "whole_binder_gradient_norm", "max_probability_row_sum_error"):
            value = update.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                return False
        if update["max_probability_row_sum_error"] > TOLERANCE:
            return False
        gradients, changes = update.get("gradient_norms"), update.get("parameter_changed")
        if (not isinstance(gradients, dict) or set(gradients) != {"input_projection", "gru"}
                or not isinstance(changes, dict) or set(changes) != {"input_projection", "gru"}
                or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                       for value in gradients.values())
                or any(type(value) is not bool for value in changes.values())):
            return False
    return True


def validate_report_record(report, *, expected):
    """Strictly validate integrity and scientific admission without rerunning GPU work."""
    if not isinstance(report, dict):
        raise ValueError("SW161 preflight report must be an object")
    equal_fields = (
        "experiment", "cache_manifest_sha256", "cache_ledger_sha256",
        "cache_bindings_file_sha256", "cache_bindings_sha256",
        "cache_implementation_fingerprint", "train_implementation_fingerprint",
        "preflight_implementation_fingerprint", "cache_count", "cache_indices", "image_ids",
    )
    expected_full = {"experiment": EXPERIMENT, **expected}
    for field in equal_fields:
        if field not in expected_full or report.get(field) != expected_full[field]:
            raise ValueError(f"SW161 preflight upstream binding mismatch: {field}")
    if (report.get("status") != "complete" or report.get("disposable_updates") != 2
            or report.get("ground_truth_used") is not True
            or report.get("ground_truth_scope") != "registered 16 TRAIN targets from complete source cache"
            or report.get("source_model_unchanged") is not True
            or report.get("source_model_state_sha256") != expected.get("source_model_state_sha256")
            or report.get("cache_unchanged") is not True
            or report.get("cache_manifest_sha256_before") != expected.get("cache_manifest_sha256")
            or report.get("cache_manifest_sha256_after") != expected.get("cache_manifest_sha256")
            or report.get("cache_ledger_sha256_before") != expected.get("cache_ledger_sha256")
            or report.get("cache_ledger_sha256_after") != expected.get("cache_ledger_sha256")
            or report.get("cache_bindings_file_sha256_before") != expected.get("cache_bindings_file_sha256")
            or report.get("cache_bindings_file_sha256_after") != expected.get("cache_bindings_file_sha256")
            or report.get("cache_implementation_fingerprint_before") != expected.get("cache_implementation_fingerprint")
            or report.get("cache_implementation_fingerprint_after") != expected.get("cache_implementation_fingerprint")
            or report.get("train_implementation_fingerprint_before") != expected.get("train_implementation_fingerprint")
            or report.get("train_implementation_fingerprint_after") != expected.get("train_implementation_fingerprint")):
        raise ValueError("SW161 preflight receipt is incomplete or cache/source changed")
    arms = report.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(train.ARMS):
        raise ValueError("SW161 preflight must record both registered arms")
    if not all(_arm_record_wellformed(arms[name]) for name in train.ARMS):
        raise ValueError("SW161 preflight disposable update record is malformed/nonfinite")
    initial_hashes = {arms[name]["initial_binder_state_sha256"] for name in train.ARMS}
    if len(initial_hashes) != 1:
        raise ValueError("SW161 preflight paired arms did not start from identical binder state")
    admitted = all(_arm_passes_admission(arms[name]) for name in train.ARMS)
    if any(type(arms[name].get("training_admission")) is not bool for name in train.ARMS):
        raise ValueError("SW161 per-arm admission summary is malformed")
    if any(arms[name].get("training_admission") is not _arm_passes_admission(arms[name])
           for name in train.ARMS):
        raise ValueError("SW161 per-arm admission summary disagrees with evidence")
    if report.get("training_admission") is not admitted:
        raise ValueError("SW161 preflight admission summary disagrees with arm evidence")
    return report


def _module_hash(module):
    return source_training.state_sha(module)


def _module_grad_norm(module):
    grads = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    if not grads:
        return 0.0, False
    finite = all(bool(torch.isfinite(grad).all().detach().cpu()) for grad in grads)
    norm_sq = sum(float(grad.detach().double().square().sum().cpu()) for grad in grads)
    return math.sqrt(norm_sq), finite


def _probability_check(binder, traces):
    with torch.no_grad():
        probabilities, slots, _ = binder(traces)
        finite = bool(torch.isfinite(probabilities).all().cpu()) and bool(torch.isfinite(slots).all().cpu())
        if not finite or tuple(probabilities.shape) != (traces.shape[0], 1024, 11):
            return False, float("inf")
        row_error = float((probabilities.sum(dim=-1) - 1.0).abs().max().cpu())
        return math.isfinite(row_error), row_error


def run_preflight(*, device="cuda:0", cache_root=train.CACHE_ROOT,
                  output=PREFLIGHT_OUTPUT):
    """Run two throwaway logical updates per arm and publish a create-only receipt."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserving existing SW161 preflight receipt: {output}")
    upstream = validate_upstream(cache_root)
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW161 preflight requires an exclusively assigned CUDA device")
    torch.cuda.set_device(dev)
    manifest = train.validate_cache(cache_root)
    store = cache._AtomicTraceCache(cache_root, manifest["bindings"],
        manifest["bindings"]["image_ids"], manifest["bindings"]["image_rgb_sha256"])
    cache_indices = list(range(train.BATCH))
    image_ids = upstream["image_ids"]
    trace_batches, target_batches = train.load_microbatches(store, cache_indices, dev)
    traces = torch.cat(trace_batches, dim=0)

    cache_manifest_path = Path(cache_root) / "manifest.json"
    cache_ledger_path = Path(cache_root) / "ledger.json"
    cache_manifest_before = sha256_file(cache_manifest_path)
    cache_ledger_before = sha256_file(cache_ledger_path)
    cache_bindings_file_before = sha256_file(Path(cache_root) / "bindings.json")
    cache_fingerprint_before = cache.implementation_fingerprint()
    train_fingerprint_before = train.implementation_fingerprint()
    source_state_sha = upstream["source_model_state_sha256"]
    arms = {}
    shared_initial_hash = None
    for arm in train.ARMS:
        binder = NativeSpikeSlotBinder(seed=train.HEAD_SEED).to(dev)
        binder.train()
        initial_hash = _module_hash(binder)
        if shared_initial_hash is None:
            shared_initial_hash = initial_hash
        initial_finite, initial_error = _probability_check(binder, traces)
        optimizer = train.make_optimizer(binder)
        updates = []
        previous_module_hashes = {
            name: _module_hash(getattr(binder, name)) for name in ("input_projection", "gru")}
        for update_ix in (1, 2):
            update_record = train.logical_update(
                binder=binder, optimizer=optimizer, arm=arm,
                trace_microbatches=trace_batches, target_microbatches=target_batches,
                update_index=update_ix, pass_index=1,
                cache_indices=cache_indices, image_ids=image_ids)
            gradients = {}
            gradient_finite = True
            changed = {}
            for name in ("input_projection", "gru"):
                norm, finite = _module_grad_norm(getattr(binder, name))
                gradients[name] = norm
                gradient_finite = gradient_finite and finite
                current_hash = _module_hash(getattr(binder, name))
                changed[name] = current_hash != previous_module_hashes[name]
                previous_module_hashes[name] = current_hash
            finite_p, row_error = _probability_check(binder, traces)
            updates.append({"update": update_ix,
                "finite_loss": math.isfinite(update_record["mean_loss"]),
                "mean_loss": float(update_record["mean_loss"]),
                "finite_probabilities": finite_p,
                "max_probability_row_sum_error": row_error,
                "gradient_norms": gradients, "gradient_finite": gradient_finite,
                "parameter_changed": changed,
                "whole_binder_gradient_norm": update_record["binder_gradient_norm"]})
        valid = (initial_finite and all(_arm_passes_admission({
            "updates_run": 2, "initial_binder_state_sha256": initial_hash,
            "finite_probabilities": initial_finite,
            "max_probability_row_sum_error": initial_error, "updates": updates}) for _ in (0,)))
        arms[arm] = {"updates_run": 2, "initial_binder_state_sha256": initial_hash,
            "finite_probabilities": initial_finite,
            "max_probability_row_sum_error": initial_error,
            "updates": updates, "training_admission": bool(valid)}
        del binder, optimizer
    del traces, trace_batches, target_batches, store
    cache_manifest_after = sha256_file(cache_manifest_path)
    cache_ledger_after = sha256_file(cache_ledger_path)
    cache_bindings_file_after = sha256_file(Path(cache_root) / "bindings.json")
    cache_fingerprint_after = cache.implementation_fingerprint()
    train_fingerprint_after = train.implementation_fingerprint()
    source_unchanged = (source_state_sha == manifest["source_model_state_after_sha256"]
                        and manifest["source_model_unchanged"] is True)
    cache_unchanged = (cache_manifest_before == cache_manifest_after
                       == upstream["cache_manifest_sha256"]
                       and cache_ledger_before == cache_ledger_after
                       == upstream["cache_ledger_sha256"]
                       and cache_bindings_file_before == cache_bindings_file_after
                       == upstream["cache_bindings_file_sha256"]
                       and cache_fingerprint_before == cache_fingerprint_after
                       == upstream["cache_implementation_fingerprint"]
                       and train_fingerprint_before == train_fingerprint_after
                       == upstream["train_implementation_fingerprint"])
    admission = all(arms[name]["training_admission"] for name in train.ARMS)
    report = {**upstream,
        "status": "complete", "disposable_updates": 2,
        "ground_truth_used": True,
        "ground_truth_scope": "registered 16 TRAIN targets from complete source cache",
        "source_model_unchanged": source_unchanged,
        "cache_unchanged": cache_unchanged,
        "cache_manifest_sha256_before": cache_manifest_before,
        "cache_manifest_sha256_after": cache_manifest_after,
        "cache_ledger_sha256_before": cache_ledger_before,
        "cache_ledger_sha256_after": cache_ledger_after,
        "cache_bindings_file_sha256_before": cache_bindings_file_before,
        "cache_bindings_file_sha256_after": cache_bindings_file_after,
        "cache_implementation_fingerprint_before": cache_fingerprint_before,
        "cache_implementation_fingerprint_after": cache_fingerprint_after,
        "train_implementation_fingerprint_before": train_fingerprint_before,
        "train_implementation_fingerprint_after": train_fingerprint_after,
        "arms": arms, "training_admission": bool(admission and source_unchanged and cache_unchanged),
    }
    # Integrity is checked even for a completed scientific rejection. A zero
    # gradient is retained as evidence and closes training without rerunning.
    validate_report_record(report, expected=upstream)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temp, output)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-root", type=Path, default=train.CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=PREFLIGHT_OUTPUT)
    args = parser.parse_args(argv)
    report = run_preflight(device=args.device, cache_root=args.cache_root, output=args.output)
    print(json.dumps({"status": report["status"],
                      "training_admission": report["training_admission"],
                      "report_sha256": sha256_file(args.output)}, sort_keys=True), flush=True)
    return 0 if report["training_admission"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
