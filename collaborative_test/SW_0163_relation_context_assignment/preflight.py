"""Two-update disposable head admission on the complete frozen-source cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np
import torch

from collaborative_test.SW_0161_supervised_spike_partition import cache as source_cache
from collaborative_test.SW_0161_supervised_spike_partition import objective, train as source_train
from collaborative_test.SW_0163_relation_context_assignment import context_cache, model

OUTPUT = HERE / "results_archive" / "preflight_20261014" / "head_seed1.json"
ARMS = model.ARMS
TOLERANCE = 2.0e-5


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint():
    rels = (
        "collaborative_test/SW_0163_relation_context_assignment/preflight.py",
        "collaborative_test/SW_0163_relation_context_assignment/context_cache.py",
        "collaborative_test/SW_0163_relation_context_assignment/model.py",
        "collaborative_test/SW_0163_relation_context_assignment/protocol.json",
        "collaborative_test/SW_0161_supervised_spike_partition/cache.py",
        "collaborative_test/SW_0161_supervised_spike_partition/objective.py",
        "collaborative_test/SW_0161_supervised_spike_partition/train.py",
    )
    result = {}
    for rel in rels:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"SW0163 preflight dependency missing: {rel}")
        result[rel] = sha256_file(path)
    return result


def _upstream_from_prepared(prepared, cache_root):
    trace_manifest = prepared["source_trace_manifest"]
    context_manifest = context_cache.validate_final_cache(
        cache_root, expected_bindings=prepared["context_bindings"])
    if trace_manifest.get("count") != 512 or context_manifest.get("count") != 512:
        raise ValueError("SW0163 preflight requires complete 512-row source and Q/A caches")
    return {
        "experiment": "SW0163_relation_context_assignment",
        "context_cache_manifest_sha256": context_manifest["manifest_sha256"],
        "context_cache_bindings_sha256": context_manifest["bindings_sha256"],
        "source_trace_manifest_sha256": sha256_file(source_cache.CACHE_ROOT / "manifest.json"),
        "source_trace_bindings_sha256": trace_manifest["bindings_sha256"],
        "source_checkpoint_sha256": prepared["context_bindings"]["source_checkpoint_sha256"],
        "source_manifest_sha256": prepared["context_bindings"]["source_manifest_sha256"],
        "ordered_image_ids": prepared["context_bindings"]["ordered_image_ids"],
        "context_implementation_fingerprint": context_cache._fingerprint(),
        "preflight_implementation_fingerprint": implementation_fingerprint(),
        "source_model_state_sha256": context_manifest.get("source_model_state_before_sha256"),
        "source_model_unchanged": context_manifest.get("source_model_unchanged") is True,
    }


def validate_upstream(cache_root=context_cache.CACHE_ROOT):
    """Deeply validate source/cache/contexts before an owner acquires a GPU."""
    return _upstream_from_prepared(context_cache.prepare_context_inputs_cpu(), cache_root)


def _state_sha(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _norm(parameters):
    values = [p.grad for p in parameters if p.grad is not None]
    if not values:
        return 0.0, True
    finite = all(bool(torch.isfinite(g).all().detach().cpu()) for g in values)
    square = sum(float(g.detach().double().square().sum().cpu()) for g in values)
    return math.sqrt(square), finite


def _changed(before, parameters):
    delta = 0.0
    for (old_name, old), (name, new) in zip(before.items(),
                                            ((n, p.detach()) for n, p in parameters)):
        if old_name != name:
            raise AssertionError("parameter order changed during SW0163 disposable update")
        delta += float((old.cpu().double() - new.detach().cpu().double()).square().sum())
    return math.sqrt(delta)


def _wellformed_arm(arm):
    if not isinstance(arm, dict) or arm.get("updates_run") != 2:
        return False
    if not isinstance(arm.get("initial_model_sha256"), str) or len(arm["initial_model_sha256"]) != 64:
        return False
    updates = arm.get("updates")
    if not isinstance(updates, list) or len(updates) != 2:
        return False
    for i, row in enumerate(updates, 1):
        if not isinstance(row, dict) or row.get("update") != i:
            return False
        for key in ("loss", "gradient_norm", "row_sum_error", "parameter_delta"):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                return False
        if (row["loss"] < 0 or row["gradient_norm"] < 0 or row["parameter_delta"] < 0
                or row["row_sum_error"] < 0):
            return False
        if row.get("finite") is not True or row["row_sum_error"] > TOLERANCE:
            return False
        norms = row.get("gradient_norms")
        if not isinstance(norms, dict) or set(norms) != {"binder", "relation_output", "relation_hidden"}:
            return False
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
               for v in norms.values()):
            return False
    return True


def validate_report_record(report, *, expected):
    if not isinstance(report, dict):
        raise ValueError("SW0163 preflight receipt must be an object")
    fields = ("experiment", "context_cache_manifest_sha256", "context_cache_bindings_sha256",
              "source_trace_manifest_sha256", "source_trace_bindings_sha256",
              "source_checkpoint_sha256", "source_manifest_sha256", "ordered_image_ids",
              "context_implementation_fingerprint", "preflight_implementation_fingerprint",
              "source_model_state_sha256", "source_model_unchanged")
    for field in fields:
        if field not in expected or report.get(field) != expected[field]:
            raise ValueError(f"SW0163 preflight upstream mismatch: {field}")
    if (report.get("status") != "complete" or report.get("disposable_updates") != 2
            or report.get("ground_truth_used") is not True
            or report.get("ground_truth_scope") != "registered sixteen TRAIN modal32 targets"
            or report.get("source_model_unchanged") is not True
            or report.get("frozen_caches_unchanged") is not True
            or report.get("context_cache_manifest_sha256_before") != expected["context_cache_manifest_sha256"]
            or report.get("context_cache_manifest_sha256_after") != expected["context_cache_manifest_sha256"]
            or report.get("source_trace_manifest_sha256_before") != expected["source_trace_manifest_sha256"]
            or report.get("source_trace_manifest_sha256_after") != expected["source_trace_manifest_sha256"]):
        raise ValueError("SW0163 preflight scope or source-integrity receipt is invalid")
    arms = report.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("SW0163 preflight must contain exactly its four registered arms")
    if not all(_wellformed_arm(arms[name]) for name in ARMS):
        raise ValueError("SW0163 disposable update evidence is malformed")
    if len({arms[name]["initial_model_sha256"] for name in ARMS}) != 1:
        raise ValueError("SW0163 arms did not start from the same initialization")
    admitted = all(all(update["finite"] and update["gradient_norm"] > 0
                       and update["parameter_delta"] > 0
                       and update["gradient_norms"]["binder"] > 0
                       and update["gradient_norms"]["relation_output"] > 0
                       for update in arms[name]["updates"])
                   and arms[name]["updates"][1]["gradient_norms"]["relation_hidden"] > 0
                   for name in ARMS)
    if report.get("training_admission") is not admitted:
        raise ValueError("SW0163 training admission disagrees with its measured evidence")
    return report


def run_preflight(*, device="cuda:0", output=OUTPUT, cache_root=context_cache.CACHE_ROOT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserving existing SW0163 preflight receipt: {output}")
    prepared = context_cache.prepare_context_inputs_cpu()
    upstream = _upstream_from_prepared(prepared, cache_root)
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0163 disposable preflight requires its exclusive CUDA lease")
    torch.cuda.set_device(dev)
    ctx = context_cache.AtomicContextCache(cache_root, prepared["context_bindings"])
    source_manifest = prepared["source_trace_manifest"]
    source_store = source_cache._AtomicTraceCache(
        source_cache.CACHE_ROOT, source_manifest["bindings"],
        source_manifest["bindings"]["image_ids"], source_manifest["bindings"]["image_rgb_sha256"])
    indices = list(range(16))
    trace_micros, target_micros = source_train.load_microbatches(source_store, indices, dev)
    q_rows, a_rows = [], []
    for index in indices:
        q, a = ctx.load(index)
        q_rows.append(q); a_rows.append(a)
    q_batch = torch.from_numpy(np.stack(q_rows)).to(dev)
    a_batch = torch.from_numpy(np.stack(a_rows)).to(dev)
    traces = torch.cat(trace_micros, dim=0)
    targets = torch.cat(target_micros, dim=0)
    cache_manifest_before = sha256_file(context_cache.CACHE_ROOT / "manifest.json")
    source_manifest_before = sha256_file(source_cache.CACHE_ROOT / "manifest.json")

    models = {}
    initial = None
    initial_outputs = {}
    for arm in ARMS:
        candidate = model.RelationContextAssignment(arm).to(dev)
        digest = _state_sha(candidate)
        if initial is None:
            initial = digest
        elif digest != initial:
            raise AssertionError("SW0163 paired arms did not share exact initial parameters")
        with torch.no_grad():
            p, _, _ = candidate(traces, q_batch, a_batch)
        initial_outputs[arm] = p.detach().cpu()
        models[arm] = candidate
    for arm in ARMS[1:]:
        if not torch.equal(initial_outputs[ARMS[0]], initial_outputs[arm]):
            raise AssertionError("zero-initialized relation messages changed raw initialization")

    records = {}
    for arm in ARMS:
        candidate = models[arm]
        optimizer = torch.optim.Adam(candidate.parameters(), lr=3e-4)
        named = dict(candidate.named_parameters())
        arm_updates = []
        for update_index in (1, 2):
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            max_row_error = 0.0
            finite = True
            # Identical sixteen TRAIN examples, in four fixed microbatches.
            for start in range(0, 16, 4):
                p, _, _ = candidate(traces[start:start+4], q_batch[start:start+4],
                                     a_batch[start:start+4])
                loss, _ = objective.mask_hungarian_loss(p, targets[start:start+4])
                if loss.ndim != 0 or not bool(torch.isfinite(loss).detach().cpu()):
                    finite = False
                    raise FloatingPointError("SW0163 disposable mask loss is nonfinite")
                (loss / 4).backward()
                batch_losses.append(float(loss.detach().cpu()))
                max_row_error = max(max_row_error, float((p.sum(-1)-1).abs().max().cpu()))
                finite = finite and bool(torch.isfinite(p).all().detach().cpu())
            output_ids = {id(candidate.relation_mlp[-1].weight), id(candidate.relation_mlp[-1].bias)}
            groups = {
                "binder": list(candidate.binder.parameters()),
                "relation_output": [candidate.relation_mlp[-1].weight,
                                    candidate.relation_mlp[-1].bias],
                "relation_hidden": [p for name, p in named.items()
                                    if name.startswith("relation_mlp") and id(p) not in output_ids],
            }
            norms = {}
            for group, parameters in groups.items():
                norms[group], grads_finite = _norm(parameters)
                finite = finite and grads_finite
            if not finite:
                raise FloatingPointError("SW0163 disposable gradients/probabilities are nonfinite")
            before = {name: p.detach().cpu().clone() for name, p in candidate.named_parameters()}
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
            optimizer.step()
            delta = _changed(before, candidate.named_parameters())
            with torch.no_grad():
                check_p, _, _ = candidate(traces[:4], q_batch[:4], a_batch[:4])
            err = max(max_row_error, float((check_p.sum(-1)-1).abs().max().cpu()))
            arm_updates.append({"update": update_index, "loss": float(np.mean(batch_losses)),
                "gradient_norm": math.sqrt(sum(v*v for v in norms.values())),
                "gradient_norms": norms, "parameter_delta": delta,
                "row_sum_error": err, "finite": finite and bool(torch.isfinite(check_p).all().cpu())})
        records[arm] = {"updates_run": 2, "initial_model_sha256": initial,
                        "updates": arm_updates}

    frozen_caches_unchanged = (
        sha256_file(context_cache.CACHE_ROOT / "manifest.json") == cache_manifest_before
        and sha256_file(source_cache.CACHE_ROOT / "manifest.json") == source_manifest_before)
    if not frozen_caches_unchanged:
        raise AssertionError("SW0163 preflight modified frozen cache metadata")
    admitted = all(_wellformed_arm(records[name]) and all(
        row["gradient_norm"] > 0 and row["parameter_delta"] > 0
        and row["gradient_norms"]["binder"] > 0
        and row["gradient_norms"]["relation_output"] > 0
        for row in records[name]["updates"])
        and records[name]["updates"][1]["gradient_norms"]["relation_hidden"] > 0
        for name in ARMS)
    report = {**upstream, "status": "complete", "disposable_updates": 2,
        "ground_truth_used": True, "ground_truth_scope": "registered sixteen TRAIN modal32 targets",
        "frozen_caches_unchanged": frozen_caches_unchanged,
        "training_admission": admitted,
        "arms": records, "cache_manifest_sha256_before": cache_manifest_before,
        "cache_manifest_sha256_after": sha256_file(context_cache.CACHE_ROOT / "manifest.json"),
        "context_cache_manifest_sha256_before": cache_manifest_before,
        "context_cache_manifest_sha256_after": sha256_file(context_cache.CACHE_ROOT / "manifest.json"),
        "source_trace_manifest_sha256_before": source_manifest_before,
        "source_trace_manifest_sha256_after": sha256_file(source_cache.CACHE_ROOT / "manifest.json"),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev))}
    if not _wellformed_receipt(report, upstream):
        raise AssertionError("SW0163 preflight generated an internally invalid receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + f".{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return report


def _wellformed_receipt(report, upstream):
    try:
        validate_report_record(report, expected=upstream)
        return True
    except (ValueError, TypeError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    result = run_preflight(device=args.device, output=args.output)
    print(json.dumps({"status": result["status"], "training_admission": result["training_admission"],
        "arms": list(result["arms"]), "ground_truth_used": result["ground_truth_used"],
        "output": str(args.output)}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
