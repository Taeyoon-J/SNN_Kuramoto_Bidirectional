"""Paired, no-update SW0180 replay on the registered eight TRAIN images."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from collaborative_test.SW_0179_qce_seed_replication import train as train179
from collaborative_test.SW_0179_qce_seed_replication import predict_evaluate as eval179
from collaborative_test.SW_0178_temporal_class_competition import train as train178
from collaborative_test.SW_0178_temporal_class_competition import predict_evaluate as eval178
from collaborative_test.SW_0175_critical_component_binding import train_v2 as native
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136
from collaborative_test.SW_0164_foreground_preserving_partition import (
    foreground_signed as solver,
)
from snn_kuramoto_bidirectional.evaluation import evaluate_patch_masks

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = "SW0180_qce_train_validation_triage"
SEEDS = (0, 1, 2)
POSITIONS = tuple(range(256, 264))
BATCH = 4
STEPS, SETTLE = 1024, 512
TEMPERATURE = 0.1
ENDPOINTS = ("conditional", "standalone")
ARM = "product_q_ce"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(array):
    arr = np.ascontiguousarray(array)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _json_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def implementation_fingerprint():
    dependencies = (
        "collaborative_test/SW_0180_qce_train_validation_triage/DESIGN_KO.md",
        "collaborative_test/SW_0180_qce_train_validation_triage/run.py",
        "collaborative_test/SW_0179_qce_seed_replication/train.py",
        "collaborative_test/SW_0179_qce_seed_replication/training_queue.py",
        "collaborative_test/SW_0179_qce_seed_replication/predict_evaluate.py",
        "collaborative_test/SW_0178_temporal_class_competition/train.py",
        "collaborative_test/SW_0178_temporal_class_competition/training_queue.py",
        "collaborative_test/SW_0178_temporal_class_competition/predict_evaluate.py",
        "collaborative_test/SW_0171_contextual_instance_graph/preflight.py",
        "collaborative_test/SW_0175_critical_component_binding/train_v2.py",
        "collaborative_test/SW_0162_connected_signed_partition/preflight.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0164_foreground_preserving_partition/foreground_signed.py",
        "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
        "snn_kuramoto_bidirectional/evaluation.py",
    )
    files = {}
    for relative in dependencies:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"SW0180 fingerprint dependency missing: {path}")
        files[relative] = sha256_file(path)
    return {"experiment": EXPERIMENT, "files": files,
            "positions": list(POSITIONS), "seeds": list(SEEDS),
            "steps": STEPS, "settle": SETTLE, "batch": BATCH,
            "temperature": TEMPERATURE, "optimizer_updates": 0,
            "ground_truth_scope": "common_SW0171_TRAIN512_modal_gt_only"}


def _assert_cache(prepared):
    arrays = prepared["arrays"]
    ids = list(map(int, prepared["ordered_train_ids"]))
    if len(ids) != 512 or len(set(ids)) != 512:
        raise ValueError("SW0180 requires the admitted ordered TRAIN512 IDs")
    rgb, modal = np.asarray(arrays["rgb"]), np.asarray(arrays["modal_gt"])
    if rgb.dtype != np.uint8 or rgb.shape != (512, 128, 128, 3):
        raise ValueError("SW0180 RGB cache must be native uint8 TRAIN512")
    if modal.dtype != np.int32 or modal.shape != (512, 32, 32) or modal.min() < 0:
        raise ValueError("SW0180 modal GT cache must be nonnegative int32 TRAIN512")
    return {"ids": ids, "rgb": rgb, "modal_gt": modal}


def _terminal_artifact(seed):
    """Validate exactly one completed producer; never reads validation data."""
    if seed == 1:
        trainer, queue = train178, train178._module(
            "collaborative_test.SW_0178_temporal_class_competition.training_queue")
        state_path = queue._state_path(ARM)
        artifact = trainer.OUTPUT_ROOT / f"{ARM}_seed1"
        manifest = trainer.validate_training_artifact(ARM, artifact)
        expected_experiment, expected_steps = "SW0178_training_queue", trainer.UPDATES
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        trainer, queue = train179, train179._module(
            "collaborative_test.SW_0179_qce_seed_replication.training_queue")
        state_path = queue._state_path(ARM, seed)
        artifact = trainer.OUTPUT_ROOT / f"{ARM}_seed{seed}"
        manifest = trainer.validate_training_artifact(ARM, artifact, seed=seed)
        expected_experiment, expected_steps = "SW0179_training_queue", trainer.UPDATES
        state = json.loads(state_path.read_text(encoding="utf-8"))
    manifest_path = artifact / "manifest.json"
    if (state.get("experiment") != expected_experiment or state.get("arm") != ARM
            or state.get("status") != "complete" or state.get("completed_updates") != expected_steps
            or state.get("manifest_sha256") != sha256_file(manifest_path)
            or state.get("validation_data_read") is not False
            or manifest.get("completed_updates") != expected_steps
            or manifest.get("validation_data_read") is not False
            or manifest.get("seed") != seed):
        raise ValueError(f"SW0180 seed {seed} final checkpoint lacks its complete TRAIN-only receipt")
    return {"artifact": artifact, "manifest": manifest, "state": state,
            "state_path": state_path, "manifest_sha256": sha256_file(manifest_path),
            "state_sha256": sha256_file(state_path),
            "checkpoint_sha256": sha256_file(artifact / "checkpoint.pt")}


def prepare_inputs_cpu(seed_subset=SEEDS):
    """CPU-only cache/artifact/factory admission; no validation assets are opened."""
    if torch.cuda.is_initialized():
        raise RuntimeError("SW0180 preparation must precede CUDA initialization")
    selected_seeds = tuple(int(seed) for seed in seed_subset)
    if not selected_seeds or any(seed not in SEEDS for seed in selected_seeds) or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("SW0180 CPU admission seed subset must be a nonempty subset of {0,1,2}")
    source_inputs = train179.prepare_training_inputs_cpu()
    source_data = _assert_cache(source_inputs)
    seed1_inputs = train178.prepare_training_inputs_cpu() if 1 in selected_seeds else None
    if seed1_inputs is not None:
        seed1_data = _assert_cache(seed1_inputs)
        if (source_data["ids"] != seed1_data["ids"]
                or not np.array_equal(source_data["rgb"], seed1_data["rgb"])
                or not np.array_equal(source_data["modal_gt"], seed1_data["modal_gt"])):
            raise ValueError("SW0180 SW0178/179 common TRAIN512 cache views differ")
    records = {seed: _terminal_artifact(seed) for seed in selected_seeds}
    fresh_sha, final_cpu_sha = {}, {}
    for seed in selected_seeds:
        if seed == 1:
            found, _opt, _families, _binding, _parts = train178._make_fresh(ARM, seed1_inputs, "cpu")
            initial = train178._state_sha(found.wrapped, found.encoder)
            minimal = {"training_inputs": seed1_inputs,
                       "artifacts": {ARM: records[seed]["artifact"]},
                       "manifests": {ARM: records[seed]["manifest"]}}
            loaded, final_sha = eval178._load_model(ARM, minimal, "cpu")
            del found, loaded
        else:
            found, _opt, _families, _binding, _parts = train179._make_fresh(
                ARM, seed, source_inputs, "cpu")
            initial = train179._state_sha(found.wrapped, found.encoder)
            minimal = {"training_inputs": source_inputs,
                       "artifacts": {ARM: records[seed]["artifact"]},
                       "manifests": {ARM: records[seed]["manifest"]}}
            loaded, final_sha = eval179._load_model(ARM, seed, minimal, "cpu")
            del found, loaded
        if initial != records[seed]["manifest"].get("initial_model_state_sha256"):
            raise ValueError(f"SW0180 seed {seed} fresh SourceT+zero-context state differs")
        if final_sha != records[seed]["manifest"].get("final_model_state_sha256"):
            raise ValueError(f"SW0180 seed {seed} strict CPU checkpoint load differs")
        fresh_sha[seed], final_cpu_sha[seed] = initial, final_sha
        if torch.cuda.is_initialized():
            raise RuntimeError("SW0180 CPU admission unexpectedly initialized CUDA")
    if 1 in fresh_sha and fresh_sha[1] != "f372b3a450ebeaf0d49e20d019d3c0676d4998cc1c49586e4c4e4c89e969d7f5":
        raise ValueError("SW0180 seed-1 initial state no longer matches SW0178")
    return {"source_inputs": source_inputs, "seed1_inputs": seed1_inputs,
            "selected_seeds": list(selected_seeds),
            "arrays": source_data, "records": records,
            "fresh_state_sha256": fresh_sha, "final_cpu_state_sha256": final_cpu_sha,
            "cache_proof": {"dataset_manifest_sha256": source_inputs["dataset_manifest_sha256"],
                "dataset_arrays_sha256": source_inputs["dataset_arrays_sha256"],
                "ordered_train_ids": source_data["ids"],
                "rgb_sha256": _sha_array(source_data["rgb"]),
                "modal_gt_sha256": _sha_array(source_data["modal_gt"])},
            "cuda_initialized": False}


def _paired_seed_rng(seed, batch_index):
    value = 180017 + int(seed) * 100 + int(batch_index)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _class_loss_and_margin(affinity, labels):
    """Registered LOO class-balanced QCE plus same-prototype true-vs-best margin."""
    losses, margins = [], []
    for image in range(affinity.shape[0]):
        q, y = affinity[image], labels[image]
        classes, inverse, counts = torch.unique(y, sorted=True, return_inverse=True,
                                                return_counts=True)
        if classes.numel() < 2:
            losses.append(q.sum() * 0.0); margins.append(q.sum() * 0.0); continue
        k = int(classes.numel())
        membership = torch.nn.functional.one_hot(inverse, num_classes=k).to(q.dtype)
        totals = q @ membership
        means = totals / counts.to(q.dtype).unsqueeze(0)
        own_count = counts.index_select(0, inverse)
        own_total = totals.gather(1, inverse[:, None]).squeeze(1)
        own = (own_total - torch.diagonal(q)) / (own_count - 1).clamp_min(1).to(q.dtype)
        logits = means.scatter(1, inverse[:, None], own[:, None])
        valid = own_count >= 2
        if not bool(valid.any()):
            losses.append(q.sum() * 0.0); margins.append(q.sum() * 0.0); continue
        scaled = logits / TEMPERATURE
        per_query = torch.nn.functional.cross_entropy(scaled, inverse, reduction="none")
        valid_classes = inverse[valid]
        class_sums = torch.zeros(k, device=q.device, dtype=q.dtype).scatter_add_(
            0, valid_classes, per_query[valid])
        class_counts = torch.zeros(k, device=q.device, dtype=q.dtype).scatter_add_(
            0, valid_classes, torch.ones_like(per_query[valid]))
        used = class_counts > 0
        losses.append((class_sums[used] / class_counts[used]).mean())
        competing = scaled.masked_fill(torch.nn.functional.one_hot(inverse, k).bool(), -torch.inf)
        per_margin = scaled.gather(1, inverse[:, None]).squeeze(1) - competing.max(dim=1).values
        class_margin_sums = torch.zeros(k, device=q.device, dtype=q.dtype).scatter_add_(
            0, valid_classes, per_margin[valid])
        margins.append((class_margin_sums[used] / class_counts[used]).mean())
    return torch.stack(losses), torch.stack(margins)


def _metric_arrays(predictions, targets):
    pred = torch.from_numpy(np.ascontiguousarray(predictions)).to(torch.int64)
    target = torch.from_numpy(np.ascontiguousarray(targets)).to(torch.int64)
    measured = evaluate_patch_masks(pred, target)
    result = {name: measured["per_image"][name].detach().cpu().numpy().astype(float).tolist()
              for name in ("fg_ari", "foreground_iou", "matched_object_iou")}
    for name, values in result.items():
        if len(values) != len(predictions) or not np.isfinite(values).all():
            raise ValueError(f"SW0180 {name} metrics are nonfinite or incomplete")
    gt_fg, pr_fg = targets > 0, predictions > 0
    tp = np.logical_and(gt_fg, pr_fg).sum(axis=(1, 2)).astype(np.float64)
    predicted = pr_fg.sum(axis=(1, 2)).astype(np.float64)
    actual = gt_fg.sum(axis=(1, 2)).astype(np.float64)
    result["foreground_precision"] = (tp / np.maximum(predicted, 1)).tolist()
    result["foreground_recall"] = (tp / np.maximum(actual, 1)).tolist()
    return result


def _summarize_pair(source_rows, final_rows, targets):
    if len(source_rows) != 2 or len(final_rows) != 2:
        raise ValueError("SW0180 paired summary requires two B4 batches per model")
    for rows in (source_rows, final_rows):
        if any(len(row["q_loss"]) != BATCH or len(row["true_minus_best_other_margin"]) != BATCH
               for row in rows):
            raise ValueError("SW0180 paired summary has incomplete B4 diagnostics")
    qce_source = [x for row in source_rows for x in row["q_loss"]]
    qce_final = [x for row in final_rows for x in row["q_loss"]]
    margin_source = [x for row in source_rows for x in row["true_minus_best_other_margin"]]
    margin_final = [x for row in final_rows for x in row["true_minus_best_other_margin"]]
    source_masks = {ep: np.concatenate([row["outputs"][ep] for row in source_rows]) for ep in ENDPOINTS}
    final_masks = {ep: np.concatenate([row["outputs"][ep] for row in final_rows]) for ep in ENDPOINTS}
    source_qcc = np.concatenate([row["qcc"] for row in source_rows])
    final_qcc = np.concatenate([row["qcc"] for row in final_rows])
    source_metrics = {ep: _metric_arrays(source_masks[ep], targets) for ep in ENDPOINTS}
    final_metrics = {ep: _metric_arrays(final_masks[ep], targets) for ep in ENDPOINTS}
    ce_delta = float(np.mean(qce_final) - np.mean(qce_source))
    fg_delta = float(np.mean(final_metrics["standalone"]["fg_ari"])
                      - np.mean(source_metrics["standalone"]["fg_ari"]))
    if ce_delta > 0 and fg_delta < 0:
        interpretation = "TRAIN CE rose and standalone FG-ARI fell; consistent with optimization/representation instability."
    elif ce_delta < 0 and fg_delta < 0:
        interpretation = "TRAIN CE fell while standalone FG-ARI fell; consistent with objective/readout mismatch."
    elif ce_delta < 0 and fg_delta > 0:
        interpretation = "TRAIN CE and standalone FG-ARI improved; compare separately frozen fixed320 scores for generalization evidence."
    else:
        interpretation = "TRAIN8 changes are mixed or tied; no diagnostic branch is supported."
    return {"qce_per_image": {"fresh": qce_source, "final": qce_final},
        "true_minus_best_other_margin_per_image": {"fresh": margin_source, "final": margin_final},
        "readout_metrics": {"fresh": source_metrics, "final": final_metrics},
        "qcc_positive_count": {"fresh": [int(np.unique(x[x > 0]).size) for x in source_qcc],
                               "final": [int(np.unique(x[x > 0]).size) for x in final_qcc]},
        "graph_gain": {"fresh": [float(x) for row in source_rows for x in row["gain"]],
                       "final": [float(x) for row in final_rows for x in row["gain"]]},
        "qce_delta_mean": ce_delta,
        "margin_delta_mean": float(np.mean(margin_final) - np.mean(margin_source)),
        "readout_metric_delta": {ep: {metric: float(np.mean(final_metrics[ep][metric])
            - np.mean(source_metrics[ep][metric])) for metric in source_metrics[ep]}
            for ep in ENDPOINTS},
        "triage_interpretation": interpretation}


@torch.no_grad()
def _forward_batch(found, rgb, target, conditional_support, *, seed, batch_index):
    _paired_seed_rng(seed, batch_index)
    gamma, trace, actual_a, _z, q, rho = native._native_forward(
        found, rgb, live_tail_steps=0)
    if q.shape != (BATCH, 1024, 1024) or rho.shape != (BATCH, 4, 1024, 1024):
        raise ValueError("SW0180 native production Q/rho geometry changed")
    qcc = np.asarray(sw136._qcc_labels(trace), dtype=np.int32)
    if qcc.shape != (BATCH, 32, 32) or not np.issubdtype(qcc.dtype, np.integer):
        raise ValueError("SW0180 native QCC labels are malformed")
    labels = target.reshape(BATCH, -1).to(device=q.device, dtype=torch.long)
    q_losses, margins = _class_loss_and_margin(q, labels)
    qcc_np = qcc.copy()
    outputs = {"conditional": [], "standalone": []}
    gains = []
    graph_gain = float(torch.exp(found.wrapped.core.graph_generator.log_coupling_gain.detach()).cpu())
    if not np.isfinite(graph_gain) or graph_gain <= 0:
        raise FloatingPointError("SW0180 trained graph coupling gain is invalid")
    if conditional_support is None:
        conditional_support = qcc_np
    for row in range(BATCH):
        q_np = q[row].detach().cpu().numpy().astype(np.float64, copy=True)
        a_np = actual_a[row].detach().cpu().numpy().astype(np.float64, copy=True)
        for endpoint, support in (("conditional", conditional_support[row]),
                                  ("standalone", qcc_np[row])):
            merged = solver.foreground_induced_merge(q_np, a_np, graph_gain,
                np.asarray(support, dtype=np.int32), grid_shape=(32, 32))
            mask = np.asarray(merged["labels"], dtype=np.int32)
            if mask.shape != (32, 32) or not np.array_equal(mask > 0, np.asarray(support) > 0):
                raise ValueError(f"SW0180 {endpoint} mask changed its designated support")
            outputs[endpoint].append(mask)
        gains.append(graph_gain)
    return {"q_loss": q_losses.detach().cpu().tolist(),
            "true_minus_best_other_margin": margins.detach().cpu().tolist(),
            "qcc": qcc_np, "outputs": {key: np.stack(value) for key, value in outputs.items()},
            "gain": gains, "gamma_sha256": _sha_array(gamma.detach().cpu().numpy()),
            "q_sha256": _sha_array(q.detach().cpu().numpy()),
            "rho_sha256": _sha_array(rho.detach().cpu().numpy()),
            "actual_a_sha256": _sha_array(actual_a.detach().cpu().numpy()),
            "trace_sha256": {name: _sha_array(trace[name].detach().cpu().numpy())
                             for name in ("theta", "membrane", "spikes", "component_membrane",
                                          "component_spikes", "component_gates")}}


def run_diagnostic(prepared, *, seed_subset=SEEDS, device="cuda:0"):
    """Run the registered 48 TRAIN forwards, without optimizer steps or VAL reads."""
    if not isinstance(prepared, dict) or prepared.get("cuda_initialized") is not False:
        raise ValueError("SW0180 requires validated CPU-prepared TRAIN inputs")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0180 paired replay requires a leased CUDA device")
    selected_seeds = tuple(int(seed) for seed in seed_subset)
    if not selected_seeds or any(seed not in SEEDS for seed in selected_seeds) or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("SW0180 seed subset must be a nonempty unique subset of {0,1,2}")
    if not torch.cuda.is_initialized():
        torch.cuda.set_device(dev)
    inputs = prepared["source_inputs"]
    if not set(selected_seeds).issubset(set(prepared.get("selected_seeds", SEEDS))):
        raise ValueError("SW0180 requested seed was not admitted by the CPU preparation")
    arrays = prepared["arrays"]
    targets = np.asarray(arrays["modal_gt"])[list(POSITIONS)]
    rgb_data = np.asarray(arrays["rgb"])
    ids = [prepared["cache_proof"]["ordered_train_ids"][i] for i in POSITIONS]
    result = {"experiment": EXPERIMENT, "status": "complete", "seeds": {},
        "image_positions": list(POSITIONS), "image_ids": ids,
        "cache_proof": prepared["cache_proof"],
        "implementation_fingerprint": implementation_fingerprint(),
        "steps": STEPS, "settle": SETTLE, "live_tail": 0,
        "optimizer_updates": 0, "ground_truth_scope": "TRAIN modal labels positions 256..263 only",
        "validation_data_read": False, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    for seed in selected_seeds:
        record = prepared["records"][seed]
        manifest = record["manifest"]
        if seed == 1:
            factory_inputs = prepared["seed1_inputs"]
            fresh_found, _opt, _families, fresh_binding, _parts = train178._make_fresh(ARM, factory_inputs, dev)
            initial_sha = train178._state_sha(fresh_found.wrapped, fresh_found.encoder)
            final_prepared = {"training_inputs": factory_inputs,
                "artifacts": {ARM: record["artifact"]}, "manifests": {ARM: manifest}}
            final_found, final_sha = eval178._load_model(ARM, final_prepared, dev)
        else:
            factory_inputs = inputs
            fresh_found, _opt, _families, fresh_binding, _parts = train179._make_fresh(ARM, seed, factory_inputs, dev)
            initial_sha = train179._state_sha(fresh_found.wrapped, fresh_found.encoder)
            final_prepared = {"training_inputs": factory_inputs,
                "artifacts": {ARM: record["artifact"]}, "manifests": {ARM: manifest}}
            final_found, final_sha = eval179._load_model(ARM, seed, final_prepared, dev)
        if initial_sha != prepared["fresh_state_sha256"][seed]:
            raise ValueError(f"SW0180 seed {seed} initial source state differs from CPU admission")
        if final_sha != prepared["final_cpu_state_sha256"][seed]:
            raise ValueError(f"SW0180 seed {seed} final model differs from CPU strict-load proof")
        conditional_support = []
        before_state_sha = initial_sha
        source_rows, final_rows = [], []
        for batch_index, offset in enumerate((0, 4)):
            pos = list(POSITIONS[offset:offset + 4])
            rgb = torch.from_numpy(np.ascontiguousarray(rgb_data[pos])).to(dev)
            gt = torch.from_numpy(np.ascontiguousarray(targets[offset:offset + 4])).to(dev)
            src = _forward_batch(fresh_found, rgb, gt, None,
                seed=seed, batch_index=batch_index)
            conditional_support.extend(src["qcc"])
            source_rows.append(src)
            fin = _forward_batch(final_found, rgb, gt, np.asarray(conditional_support[-BATCH:], dtype=np.int32),
                seed=seed, batch_index=batch_index)
            final_rows.append(fin)
            del rgb, gt
        fresh_after = (train178._state_sha(fresh_found.wrapped, fresh_found.encoder)
                       if seed == 1 else train179._state_sha(fresh_found.wrapped, fresh_found.encoder))
        final_after = (train178._state_sha(final_found.wrapped, final_found.encoder)
                       if seed == 1 else train179._state_sha(final_found.wrapped, final_found.encoder))
        if fresh_after != before_state_sha or final_after != final_sha:
            raise ValueError(f"SW0180 seed {seed} model state mutated during no-grad replay")
        gt_np = targets.copy()
        summary = _summarize_pair(source_rows, final_rows, gt_np)
        result["seeds"][str(seed)] = {
            "source_model_state_sha256": before_state_sha,
            "final_model_state_sha256": final_sha,
            "training_manifest_sha256": record["manifest_sha256"],
            "training_state_sha256": record["state_sha256"],
            "training_checkpoint_sha256": record["checkpoint_sha256"],
            "source_binding": fresh_binding.get("source_binding", fresh_binding.get("source_proof", fresh_binding)),
            "initial_model_state_sha256": manifest.get("initial_model_state_sha256"),
            "fresh": {"forward_digests": [{k: r[k] for k in ("gamma_sha256", "q_sha256", "rho_sha256", "actual_a_sha256", "trace_sha256")} for r in source_rows]},
            "final": {"forward_digests": [{k: r[k] for k in ("gamma_sha256", "q_sha256", "rho_sha256", "actual_a_sha256", "trace_sha256")} for r in final_rows]},
            "state_unchanged": {"fresh": fresh_after == before_state_sha, "final": final_after == final_sha},
            **summary,
        }
        del fresh_found, final_found
    if not all(row["state_unchanged"][k] for row in result["seeds"].values() for k in ("fresh", "final")):
        raise AssertionError("SW0180 model immutability check failed")
    return result


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        if temp.exists(): temp.unlink()
