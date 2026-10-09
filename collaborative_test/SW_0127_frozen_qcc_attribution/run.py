"""Read-only four-arm SW0126 checkpoint attribution with the original QCC readout.

All predictions are saved and hash-verified before the CLEVR masks are opened.
This is an attribution diagnostic, not a training run or evidence that any
individual SW0126 mechanism caused a score change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np  # Import before torch in the shared Windows OpenMP runtime.
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
IDS = tuple(range(1320, 1640))
BATCH = 8
STEPS = 1024
SETTLE = 512
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
DATASET = Path("/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5")
ARMS = ("source97_native", "source97_untrained_history_adapter",
        "sw0126_history_event", "sw0126_gate_only")

sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0115_analytic_partition_rgb import loss as rgb_loss
from collaborative_test.SW_0117_joint_analytic_rgb import run as sw117
from collaborative_test.SW_0118_allowed_patch_labels import run as sw118
from collaborative_test.SW_0122_joint_rgb_seed_replication import run as sw122
from collaborative_test.SW_0126_history_event_binding import pilot_run, screen
from collaborative_test.SW_0126_history_event_binding.history_event import attach_history_event_membrane
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _implementation_fingerprint() -> dict:
    paths = (
        HERE / "run.py", HERE / "protocol.json",
        ROOT / "collaborative_test/SW_0110_xy_graph_route/run.py",
        ROOT / "collaborative_test/SW_0115_analytic_partition_rgb/loss.py",
        sw117.RUNNER,
        ROOT / "collaborative_test/SW_0118_allowed_patch_labels/run.py",
        ROOT / "collaborative_test/SW_0122_joint_rgb_seed_replication/run.py",
        ROOT / "collaborative_test/SW_0125_late_rollout_credit/late_rollout.py",
        ROOT / "collaborative_test/SW_0126_history_event_binding/screen.py",
        ROOT / "collaborative_test/SW_0126_history_event_binding/pilot_run.py",
        ROOT / "collaborative_test/SW_0126_history_event_binding/pilot_rollout.py",
        ROOT / "collaborative_test/SW_0126_history_event_binding/history_event.py",
        ROOT / "snn_kuramoto_bidirectional/s2net_cls.py",
        ROOT / "snn_kuramoto_bidirectional/membrane_layer.py",
        ROOT / "snn_kuramoto_bidirectional/dendric_layer.py",
        ROOT / "snn_kuramoto_bidirectional/kuramoto_layer.py",
        ROOT / "snn_kuramoto_bidirectional/sinusoidal_gating.py",
        ROOT / "snn_kuramoto_bidirectional/spike_classifier.py",
        ROOT / "snn_kuramoto_bidirectional/evaluation.py",
        ROOT / "collaborative_test/SW_0040_peer_transfer/evaluate.py",
    )
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in paths}


def _source_reference():
    ref = sw122.validate_source_reference(1)
    report = json.loads(Path(ref["source_evaluation"]).read_text(encoding="utf-8"))
    try:
        score = report["sweep"][0]["scored_targets"]["our_hdf5"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("registered SW0097 native evaluation schema changed") from exc
    if report.get("ground_truth_used_for_prediction") is not False:
        raise ValueError("SW0097 native reference must have prediction GT=false")
    values = {}
    for metric in METRICS:
        row = score.get("per_image", {}).get(metric)
        if (not isinstance(row, list) or len(row) != len(IDS)
                or score.get("valid_count", {}).get(metric) != len(IDS)
                or not math.isfinite(float(score.get("metrics", {}).get(metric, float("nan"))))
                or not all(math.isfinite(float(v)) for v in row)):
            raise ValueError(f"invalid SW0097 native reference metric {metric}")
        values[metric] = [float(v) for v in row]
    return ref, values


def _source_core(device):
    source, source_manifest, _metadata, ids, rows = sw122.source_contract(1)
    core = base.make_core(device, steps=64)
    state = torch.load(source, map_location=device, weights_only=True)
    core.load_state_dict(state, strict=True)
    core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.shape[0])]
    core.eval().requires_grad_(False)
    if core.osc_dim != 4 or core.graph_generator.uses_feedback or core.kuramoto.spike_pulse_gain is not None:
        raise AssertionError("source core differs from the registered static D4/no-feedback contract")
    return core, source, source_manifest, np.asarray(ids), np.asarray(rows)


def _source_gamma_caches():
    train, train_meta = base.validate_gamma_cache(base.GAMMA_TRAIN, base.GAMMA_TRAIN_MANIFEST)
    val, val_meta = base.validate_gamma_cache(base.GAMMA_VAL, base.GAMMA_VAL_MANIFEST,
                                              validation=True)
    if tuple(train.shape[1:]) != (8, 256) or tuple(val.shape) != (320, 8, 256):
        raise ValueError("registered cached gamma shapes changed")
    return train, train_meta, val, val_meta


def _preflight_rollout_parity(device: str) -> dict:
    core, source, source_manifest, ids, rows = _source_core(device)
    train_gamma, train_meta, val_gamma, val_meta = _source_gamma_caches()
    records = []
    native_val_trace = None
    validation_gamma_two = val_gamma[:2].to(device)
    for name, cache, row_ids in (
        ("train", train_gamma, rows[:2]),
        ("validation", val_gamma, np.arange(2, dtype=np.int64)),
    ):
        gamma = cache[torch.as_tensor(row_ids, dtype=torch.long)].to(device)
        manual = screen._full_rollout(core, gamma, live_tail=0)
        parity = screen.assert_native_source_rollout_parity(core, gamma, manual)
        if name == "validation":
            native_val_trace = manual
        native_components = core.last_component_spikes
        manual_labels, _ = rgb_loss.production_partition(
            manual["component_spikes"].mean(dim=1), manual["component_spikes"], settle=SETTLE)
        native_labels, _ = rgb_loss.production_partition(
            native_components.mean(dim=1), native_components, settle=SETTLE)
        labels_exact = torch.equal(manual_labels, native_labels)
        if not labels_exact:
            raise AssertionError(f"{name} native/manual QCC labels differ")
        records.append({"split": name, "ids": [int(v) for v in ids[:2]] if name == "train" else [1320, 1321],
                        "traces_exact": parity, "qcc_labels_exact": labels_exact})
    arm_smokes = []
    for arm in ARMS:
        if arm == "source97_native":
            trace = native_val_trace
            core = None
        else:
            core, _provenance = _core_for_arm(arm, device, source)
            with torch.no_grad():
                trace = screen._full_rollout(core, validation_gamma_two, live_tail=0)
        component_spikes = trace["component_spikes"]
        if (tuple(component_spikes.shape) != (2, 4, 256, STEPS)
                or not torch.isfinite(component_spikes).all()):
            raise AssertionError(f"{arm} two-image smoke produced invalid component spikes")
        adapter_trace_exact = (_assert_adapter_spikes(core, component_spikes, 2)
                               if core is not None else None)
        labels = _qcc_labels(component_spikes)
        arm_smokes.append({"arm": arm, "count": 2, "ids": [1320, 1321],
                           "actual_spikes_finite": True,
                           "adapter_spikes_equal_gate_times_event": adapter_trace_exact,
                           "qcc_labels_shape": list(labels.shape),
                           "qcc_label_sha256": hashlib.sha256(
                               labels.contiguous().numpy().tobytes()).hexdigest()})
    return {
        "status": "passed", "source_core_sha256": sha256_file(source),
        "source_manifest_sha256": sha256_file(source_manifest),
        "train_gamma_sha256": sha256_file(base.GAMMA_TRAIN),
        "train_gamma_manifest_sha256": sha256_file(base.GAMMA_TRAIN_MANIFEST),
        "validation_gamma_sha256": sha256_file(base.GAMMA_VAL),
        "validation_gamma_manifest_sha256": sha256_file(base.GAMMA_VAL_MANIFEST),
        "batches": records, "four_arm_validation_smokes": arm_smokes,
    }


def _core_for_arm(arm: str, device, source: Path):
    core = base.make_core(device, steps=64)
    core.load_state_dict(torch.load(source, map_location=device, weights_only=True), strict=True)
    core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.shape[0])]
    adapter = None
    provenance = {"arm": arm, "source_core_sha256": sha256_file(source),
                  "adapter_attached": False, "trained_core_sha256": None,
                  "training_manifest_sha256": None, "training_history_sha256": None}
    if arm != "source97_native":
        adapter = attach_history_event_membrane(core, capture_gate_trace=True)
        provenance["adapter_attached"] = True
        provenance["beta_initial"] = 0.95
    if arm in ("sw0126_history_event", "sw0126_gate_only"):
        train_arm = "history_event" if arm.endswith("history_event") else "gate_only"
        folder = pilot_run.training_dir(train_arm)
        if not pilot_run.valid_training(folder, train_arm):
            raise AssertionError(f"completed SW0126 training artifact is invalid: seed1/{train_arm}")
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("source_core_sha256") != sha256_file(source)
                or manifest.get("updates") != 256 or manifest.get("batch_size") != 16
                or manifest.get("ground_truth_used_for_training") is not False):
            raise AssertionError("SW0126 trained checkpoint has an unexpected recipe/source")
        checkpoint = folder / "core.pt"
        core.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        provenance.update(trained_core_sha256=sha256_file(checkpoint),
                          training_manifest_sha256=sha256_file(manifest_path),
                          training_history_sha256=sha256_file(folder / "history.json"),
                          training_arm=train_arm,
                          training_implementation_fingerprint=manifest["implementation_fingerprint"])
    core.eval().requires_grad_(False)
    return core, provenance


def _qcc_labels(component_spikes: torch.Tensor) -> torch.Tensor:
    if (component_spikes.ndim != 4 or component_spikes.shape[1] != 4
            or component_spikes.shape[2] != 256 or component_spikes.shape[-1] != STEPS):
        raise ValueError("QCC requires actual [B,4,256,1024] emitted component spike traces")
    if not torch.isfinite(component_spikes).all():
        raise ValueError("QCC actual component spike traces must be finite")
    groups = spike_synchrony_components(
        component_spikes.mean(dim=1).detach().cpu(),
        foreground_threshold=0.15,
        synchrony_threshold=0.50,
        min_group_size=2,
        settle=SETTLE,
        components=component_spikes.detach().cpu(),
        background="largest_component",
        synchrony_quantile=None,
        target_foreground=None,
        spatial_sigma=None,
        spatial_grid_size=16,
        affinity_mode="spike",
    )
    labels = spatial_components_to_patch_labels(groups, 16, device="cpu").reshape(
        component_spikes.shape[0], 16, 16)
    return labels


def _assert_adapter_spikes(core, component_spikes, batch_size: int) -> bool | None:
    layer = core.membrane_layer
    if not hasattr(layer, "component_event_trace"):
        return None
    events = layer.component_event_trace(batch_size)
    gates = layer.component_gate_trace(batch_size)
    if (tuple(events.shape) != tuple(component_spikes.shape)
            or tuple(gates.shape) != tuple(component_spikes.shape)
            or not torch.logical_or(events == 0, events == 1).all()
            or not torch.equal(component_spikes, gates * events)):
        raise AssertionError("adapter output must be the actual gate-times-binary-event trace")
    return True


def _predict_arm(arm, source, gamma_cache, rows, device):
    core, provenance = _core_for_arm(arm, device, source)
    predictions = []
    for start in range(0, len(IDS), BATCH):
        selected_rows = rows[start:start + BATCH]
        gamma = gamma_cache[torch.as_tensor(selected_rows, dtype=torch.long)].to(device)
        with torch.no_grad():
            trace = screen._full_rollout(core, gamma, live_tail=0)
        spikes = trace["component_spikes"]
        labels = _qcc_labels(spikes)
        predictions.append(labels)
    result = torch.cat(predictions, dim=0).to(dtype=torch.int64, device="cpu")
    if tuple(result.shape) != (len(IDS), 16, 16):
        raise AssertionError(f"{arm} did not produce the full fixed 320-grid prediction")
    provenance.update({"image_ids": [IDS[0], IDS[-1]], "count": len(IDS),
                       "ground_truth_used_for_prediction": False,
                       "gamma_validation_sha256": sha256_file(base.GAMMA_VAL),
                       "gamma_validation_manifest_sha256": sha256_file(base.GAMMA_VAL_MANIFEST)})
    return result, provenance


def _commit_predictions(output: Path, predictions: dict, provenance: dict) -> tuple[Path, str]:
    prediction_path = output / "frozen_predictions.pt"
    with prediction_path.open("xb") as stream:
        torch.save({"image_ids": list(IDS), "predictions": predictions,
                    "provenance": provenance}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    prediction_sha = sha256_file(prediction_path)
    manifest = {"status": "predictions_complete", "experiment": "SW0127",
                "image_ids": [IDS[0], IDS[-1]], "count": len(IDS),
                "arms": list(ARMS), "prediction_sha256": prediction_sha,
                "ground_truth_used_for_prediction": False,
                "prediction_contract": {"steps": STEPS, "settle": SETTLE, "batch_size": BATCH,
                    "readout": "QCC actual emitted component spikes", "threshold": 0.50,
                    "foreground_threshold": 0.15, "min_group_size": 2,
                    "background": "largest_component", "quantile": None,
                    "target_foreground": None, "spatial_sigma": None,
                    "affinity_mode": "spike", "grid_size": 16},
                "provenance": provenance}
    manifest_path = output / "prediction_manifest.json"
    _write_once(manifest_path, manifest)
    # Re-read and verify the entire saved prediction bundle before GT access.
    if sha256_file(prediction_path) != prediction_sha:
        raise AssertionError("frozen prediction SHA changed after commit")
    loaded = torch.load(prediction_path, map_location="cpu", weights_only=True)
    if (loaded.get("image_ids") != list(IDS) or set(loaded.get("predictions", {})) != set(ARMS)
            or not all(tuple(loaded["predictions"][arm].shape) == (320, 16, 16) for arm in ARMS)
            or manifest.get("prediction_sha256") != prediction_sha):
        raise AssertionError("committed prediction payload failed its frozen contract")
    return prediction_path, prediction_sha


def _compare_source_score(actual: dict, source_values: dict) -> dict:
    deltas = {}
    passed = True
    for metric in METRICS:
        got = np.asarray(actual[metric]["per_image"], dtype=np.float64)
        expected = np.asarray(source_values[metric], dtype=np.float64)
        if got.shape != (len(IDS),) or expected.shape != (len(IDS),):
            deltas[metric] = {"max_abs_diff": None, "passed": False}
            passed = False
            continue
        if not np.isfinite(got).all() or not np.isfinite(expected).all():
            deltas[metric] = {"max_abs_diff": None, "passed": False}
            passed = False
            continue
        max_abs = float(np.max(np.abs(got - expected)))
        ok = max_abs <= 1e-10
        deltas[metric] = {"max_abs_diff": max_abs, "passed": ok}
        passed = passed and ok
    return {"passed": passed, "metrics": deltas}


def _score_committed_predictions(output: Path, target_loader, source_values: dict) -> dict:
    prediction_path = output / "frozen_predictions.pt"
    manifest_path = output / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha = manifest.get("prediction_sha256")
    if (manifest.get("status") != "predictions_complete"
            or manifest.get("image_ids") != [IDS[0], IDS[-1]]
            or manifest.get("count") != len(IDS)
            or manifest.get("ground_truth_used_for_prediction") is not False
            or not isinstance(expected_sha, str) or sha256_file(prediction_path) != expected_sha):
        raise ValueError("frozen prediction bundle is invalid; GT access refused")
    bundle = torch.load(prediction_path, map_location="cpu", weights_only=True)
    if (bundle.get("image_ids") != list(IDS) or set(bundle.get("predictions", {})) != set(ARMS)
            or any(tuple(bundle["predictions"][arm].shape) != (320, 16, 16) for arm in ARMS)):
        raise ValueError("prediction payload is incomplete; GT access refused")
    target = target_loader()
    if tuple(target.shape) != (320, 16, 16):
        raise ValueError("modal-8x8 target labels have the wrong shape")
    scores = {}
    for arm in ARMS:
        scores[arm] = sw118._score_standard(bundle["predictions"][arm], target)
    reproduction = _compare_source_score(scores["source97_native"], source_values)
    return {"scores": scores, "source97_per_image_reproduction": reproduction,
            "prediction_sha256": expected_sha, "prediction_manifest_sha256": sha256_file(manifest_path)}


def _load_targets_after_prediction_commit(dataset=DATASET):
    import h5py
    if Path(dataset).resolve() != DATASET.resolve():
        raise ValueError(f"only the canonical CLEVR dataset is permitted: {DATASET}")
    with h5py.File(dataset, "r") as handle:
        masks = np.asarray(handle["mask"][IDS[0]:IDS[-1] + 1])
    return __import__("snn_kuramoto_bidirectional.evaluation", fromlist=["clevr_mask_patch"]).clevr_mask_patch(
        torch.from_numpy(masks), 8)["patch_labels"].to(dtype=torch.int64, device="cpu")


def preflight(output: Path, device="cuda:0") -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserving existing SW0127 preflight output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    checks = _preflight_rollout_parity(device)
    result = {"status": "passed", "experiment": "SW0127", "device": str(device),
              "ground_truth_used": False, "optimizer_updates": 0,
              "implementation_fingerprint": _implementation_fingerprint(),
              "native_rollout_parity": checks,
              "readout": {"threshold": 0.50, "foreground_threshold": 0.15,
                          "min_group_size": 2, "background": "largest_component",
                          "affinity_mode": "spike", "grid_size": 16}}
    _write_once(output, result)
    return result


def evaluate(output: Path, device="cuda:0", dataset=DATASET) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserving existing SW0127 output: {output}")
    if Path(dataset).resolve() != DATASET.resolve():
        raise ValueError(f"only the canonical CLEVR dataset is permitted: {DATASET}")
    output.mkdir(parents=True, exist_ok=False)
    preflight_record = _preflight_rollout_parity(device)
    source, _manifest, _metadata, ids, rows = sw122.source_contract(1)
    gamma, gamma_meta = base.validate_gamma_cache(base.GAMMA_VAL, base.GAMMA_VAL_MANIFEST,
                                                   validation=True)
    predictions, provenance = {}, {}
    validation_rows = np.arange(len(IDS), dtype=np.int64)
    for arm in ARMS:
        predictions[arm], provenance[arm] = _predict_arm(arm, source, gamma, validation_rows, device)
    prediction_path, prediction_sha = _commit_predictions(output, predictions, provenance)
    source_ref, source_values = _source_reference()
    # This is the first point in the evaluation path that reads ground truth.
    result = _score_committed_predictions(
        output, lambda: _load_targets_after_prediction_commit(dataset), source_values)
    baseline_ok = result["source97_per_image_reproduction"]["passed"]
    report = {
        "status": "complete" if baseline_ok else "failed_source97_reference_reproduction",
        "experiment": "SW0127", "seed": 1, "image_ids": [IDS[0], IDS[-1]], "count": len(IDS),
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_training": False,
        "ground_truth_used_for_scoring": True, "optimizer_updates": 0,
        "prediction_path": str(prediction_path), "prediction_sha256": prediction_sha,
        "prediction_manifest_sha256": result["prediction_manifest_sha256"],
        "source_reference_evaluation_sha256": source_ref["source_evaluation_sha256"],
        "source_core_sha256": source_ref["source_core_sha256"],
        "source_manifest_sha256": source_ref["source_manifest_sha256"],
        "validation_gamma_sha256": sha256_file(base.GAMMA_VAL),
        "validation_gamma_manifest_sha256": sha256_file(base.GAMMA_VAL_MANIFEST),
        "native_rollout_preflight": preflight_record,
        "arm_provenance": provenance,
        "scores": result["scores"],
        "source97_per_image_reproduction": result["source97_per_image_reproduction"],
        "interpretation": "Frozen QCC attribution only; this comparison does not establish individual gate causality.",
        "implementation_fingerprint": _implementation_fingerprint(),
    }
    _write_once(output / "evaluation.json", report)
    _write_once(output / "evaluation_manifest.json", {
        "experiment": "SW0127", "seed": 1, "evaluation_sha256": sha256_file(output / "evaluation.json"),
        "prediction_sha256": prediction_sha, "source_reference_evaluation_sha256": source_ref["source_evaluation_sha256"],
        "implementation_fingerprint": report["implementation_fingerprint"],
        "ground_truth_used_for_prediction": False, "optimizer_updates": 0,
    })
    if not baseline_ok:
        raise AssertionError("source97 reference did not reproduce; failed report and predictions preserved")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args(argv)
    result = (preflight(args.output, args.device) if args.preflight_only
              else evaluate(args.output, args.device, args.dataset))
    print(json.dumps({"status": result["status"], "experiment": "SW0127",
                      "output": str(args.output), "preflight_only": args.preflight_only},
                     allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
