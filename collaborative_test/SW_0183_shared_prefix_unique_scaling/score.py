"""GT-last paired 320-image evaluation for SW0183 control512/candidate1024."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

if "--worker" not in __import__("sys").argv: os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1"); os.environ["USE_PYTORCH_KERNEL_CACHE"] = "0"
import numpy as np
import h5py

from collaborative_test.SW_0183_shared_prefix_unique_scaling import predict_evaluate as pe
from collaborative_test.SW_0183_shared_prefix_unique_scaling import train
from collaborative_test.SW_0179_qce_seed_replication import score as reference
from collaborative_test.SW_0182_unique1024_actual_descent import score as prior_score
from collaborative_test.SW_0165_foreground_partition_replication import predict_evaluate as sw165
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
SEEDS, ARMS, ENDPOINTS = (0, 1, 2), ("control512", "candidate1024"), ("conditional", "standalone")
METRICS = tuple(contract.METRICS); COUNT = int(contract.COUNT)
IMAGE_IDS = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES = 178, 10_000
SCORE_PATH = HERE / "results_archive" / "evaluation_seedmean_20261016.json"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def implementation_fingerprint():
    rels = (
        "collaborative_test/SW_0183_shared_prefix_unique_scaling/DESIGN_KO.md",
        "collaborative_test/SW_0183_shared_prefix_unique_scaling/train.py",
        "collaborative_test/SW_0183_shared_prefix_unique_scaling/training_queue.py",
        "collaborative_test/SW_0183_shared_prefix_unique_scaling/predict_evaluate.py",
        "collaborative_test/SW_0183_shared_prefix_unique_scaling/prediction_queue.py",
        "collaborative_test/SW_0183_shared_prefix_unique_scaling/score.py",
        "collaborative_test/SW_0183_shared_prefix_unique_scaling/score_queue.py",
        "collaborative_test/SW_0183_shared_prefix_unique_scaling/test_predict_evaluate.py",
        "collaborative_test/SW_0183_shared_prefix_unique_scaling/test_prediction_queue.py",
        "collaborative_test/SW_0183_shared_prefix_unique_scaling/test_score.py",
        "collaborative_test/SW_0183_shared_prefix_unique_scaling/test_score_queue.py",
        "collaborative_test/SW_0179_qce_seed_replication/predict_evaluate.py",
        "collaborative_test/SW_0179_qce_seed_replication/score.py",
        "collaborative_test/SW_0182_unique1024_actual_descent/score.py",
        "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
    )
    values = {name: sha256_file(ROOT / name) for name in rels if (ROOT / name).is_file()}
    if len(values) != len(rels): raise FileNotFoundError("SW0183 score dependency missing")
    values["SW0181_score_sha256"] = prior_score.SW0181_SCORE_SHA256
    values["SW0165_summary_sha256"] = prior_score.SW165_SUMMARY_SHA256
    return values


def _state(seed):
    queue = train._module(pe.QUEUE_MODULE); path = queue.state_path(seed)
    if not path.is_file(): raise FileNotFoundError(f"SW0183 prediction state missing for seed {seed}")
    return queue, path, json.loads(path.read_text(encoding="utf-8"))


def _validate_prediction(seed):
    queue, state_path, state = _state(seed)
    if (state.get("experiment") != "SW0183_prediction_queue" or state.get("seed") != seed
            or state.get("status") != "complete" or state.get("ground_truth_used_for_prediction") is not False
            or state.get("validation_data_read") is not False
            or set(state.get("predictions", {})) != set(ARMS)):
        raise RuntimeError(f"SW0183 seed{seed} does not have both terminal frozen predictions")
    prepared = pe.prepare_prediction_inputs_cpu(seed)
    results = {arm: pe.validate_prediction(seed, arm, prepared=prepared) for arm in ARMS}
    manifests = {arm: prepared["artifacts"][arm]["manifest_sha256"] for arm in ARMS}
    for arm in ARMS:
        item = state["predictions"][arm]; result = results[arm]
        if (item.get("prediction_sha256") != result["prediction_sha256"]
                or item.get("protocol_sha256") != result["protocol_sha256"]
                or state.get("training_manifest_sha256", {}).get(arm) != manifests[arm]):
            raise ValueError(f"SW0183 queue receipt does not bind {arm} prediction/training artifacts")
    source_foreground = prepared["references"]["source_T"]["labels"] > 0
    for arm in ARMS:
        labels, support = results[arm]["labels"], results[arm]["standalone_support"]
        if (not np.array_equal(labels["conditional"] > 0, source_foreground)
                or not np.array_equal(labels["standalone"] > 0, support > 0)):
            raise ValueError("SW0183 endpoint-specific foreground contract failed")
    return {"prepared": prepared, "results": results, "state": state,
        "state_path": state_path, "state_sha256": sha256_file(state_path)}


def prepare_score_inputs_cpu():
    if pe.torch.cuda.is_initialized(): raise RuntimeError("SW0183 score preparation must precede CUDA initialization")
    frozen = {seed: _validate_prediction(seed) for seed in SEEDS}
    baseline = reference._baseline_reports()
    sw0181 = prior_score._read_sw0181_score()
    dataset = Path(sw165.DATASET).resolve(); stat = dataset.stat()
    return {"frozen": frozen, "baseline": baseline, "sw0181": sw0181,
        "dataset": dataset, "dataset_path": str(dataset), "dataset_size_bytes": stat.st_size,
        "dataset_mtime_ns": stat.st_mtime_ns, "ground_truth_opened": False}


def _metrics_by_seed(predictions, gt):
    result = {}
    for group, seed_labels in predictions.items():
        per_seed = {seed: contract.production_metrics32(seed_labels[seed], gt) for seed in SEEDS}
        metrics = {}
        for metric in METRICS:
            matrix = np.stack([np.asarray(per_seed[seed][metric]["per_image"], dtype=np.float64)
                               for seed in SEEDS])
            if matrix.shape != (3, COUNT) or not np.isfinite(matrix).all():
                raise ValueError("SW0183 requires finite paired 3x320 metric rows")
            mean_images = matrix.mean(axis=0)
            metrics[metric] = {"per_image": mean_images, "mean": float(mean_images.mean()),
                "per_seed_mean": {str(seed): float(per_seed[seed][metric]["mean"]) for seed in SEEDS},
                "per_seed_per_image": {str(seed): matrix[index] for index, seed in enumerate(SEEDS)}}
        result[group] = metrics
    return result


def _bootstrap(left, right, sample_indices):
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if delta.shape != (COUNT,) or not np.isfinite(delta).all():
        raise ValueError("SW0183 paired bootstrap requires finite fixed320 image vectors")
    sampled = delta[sample_indices].mean(axis=1)
    return {"mean": float(delta.mean()), "ci95": [float(v) for v in np.percentile(sampled, [2.5, 97.5])],
        "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
        "resampling_unit": "paired_image_after_three_seed_average"}


def aggregate_and_gate(scores, sample_indices):
    """Aggregate paired image vectors and derive registered primary/benchmark gates."""
    paired = {}
    for endpoint in ENDPOINTS:
        for metric in METRICS:
            left = scores[f"candidate1024_{endpoint}"][metric]["per_image"]
            right = scores[f"control512_{endpoint}"][metric]["per_image"]
            paired[f"candidate1024_minus_control512_{endpoint}_{metric}"] = _bootstrap(left, right, sample_indices)
    for endpoint in ENDPOINTS:
        for group in ("SW0165_foreground_induced", "Slot70k", "SW0181"):
            for metric in METRICS:
                paired[f"candidate1024_minus_{group}_{endpoint}_{metric}"] = _bootstrap(
                    scores[f"candidate1024_{endpoint}"][metric]["per_image"],
                    scores[f"{group}_{endpoint}"][metric]["per_image"], sample_indices)
    primary = scores["candidate1024_standalone"]
    control = scores["control512_standalone"]
    slot = scores["Slot70k_standalone"]
    gates = {
        "primary_endpoint": "standalone",
        "unique_data_fg_mean_above_control": primary["fg_ari"]["mean"] > control["fg_ari"]["mean"],
        "unique_data_fg_paired_ci95_lower_above_zero": paired["candidate1024_minus_control512_standalone_fg_ari"]["ci95"][0] > 0,
        "slot_fg_mean_strictly_exceeded": primary["fg_ari"]["mean"] > slot["fg_ari"]["mean"],
        "slot_fg_paired_ci95_lower_above_zero": paired["candidate1024_minus_Slot70k_standalone_fg_ari"]["ci95"][0] > 0,
        "slot_foreground_iou_mean_strictly_exceeded": primary["foreground_iou"]["mean"] > slot["foreground_iou"]["mean"],
        "slot_matched_object_iou_mean_strictly_exceeded": primary["matched_object_iou"]["mean"] > slot["matched_object_iou"]["mean"],
        "sw0165_fg_mean_strictly_exceeded": primary["fg_ari"]["mean"] > scores["SW0165_foreground_induced_standalone"]["fg_ari"]["mean"],
        "sw0165_fg_paired_ci95_lower_above_zero": paired["candidate1024_minus_SW0165_foreground_induced_standalone_fg_ari"]["ci95"][0] > 0,
        "sw0181_fg_mean_strictly_exceeded": primary["fg_ari"]["mean"] > scores["SW0181_standalone"]["fg_ari"]["mean"],
        "sw0181_fg_paired_ci95_lower_above_zero": paired["candidate1024_minus_SW0181_standalone_fg_ari"]["ci95"][0] > 0,
        "fg_0_87_goal": primary["fg_ari"]["mean"] >= 0.87,
    }
    gates["unique_data_gate_passed"] = gates["unique_data_fg_mean_above_control"] and gates["unique_data_fg_paired_ci95_lower_above_zero"]
    gates["slot_benchmark_passed"] = all(gates[key] for key in (
        "slot_fg_mean_strictly_exceeded", "slot_fg_paired_ci95_lower_above_zero",
        "slot_foreground_iou_mean_strictly_exceeded", "slot_matched_object_iou_mean_strictly_exceeded"))
    gates["mask_supervised"] = True; gates["unsupervised_leaderboard_eligible"] = False
    return paired, gates


def score_all(*, output=SCORE_PATH, prepared=None):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    p = prepare_score_inputs_cpu() if prepared is None else prepared
    frozen = p["frozen"]
    if set(frozen) != set(SEEDS): raise ValueError("SW0183 requires all three seeds before validation GT")
    for seed in SEEDS:
        item = frozen[seed]
        if sha256_file(item["state_path"]) != item["state_sha256"]: raise ValueError("prediction queue state changed before GT")
        current = {arm: pe.validate_prediction(seed, arm, prepared=item["prepared"]) for arm in ARMS}
        for arm in ARMS:
            old, new = item["results"][arm], current[arm]
            if old["prediction_sha256"] != new["prediction_sha256"] or old["protocol_sha256"] != new["protocol_sha256"]:
                raise ValueError("prediction archive/protocol changed before GT access")
    dataset = Path(p["dataset"]); before = dataset.stat()
    if (str(dataset.resolve()) != p["dataset_path"] or before.st_size != p["dataset_size_bytes"]
            or before.st_mtime_ns != p["dataset_mtime_ns"]): raise ValueError("canonical dataset changed before score")
    # No mask path is opened until all six prediction archives and protocols are revalidated.
    with h5py.File(dataset, "r") as hdf:
        gt = contract.modal_native32(np.asarray(hdf["mask"][int(IMAGE_IDS[0]):int(IMAGE_IDS[-1]) + 1]))
    predictions = {f"{arm}_{endpoint}": {seed: frozen[seed]["results"][arm]["labels"][endpoint]
        for seed in SEEDS} for arm in ARMS for endpoint in ENDPOINTS}
    scores = _metrics_by_seed(predictions, gt)
    base = p["baseline"]
    for group in ("SW0165_foreground_induced", "Slot70k"):
        for endpoint in ENDPOINTS:
            rows = {}
            for metric in METRICS:
                matrix = np.stack([base["reports"][seed]["metric_vectors"][group][metric] for seed in SEEDS]).astype(np.float64)
                vector = matrix.mean(axis=0)
                rows[metric] = {"per_image": vector, "mean": float(vector.mean()),
                    "per_seed_mean": {str(seed): float(matrix[i].mean()) for i, seed in enumerate(SEEDS)},
                    "per_seed_per_image": {str(seed): matrix[i] for i, seed in enumerate(SEEDS)}}
            scores[f"{group}_{endpoint}"] = rows
    sw181 = p["sw0181"]
    for endpoint in ENDPOINTS:
        rows = {}; reference_rows = sw181["payload"]["scores"][f"candidate_{endpoint}"]
        for metric in METRICS:
            item = reference_rows[metric]; matrix = np.stack([np.asarray(item["per_seed_per_image"][str(seed)], dtype=np.float64) for seed in SEEDS])
            vector = matrix.mean(axis=0)
            rows[metric] = {"per_image": vector, "mean": float(vector.mean()),
                "per_seed_mean": item["per_seed_mean"], "per_seed_per_image": {str(seed): matrix[i] for i, seed in enumerate(SEEDS)}}
        scores[f"SW0181_{endpoint}"] = rows
    sample = np.random.RandomState(BOOTSTRAP_SEED).randint(0, COUNT, size=(BOOTSTRAP_REPLICATES, COUNT))
    paired, gates = aggregate_and_gate(scores, sample)
    for seed in SEEDS:
        item = frozen[seed]
        if sha256_file(item["state_path"]) != item["state_sha256"]: raise ValueError("prediction state changed during score")
        for arm in ARMS:
            again = pe.validate_prediction(seed, arm, prepared=item["prepared"])
            if (again["prediction_sha256"] != item["results"][arm]["prediction_sha256"]
                    or again["protocol_sha256"] != item["results"][arm]["protocol_sha256"]):
                raise ValueError("prediction bytes changed during scoring")
    if reference._baseline_reports()["summary_sha256"] != prior_score.SW165_SUMMARY_SHA256:
        raise ValueError("SW0165 baseline changed during scoring")
    if prior_score._read_sw0181_score()["sha256"] != prior_score.SW0181_SCORE_SHA256:
        raise ValueError("SW0181 reference score changed during scoring")
    after = dataset.stat()
    if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns: raise ValueError("dataset changed during score")
    serialized = {group: {metric: {**row, "per_image": row["per_image"].tolist(),
        "per_seed_per_image": {seed: values.tolist() for seed, values in row["per_seed_per_image"].items()}}
        for metric, row in content.items()} for group, content in scores.items()}
    report = {"experiment": "SW0183_shared_prefix_unique_scaling_three_seed_score", "status": "complete",
        "seed_scope": list(SEEDS), "arm_scope": list(ARMS), "endpoint_scope": list(ENDPOINTS), "metric_scope": list(METRICS),
        "candidate_prediction_hashes": {str(seed): {arm: {
            "prediction_sha256": frozen[seed]["results"][arm]["prediction_sha256"],
            "protocol_sha256": frozen[seed]["results"][arm]["protocol_sha256"]} for arm in ARMS}
            for seed in SEEDS},
        "prediction_queue_state_sha256": {str(seed): frozen[seed]["state_sha256"] for seed in SEEDS},
        "training_manifest_sha256": {str(seed): {arm: frozen[seed]["prepared"]["artifacts"][arm]["manifest_sha256"] for arm in ARMS} for seed in SEEDS},
        "baseline_summary_sha256": prior_score.SW165_SUMMARY_SHA256,
        "SW0181_control_score_sha256": prior_score.SW0181_SCORE_SHA256,
        "scores": serialized, "paired_bootstrap": paired, "gates": gates,
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
        "ground_truth_scope": "fixed320 canonical validation modal-native32 masks only",
        "mask_supervised": True, "unsupervised_leaderboard_eligible": False,
        "dataset": {"path": p["dataset_path"], "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns},
        "bootstrap": {"rng": "numpy RandomState", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
            "resampling_unit": "320 paired images after within-image three-seed averaging"},
        "implementation_fingerprint": implementation_fingerprint()}
    _atomic_json_create(output, report); report["score_sha256"] = sha256_file(output)
    return report


def _atomic_json_create(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        if temp.exists(): temp.unlink()

