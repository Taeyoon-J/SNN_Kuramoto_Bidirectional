"""GT-last score for frozen SW0185 predictions and pinned controls."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

if "--worker" not in __import__("sys").argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ["USE_PYTORCH_KERNEL_CACHE"] = "0"

import numpy as np
import h5py

from collaborative_test.SW_0185_qce_spatial_connectivity import predict_evaluate as pe
from collaborative_test.SW_0179_qce_seed_replication import score as reference_score
from collaborative_test.SW_0165_foreground_partition_replication import predict_evaluate as sw165
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0130_phase_state_integration import run as sw130

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS, ENDPOINTS = (0, 1, 2), ("conditional", "standalone")
METRICS = tuple(contract.METRICS)
COUNT = int(contract.COUNT)
IMAGE_IDS = np.asarray(contract.IMAGE_IDS, dtype=np.int64)
BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES = 178, 10_000
SCORE_PATH = HERE / "results_archive" / "evaluation_seedmean_20261016.json"
SW0181_SCORE_CANDIDATES = (
    ROOT / "collaborative_test" / "SW_0181_actual_descent_qce" / "results_archive" / "score_v1" / "evaluation.json",
    ROOT / "collaborative_test" / "SW_0181_actual_descent_qce" / "results_archive" / "evaluation_seedmean_20261016.json",
)
SW0181_SCORE_SHA256 = "3f69a74a5729341b5fd11f2bd74f2d51ff245c0839f76053b8733af9523ddd54"
SW165_SUMMARY_SHA256 = reference_score.SW165_SUMMARY_SHA256
SW165_FG_BASELINE = reference_score.SW165_FG_BASELINE
SLOT_FG_BASELINE = reference_score.SLOT_FG_BASELINE
SLOT_FOREGROUND_IOU_BASELINE = reference_score.SLOT_FOREGROUND_IOU_BASELINE
SLOT_MATCHED_IOU_BASELINE = reference_score.SLOT_MATCHED_IOU_BASELINE


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def implementation_fingerprint():
    rels = ("collaborative_test/SW_0185_qce_spatial_connectivity/DESIGN_KO.md",
        "collaborative_test/SW_0185_qce_spatial_connectivity/train.py",
        "collaborative_test/SW_0185_qce_spatial_connectivity/training_queue.py",
        "collaborative_test/SW_0185_qce_spatial_connectivity/predict_evaluate.py",
        "collaborative_test/SW_0185_qce_spatial_connectivity/prediction_queue.py",
        "collaborative_test/SW_0185_qce_spatial_connectivity/score.py",
        "collaborative_test/SW_0185_qce_spatial_connectivity/test_predict_evaluate.py",
        "collaborative_test/SW_0185_qce_spatial_connectivity/test_prediction_queue.py",
        "collaborative_test/SW_0185_qce_spatial_connectivity/test_score.py",
        "collaborative_test/SW_0185_qce_spatial_connectivity/score_queue.py",
        "collaborative_test/SW_0185_qce_spatial_connectivity/test_score_queue.py",
        "collaborative_test/SW_0181_actual_descent_qce/score.py",
        "collaborative_test/SW_0179_qce_seed_replication/predict_evaluate.py",
        "collaborative_test/SW_0179_qce_seed_replication/score.py",
        "collaborative_test/SW_0165_foreground_partition_replication/replication_summary.py",
        "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py")
    result = {rel: sha256_file(ROOT / rel) for rel in rels if (ROOT / rel).is_file()}
    if len(result) != len(rels): raise FileNotFoundError("SW0185 score fingerprint dependency missing")
    result["sw0181_control_score_sha256"] = SW0181_SCORE_SHA256
    return result


def _prediction_queue_state(seed):
    queue = pe.train._module(pe.QUEUE_MODULE)
    path = queue._state_path(seed)
    if not path.is_file(): raise FileNotFoundError(f"SW0181 prediction state missing for seed {seed}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _select_sw0181_score(candidates=SW0181_SCORE_CANDIDATES, expected=SW0181_SCORE_SHA256,
                          *, is_file=None, sha_file=None):
    is_file = Path.is_file if is_file is None else is_file
    sha_file = sha256_file if sha_file is None else sha_file
    matches = [Path(path) for path in candidates
               if is_file(Path(path)) and sha_file(Path(path)) == expected]
    if len(matches) != 1:
        raise ValueError("pinned SW0181 three-seed score is missing, changed, or ambiguous")
    return matches[0]


def _read_sw0181_score():
    path = _select_sw0181_score()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("experiment") != "SW0181_actual_descent_qce_three_seed_score"
            or payload.get("status") != "complete"
            or payload.get("seed_scope") != list(SEEDS)
            or payload.get("endpoint_scope") != list(ENDPOINTS)
            or payload.get("ground_truth_used_for_scoring") is not True):
        raise ValueError("pinned SW0181 score has an unexpected producer schema")
    vectors = {}
    for endpoint in ENDPOINTS:
        group = payload.get("scores", {}).get(f"candidate_{endpoint}")
        if not isinstance(group, dict):
            raise ValueError(f"SW0181 score lacks candidate {endpoint} metrics")
        vectors[endpoint] = {}
        for metric in METRICS:
            values = np.asarray(group.get(metric, {}).get("per_image"), dtype=np.float64)
            if values.shape != (COUNT,) or not np.isfinite(values).all():
                raise ValueError(f"SW0181 {endpoint}/{metric} vector is malformed")
            vectors[endpoint][metric] = values
    return {"path": path, "sha256": sha256_file(path), "payload": payload,
            "vectors": vectors}


def _validate_prediction(seed):
    state_path, state = _prediction_queue_state(seed)
    if (state.get("experiment") != "SW0185_prediction_queue" or state.get("seed") != seed
            or state.get("arm") != pe.ARM or state.get("status") != "complete"
            or state.get("ground_truth_used_for_prediction") is not False
            or state.get("validation_data_read") is not False):
        raise RuntimeError(f"SW0185 seed{seed} prediction is not terminal frozen complete")
    prepared = pe.prepare_prediction_inputs_cpu(seed)
    result = pe.validate_prediction(seed, prepared=prepared)
    if (state.get("prediction_sha256") != result["prediction_sha256"]
            or state.get("protocol_sha256") != result["protocol_sha256"]
            or state.get("training_manifest_sha256") != prepared["manifest_sha256"]
            or state.get("training_state_sha256") != prepared["training_state_sha256"]):
        raise ValueError(f"SW0185 seed{seed} prediction queue receipt does not bind artifacts")
    labels, support = result["labels"], result["standalone_support"]
    if (not np.array_equal(labels["conditional"] > 0,
                           prepared["references"]["source_T"]["labels"] > 0)
            or not np.array_equal(labels["standalone"] > 0, support > 0)):
        raise ValueError("SW0185 endpoint-specific foreground contracts failed")
    return {"prepared": prepared, "result": result, "state": state,
        "state_path": state_path, "state_sha256": sha256_file(state_path)}


def _assert_prediction_unchanged(seed, current, frozen):
    if (current.get("prediction_sha256") != frozen.get("prediction_sha256")
            or current.get("protocol_sha256") != frozen.get("protocol_sha256")):
        raise ValueError(f"SW0185 seed{int(seed)} prediction or protocol changed")


def prepare_score_inputs_cpu():
    if pe.torch.cuda.is_initialized():
        raise RuntimeError("SW0181 score preparation must precede CUDA initialization")
    frozen = {seed: _validate_prediction(seed) for seed in SEEDS}
    baseline = reference_score._baseline_reports()
    sw0181 = _read_sw0181_score()
    dataset = Path(sw165.DATASET).resolve(); stat = dataset.stat()
    return {"frozen": frozen, "baseline": baseline, "sw0181": sw0181, "dataset": dataset,
        "dataset_path": str(dataset), "dataset_size_bytes": stat.st_size,
        "dataset_mtime_ns": stat.st_mtime_ns, "implementation_fingerprint": implementation_fingerprint(),
        "ground_truth_opened": False}


def _metrics(predictions, gt):
    result = {}
    for group, seed_labels in predictions.items():
        rows = {seed: contract.production_metrics32(seed_labels[seed], gt) for seed in SEEDS}
        group_result = {}
        for metric in METRICS:
            matrix = np.stack([np.asarray(rows[seed][metric]["per_image"], dtype=np.float64)
                               for seed in SEEDS])
            if matrix.shape != (3, COUNT) or not np.isfinite(matrix).all():
                raise ValueError("SW0181 metric rows must be finite 3x320 paired values")
            per_image = matrix.mean(axis=0)
            group_result[metric] = {"per_image": per_image, "mean": float(per_image.mean()),
                "per_seed_mean": {str(seed): float(rows[seed][metric]["mean"]) for seed in SEEDS},
                "per_seed_per_image": {str(seed): matrix[i] for i, seed in enumerate(SEEDS)}}
        result[group] = group_result
    return result


def _average_seed(vectors):
    if not isinstance(vectors, dict) or set(vectors) != set(SEEDS):
        raise ValueError("SW0181 aggregation requires all three source seeds")
    matrix = np.stack([np.asarray(vectors[seed], dtype=np.float64) for seed in SEEDS])
    if matrix.shape != (3, COUNT) or not np.isfinite(matrix).all():
        raise ValueError("SW0181 requires shared finite per-image vectors across all source seeds")
    return matrix.mean(axis=0)


def _bootstrap(candidate, baseline, sample_indices):
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if delta.shape != (COUNT,) or not np.isfinite(delta).all():
        raise ValueError("SW0181 paired bootstrap needs finite fixed320 vectors")
    sampled = delta[sample_indices].mean(axis=1)
    return {"mean": float(delta.mean()),
        "ci95": [float(x) for x in np.percentile(sampled, [2.5, 97.5])],
        "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
        "resampling_unit": "paired_image_after_three_seed_average"}


def _gate(scores, paired):
    primary = scores["candidate_standalone"]
    sw = scores["SW0165_foreground_induced_standalone"]
    slot = scores["Slot70k_standalone"]
    sw0181 = scores["SW0181_standalone"]
    gate = {
        "primary_endpoint": "standalone",
        "fg_strictly_above_sw0165_three_seed_mean": primary["fg_ari"]["mean"] > SW165_FG_BASELINE,
        "fg_paired_ci95_lower_above_zero_vs_sw0165": paired["standalone_candidate_minus_SW0165_fg_ari"]["ci95"][0] > 0,
        "fg_strictly_above_slot_three_seed_mean": primary["fg_ari"]["mean"] > SLOT_FG_BASELINE,
        "fg_paired_ci95_lower_above_zero_vs_slot": paired["standalone_candidate_minus_Slot70k_fg_ari"]["ci95"][0] > 0,
        "fg_strictly_above_sw0181_three_seed_mean": primary["fg_ari"]["mean"] > sw0181["fg_ari"]["mean"],
        "fg_paired_ci95_lower_above_zero_vs_sw0181": paired["standalone_candidate_minus_SW0181_fg_ari"]["ci95"][0] > 0,
        "foreground_iou_strictly_above_slot_three_seed_mean": primary["foreground_iou"]["mean"] > slot["foreground_iou"]["mean"],
        "matched_object_iou_strictly_above_slot_three_seed_mean": primary["matched_object_iou"]["mean"] > slot["matched_object_iou"]["mean"],
        "fg_ari_at_least_0_87_goal": primary["fg_ari"]["mean"] >= 0.87,
        "mask_supervised": True, "unsupervised_leaderboard_eligible": False}
    gate["primary_gate_passed"] = all(gate[key] for key in (
        "fg_strictly_above_sw0181_three_seed_mean",
        "fg_paired_ci95_lower_above_zero_vs_sw0181",
        "fg_strictly_above_slot_three_seed_mean",
        "foreground_iou_strictly_above_slot_three_seed_mean",
        "matched_object_iou_strictly_above_slot_three_seed_mean"))
    return gate


def _aggregate_paired_and_gate(scores, sample_indices):
    """Use the production pair names and gate inputs for a complete score payload."""
    paired = {}
    for endpoint in ENDPOINTS:
        for group in ("SW0165_foreground_induced", "Slot70k", "SW0181"):
            key = f"{endpoint}_candidate_minus_{group}"
            left, right = scores[f"candidate_{endpoint}"], scores[f"{group}_{endpoint}"]
            paired[key] = {metric: _bootstrap(left[metric]["per_image"],
                right[metric]["per_image"], sample_indices) for metric in METRICS}
    paired_gate = {
        "standalone_candidate_minus_SW0165_fg_ari": paired["standalone_candidate_minus_SW0165_foreground_induced"]["fg_ari"],
        "standalone_candidate_minus_Slot70k_fg_ari": paired["standalone_candidate_minus_Slot70k"]["fg_ari"],
        "standalone_candidate_minus_SW0181_fg_ari": paired["standalone_candidate_minus_SW0181"]["fg_ari"],
    }
    return paired, _gate(scores, paired_gate)


def score_all(*, output=SCORE_PATH, prepared=None):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    p = prepare_score_inputs_cpu() if prepared is None else prepared
    frozen = p["frozen"]
    if set(frozen) != set(SEEDS): raise ValueError("SW0181 requires all three seeds before GT")
    for seed in SEEDS:
        item = frozen[seed]
        if sha256_file(item["state_path"]) != item["state_sha256"]:
            raise ValueError("SW0181 prediction state changed before GT access")
        current = pe.validate_prediction(seed, prepared=item["prepared"])
        _assert_prediction_unchanged(seed, current, item["result"])
    if sha256_file(p["sw0181"]["path"]) != SW0181_SCORE_SHA256:
        raise ValueError("pinned SW0181 score changed before GT access")
    dataset = Path(p["dataset"])
    before = dataset.stat()
    if (str(dataset.resolve()) != p["dataset_path"] or before.st_size != p["dataset_size_bytes"]
            or before.st_mtime_ns != p["dataset_mtime_ns"]):
        raise ValueError("SW0181 canonical dataset changed before scoring")

    # First validation mask access occurs only after all three final prediction archives,
    # protocols, training manifests, and queue states have passed validation above.
    with h5py.File(dataset, "r") as hdf:
        if "mask" not in hdf or hdf["mask"].shape[0] <= int(IMAGE_IDS[-1]):
            raise ValueError("SW0181 fixed320 validation masks are unavailable")
        gt = contract.modal_native32(np.asarray(hdf["mask"][int(IMAGE_IDS[0]):int(IMAGE_IDS[-1]) + 1]))
    candidate = {endpoint: {seed: frozen[seed]["result"]["labels"][endpoint] for seed in SEEDS}
                 for endpoint in ENDPOINTS}
    scores = _metrics({f"candidate_{endpoint}": candidate[endpoint] for endpoint in ENDPOINTS}, gt)
    baseline = p["baseline"]
    for group in ("SW0165_foreground_induced", "Slot70k"):
        for endpoint in ENDPOINTS:
            rows = {}
            for metric in METRICS:
                vectors = {seed: baseline["reports"][seed]["metric_vectors"][group][metric]
                           for seed in SEEDS}
                matrix = np.stack([vectors[seed] for seed in SEEDS])
                average = _average_seed(vectors)
                rows[metric] = {"per_image": average, "mean": float(average.mean()),
                    "per_seed_mean": {str(seed): float(matrix[i].mean()) for i, seed in enumerate(SEEDS)},
                    "per_seed_per_image": {str(seed): matrix[i] for i, seed in enumerate(SEEDS)}}
            scores[f"{group}_{endpoint}"] = rows

    sw0181 = p["sw0181"]
    for endpoint in ENDPOINTS:
        rows = {}
        reference = sw0181["payload"]["scores"][f"candidate_{endpoint}"]
        for metric in METRICS:
            vector = sw0181["vectors"][endpoint][metric]
            per_seed = reference[metric].get("per_seed_mean", {})
            per_seed_vectors = reference[metric].get("per_seed_per_image", {})
            if set(per_seed_vectors) != {str(seed) for seed in SEEDS}:
                raise ValueError("pinned SW0181 score lacks its three seed-specific image vectors")
            matrix = np.stack([np.asarray(per_seed_vectors[str(seed)], dtype=np.float64)
                               for seed in SEEDS])
            if matrix.shape != (3, COUNT) or not np.isfinite(matrix).all():
                raise ValueError("pinned SW0181 per-seed metric vectors are malformed")
            rows[metric] = {"per_image": vector, "mean": float(vector.mean()),
                "per_seed_mean": {str(seed): float(per_seed[str(seed)]) for seed in SEEDS},
                "per_seed_per_image": {str(seed): matrix[i] for i, seed in enumerate(SEEDS)}}
        scores[f"SW0181_{endpoint}"] = rows

    indices = np.random.RandomState(BOOTSTRAP_SEED).randint(0, COUNT,
        size=(BOOTSTRAP_REPLICATES, COUNT))
    paired, gates = _aggregate_paired_and_gate(scores, indices)

    # Revalidate every immutable prediction and baseline after GT scoring.
    for seed in SEEDS:
        item = frozen[seed]
        if sha256_file(item["state_path"]) != item["state_sha256"]:
            raise ValueError("SW0181 prediction queue state changed during scoring")
        current = pe.validate_prediction(seed, prepared=item["prepared"])
        _assert_prediction_unchanged(seed, current, item["result"])
    if reference_score._baseline_reports()["summary_sha256"] != SW165_SUMMARY_SHA256:
        raise ValueError("SW0165 baseline changed during SW0181 scoring")
    if sha256_file(p["sw0181"]["path"]) != SW0181_SCORE_SHA256:
        raise ValueError("pinned SW0181 score changed during SW0185 scoring")
    after = dataset.stat()
    if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
        raise ValueError("canonical dataset changed during scoring")
    json_scores = {group: {metric: {**row, "per_image": row["per_image"].tolist(),
        "per_seed_per_image": {seed: values.tolist() for seed, values in row["per_seed_per_image"].items()}}
        for metric, row in values.items()} for group, values in scores.items()}
    report = {"experiment": "SW0185_qce_spatial_connectivity_three_seed_score", "status": "complete",
        "seed_scope": list(SEEDS), "endpoint_scope": list(ENDPOINTS), "metric_scope": list(METRICS),
        "candidate_prediction_hashes": {str(seed): {"prediction_sha256": frozen[seed]["result"]["prediction_sha256"],
            "protocol_sha256": frozen[seed]["result"]["protocol_sha256"],
            "queue_state_sha256": frozen[seed]["state_sha256"]} for seed in SEEDS},
        "training_manifest_sha256": {str(seed): frozen[seed]["prepared"]["manifest_sha256"] for seed in SEEDS},
        "baseline_summary_sha256": SW165_SUMMARY_SHA256,
        "baseline_score_sha256": {str(seed): baseline["reports"][seed]["sha256"] for seed in SEEDS},
        "SW0181_control_score": {"sha256": p["sw0181"]["sha256"],
            "path": str(p["sw0181"]["path"]), "experiment": p["sw0181"]["payload"]["experiment"]},
        "scores": json_scores, "paired_bootstrap": paired, "gate": gates,
        "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
        "ground_truth_scope": "fixed320 canonical validation modal-native32 masks only",
        "mask_supervised": True, "unsupervised_leaderboard_eligible": False,
        "dataset": {"path": p["dataset_path"], "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns},
        "bootstrap": {"rng": "numpy RandomState", "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "resampling_unit": "320 paired images after candidate three-seed average"},
        "implementation_fingerprint": implementation_fingerprint()}
    _atomic_json_create(output, report)
    report["score_sha256"] = sha256_file(output)
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


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=SCORE_PATH)
    args = parser.parse_args(argv)
    report = score_all(output=args.output)
    print(json.dumps({"status": report["status"], "gate": report["gate"],
        "score_sha256": report["score_sha256"], "output": str(args.output)}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
