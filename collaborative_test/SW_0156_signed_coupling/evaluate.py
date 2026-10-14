"""GT-last fixed320 native32 score for the SW0156 seed1 signed-coupling screen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import evaluation_contract32 as contract
from collaborative_test.SW_0135_native32_spike_binding import score_slot32
from collaborative_test.SW_0143_native32_selective_geodesic import run as sw143
from collaborative_test.SW_0156_signed_coupling import predict

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(sw130.source97.DATASET)
SLOT_BASELINE = sw143.SLOT_BASELINE
SLOT_ROOT = sw143.SLOT_ROOT
OUTPUT = HERE / "results_archive" / "evaluation_seed1.json"
ARMS = ("source_T", "signed_candidate", "matched_control")
BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES = 139, 10_000


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_identity(path):
    requested = Path(path).resolve(strict=True)
    registered = Path(sw130.source97.DATASET).resolve(strict=True)
    if requested != registered:
        raise ValueError("score target must be the registered SW0097 HDF5 dataset")
    stat = requested.stat()
    return {"resolved_path": str(requested), "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns)}


def implementation_fingerprint():
    paths = (HERE / "evaluate.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
             ROOT / "collaborative_test/SW_0135_native32_spike_binding/score_slot32.py")
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("SW0156 evaluation dependency missing")
    result = predict.fingerprint()
    result.update({path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths})
    return result


def paired_image_bootstrap(left, right, *, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != (320,) or right.shape != (320,) or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired bootstrap requires two finite 320-image vectors")
    delta = left - right
    rng = np.random.RandomState(seed)
    draw = rng.randint(0, 320, size=(replicates, 320))
    samples = delta[draw].mean(axis=1)
    return {"mean": float(delta.mean()),
            "ci95": [float(value) for value in np.percentile(samples, [2.5, 97.5])],
            "per_image_count": 320, "common_image_resamples": True,
            "seed": int(seed), "replicates": int(replicates)}


def _read_slot_seed1():
    directory = SLOT_ROOT / "seed1_epoch10"
    prediction_path = directory / "predictions.npz"
    protocol_path = directory / "protocol.json"
    protocol = score_slot32._read_json(protocol_path)
    slot_sha = contract.validate_slot_protocol(protocol, seed=1, prediction_path=prediction_path)
    with np.load(prediction_path, allow_pickle=False) as saved:
        if not {"labels", "image_ids"}.issubset(saved.files):
            raise ValueError("Slot seed1 archive schema invalid")
        ids = contract.require_fixed_image_ids(saved["image_ids"], name="Slot IDs")
        labels_native = np.asarray(saved["labels"])
    labels32 = contract.slot_native_pixels_to32(labels_native)
    return {"ids": ids, "labels": labels32, "native_path": prediction_path,
            "protocol_path": protocol_path, "npz_sha": slot_sha,
            "protocol_sha": sha256_file(protocol_path), "protocol": protocol}


def score_all(*, dataset=DATASET, output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing SW0156 evaluation: {output}")
    expected_fp = predict.fingerprint()
    loaded = {arm: predict.validate_prediction(
        predict.prediction_dir(arm), arm=arm, expected_fingerprint=expected_fp) for arm in ARMS}
    ids = contract.require_fixed_image_ids(loaded["source_T"]["ids"])
    for arm in ARMS:
        if not np.array_equal(loaded[arm]["ids"], ids):
            raise ValueError("SW0156 prediction IDs differ across arms")
    slot = _read_slot_seed1()
    if not np.array_equal(slot["ids"], ids):
        raise ValueError("Slot seed1 and SW0156 fixed320 IDs differ")
    dataset_identity = _dataset_identity(dataset)
    if sha256_file(SLOT_BASELINE) != sw143.SLOT_BASELINE_SHA256:
        raise ValueError("registered Slot32 baseline summary SHA changed")
    baseline = json.loads(SLOT_BASELINE.read_text(encoding="utf-8"))

    # Re-hash every required prediction/reference immediately before the one
    # ground-truth open. No masks are used by prediction or admission.
    for arm, row in loaded.items():
        if (sha256_file(row["npz_path"]) != row["prediction_sha256"]
                or sha256_file(row["protocol_path"]) != row["protocol_sha256"]):
            raise ValueError(f"frozen {arm} prediction changed before GT access")
    if (sha256_file(slot["native_path"]) != slot["npz_sha"]
            or sha256_file(slot["protocol_path"]) != slot["protocol_sha"]
            or sha256_file(SLOT_BASELINE) != sw143.SLOT_BASELINE_SHA256):
        raise ValueError("frozen Slot reference changed before GT access")
    if _dataset_identity(dataset) != dataset_identity:
        raise ValueError("registered HDF5 identity changed before GT access")
    with h5py.File(dataset, "r") as h5:
        if "mask" not in h5 or h5["mask"].shape[0] < 1640:
            raise ValueError("registered HDF5 mask rows through1639 are required")
        pixel_gt = np.asarray(h5["mask"][1320:1640])
    if _dataset_identity(dataset) != dataset_identity:
        raise ValueError("registered HDF5 identity changed during GT read")
    gt = contract.modal_native32(pixel_gt)

    scored = {arm: contract.production_metrics32(loaded[arm]["labels"], gt) for arm in ARMS}
    slot_scores = contract.production_metrics32(slot["labels"], gt)
    means = {arm: {metric: float(scored[arm][metric]["mean"]) for metric in contract.METRICS}
             for arm in ARMS}
    slot_three_seed_means = {metric: float(baseline["scores"]["slot32"][metric]["three_seed_mean"])
                             for metric in contract.METRICS}
    paired = {}
    for other in ("source_T", "matched_control"):
        paired[f"signed_candidate-{other}"] = {
            metric: paired_image_bootstrap(
                scored["signed_candidate"][metric]["per_image"],
                scored[other][metric]["per_image"])
            for metric in contract.METRICS}
    gates = {
        "candidate_fg_ari_at_least_0.01_above_source_T":
            means["signed_candidate"]["fg_ari"] >= means["source_T"]["fg_ari"] + .01,
        "candidate_fg_ari_at_least_0.01_above_control":
            means["signed_candidate"]["fg_ari"] >= means["matched_control"]["fg_ari"] + .01,
        "candidate_minus_source_T_fg_ci_lower_positive":
            paired["signed_candidate-source_T"]["fg_ari"]["ci95"][0] > 0,
        "candidate_minus_control_fg_ci_lower_positive":
            paired["signed_candidate-matched_control"]["fg_ari"]["ci95"][0] > 0,
    }
    report = {
        "experiment": "SW0156_native32_signed_coupling", "status": "complete",
        "seed": 1, "arms": list(ARMS), "implementation": implementation_fingerprint(),
        "prediction_implementation": expected_fp,
        "evaluation_contract": {"image_ids": [1320, 1639, 320], "grid_size": [32, 32],
            "metrics": list(contract.METRICS), "readout": "unchanged production actual-spike QCC",
            "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True},
        "scores": {arm: scored[arm] for arm in ARMS}, "slot_seed1": slot_scores,
        "means": means, "slot32_registered_three_seed_means": slot_three_seed_means,
        "paired_bootstrap": paired,
        "bootstrap": {"rng": "numpy RandomState", "seed": BOOTSTRAP_SEED,
                      "replicates": BOOTSTRAP_REPLICATES, "common_fixed320_indices": True},
        "registered_gate": gates, "all_registered_gate_conditions_pass": all(gates.values()),
        "prediction_provenance": {arm: {"prediction_sha256": loaded[arm]["prediction_sha256"],
            "protocol_sha256": loaded[arm]["protocol_sha256"]} for arm in ARMS},
        "slot_provenance": {"prediction_sha256": slot["npz_sha"],
            "protocol_sha256": slot["protocol_sha"],
            "checkpoint_sha256": slot["protocol"].get("checkpoint_sha256"),
            "registered_baseline_sha256": sw143.SLOT_BASELINE_SHA256},
        "ground_truth_dataset_identity": dataset_identity,
        "ground_truth_used_for_prediction": False,
        "ground_truth_used_for_scoring": True,
        "ground_truth_opened_after_all_predictions_validated": True,
        "optimizer_updates": 0,
        "interpretation_limits": "Single seed fixed320 evaluation; no scaling or mechanism-necessity claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def validate_score(path=OUTPUT):
    """Validate the completed receipt and rebind it to current frozen inputs."""
    try:
        path = Path(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("experiment") != "SW0156_native32_signed_coupling"
                or record.get("status") != "complete" or record.get("seed") != 1
                or record.get("arms") != list(ARMS)
                or record.get("implementation") != implementation_fingerprint()
                or record.get("ground_truth_opened_after_all_predictions_validated") is not True
                or record.get("ground_truth_used_for_prediction") is not False
                or record.get("ground_truth_used_for_scoring") is not True
                or record.get("optimizer_updates") != 0):
            return False
        if sha256_file(SLOT_BASELINE) != sw143.SLOT_BASELINE_SHA256:
            return False
        if record.get("ground_truth_dataset_identity") != _dataset_identity(DATASET):
            return False
        provenance = record.get("prediction_provenance")
        if not isinstance(provenance, dict) or set(provenance) != set(ARMS):
            return False
        for arm in ARMS:
            row = predict.validate_prediction(predict.prediction_dir(arm), arm=arm)
            if (provenance[arm].get("prediction_sha256") != row["prediction_sha256"]
                    or provenance[arm].get("protocol_sha256") != row["protocol_sha256"]):
                return False
        scores = record.get("scores")
        if not isinstance(scores, dict) or set(scores) != set(ARMS):
            return False
        for arm in ARMS:
            if set(scores[arm]) != set(contract.METRICS):
                return False
            for metric in contract.METRICS:
                item = scores[arm].get(metric)
                if (not isinstance(item, dict) or not np.isfinite(item.get("mean"))
                        or np.asarray(item.get("per_image")).shape != (320,)
                        or not np.isfinite(np.asarray(item["per_image"], dtype=np.float64)).all()):
                    return False
        gates = record.get("registered_gate")
        return isinstance(gates, dict) and set(gates) == {
            "candidate_fg_ari_at_least_0.01_above_source_T",
            "candidate_fg_ari_at_least_0.01_above_control",
            "candidate_minus_source_T_fg_ci_lower_positive",
            "candidate_minus_control_fg_ci_lower_positive"} \
            and all(type(value) is bool for value in gates.values()) \
            and record.get("all_registered_gate_conditions_pass") is all(gates.values())
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


if __name__ == "__main__":
    result = score_all()
    print(json.dumps({"status": result["status"], "gates": result["registered_gate"]},
                     allow_nan=False), flush=True)
