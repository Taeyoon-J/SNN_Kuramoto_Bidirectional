"""GT-last CPU score coordinator for the SW0179 three-seed predictions."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

if "--worker" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ["USE_PYTORCH_KERNEL_CACHE"] = "0"

import numpy as np  # Initialize NumPy before legacy Torch imports.

from collaborative_test.SW_0179_qce_seed_replication import score
from collaborative_test.SW_0179_qce_seed_replication import train
from collaborative_test.SW_0179_qce_seed_replication import prediction_queue as predq
from collaborative_test.SW_0178_temporal_class_competition import prediction_queue as seed1q

HERE = Path(__file__).resolve().parent
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
RUNTIME = HERE / "results_archive" / "score_queue_20261016"
EXPERIMENT = "SW0179_score_queue"
ACTIVE_PREDICTION = {"waiting_for_training", "waiting_for_exclusive_gpu",
    "waiting_for_stable_gpu", "running_prediction", "yielding_foreign_owner"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value, *, replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if replace: os.replace(tmp, path)
        else: os.link(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def _state_path():
    return RUNTIME / "state.json"


def _write_state(status, **fields):
    _atomic_json(_state_path(), {"experiment": EXPERIMENT, "status": status,
        "supervisor_pid": os.getpid(), "updated_unix": time.time(), **fields}, replace=True)


def _load(name):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module(name)
    finally:
        if visible is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else: os.environ["CUDA_VISIBLE_DEVICES"] = visible


def _module_live(owner, pid, module, *, seed=None):
    if type(pid) is not int or pid <= 0: return False
    record = owner._proc_record(pid)
    if not record or record.get("state") == "Z": return False
    argv = list(record.get("argv") or [])
    if argv.count("-m") != 1 or "--worker" in argv: return False
    pos = argv.index("-m")
    if pos + 1 >= len(argv) or argv[pos + 1] != module: return False
    if argv.count("--arm") != 1 or argv[argv.index("--arm") + 1:argv.index("--arm") + 2] != ["product_q_ce"]:
        return False
    if seed is not None and (argv.count("--source-seed") != 1
            or argv[argv.index("--source-seed") + 1:argv.index("--source-seed") + 2] != [str(int(seed))]):
        return False
    if seed is None and "--source-seed" in argv: return False
    return True


def _state_record(path, *, experiment, arm, seed=None):
    if not path.is_file(): return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("experiment") != experiment or state.get("arm") != arm:
        raise ValueError("prediction queue experiment/arm identity mismatch")
    if seed is not None and state.get("seed") != int(seed):
        raise ValueError("prediction queue source-seed identity mismatch")
    return {"path": path, "state": state, "state_sha256": _sha(path)}


def _complete_prediction(record, *, seed, queue_experiment):
    state = record["state"]
    if state.get("status") != "complete" or state.get("ground_truth_used_for_prediction") is not False:
        raise ValueError(f"seed{seed} prediction is not a GT-free complete artifact")
    output = Path(state.get("output", ""))
    archive, protocol = output / "predictions.npz", output / "protocol.json"
    if not output.is_dir() or not archive.is_file() or not protocol.is_file():
        raise ValueError(f"seed{seed} prediction archive/protocol is missing")
    pred_sha, proto_sha = _sha(archive), _sha(protocol)
    protocol_data = json.loads(protocol.read_text(encoding="utf-8"))
    if (state.get("prediction_sha256") != pred_sha or state.get("protocol_sha256") != proto_sha
            or protocol_data.get("status") != "complete"
            or protocol_data.get("ground_truth_used_for_prediction") is not False
            or protocol_data.get("prediction_sha256") != pred_sha):
        raise ValueError(f"seed{seed} prediction receipt/hash mismatch")
    if seed in (0, 2):
        if (protocol_data.get("experiment") != "SW0179_fixed320_prediction"
                or protocol_data.get("seed") != seed or protocol_data.get("arm") != train.ARMS[0]):
            raise ValueError(f"seed{seed} SW0179 prediction protocol identity mismatch")
    else:
        if (queue_experiment != "SW0178_prediction_queue_v1"
                or protocol_data.get("experiment") != "SW0178_temporal_class_competition_fixed320_prediction"
                or protocol_data.get("arm") != "product_q_ce"):
            raise ValueError("seed1 immutable SW0178 prediction protocol identity mismatch")
    return {**record, "prediction_sha256": pred_sha, "protocol_sha256": proto_sha,
        "prediction_path": archive, "protocol_path": protocol}


def _prediction_dependencies(owner):
    found, rejected = {}, {}
    new_queue = _load(predq.MODULE)
    for seed in (0, 2):
        record = _state_record(new_queue._state_path(seed), experiment="SW0179_prediction_queue",
            arm=train.ARMS[0], seed=seed)
        if record is None: return None
        state = record["state"]
        if state.get("status") == "complete_training_rejected":
            if state.get("ground_truth_used_for_prediction") is not False:
                raise ValueError(f"seed{seed} rejected prediction state has invalid GT scope")
            rejected[seed] = record
            continue
        if state.get("status") != "complete":
            if (state.get("status") in ACTIVE_PREDICTION and state.get("seed") == seed
                    and _module_live(owner, state.get("supervisor_pid"), predq.MODULE, seed=seed)):
                return False
            raise RuntimeError(f"SW0179 seed{seed} prediction failed/orphaned: {state.get('status')}")
        found[seed] = _complete_prediction(record, seed=seed, queue_experiment="SW0179_prediction_queue")

    old_queue = _load(seed1q.MODULE)
    arm = "product_q_ce"
    record = _state_record(old_queue._state_path(arm), experiment="SW0178_prediction_queue_v1", arm=arm)
    if record is None: return None
    state = record["state"]
    if state.get("status") != "complete":
        if state.get("status") in ACTIVE_PREDICTION and _module_live(
                owner, state.get("supervisor_pid"), seed1q.MODULE):
            return False
        raise RuntimeError(f"SW0178 seed1 prediction failed/orphaned: {state.get('status')}")
    found[1] = _complete_prediction(record, seed=1, queue_experiment="SW0178_prediction_queue_v1")
    if rejected:
        return {"rejected": rejected, "completed": found}
    return found


def _queue_fingerprint(prepared):
    return {"experiment": EXPERIMENT,
        "score_implementation_fingerprint": score.implementation_fingerprint(),
        "score_queue_sha256": _sha(__file__),
        "frozen_candidate_predictions": {str(seed): {
            "prediction_sha256": row["result"]["prediction_sha256"],
            "protocol_sha256": row["result"]["protocol_sha256"],
            "state_sha256": row["state_sha256"]}
            for seed, row in prepared["frozen_predictions"].items()},
        "baseline_summary_sha256": prepared["baseline"]["summary_sha256"],
        "cache_mode": "USE_PYTORCH_KERNEL_CACHE=0"}


def run():
    import fcntl
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0179 score coordinator must remain CUDA hidden")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = _state_path()
    if state_path.exists() or (RUNTIME / "score.log").exists():
        raise FileExistsError("preserving existing SW0179 score queue state/log")
    lock = (RUNTIME / "score.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    own_state = False
    try:
        owner = _load(OWNER_MODULE)
        _write_state("waiting_for_predictions", ground_truth_used_for_prediction=False,
            ground_truth_used_for_scoring=False)
        own_state = True
        while True:
            dependencies = _prediction_dependencies(owner)
            if dependencies is not None and dependencies is not False: break
            _write_state("waiting_for_predictions", ground_truth_used_for_prediction=False,
                ground_truth_used_for_scoring=False)
            time.sleep(5)
        if dependencies.get("rejected"):
            final = {"status": "complete_candidate_rejected", "rejected_seeds": sorted(dependencies["rejected"]),
                "prediction_state_sha256": {str(seed): row["state_sha256"]
                    for seed, row in dependencies["completed"].items()},
                "rejection_state_sha256": {str(seed): row["state_sha256"]
                    for seed, row in dependencies["rejected"].items()},
                "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": False}
            _write_state(final["status"], **{key: value for key, value in final.items() if key != "status"})
            return final

        # CPU admission validates model, prediction and baseline contracts without reading GT.
        prepared = score.prepare_score_inputs_cpu()
        dependency_hashes = {str(seed): {"prediction_sha256": row["prediction_sha256"],
            "protocol_sha256": row["protocol_sha256"], "state_sha256": row["state_sha256"]}
            for seed, row in dependencies.items()}
        actual_hashes = {str(seed): {"prediction_sha256": row["result"]["prediction_sha256"],
            "protocol_sha256": row["result"]["protocol_sha256"], "state_sha256": row["state_sha256"]}
            for seed, row in prepared["frozen_predictions"].items()}
        if dependency_hashes != actual_hashes:
            raise ValueError("prediction queue dependencies changed during CPU score admission")
        fingerprint = _queue_fingerprint(prepared)
        _write_state("scoring_gt_last", queue_fingerprint=fingerprint,
            candidate_prediction_hashes={seed: values for seed, values in fingerprint["frozen_candidate_predictions"].items()},
            ground_truth_used_for_prediction=False, ground_truth_used_for_scoring=True,
            optimizer_updates=0)
        report = score.score_all(prepared=prepared)
        after = _prediction_dependencies(owner)
        if after is None or after is False or after.get("rejected"):
            raise ValueError("SW0179 prediction dependencies changed during GT score")
        after_hashes = {str(seed): {"prediction_sha256": row["prediction_sha256"],
            "protocol_sha256": row["protocol_sha256"], "state_sha256": row["state_sha256"]}
            for seed, row in after.items()}
        if after_hashes != dependency_hashes:
            raise ValueError("SW0179 prediction queue receipts changed during GT scoring")
        score_sha = _sha(score.SCORE_PATH)
        final = {"status": "complete", "score_path": str(score.SCORE_PATH),
            "score_sha256": score_sha, "gate": report["gate"],
            "queue_fingerprint": fingerprint,
            "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
            "finished_unix": time.time()}
        _write_state("complete", **{key: value for key, value in final.items() if key != "status"})
        return final
    except BaseException as exc:
        if own_state:
            _write_state("failed", error=repr(exc), ground_truth_used_for_prediction=False,
                finished_unix=time.time())
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.worker: parser.error("SW0179 score queue has no worker mode")
    result = run()
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
