"""GT-last score coordinator consuming both frozen SW0178 predictions."""
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

import numpy as np  # Keep NumPy/OpenMP initialization ahead of legacy Torch imports.

from collaborative_test.SW_0178_temporal_class_competition import predict_evaluate as evaluate
from collaborative_test.SW_0178_temporal_class_competition import train

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "results_archive" / "score_queue_v1_20261016"
PREDICTION_RUNTIME = HERE / "results_archive" / "prediction_queue_v1_20261016"
PREDICTION_MODULE = "collaborative_test.SW_0178_temporal_class_competition.prediction_queue"
OWNER_MODULE = "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4"
EXPERIMENT = "SW0178_score_queue_v1"
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
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if replace: os.replace(temp, path)
        else: os.link(temp, path)
    finally:
        if temp.exists(): temp.unlink()


def _state_path():
    return RUNTIME / "state.json"


def _write_state(status, **fields):
    _atomic_json(_state_path(), {"experiment": EXPERIMENT, "status": status,
        "supervisor_pid": os.getpid(), "updated_unix": time.time(), **fields}, replace=True)


def _prediction_state_path(arm):
    if arm not in train.ARMS: raise ValueError("unregistered SW0178 prediction arm")
    return PREDICTION_RUNTIME / f"{arm}_state.json"


def _module_live(owner, pid, module, arm):
    if type(pid) is not int or pid <= 0: return False
    record = owner._proc_record(pid)
    if not record or record.get("state") == "Z": return False
    argv = record.get("argv", [])
    if argv.count("-m") != 1: return False
    pos = argv.index("-m")
    if pos + 1 >= len(argv) or argv[pos + 1] != module or "--worker" in argv:
        return False
    return argv.count("--arm") == 1 and argv[argv.index("--arm") + 1:argv.index("--arm") + 2] == [arm]


def _prediction_is_pending(owner, state, arm):
    return (state.get("status") in ACTIVE_PREDICTION
            and _module_live(owner, state.get("supervisor_pid"), PREDICTION_MODULE, arm))


def _validate_prediction_receipt(arm, state, *, prediction_path, protocol_path,
                                prediction_sha256, protocol_sha256, protocol_data):
    """Pure producer/consumer check; all bytes and protocol fields are explicit."""
    if (state.get("experiment") != "SW0178_prediction_queue_v1"
            or state.get("arm") != arm or state.get("status") != "complete"
            or state.get("prediction_sha256") != prediction_sha256
            or state.get("protocol_sha256") != protocol_sha256
            or state.get("ground_truth_used_for_prediction") is not False):
        raise ValueError(f"SW0178 {arm} complete prediction receipt/hash mismatch")
    if (protocol_data.get("experiment") != "SW0178_temporal_class_competition_fixed320_prediction"
            or protocol_data.get("arm") != arm or protocol_data.get("status") != "complete"
            or protocol_data.get("ground_truth_used_for_prediction") is not False
            or protocol_data.get("prediction_sha256") != state["prediction_sha256"]):
        raise ValueError(f"SW0178 {arm} prediction protocol does not match queue state")
    return {"state": state, "prediction_path": str(prediction_path),
            "protocol_path": str(protocol_path)}


def _prediction_dependency(owner):
    """Return None while not started, False while either arm is active, else receipts."""
    found, rejected = {}, []
    for arm in train.ARMS:
        path = _prediction_state_path(arm)
        if not path.is_file():
            return None
        state = json.loads(path.read_text(encoding="utf-8"))
        if (state.get("experiment") != "SW0178_prediction_queue_v1"
                or state.get("arm") != arm):
            raise ValueError("SW0178 prediction queue state identity mismatch")
        if state.get("status") == "complete_training_rejected":
            if state.get("ground_truth_used_for_prediction") is not False:
                raise ValueError("SW0178 rejected-training prediction state has invalid GT scope")
            rejected.append({"arm": arm, "state_sha256": _sha(path),
                             "training_rejections": state.get("training_rejections", [])})
            continue
        if state.get("status") != "complete":
            if _prediction_is_pending(owner, state, arm):
                return False
            raise RuntimeError(f"SW0178 {arm} prediction failed or is orphaned: {state.get('status')}")
        output = Path(state.get("output", ""))
        archive, protocol = output / "predictions.npz", output / "protocol.json"
        if not output.is_dir() or not archive.is_file() or not protocol.is_file():
            raise ValueError(f"SW0178 {arm} complete prediction receipt/hash mismatch")
        protocol_data = json.loads(protocol.read_text(encoding="utf-8"))
        receipt = _validate_prediction_receipt(arm, state, prediction_path=archive,
            protocol_path=protocol, prediction_sha256=_sha(archive),
            protocol_sha256=_sha(protocol), protocol_data=protocol_data)
        found[arm] = {**receipt, "state_sha256": _sha(path)}
    rejected_by_arm = {row["arm"]: row for row in rejected}
    if set(found) | set(rejected_by_arm) != set(train.ARMS):
        return None
    return {**found, "_rejected": rejected_by_arm}


def run(*, active_timeout_seconds=4 * 3600):
    import fcntl
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("SW0178 score coordinator must remain CUDA hidden")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    state_path = _state_path()
    if state_path.exists() or (RUNTIME / "score.log").exists():
        raise FileExistsError("preserving existing SW0178 score queue state/log")
    lock = (RUNTIME / "score.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    owned_state = False
    try:
        owner = importlib.import_module(OWNER_MODULE)
        _write_state("waiting_for_predictions", ground_truth_used_for_prediction=False)
        owned_state = True
        # No wall-clock timeout while predictions are queued/blocked by GPU owners.
        while True:
            deps = _prediction_dependency(owner)
            if deps is not None and deps is not False:
                break
            _write_state("waiting_for_predictions", ground_truth_used_for_prediction=False)
            time.sleep(5)

        if deps.get("_rejected") and not any(arm in deps for arm in train.ARMS):
            final = {"experiment": EXPERIMENT, "status": "complete_training_rejected",
                "supervisor_pid": os.getpid(), "prediction_state_sha256": {
                    arm: deps["_rejected"][arm]["state_sha256"] for arm in deps["_rejected"]},
                "training_rejections": list(deps["_rejected"].values()),
                "ground_truth_used_for_prediction": False,
                "ground_truth_used_for_scoring": False, "finished_unix": time.time()}
            _write_state(final["status"], **{k: v for k, v in final.items()
                if k not in {"experiment", "status", "supervisor_pid"}})
            return final

        # This validation loads references and model manifests but does not open GT.
        prepared = evaluate.prepare_prediction_inputs_cpu()
        before = evaluate._prediction_receipts(prepared)
        for arm in prepared["manifests"]:
            if before[arm]["state_sha256"] != deps[arm]["state_sha256"]:
                raise ValueError("SW0178 prediction queue state changed before GT validation")
        for arm in prepared["rejected"]:
            if before["_rejected"][arm]["state_sha256"] != deps["_rejected"][arm]["state_sha256"]:
                raise ValueError("SW0178 rejected prediction state changed before GT validation")
        evaluate.implementation_fingerprint()
        _write_state("scoring_gt_last", prediction_state_sha256={
            arm: deps[arm]["state_sha256"] for arm in prepared["manifests"]},
            prediction_sha256={arm: deps[arm]["state"]["prediction_sha256"] for arm in prepared["manifests"]},
            rejected_training_state_sha256={arm: deps["_rejected"][arm]["state_sha256"]
                                             for arm in prepared["rejected"]},
            optimizer_updates=0, ground_truth_used_for_prediction=False)
        report = evaluate.score_all(prepared=prepared)
        # Scoring revalidates every prediction after the HDF5 read. Recheck queue
        # receipts as well, so a replaced state file cannot be hidden by the report.
        after = _prediction_dependency(owner)
        if after is None or after is False or any(
                after[arm]["state_sha256"] != deps[arm]["state_sha256"] for arm in prepared["manifests"]):
            raise ValueError("SW0178 prediction queue receipts changed during GT scoring")
        if any(after["_rejected"][arm]["state_sha256"] != deps["_rejected"][arm]["state_sha256"]
               for arm in prepared["rejected"]):
            raise ValueError("SW0178 rejected prediction state changed during GT scoring")
        score_sha = _sha(evaluate.SCORE_PATH)
        final = {"experiment": EXPERIMENT, "status": "complete", "supervisor_pid": os.getpid(),
            "prediction_state_sha256": {arm: deps[arm]["state_sha256"] for arm in prepared["manifests"]},
            "rejected_training_state_sha256": {arm: deps["_rejected"][arm]["state_sha256"]
                                               for arm in prepared["rejected"]},
            "prediction_sha256": {arm: deps[arm]["state"]["prediction_sha256"]
                                  for arm in prepared["manifests"]},
            "score_sha256": score_sha, "score_path": str(evaluate.SCORE_PATH),
            "implementation_fingerprint": evaluate.implementation_fingerprint(),
            "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
            "gate": report["gate"], "finished_unix": time.time()}
        _write_state("complete", **{k: v for k, v in final.items()
                                     if k not in {"experiment", "status", "supervisor_pid"}})
        return final
    except BaseException as exc:
        if owned_state:
            _write_state("failed", error=repr(exc), finished_unix=time.time())
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.worker: parser.error("SW0178 score queue has no worker mode")
    print(json.dumps(run(), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
