"""Process-local SW0148 parity-binding adapter for safe checkpoint resumes.

The frozen trainer and checkpoint schema remain unchanged. Only the transient
fresh-binding placeholder ``initial_source_Q_parity: None`` is removed before
the original train_arm resume comparison; the saved exact parity record is
then reattached by the original trainer and checked by load_latest.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
               str(ROOT / "snn_kuramoto_bidirectional")]

from collaborative_test.SW_0148_checkpointable_boundary_q import train as trainer  # noqa: E402
from collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint import (  # noqa: E402
    inspect_committed_metadata,
)
from collaborative_test.SW_0148_checkpointable_boundary_q.resume_lineage_v2 import (  # noqa: E402
    current_resume_bindings,
)

PARITY_KEY = "initial_source_Q_parity"
RECEIPT_ROOT = HERE / "results_archive" / "recovery_v2_20261012" / "attempt_receipts"
BOUND_FILES = (
    "checkpoint.py", "run.py", "train.py", "protocol.json", "stage_queue.py",
    "test_checkpoint.py", "test_queue.py", "README.md",
)
EXPECTED_FROZEN_HASHES = {
    "checkpoint.py": "75d5076c051f81a51a39bbafedd363e589aa58ffc41ba4a662a87ffb91601f53",
    "train.py": "c8e2e1081224ed92f4d1158ee6d979b63e881db65c445bf730ec23fed394534b",
    "run.py": "bc895543dba46ea4a61f0761367763f927d6f543d76b9f7553392f91fea2c679",
    "stage_queue.py": "124a05c3f5091f6969d9e852a89fbb5f1fabf97de05af3d60a19b08cd63b2b47",
    "protocol.json": "8a34122b30f8d4fd34e519b7dd20f165588fca1d3aba48f570b270406d5abccb",
    "README.md": "07eb1697fc6242746807b1c3f52e44c67de94894f1a586eba60b5da3a6a23264",
    "test_checkpoint.py": "f71d91b6da7e653d23f7b2132cf1b2744e1f25faff8016f09360ee4a9f3a24cc",
    "test_queue.py": "bc5bbdae2df21d1dfb2333922e59e886dbc41011b62d5caf8d0173a38e9f4a25",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_sha(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verified_parent_hashes() -> dict:
    actual = {name: sha256_file(HERE / name) for name in BOUND_FILES}
    if actual != EXPECTED_FROZEN_HASHES:
        raise RuntimeError("frozen SW0148 source bundle differs from the reviewed original eight files")
    return actual


def _atomic_create_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, path)  # atomic create-only publication
    except FileExistsError:
        temp.unlink()
        raise
    temp.unlink()
    if os.name == "posix":
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


@contextmanager
def install_parity_normalizer(train_module):
    """Temporarily wrap the frozen source-binding builder in this process only."""
    original = train_module._train_source_bindings

    def normalized(*args, **kwargs):
        fresh = original(*args, **kwargs)
        if not isinstance(fresh, dict):
            raise TypeError("trainer source bindings must be a dictionary")
        return current_resume_bindings(fresh)

    train_module._train_source_bindings = normalized
    try:
        yield
    finally:
        train_module._train_source_bindings = original


def _resume_receipt(arm: str, work_root: Path, train_module) -> dict:
    checkpoint_dir = work_root / f"{arm}_seed1" / "checkpoints"
    boundary = checkpoint_dir / "boundary.ptx"
    metadata = None
    if boundary.is_file():
        metadata = inspect_committed_metadata(checkpoint_dir)
        bindings = metadata.get("bindings")
        parity = bindings.get(PARITY_KEY) if isinstance(bindings, dict) else None
        if not isinstance(parity, dict) or parity.get("exact") is not True:
            raise ValueError("resume checkpoint lacks exact original source-Q parity")
        completed = metadata.get("completed_updates")
        runtime = metadata.get("runtime")
        visibility = runtime.get("cuda_visible_devices") if isinstance(runtime, dict) else None
        if (not isinstance(completed, int) or isinstance(completed, bool)
                or not 0 <= completed <= train_module.sw146.UPDATES):
            raise ValueError("resume checkpoint update boundary is invalid")
        if (not isinstance(visibility, str) or visibility not in {"0", "1", "2", "3"}
                or runtime.get("device_type") != "cuda" or runtime.get("device_index") != 0):
            raise ValueError("resume checkpoint runtime is not bound to one mapped CUDA device")
    else:
        completed = 0
        runtime = None
        bindings = None

    frozen = _verified_parent_hashes()
    return {
        "status": "resume_attempt_prepared",
        "experiment": "SW0148_checkpointable_boundary_q",
        "arm": arm,
        "normalization": "drop only current None initial_source_Q_parity; saved exact record is reattached and checked by frozen trainer/load_latest",
        "adapter_sha256": sha256_file(Path(__file__)),
        "lineage_helper_sha256": sha256_file(HERE / "resume_lineage_v2.py"),
        "frozen_sw0148_file_sha256": frozen,
        "checkpoint_boundary_sha256": sha256_file(boundary) if boundary.is_file() else None,
        "completed_updates_at_start": completed,
        "saved_parity_exact": (metadata["bindings"][PARITY_KEY]["exact"]
                                if metadata is not None else None),
        "source_bindings_sha256": _canonical_sha(bindings) if bindings is not None else None,
        "runtime_sha256": _canonical_sha(runtime) if runtime is not None else None,
        "cuda_visible_devices": runtime.get("cuda_visible_devices") if runtime is not None else None,
        "preparing_pid": os.getpid(),
    }


def train_arm(arm: str, *, device="cuda:0", receipt_root=RECEIPT_ROOT, **kwargs):
    """Prepare a durable provenance receipt, then call the unchanged trainer."""
    if arm not in trainer.sw146.ARMS:
        raise ValueError("unregistered SW0148 arm")
    _verified_parent_hashes()
    work_root = Path(kwargs.get("work_root", trainer.WORK))
    receipt = _resume_receipt(arm, work_root, trainer)
    receipt_path = Path(receipt_root) / f"{arm}_seed1_{uuid.uuid4().hex}.json"
    _atomic_create_json(receipt_path, receipt)
    train_kwargs = dict(kwargs)
    train_kwargs["device"] = device
    with install_parity_normalizer(trainer):
        return trainer.train_arm(arm, **train_kwargs)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=trainer.sw146.ARMS, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    result = train_arm(args.arm, device=args.device)
    print(json.dumps({"status": result.get("status", "training_complete"),
                      "arm": args.arm, "completed_updates": result.get("completed_updates")},
                     allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
