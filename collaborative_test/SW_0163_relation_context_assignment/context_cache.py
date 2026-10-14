"""Create-once frozen-source Q/A contexts for the registered first512 TRAIN rows.

Actual component spikes are read from the validated SW0161 cache. Q is computed
with the production spike-affinity function. A is captured from the unchanged
SW0143-T graph and native ``prepare_coupling`` path, in original B4 groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]

import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_operator, run as sw143
from collaborative_test.SW_0155_supervised_spike_relations import train as sw155
from collaborative_test.SW_0161_supervised_spike_partition import cache as source_cache
from collaborative_test.SW_0162_connected_signed_partition.preflight import capture_source_coupling
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity

COUNT, BATCH, N, SETTLE, STEPS = 512, 4, 1024, 512, 1024
EXPECTED_SOURCE_CACHE_MANIFEST_SHA256 = "b4125a81012bf46cf38ff512bf251452f511524c42a2d2364c34783de04302b3"
CACHE_ROOT = HERE / "results_archive" / "relation_context_cache_20261014"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _json_sha(value):
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8"))


def _atomic_json(path, value, *, create_only):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{uuid.uuid4().hex[:8]}.tmp")
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    try:
        with temp.open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        if create_only:
            os.link(temp, path)
        else:
            os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _fingerprint():
    relatives = (
        "collaborative_test/SW_0163_relation_context_assignment/context_cache.py",
        "collaborative_test/SW_0163_relation_context_assignment/model.py",
        "collaborative_test/SW_0163_relation_context_assignment/protocol.json",
        "collaborative_test/SW_0161_supervised_spike_partition/cache.py",
        "collaborative_test/SW_0161_supervised_spike_partition/protocol.json",
        "collaborative_test/SW_0130_phase_state_integration/run.py",
        "collaborative_test/SW_0134_native_spike_binding/rollout.py",
        "collaborative_test/SW_0135_native32_spike_binding/foundation.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/run.py",
        "collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
        "collaborative_test/SW_0162_connected_signed_partition/preflight.py",
        "collaborative_test/SW_0155_supervised_spike_relations/train.py",
        "collaborative_test/SW_0155_supervised_spike_relations/objective.py",
        "snn_kuramoto_bidirectional/spike_classifier.py",
    )
    paths = [ROOT / rel for rel in relatives]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"SW0163 context-cache dependency missing: {missing[0]}")
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _make_context_bindings(prepared, source_trace_manifest):
    source = prepared["source_contract"]
    return {
        "experiment": "SW0163_relation_context_assignment",
        "source_seed": 1,
        "source_checkpoint_sha256": source["source_sha256"],
        "source_manifest_sha256": sha256_file(source["manifest_path"]),
        "ordered_image_ids": [int(value) for value in prepared["image_ids"]],
        "source_rows": np.asarray(source["rows"][:COUNT], dtype=np.int64).tolist(),
        "rgb_assets": prepared["rgb_asset_validation"],
        "source_trace_cache_manifest_sha256": sha256_file(source_cache.CACHE_ROOT / "manifest.json"),
        "source_trace_cache_bindings_sha256": source_trace_manifest["bindings_sha256"],
        "source_trace_cache_implementation_fingerprint": source_trace_manifest["bindings"].get(
            "cache_implementation_fingerprint"),
        "rgb_sha256": prepared["rgb_sha256"],
        "train_rgb_size": Path(sw130.TRAIN_RGB).stat().st_size,
        "train_rgb_mtime_ns": Path(sw130.TRAIN_RGB).stat().st_mtime_ns,
        "hdf5_path": str(Path(sw155.HDF5_PATH).resolve()),
        "hdf5_size": Path(sw155.HDF5_PATH).stat().st_size,
        "hdf5_mtime_ns": Path(sw155.HDF5_PATH).stat().st_mtime_ns,
        "code_fingerprint": _fingerprint(),
        "protocol_sha256": sha256_file(HERE / "protocol.json"),
        "context_semantics": {
            "Q": "production spike_synchrony_affinity over saved settled float32 S; settle=0",
            "A": "captured SW0143-T graph output passed through native prepare_coupling tuple[0]",
            "A_method": "static source RGB encode and graph forward in original B4 groups; no extra rollout",
        },
    }


def prepare_context_inputs_cpu():
    """Validate actual source/cache/RGB/TRAIN inputs before CUDA initialization."""
    prepared = source_cache.prepare_cache_cpu()
    manifest = source_cache.validate_final_cache(
        source_cache.CACHE_ROOT, expected_bindings=prepared["bindings"])
    if manifest.get("status") != "complete" or manifest.get("count") != COUNT:
        raise ValueError("SW0161 source spike cache is incomplete")
    actual_manifest_sha = sha256_file(source_cache.CACHE_ROOT / "manifest.json")
    if actual_manifest_sha != EXPECTED_SOURCE_CACHE_MANIFEST_SHA256:
        raise ValueError("SW0161 source-cache manifest differs from the registered exact bytes")
    prepared["source_trace_manifest"] = manifest
    prepared["context_bindings"] = _make_context_bindings(prepared, manifest)
    return prepared


def source_bindings():
    """Compatibility API backed by the exact production source/cache validator."""
    return prepare_context_inputs_cpu()["context_bindings"]


def _context_arrays(q, adjacency):
    q = np.asarray(q)
    adjacency = np.asarray(adjacency)
    if q.dtype != np.float32 or adjacency.dtype != np.float32:
        raise ValueError("context cache Q/A must remain float32")
    if q.shape != (N, N) or adjacency.shape != (N, N):
        raise ValueError("context cache rows must each be [1024,1024]")
    if not np.isfinite(q).all() or not np.isfinite(adjacency).all():
        raise ValueError("context cache contains nonfinite Q/A")
    if q.min() < 0 or q.max() > 1 or adjacency.min() < 0:
        raise ValueError("context Q/A affinities must be nonnegative and Q in [0,1]")
    if not np.array_equal(q, q.T) or not np.array_equal(adjacency, adjacency.T):
        raise ValueError("context Q/A violates native symmetric affinity schema")
    return np.ascontiguousarray(q), np.ascontiguousarray(adjacency)


class AtomicContextCache:
    """Create-only per-image Q/A storage bound to exact source IDs and RGB."""
    def __init__(self, root, bindings):
        self.root = Path(root)
        self.rows_dir = self.root / "rows"
        self.bindings = bindings
        self.bindings_sha256 = _json_sha(bindings)
        self.ids = bindings["ordered_image_ids"]
        self.records = []
        if self.root.exists():
            self._open_existing()
        else:
            self.rows_dir.mkdir(parents=True, exist_ok=False)
            _atomic_json(self.root / "bindings.json", {"bindings": bindings,
                         "bindings_sha256": self.bindings_sha256}, create_only=True)
            _atomic_json(self.root / "ledger.json", {"rows": []}, create_only=True)

    def row_path(self, index):
        return self.rows_dir / f"{index:04d}.npz"

    def _verify(self, index, record):
        path = self.row_path(index)
        if not path.is_file():
            raise ValueError(f"missing SW0163 Q/A row {index}")
        if record is not None and (record.get("index") != index
                or record.get("image_id") != self.ids[index]
                or record.get("row_file_sha256") != sha256_file(path)):
            raise ValueError(f"SW0163 Q/A row {index} ledger identity/hash mismatch")
        with np.load(path, allow_pickle=False) as row:
            q, a = _context_arrays(row["q"], row["adjacency"])
            rgb_sha = self.bindings["rgb_sha256"][index]
            if (int(row["index"]) != index or int(row["image_id"]) != self.ids[index]
                    or str(row["rgb_sha256"].item()) != rgb_sha
                    or str(row["bindings_sha256"].item()) != self.bindings_sha256):
                raise ValueError(f"SW0163 Q/A row {index} source binding changed")
        result = {"index": index, "image_id": self.ids[index], "rgb_sha256": rgb_sha,
                  "q_sha256": _sha_bytes(q.tobytes()),
                  "adjacency_sha256": _sha_bytes(a.tobytes()),
                  "row_file_sha256": sha256_file(path)}
        if record is not None and result != record:
            raise ValueError(f"SW0163 Q/A row {index} payload digest mismatch")
        return result

    def _open_existing(self):
        bindings_path, ledger_path = self.root / "bindings.json", self.root / "ledger.json"
        if not bindings_path.is_file() or not ledger_path.is_file() or not self.rows_dir.is_dir():
            raise ValueError("preserving incomplete SW0163 context-cache root")
        saved = json.loads(bindings_path.read_text(encoding="utf-8"))
        if saved != {"bindings": self.bindings, "bindings_sha256": self.bindings_sha256}:
            raise ValueError("existing SW0163 Q/A cache source/RGB binding differs")
        records = json.loads(ledger_path.read_text(encoding="utf-8")).get("rows")
        if not isinstance(records, list) or len(records) > COUNT:
            raise ValueError("invalid SW0163 context-cache ledger")
        for index, record in enumerate(records):
            self.records.append(self._verify(index, record))
        while len(self.records) < COUNT and self.row_path(len(self.records)).is_file():
            self.records.append(self._verify(len(self.records), None))
            _atomic_json(ledger_path, {"rows": self.records}, create_only=False)
        expected = {self.row_path(index).name for index in range(len(self.records))}
        if any(path.name not in expected for path in self.rows_dir.glob("*.npz")):
            raise ValueError("unexpected SW0163 context row; preserving existing artifacts")

    def load(self, index):
        if not 0 <= index < len(self.records):
            raise ValueError("SW0163 context row is not committed")
        self._verify(index, self.records[index])
        with np.load(self.row_path(index), allow_pickle=False) as row:
            return row["q"].copy(), row["adjacency"].copy()

    def write(self, index, q, adjacency, rgb_sha256):
        if not 0 <= index < COUNT or rgb_sha256 != self.bindings["rgb_sha256"][index]:
            raise ValueError("context row does not match the registered source index/RGB")
        if index < len(self.records):
            old_q, old_a = self.load(index)
            new_q, new_a = _context_arrays(q, adjacency)
            if not np.array_equal(old_q, new_q) or not np.array_equal(old_a, new_a):
                raise ValueError("live SW0163 context differs from committed Q/A row")
            return self.records[index]
        if index != len(self.records):
            raise ValueError("context rows must publish exactly once in source order")
        q, adjacency = _context_arrays(q, adjacency)
        path = self.row_path(index)
        temp = path.with_name(f".{path.stem}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with temp.open("xb") as stream:
                np.savez(stream, q=q, adjacency=adjacency,
                         image_id=np.int64(self.ids[index]), index=np.int64(index),
                         rgb_sha256=np.asarray(rgb_sha256),
                         bindings_sha256=np.asarray(self.bindings_sha256))
                stream.flush(); os.fsync(stream.fileno())
            os.link(temp, path)
        finally:
            if temp.exists():
                temp.unlink()
        record = {"index": index, "image_id": self.ids[index], "rgb_sha256": rgb_sha256,
                  "q_sha256": _sha_bytes(q.tobytes()), "adjacency_sha256": _sha_bytes(adjacency.tobytes()),
                  "row_file_sha256": sha256_file(path)}
        self.records.append(record)
        _atomic_json(self.root / "ledger.json", {"rows": self.records}, create_only=False)

    def finalize(self, *, source_state_before, source_state_after, b4_replay):
        if len(self.records) != COUNT:
            raise ValueError("cannot finalize incomplete SW0163 context cache")
        if (not _is_sha256(source_state_before) or not _is_sha256(source_state_after)
                or source_state_before != source_state_after):
            raise ValueError("frozen source model changed while generating Q/A context")
        if b4_replay is not True:
            raise ValueError("required saved-spike B4/native-coupling replay proof is missing")
        manifest = {"experiment": "SW0163_relation_context_cache", "status": "complete",
                    "count": COUNT, "dtype": "float32", "bindings": self.bindings,
                    "bindings_sha256": self.bindings_sha256,
                    "ledger_sha256": sha256_file(self.root / "ledger.json"),
                    "source_model_state_before_sha256": source_state_before,
                    "source_model_state_after_sha256": source_state_after,
                    "source_model_unchanged": True,
                    "b4_saved_spikes_and_coupling_replay_exact": True,
                    "context_rows": self.records}
        _atomic_json(self.root / "manifest.json", manifest, create_only=True)
        manifest["manifest_sha256"] = sha256_file(self.root / "manifest.json")
        return manifest


def validate_final_cache(root=CACHE_ROOT, *, expected_bindings=None):
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("SW0163 context cache is not complete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or manifest.get("count") != COUNT
            or manifest.get("dtype") != "float32"
            or manifest.get("source_model_state_before_sha256") != manifest.get("source_model_state_after_sha256")
            or not _is_sha256(manifest.get("source_model_state_before_sha256"))
            or manifest.get("source_model_unchanged") is not True
            or manifest.get("b4_saved_spikes_and_coupling_replay_exact") is not True
            or (expected_bindings is not None and manifest.get("bindings") != expected_bindings)):
        raise ValueError("SW0163 context-cache manifest/source binding is invalid")
    if sha256_file(root / "ledger.json") != manifest.get("ledger_sha256"):
        raise ValueError("SW0163 context-cache ledger digest changed")
    ledger = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    records = ledger.get("rows")
    if not isinstance(records, list) or len(records) != COUNT or records != manifest.get("context_rows"):
        raise ValueError("SW0163 context-cache ledger does not contain the registered512 rows")
    bindings_sha = _json_sha(manifest["bindings"])
    if bindings_sha != manifest.get("bindings_sha256"):
        raise ValueError("SW0163 context-cache bindings digest changed")
    for index, record in enumerate(records):
        path = root / "rows" / f"{index:04d}.npz"
        if (record.get("index") != index or record.get("image_id") != manifest["bindings"]["ordered_image_ids"][index]
                or record.get("row_file_sha256") != sha256_file(path)):
            raise ValueError(f"SW0163 context-cache row {index} identity/hash mismatch")
        with np.load(path, allow_pickle=False) as row:
            q, a = _context_arrays(row["q"], row["adjacency"])
            if (int(row["index"]) != index or int(row["image_id"]) != record["image_id"]
                    or str(row["rgb_sha256"].item()) != record["rgb_sha256"]
                    or str(row["bindings_sha256"].item()) != bindings_sha):
                raise ValueError(f"SW0163 context-cache row {index} archived identity mismatch")
            if (_sha_bytes(q.tobytes()) != record["q_sha256"]
                    or _sha_bytes(a.tobytes()) != record["adjacency_sha256"]):
                raise ValueError(f"SW0163 context-cache row {index} payload hash mismatch")
    result = dict(manifest)
    result["manifest_sha256"] = sha256_file(manifest_path)
    return result


def _source_state_digest(foundation):
    from collaborative_test.SW_0161_supervised_spike_partition.cache import _state_digest
    return _state_digest(foundation)


def run_cache(*, device="cuda:0", output=CACHE_ROOT):
    prepared = prepare_context_inputs_cpu()
    src = prepared["source_contract"]
    rgb_rows, ids = prepared["rgb_rows"], prepared["image_ids"]
    source_trace_manifest = prepared["source_trace_manifest"]
    bindings = prepared["context_bindings"]
    output = Path(output)
    if (output / "manifest.json").is_file():
        return validate_final_cache(output, expected_bindings=bindings)
    store = AtomicContextCache(output, bindings)

    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0163 context-cache generation requires assigned CUDA")
    torch.cuda.set_device(dev)
    foundation = sw155._load_foundation(dev)
    source_cache._require_frozen_source(foundation, src)
    foundation.encoder.eval(); foundation.patcher.eval(); foundation.wrapped.eval()
    foundation.encoder.requires_grad_(False); foundation.patcher.requires_grad_(False)
    foundation.wrapped.requires_grad_(False)
    graph = foundation.wrapped.core.graph_generator
    geodesic_operator.install_operator(graph, "T")
    state_before = _source_state_digest(foundation)
    original_qcc = None
    b4_replay = False
    source_rows = source_cache.CACHE_ROOT / "rows"

    def load_saved_spikes(index):
        path = source_rows / f"{index:04d}.npz"
        with np.load(path, allow_pickle=False) as row:
            if (int(row["index"]) != index or int(row["image_id"]) != ids[index]
                    or str(row["rgb_sha256"].item()) != prepared["rgb_sha256"][index]):
                raise ValueError(f"SW0161 cached source row {index} identity changed")
            return source_cache._trace_row_array(row["component_spikes"])

    for start in range(0, COUNT, BATCH):
        if start > 0 and len(store.records) >= start + BATCH:
            continue
        rgb_device = torch.from_numpy(rgb_rows[start:start+BATCH].copy()).to(dev)
        with torch.no_grad():
            gamma = foundation.encode(rgb_device)
            if start == 0:
                captured = {}
                hook = graph.register_forward_hook(
                    lambda _module, _args, output: captured.__setitem__("graph", output.detach().clone()))
                try:
                    trace, adjacency_capture = capture_source_coupling(
                        foundation.wrapped.core.kuramoto,
                        lambda: late_rollout(foundation.wrapped, gamma,
                                             total_steps=STEPS, live_tail_steps=0))
                finally:
                    hook.remove()
                if "graph" not in captured:
                    raise RuntimeError("source-T rollout did not execute the registered graph")
                first_graph = captured["graph"]
                expected_graph = graph(gamma).detach()
                if not torch.equal(first_graph, expected_graph):
                    raise AssertionError("captured B4 graph differs from direct native source graph output")
                coupling = foundation.wrapped.core.kuramoto.prepare_coupling(
                    first_graph, batch_size=BATCH, num_units=N, device=dev)
                if not isinstance(coupling, tuple) or len(coupling) != 3:
                    raise ValueError("native source prepare_coupling tuple contract changed")
                adjacency = coupling[0].detach().float()
                if not torch.equal(adjacency, adjacency_capture):
                    raise AssertionError("captured native rollout A differs from static B4 preparation")
                source_components = trace["component_spikes"].detach()
                cached = np.stack([load_saved_spikes(i) for i in range(BATCH)])
                if not np.array_equal(source_components[..., SETTLE:].cpu().numpy(), cached):
                    raise AssertionError("first actual B4 spikes differ from SW0161 source-cache rows")
                b4_replay = True
            else:
                graph_output = graph(gamma)
                coupling = foundation.wrapped.core.kuramoto.prepare_coupling(
                    graph_output, batch_size=BATCH, num_units=N, device=dev)
                adjacency = coupling[0].detach().float()
            components = (trace["component_spikes"].detach()[..., SETTLE:]
                          if start == 0 else torch.from_numpy(np.stack([
                              load_saved_spikes(i) for i in range(start, start+BATCH)]
                          )).to(dev))
            q = spike_synchrony_affinity(components.mean(dim=1), components,
                                         settle=0, affinity_mode="spike")
        for offset in range(BATCH):
            q_np, a_np = _context_arrays(q[offset].cpu().numpy(), adjacency[offset].cpu().numpy())
            store.write(start + offset, q_np, a_np,
                        prepared["rgb_sha256"][start + offset])
        if start == 0:
            del trace, first_graph, expected_graph, coupling, source_components, cached
        del rgb_device, gamma, adjacency, components, q
        if start % 32 == 0:
            print(json.dumps({"stage": "SW0163_context_cache", "rows": start + BATCH,
                              "total": COUNT}), flush=True)
    state_after = _source_state_digest(foundation)
    result = store.finalize(source_state_before=state_before,
                            source_state_after=state_after, b4_replay=b4_replay)
    validate_final_cache(output, expected_bindings=bindings)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=CACHE_ROOT)
    args = parser.parse_args(argv)
    result = run_cache(device=args.device, output=args.output)
    print(json.dumps({"status": result["status"], "count": result["count"],
                      "manifest_sha256": result["manifest_sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
