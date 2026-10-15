"""Read-only CPU audit of SW0161/SW0163 TRAIN cache rows 0..7 for SW0167.

This audits the arithmetic used to create the fixed B4 caches. It never loads
TRAIN masks, validation data, or GT, and it does not run a model or create files.
The CPU-Q comparison is diagnostic; it is not a B4/B1 parity assertion.
"""
from __future__ import annotations

import os

# Hide CUDA before importing PyTorch. This script is intentionally CPU-only.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_root_override = os.environ.get("SW0167_AUDIT_ROOT")
if _root_override:
    ROOT = Path(_root_override).expanduser().resolve()
elif Path(__file__).name not in {"<stdin>", "-c"}:
    ROOT = next((parent for parent in Path(__file__).resolve().parents
                 if (parent / "collaborative_test").is_dir()
                 and (parent / "snn_kuramoto_bidirectional").is_dir()), None)
    if ROOT is None:
        raise RuntimeError("cannot locate patch_v2_sw; set SW0167_AUDIT_ROOT explicitly")
else:
    raise RuntimeError("set SW0167_AUDIT_ROOT to the patch_v2_sw repository root")
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"),
                str(ROOT / "snn_kuramoto_bidirectional")]

import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0161_supervised_spike_partition import cache as source_cache
from collaborative_test.SW_0162_connected_signed_partition import connected_signed as signed
from collaborative_test.SW_0163_relation_context_assignment import context_cache
from collaborative_test.SW_0164_foreground_preserving_partition import foreground_signed
from collaborative_test.SW_0164_foreground_preserving_partition import predict_evaluate as sw164_eval
from collaborative_test.SW_0167_structured_relation_energy import qcc_readout, structured_energy
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity

COUNT = 8
BATCH = 4
N = 1024
SETTLE = 512
SW164_DIR = ROOT / "collaborative_test/SW_0164_foreground_preserving_partition"
SW164_WORK = SW164_DIR / "results_archive/prediction_work_20261014/foreground_induced_seed1"
EXPECTED_SOURCE_TRACE_MANIFEST_SHA256 = (
    "b4125a81012bf46cf38ff512bf251452f511524c42a2d2364c34783de04302b3"
)
EXPECTED_SOURCE_STATE_SHA256 = "b205d21a917299089b9b5d373eb94c9eb5eb971471e9428072e0fc0229c70749"


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_sha(value) -> str:
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8"))


def _array_sha(value: np.ndarray) -> str:
    return _sha_bytes(np.ascontiguousarray(value).tobytes())


def _require_sha(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value.lower())):
        raise ValueError(f"{label} is not a SHA-256 digest")
    return value


def _ledger_sha(work: Path) -> str:
    hashes = []
    for index in range(320):
        path = work / "rows" / f"{index:04d}" / "row.json"
        if not path.is_file():
            raise ValueError("SW0164 source-T row ledger is incomplete")
        hashes.append(_sha_file(path))
    return _sha_bytes(json.dumps(hashes, separators=(",", ":")).encode("utf-8"))


def _load_registered_gain(source_sha: str, source_manifest_sha: str) -> tuple[float, str]:
    """Read the exact GPU-recorded gain from SW0164's immutable source-T row 0."""
    binding_path = SW164_WORK / "bindings.json"
    state_path = SW164_WORK / "source_state.json"
    row_path = SW164_WORK / "rows/0000/row.json"
    row_npz = SW164_WORK / "rows/0000/row.npz"
    if not all(path.is_file() for path in (binding_path, state_path, row_path, row_npz)):
        raise FileNotFoundError("registered SW0164 source-T prediction work/gain row is missing")
    binding_wrapper = json.loads(binding_path.read_text(encoding="utf-8"))
    bindings = binding_wrapper.get("bindings")
    binding_sha = _json_sha(bindings)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    row = json.loads(row_path.read_text(encoding="utf-8"))
    row_npz_sha = _sha_file(row_npz)
    if (not isinstance(bindings, dict)
            or binding_wrapper.get("bindings_sha256") != binding_sha
            or bindings.get("experiment") != "SW0164_foreground_preserving_partition"
            or bindings.get("seed") != 1
            or bindings.get("source_checkpoint_sha256") != source_sha
            or bindings.get("source_manifest_sha256") != source_manifest_sha
            or bindings.get("steps") != 1024 or bindings.get("settle") != 512
            or bindings.get("live_tail") != 0 or bindings.get("optimizer_updates") != 0
            or bindings.get("ground_truth_used_for_prediction") is not False
            or state.get("source_unchanged") is not True
            or state.get("bindings_sha256") != binding_sha
            or state.get("source_state_sha256_before") != bindings.get("source_model_state_sha256")
            or state.get("source_state_sha256_after") != bindings.get("source_model_state_sha256")
            or state.get("row_commits_sha256") != _ledger_sha(SW164_WORK)):
        raise ValueError("SW0164 source-T work binding/immutability receipt is invalid")
    current_fp = sw164_eval.implementation_fingerprint()
    if bindings.get("implementation_fingerprint") != current_fp:
        raise ValueError("SW0164 source-T work was not produced by the currently registered implementation")
    if (row.get("index") != 0 or row.get("bindings_sha256") != binding_sha
            or row.get("npz_sha256") != row_npz_sha
            or row.get("q_sha256") is None or row.get("actual_a_sha256") is None
            or not isinstance(row.get("trace_sha256"), dict)
            or row.get("foreground_support_exact") is not True):
        raise ValueError("SW0164 source-T gain row 0 payload receipt is invalid")
    gain = row.get("gain")
    if type(gain) not in (int, float) or not np.isfinite(gain) or gain <= 0:
        raise ValueError("SW0164 source-T row lacks the actual positive learned gain")
    return float(gain), _sha_file(row_path)


def _validate_upstream_and_rows():
    source_manifest_path = source_cache.CACHE_ROOT / "manifest.json"
    source_ledger_path = source_cache.CACHE_ROOT / "ledger.json"
    context_manifest_path = context_cache.CACHE_ROOT / "manifest.json"
    context_ledger_path = context_cache.CACHE_ROOT / "ledger.json"
    for path in (source_manifest_path, source_ledger_path, context_manifest_path,
                 context_ledger_path, sw130.TRAIN_RGB):
        if not path.is_file():
            raise FileNotFoundError(f"required registered TRAIN cache is missing: {path}")

    # The source contract reads only model/checkpoint metadata and ordered IDs;
    # it does not open image masks or any validation/GT arrays.
    checkpoint, source_manifest_file, source_manifest, pool_rows, ordered_ids, source_sha = sw130.source_contract(1)
    source_manifest_sha = _sha_file(source_manifest_file)
    if source_manifest_path != source_cache.CACHE_ROOT / "manifest.json":
        raise AssertionError("unexpected SW0161 source cache location")
    source_manifest_record = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    context_manifest_record = json.loads(context_manifest_path.read_text(encoding="utf-8"))
    source_bindings = source_manifest_record.get("bindings", {})
    context_bindings = context_manifest_record.get("bindings", {})
    source_manifest_sha_actual = _sha_file(source_manifest_path)

    if (source_manifest_record.get("experiment") != source_cache.EXPERIMENT
            or source_manifest_record.get("status") != "complete"
            or source_manifest_record.get("count") != 512
            or source_manifest_record.get("source_model_unchanged") is not True
            or source_manifest_record.get("source_model_state_before_sha256") != source_manifest_record.get("source_model_state_after_sha256")
            or source_manifest_record.get("trace_dtype") != "float32"
            or source_manifest_record.get("trace_shape_per_image") != [4, N, SETTLE]
            or source_manifest_record.get("bindings_sha256") != _json_sha(source_bindings)
            or source_manifest_sha_actual != EXPECTED_SOURCE_TRACE_MANIFEST_SHA256
            or source_manifest_record.get("source_model_state_before_sha256") != EXPECTED_SOURCE_STATE_SHA256):
        raise ValueError("SW0161 final source cache manifest/source immutability receipt is invalid")
    if (source_bindings.get("source_seed") != 1
            or source_bindings.get("source_checkpoint_sha256") != source_sha
            or source_bindings.get("source_manifest_sha256") != source_manifest_sha
            or source_bindings.get("image_ids") != [int(x) for x in ordered_ids[:512]]
            or source_bindings.get("source_rows") != np.asarray(pool_rows[:512], dtype=np.int64).tolist()
            or source_bindings.get("ordered_ids_sha256") != _sha_bytes(np.asarray(ordered_ids[:512], dtype="<i8").tobytes())
            or source_bindings.get("source_train_ids_sha256") != _sha_bytes(np.asarray(ordered_ids, dtype="<i8").tobytes())
            or source_bindings.get("operator") != "SW0143-T"):
        raise ValueError("SW0161 source cache is not bound to the current seed-1 source contract")
    if _sha_file(source_ledger_path) != source_manifest_record.get("ledger_sha256"):
        raise ValueError("SW0161 source cache ledger digest mismatch")
    source_ledger = json.loads(source_ledger_path.read_text(encoding="utf-8"))
    source_records = source_ledger.get("rows")
    if not isinstance(source_records, list) or len(source_records) != 512:
        raise ValueError("SW0161 source cache ledger must contain 512 ordered TRAIN rows")
    if source_manifest_record.get("rows_sha256") != [r.get("row_file_sha256") for r in source_records]:
        raise ValueError("SW0161 source manifest row hashes differ from its ledger")

    actual_context_manifest_sha = _sha_file(context_manifest_path)
    if (context_manifest_record.get("experiment") != "SW0163_relation_context_cache"
            or context_manifest_record.get("status") != "complete"
            or context_manifest_record.get("count") != 512
            or context_manifest_record.get("source_model_unchanged") is not True
            or context_manifest_record.get("b4_saved_spikes_and_coupling_replay_exact") is not True
            or context_manifest_record.get("source_model_state_before_sha256") != context_manifest_record.get("source_model_state_after_sha256")
            or context_manifest_record.get("source_model_state_before_sha256") != EXPECTED_SOURCE_STATE_SHA256
            or context_manifest_record.get("bindings_sha256") != _json_sha(context_bindings)):
        raise ValueError("SW0163 context manifest/source immutability receipt is invalid")
    if (context_bindings.get("source_seed") != 1
            or context_bindings.get("source_checkpoint_sha256") != source_sha
            or context_bindings.get("source_manifest_sha256") != source_manifest_sha
            or context_bindings.get("source_trace_cache_manifest_sha256") != _sha_file(source_manifest_path)
            or context_bindings.get("source_trace_cache_bindings_sha256") != source_manifest_record.get("bindings_sha256")
            or context_bindings.get("ordered_image_ids") != [int(x) for x in ordered_ids[:512]]
            or context_bindings.get("source_rows") != np.asarray(pool_rows[:512], dtype=np.int64).tolist()):
        raise ValueError("SW0163 context cache source/checkpoint/order binding mismatch")
    if source_bindings.get("cache_implementation_fingerprint") != source_cache.implementation_fingerprint():
        raise ValueError("SW0161 source-cache implementation family changed")
    if context_bindings.get("code_fingerprint") != context_cache._fingerprint():
        raise ValueError("SW0163 context-cache implementation family changed")

    # Stat-only checks for bound assets. In particular, no HDF5 datasets or
    # validation RGB arrays are opened by this audit.
    hdf5_path = Path(source_bindings.get("hdf5_path", ""))
    context_hdf5_path = Path(context_bindings.get("hdf5_path", ""))
    if (not hdf5_path.is_file() or hdf5_path.resolve() != context_hdf5_path.resolve()
            or hdf5_path.stat().st_size != source_bindings.get("hdf5_size")
            or hdf5_path.stat().st_mtime_ns != source_bindings.get("hdf5_mtime_ns")
            or context_bindings.get("hdf5_size") != source_bindings.get("hdf5_size")
            or context_bindings.get("hdf5_mtime_ns") != source_bindings.get("hdf5_mtime_ns")):
        raise ValueError("bound canonical TRAIN HDF5 identity/stat changed")

    # Validate the registered TRAIN RGB metadata and selected eight RGB bytes;
    # no validation RGB file is opened.
    rgb_meta_path = sw130.TRAIN_RGB.with_suffix(sw130.TRAIN_RGB.suffix + ".complete.json")
    rgb_meta = json.loads(rgb_meta_path.read_text(encoding="utf-8-sig"))
    rgb_stat = sw130.TRAIN_RGB.stat()
    assets = source_bindings.get("source_rgb_asset_validation", {})
    expected_train_ids = np.concatenate((np.arange(1000, dtype="<i8"),
                                         np.arange(1640, 70640, dtype="<i8")))
    expected_mapping_sha = _sha_bytes(expected_train_ids.tobytes())
    if (rgb_meta.get("status") != "complete"
            or rgb_meta.get("cache_shape") != [70000, 128, 128, 3]
            or rgb_meta.get("ids_mapping_sha256") != expected_mapping_sha
            or assets.get("train_manifest_sha256") != _sha_file(rgb_meta_path)
            or source_bindings.get("train_rgb_cache_size") != rgb_stat.st_size
            or source_bindings.get("train_rgb_cache_mtime_ns") != rgb_stat.st_mtime_ns
            or context_bindings.get("train_rgb_size") != rgb_stat.st_size
            or context_bindings.get("train_rgb_mtime_ns") != rgb_stat.st_mtime_ns
            or context_bindings.get("rgb_assets", {}).get("train_manifest_sha256") != _sha_file(rgb_meta_path)):
        raise ValueError("registered TRAIN RGB asset metadata/stat binding changed")

    if _sha_file(context_ledger_path) != context_manifest_record.get("ledger_sha256"):
        raise ValueError("SW0163 context cache ledger digest mismatch")
    context_ledger = json.loads(context_ledger_path.read_text(encoding="utf-8"))
    context_records = context_ledger.get("rows")
    if (not isinstance(context_records, list) or len(context_records) != 512
            or context_records != context_manifest_record.get("context_rows")):
        raise ValueError("SW0163 context cache ledger/final manifest rows differ")

    # Verify only the selected TRAIN image bytes, without reading any mask HDF5.
    rgb = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if rgb.dtype != np.uint8 or tuple(rgb.shape) != (70000, 128, 128, 3):
        raise ValueError("registered TRAIN RGB cache schema changed")
    selected_rows = np.asarray(pool_rows[:COUNT], dtype=np.int64)
    selected_ids = [int(value) for value in ordered_ids[:COUNT]]
    if any(not (0 <= value < 1000 or 1640 <= value < 70640) for value in selected_ids):
        raise ValueError("one of the first eight source IDs is outside registered TRAIN ID ranges")
    selected_rgb_sha = []
    for pos, (row_index, image_id) in enumerate(zip(selected_rows, selected_ids)):
        image = np.ascontiguousarray(rgb[int(row_index)])
        digest = _array_sha(image)
        selected_rgb_sha.append(digest)
        if source_bindings["image_rgb_sha256"][pos] != digest:
            raise ValueError(f"SW0161 RGB byte binding mismatch at TRAIN position {pos}")
        if context_bindings["rgb_sha256"][pos] != digest:
            raise ValueError(f"SW0163 RGB byte binding mismatch at TRAIN position {pos}")
        if (source_records[pos].get("index") != pos
                or source_records[pos].get("image_id") != image_id
                or source_records[pos].get("rgb_sha256") != digest
                or source_records[pos].get("row_file_sha256") != _sha_file(source_cache.CACHE_ROOT / "rows" / f"{pos:04d}.npz")):
            raise ValueError(f"SW0161 cache ledger row {pos} ID/RGB/file hash mismatch")
        if (context_records[pos].get("index") != pos
                or context_records[pos].get("image_id") != image_id
                or context_records[pos].get("rgb_sha256") != digest
                or context_records[pos].get("row_file_sha256") != _sha_file(context_cache.CACHE_ROOT / "rows" / f"{pos:04d}.npz")):
            raise ValueError(f"SW0163 cache ledger row {pos} ID/RGB/file hash mismatch")
    del rgb

    source_bindings_sha = source_manifest_record["bindings_sha256"]
    context_bindings_sha = context_manifest_record["bindings_sha256"]
    source_rows_data, context_rows_data = [], []
    for pos, image_id in enumerate(selected_ids):
        source_path = source_cache.CACHE_ROOT / "rows" / f"{pos:04d}.npz"
        context_path = context_cache.CACHE_ROOT / "rows" / f"{pos:04d}.npz"
        with np.load(source_path, allow_pickle=False) as row:
            required = {"component_spikes", "targets32", "index", "image_id", "rgb_sha256", "bindings_sha256"}
            if set(row.files) != required:
                raise ValueError(f"SW0161 source row {pos} schema changed")
            if (int(row["index"]) != pos or int(row["image_id"]) != image_id
                    or str(row["rgb_sha256"].item()) != selected_rgb_sha[pos]
                    or str(row["bindings_sha256"].item()) != source_bindings_sha):
                raise ValueError(f"SW0161 source row {pos} embedded identity/binding mismatch")
            # Deliberately do not access targets32: it contains TRAIN GT-derived labels.
            components = row["component_spikes"].copy()
        if components.dtype != np.float32 or components.shape != (4, N, SETTLE) or not np.isfinite(components).all():
            raise ValueError(f"SW0161 selected source row {pos} spike schema/value invalid")
        if _array_sha(components) != source_records[pos].get("trace_sha256"):
            raise ValueError(f"SW0161 selected source row {pos} trace payload hash mismatch")

        with np.load(context_path, allow_pickle=False) as row:
            if set(row.files) != {"q", "adjacency", "image_id", "index", "rgb_sha256", "bindings_sha256"}:
                raise ValueError(f"SW0163 context row {pos} schema changed")
            if (int(row["index"]) != pos or int(row["image_id"]) != image_id
                    or str(row["rgb_sha256"].item()) != selected_rgb_sha[pos]
                    or str(row["bindings_sha256"].item()) != context_bindings_sha):
                raise ValueError(f"SW0163 context row {pos} embedded identity/binding mismatch")
            q, adjacency = row["q"].copy(), row["adjacency"].copy()
        q, adjacency = context_cache._context_arrays(q, adjacency)
        if (_array_sha(q) != context_records[pos].get("q_sha256")
                or _array_sha(adjacency) != context_records[pos].get("adjacency_sha256")):
            raise ValueError(f"SW0163 selected context row {pos} Q/A payload hash mismatch")
        source_rows_data.append(components)
        context_rows_data.append((q, adjacency))

    gain, gain_receipt_sha = _load_registered_gain(source_sha, source_manifest_sha)
    return {
        "source_checkpoint_sha256": source_sha,
        "source_manifest_sha256": source_manifest_sha,
        "source_checkpoint_path": str(checkpoint),
        "source_cache_manifest_sha256": _sha_file(source_manifest_path),
        "source_cache_bindings_sha256": source_bindings_sha,
        "source_state_digest_family": "SW0161 cache _state_digest; verified by complete immutable source-cache receipt",
        "source_state_sha256": source_manifest_record["source_model_state_before_sha256"],
        "context_cache_manifest_sha256": actual_context_manifest_sha,
        "context_cache_bindings_sha256": context_bindings_sha,
        "context_state_digest_family": "SW0161 cache _state_digest; verified by complete immutable SW0163 context-cache receipt",
        "context_state_sha256": context_manifest_record["source_model_state_before_sha256"],
        "source_rgb_manifest_sha256": context_bindings["rgb_assets"]["train_manifest_sha256"],
        "gain": gain,
        "gain_receipt_sha256": gain_receipt_sha,
        "rows": [{"index": i, "image_id": selected_ids[i],
                  "rgb_sha256": selected_rgb_sha[i],
                  "source_file_sha256": source_records[i]["row_file_sha256"],
                  "trace_sha256": source_records[i]["trace_sha256"],
                  "context_file_sha256": context_records[i]["row_file_sha256"],
                  "q_sha256": context_records[i]["q_sha256"],
                  "adjacency_sha256": context_records[i]["adjacency_sha256"]}
                 for i in range(COUNT)],
        "component_spikes": source_rows_data,
        "contexts": context_rows_data,
        "selected_ids": selected_ids,
    }


def _recompute_cpu_q(components_b4: np.ndarray) -> np.ndarray:
    """Use the production per-component correlation routine on CPU float32."""
    if components_b4.dtype != np.float32 or components_b4.shape != (BATCH, 4, N, SETTLE):
        raise ValueError("audit requires one original contiguous B4 group [4,4,1024,512]")
    component_tensor = torch.from_numpy(np.ascontiguousarray(components_b4))
    # The first parameter is required for API/schema compatibility. With
    # components supplied, spike_synchrony_affinity computes Q componentwise;
    # it does not use this aggregate as an alternate affinity.
    activity = component_tensor.mean(dim=1)
    q = spike_synchrony_affinity(
        activity, component_tensor, settle=0, affinity_mode="spike")
    if q.dtype != torch.float32 or tuple(q.shape) != (BATCH, N, N) or not torch.isfinite(q).all():
        raise ValueError("production CPU Q routine returned an invalid B4 affinity")
    return q.contiguous().numpy()


def run_audit() -> dict:
    prepared = _validate_upstream_and_rows()
    # Keep the historical B4 grouping from the cache recipe: Q was computed
    # on original B4 context-generation batches, not B1 validation inference.
    reports = []
    feature_admission_all = True
    for batch_start in (0, 4):
        components = np.stack(prepared["component_spikes"][batch_start:batch_start + BATCH])
        q_cpu = _recompute_cpu_q(components)
        for offset in range(BATCH):
            index = batch_start + offset
            q_cached, adjacency = prepared["contexts"][index]
            label_cpu = qcc_readout.qcc_labels_from_affinity(q_cpu[offset:offset + 1])[0].numpy().astype(np.int32)
            label_cached = qcc_readout.qcc_labels_from_affinity(q_cached[None])[0].numpy().astype(np.int32)
            max_abs_q_delta = float(np.max(np.abs(q_cpu[offset].astype(np.float64) - q_cached.astype(np.float64))))
            differing_q_entries = int(np.count_nonzero(q_cpu[offset] != q_cached))
            threshold_edge_disagreements = int(np.count_nonzero(
                (q_cpu[offset] >= 0.5) != (q_cached >= 0.5)))
            qcc_different_labels = int(np.count_nonzero(label_cpu != label_cached))
            qcc_different_foreground = int(np.count_nonzero((label_cpu > 0) != (label_cached > 0)))

            # Solver-only comparison uses the same cached GPU-B4 Q/A and
            # CPU-B4-derived source support. Feature admission is separate:
            # production pair_features must reject a cached Q that is not
            # consistent with the saved spikes; never swap in CPU Q silently.
            reference = foreground_signed.foreground_induced_merge(
                q_cached, adjacency, prepared["gain"], label_cpu, grid_shape=(32, 32))
            e0, _weight = structured_energy.energy_components(
                torch.from_numpy(q_cached), torch.from_numpy(adjacency), prepared["gain"])
            solver_only = structured_energy.solve_energy(
                e0.detach().cpu().numpy(), label_cpu, grid_shape=(32, 32))
            energy_exact = np.array_equal(reference["pair_energy"], e0.detach().cpu().numpy())
            labels_exact = np.array_equal(reference["labels"], solver_only["labels"])
            merge_log_exact = reference["merge_log"] == solver_only["merge_log"]
            if not (energy_exact and labels_exact and merge_log_exact):
                raise AssertionError(f"solver-only SW0167 parity differs from SW0164 at TRAIN row {index}")

            feature_admission_error = None
            residual_max = None
            mlp_energy_exact = mlp_labels_exact = mlp_merge_log_exact = False
            try:
                source_spikes = torch.from_numpy(prepared["component_spikes"][index])
                features = structured_energy.pair_features(
                    source_spikes, torch.from_numpy(q_cached),
                    torch.from_numpy(adjacency), prepared["gain"])
                rng_state = torch.random.get_rng_state()
                try:
                    torch.manual_seed(167001)
                    head = structured_energy.PairRelationMLP()
                finally:
                    torch.random.set_rng_state(rng_state)
                if (not torch.equal(head.output.weight, torch.zeros_like(head.output.weight))
                        or not torch.equal(head.output.bias, torch.zeros_like(head.output.bias))):
                    raise AssertionError("SW0167 zero-residual head initialization changed")
                e_theta, residual = head(features, e0, _weight)
                if not torch.equal(e_theta, e0) or torch.count_nonzero(residual).item() != 0:
                    raise AssertionError(f"SW0167 initialized head changed source energy at TRAIN row {index}")
                candidate = structured_energy.solve_energy(
                    e_theta.detach().cpu().numpy(), label_cpu, grid_shape=(32, 32))
                mlp_energy_exact = np.array_equal(reference["pair_energy"], e_theta.detach().cpu().numpy())
                mlp_labels_exact = np.array_equal(reference["labels"], candidate["labels"])
                mlp_merge_log_exact = reference["merge_log"] == candidate["merge_log"]
                residual_max = float(residual.abs().max().item())
                if not (mlp_energy_exact and mlp_labels_exact and mlp_merge_log_exact):
                    raise AssertionError(f"zero-residual SW0167 MLP solver differs at TRAIN row {index}")
            except ValueError as exc:
                feature_admission_error = f"{type(exc).__name__}: {exc}"
                feature_admission_all = False
            reports.append({
                "cache_index": index,
                "image_id": prepared["selected_ids"][index],
                "rgb_sha256": prepared["rows"][index]["rgb_sha256"],
                "cpu_q_sha256": _array_sha(q_cpu[offset]),
                "cached_q_sha256": prepared["rows"][index]["q_sha256"],
                "cpu_vs_cached_q_max_abs_difference": max_abs_q_delta,
                "cpu_vs_cached_q_exact_entry_differences": differing_q_entries,
                "cpu_vs_cached_q_threshold_edge_disagreements": threshold_edge_disagreements,
                "cpu_vs_cached_qcc_different_label_pixels": qcc_different_labels,
                "cpu_vs_cached_qcc_different_foreground_pixels": qcc_different_foreground,
                "cpu_qcc_foreground_patch_count": int(np.count_nonzero(label_cpu)),
                "cached_qcc_foreground_patch_count": int(np.count_nonzero(label_cached)),
                "pair_features_admission_passed": feature_admission_error is None,
                "pair_features_admission_error": feature_admission_error,
                "zero_initialized_pair_mlp_residual_max_abs": residual_max,
                "solver_only_energy_exact": energy_exact,
                "solver_only_labels_exact": labels_exact,
                "solver_only_merge_log_exact": merge_log_exact,
                "zero_residual_mlp_energy_exact": mlp_energy_exact,
                "zero_residual_mlp_labels_exact": mlp_labels_exact,
                "zero_residual_mlp_merge_log_exact": mlp_merge_log_exact,
                "solver_only_merge_count": int(solver_only["merge_count"]),
            })
    return {
        "experiment": "SW0167_train_cache_cpu_arithmetic_audit",
        "status": ("complete" if feature_admission_all
                   else "diagnostic_complete_with_feature_admission_failure"),
        "full_preflight_passed": feature_admission_all,
        "scope": "first8 registered SW0161/SW0163 TRAIN rows only; read-only CPU cache audit",
        "rows": reports,
        "source_checkpoint_sha256": prepared["source_checkpoint_sha256"],
        "source_manifest_sha256": prepared["source_manifest_sha256"],
        "source_cache_manifest_sha256": prepared["source_cache_manifest_sha256"],
        "context_cache_manifest_sha256": prepared["context_cache_manifest_sha256"],
        "source_state_sha256": prepared["source_state_sha256"],
        "source_state_digest_family": prepared["source_state_digest_family"],
        "context_state_sha256": prepared["context_state_sha256"],
        "context_state_digest_family": prepared["context_state_digest_family"],
        "learned_source_gain": prepared["gain"],
        "gain_receipt_file_sha256": prepared["gain_receipt_sha256"],
        "batching": "CPU Q recomputed in original B4 groups rows 0..3 and 4..7; no B1 equivalence asserted",
        "ground_truth_or_mask_data_read": False,
        "validation_data_read": False,
        "rollouts_or_optimizer_updates": 0,
        "interpretation": (
            "CPU production-Q/cache-Q deltas are descriptive. Solver parity compares SW0164 and "
            "SW0167 zero-residual solvers on identical cached Q/A and the CPU-derived B4 support; "
            "it does not establish B1/B4 QCC parity or model-performance evidence."
        ),
    }


def main() -> int:
    torch.set_num_threads(1)
    report = run_audit()
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
