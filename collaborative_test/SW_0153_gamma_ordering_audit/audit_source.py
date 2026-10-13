"""Read-only TRAIN audit of native32 gamma channel and patch ordering."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"), str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0136_native32_transfer import evaluate as sw136_eval
from collaborative_test.SW_0145_native32_boundary_credit_probe import run as sw145

SEED = 1
CHANNELS = 8
NODES = 1024
GRID = 32
DEFAULT_COUNT = 4
MAX_COUNT = 16
PERMUTATION_SEED = 153001


def validate_permutation(permutation, size):
    value = torch.as_tensor(permutation, dtype=torch.int64, device="cpu")
    if value.shape != (size,) or not torch.equal(torch.sort(value).values, torch.arange(size)):
        raise ValueError(f"permutation must contain each index 0..{size - 1} exactly once")
    return value


def permute_gamma_channels(gamma, permutation):
    if not torch.is_tensor(gamma) or gamma.ndim != 3 or gamma.shape[1] != CHANNELS:
        raise ValueError("gamma must use native [B,8,N] layout")
    permutation = validate_permutation(permutation, CHANNELS).to(gamma.device)
    return gamma.index_select(1, permutation)


def permute_gamma_patches(gamma, permutation):
    if not torch.is_tensor(gamma) or gamma.ndim != 3 or gamma.shape[1:] != (CHANNELS, NODES):
        raise ValueError("gamma must use native [B,8,1024] layout")
    permutation = torch.as_tensor(permutation, dtype=torch.int64, device="cpu")
    if permutation.ndim == 1:
        permutation = permutation.unsqueeze(0).expand(gamma.shape[0], -1)
    if permutation.shape != (gamma.shape[0], NODES):
        raise ValueError("patch permutation must be [1024] or [B,1024]")
    expected = torch.arange(NODES).expand(gamma.shape[0], -1)
    if not torch.equal(torch.sort(permutation, dim=1).values, expected):
        raise ValueError("each sample's patch permutation must be a bijection")
    return gamma.gather(2, permutation.to(gamma.device).unsqueeze(1).expand(-1, CHANNELS, -1))


def _permute_linear_columns_(linear, permutation):
    if linear is None or linear.weight.ndim != 2 or linear.weight.shape[1] != len(permutation):
        raise ValueError("projection does not consume the registered gamma channels")
    with torch.no_grad():
        linear.weight.copy_(linear.weight.detach().index_select(1, permutation.to(linear.weight.device)))


def _run_dynamics(foundation, gamma, *, cuda_state=None, cpu_state=None):
    if cpu_state is not None:
        torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, gamma.device)
    with torch.no_grad():
        drive = foundation.wrapped.core.gamma_to_drive(
            gamma, foundation.wrapped.core.gamma_channel_proj,
            foundation.wrapped.core.gamma_phase_gain)
        adjacency = foundation.wrapped.core.graph_generator(gamma)
        trace = foundation.rollout(gamma, total_steps=1024, live_tail_steps=0)
        q = sw145._actual_q(trace)
        labels = sw136_eval._qcc_labels(trace)
    return drive.detach(), adjacency.detach(), q.detach(), torch.as_tensor(labels).detach()


def _offdiagonal_mse(left, right):
    if left.shape != right.shape or left.ndim != 2 or left.shape != (NODES, NODES):
        raise ValueError("Q comparison requires two native1024 square matrices")
    offdiag = ~torch.eye(NODES, dtype=torch.bool, device=left.device)
    value = (left[offdiag].to(torch.float64) - right[offdiag].to(torch.float64)).square().mean()
    return float(value.cpu())


def _fraction_changed(left, right):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != (GRID, GRID) or b.shape != a.shape:
        raise ValueError("QCC comparison requires native32 patch labels")
    return float(np.mean(a != b))


def _model_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _reference_mapped(labels, permutation):
    """Map outputs from permuted node positions back to original patch positions."""
    index = np.asarray(permutation, dtype=np.int64)
    if index.shape != (NODES,) or not np.array_equal(np.sort(index), np.arange(NODES)):
        raise ValueError("invalid patch permutation for output restoration")
    result = np.empty_like(np.asarray(labels).reshape(-1))
    result[index] = np.asarray(labels).reshape(-1)
    return result.reshape(GRID, GRID)


def unpermute_nodes(values, permutation):
    """Restore vectors/matrices from new[j] = old[permutation[j]] ordering."""
    index = torch.as_tensor(permutation, dtype=torch.int64, device="cpu")
    if index.ndim != 1 or index.numel() < 1 or not torch.equal(
            torch.sort(index).values, torch.arange(index.numel())):
        raise ValueError("node permutation must be a bijection")
    inverse = torch.argsort(index).to(values.device)
    if values.shape[-1] != index.numel():
        raise ValueError("last tensor axis does not match node permutation")
    if values.ndim >= 2 and values.shape[-2] == index.numel():
        return values.index_select(-2, inverse).index_select(-1, inverse)
    return values.index_select(-1, inverse)


def audit_source(*, count=DEFAULT_COUNT, output, device="cuda:0"):
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_COUNT:
        raise ValueError(f"count must be an integer in 1..{MAX_COUNT}")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserve existing gamma ordering audit: {output}")
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("source ordering probe requires an assigned CUDA device")
    torch.cuda.set_device(dev)
    foundation = sw135.load_native32_foundation(SEED, dev, verify_rgb_assets=True)
    ids = [int(x) for x in foundation.image_ids[:count]]
    rows = np.asarray(foundation.pool_indices[:count], dtype=np.int64)
    checkpoint, manifest_path, _manifest, pool, contract_ids, source_sha = sw130.source_contract(SEED)
    if ids != [int(x) for x in contract_ids[:count]] or not np.array_equal(rows, pool[:count]):
        raise ValueError("loaded foundation does not match the ordered SW0097 TRAIN prefix")
    cache_path = Path(sw130.TRAIN_RGB)
    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (70000, 128, 128, 3):
        raise ValueError("registered TRAIN RGB cache schema changed")
    rgb = torch.from_numpy(np.asarray(cache[rows]).copy()).to(dev)
    gamma_batch = foundation.encode(rgb)
    if tuple(gamma_batch.shape) != (count, CHANNELS, NODES) or not torch.isfinite(gamma_batch).all():
        raise ValueError("native gamma output is not finite [B,8,1024]")
    gamma_batch = gamma_batch.detach()
    random_perm = torch.from_numpy(np.random.RandomState(PERMUTATION_SEED).permutation(CHANNELS))
    mean_order = torch.argsort(gamma_batch.mean(dim=(0, 2)).cpu(), stable=True)
    patch_perms = [np.random.RandomState(PERMUTATION_SEED + 1000 + image_id).permutation(NODES)
                   for image_id in ids]
    core = foundation.wrapped.core
    graph = core.graph_generator
    if core.gamma_channel_proj is None or graph is None or graph.projection.in_features != CHANNELS:
        raise ValueError("source checkpoint lacks both registered gamma-channel projections")
    before = {"foundation": _model_digest(foundation.wrapped), "encoder": _model_digest(foundation.encoder)}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(dev)
    per_image = []
    try:
        for index, image_id in enumerate(ids):
            gamma = gamma_batch[index:index + 1]
            # Every variant starts from the same RNG state so phase initialization is matched.
            cpu_before = torch.get_rng_state().clone()
            cuda_before = torch.cuda.get_rng_state(dev).clone()
            drive0, graph0, q0, labels0 = _run_dynamics(
                foundation, gamma, cpu_state=cpu_before, cuda_state=cuda_before)
            record = {"image_id": image_id, "variants": {}}

            for name, permutation in (("naive_fixed_channel_shuffle", random_perm),
                                      ("naive_train_mean_channel_order", mean_order)):
                gamma_variant = permute_gamma_channels(gamma, permutation)
                drive, adjacency, q, labels = _run_dynamics(
                    foundation, gamma_variant, cpu_state=cpu_before, cuda_state=cuda_before)
                record["variants"][name] = {
                    "permutation": permutation.tolist(),
                    "drive_mse": float((drive - drive0).square().mean().cpu()),
                    "graph_adjacency_mse": float((adjacency - graph0).square().mean().cpu()),
                    "actual_q_offdiag_mse": _offdiagonal_mse(q[0], q0[0]),
                    "qcc_changed_fraction": _fraction_changed(labels[0], labels0[0]),
                }

            gamma_variant = permute_gamma_channels(gamma, random_perm)
            old_channel = core.gamma_channel_proj.weight.detach().clone()
            old_graph = graph.projection.weight.detach().clone()
            try:
                _permute_linear_columns_(core.gamma_channel_proj, random_perm)
                _permute_linear_columns_(graph.projection, random_perm)
                drive, adjacency, q, labels = _run_dynamics(
                    foundation, gamma_variant, cpu_state=cpu_before, cuda_state=cuda_before)
            finally:
                with torch.no_grad():
                    core.gamma_channel_proj.weight.copy_(old_channel)
                    graph.projection.weight.copy_(old_graph)
            record["variants"]["consistent_global_channel_reindex"] = {
                "permutation": random_perm.tolist(),
                "drive_max_abs": float((drive - drive0).abs().max().cpu()),
                "graph_adjacency_max_abs": float((adjacency - graph0).abs().max().cpu()),
                "actual_q_offdiag_mse": _offdiagonal_mse(q[0], q0[0]),
                "qcc_changed_fraction": _fraction_changed(labels[0], labels0[0]),
            }

            patch_perm = patch_perms[index]
            gamma_variant = permute_gamma_patches(gamma, patch_perm)
            drive, adjacency, q, labels = _run_dynamics(
                foundation, gamma_variant, cpu_state=cpu_before, cuda_state=cuda_before)
            patch_index = torch.as_tensor(patch_perm, device=dev)
            drive_unpermuted = torch.empty_like(drive)
            drive_unpermuted[:, patch_index] = drive
            adjacency_unpermuted = unpermute_nodes(adjacency[0], patch_perm)
            q_unpermuted = unpermute_nodes(q[0], patch_perm)
            labels_unpermuted = _reference_mapped(labels[0].cpu().numpy(), patch_perm)
            record["variants"]["per_image_patch_permutation"] = {
                "permutation_sha256": hashlib.sha256(np.asarray(patch_perm, dtype="<i8").tobytes()).hexdigest(),
                "drive_mse_after_node_unpermutation": float((drive_unpermuted - drive0).square().mean().cpu()),
                "graph_adjacency_mse_after_node_unpermutation": float((adjacency_unpermuted - graph0[0]).square().mean().cpu()),
                "actual_q_offdiag_mse_after_node_unpermutation": _offdiagonal_mse(q_unpermuted, q0[0]),
                "qcc_changed_fraction_after_node_unpermutation": _fraction_changed(labels_unpermuted, labels0[0]),
            }
            per_image.append(record)
            print(json.dumps({"stage": "gamma_order_probe", "completed": index + 1,
                              "total": count}, allow_nan=False), flush=True)
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, dev)

    after = {"foundation": _model_digest(foundation.wrapped), "encoder": _model_digest(foundation.encoder)}
    if before != after:
        raise AssertionError("read-only source gamma probe changed model state")
    fingerprints = sw135.implementation_fingerprint()
    for path in (HERE / "audit_source.py", HERE / "README.md", Path(__file__).with_name("test_ordering.py")):
        fingerprints[path.relative_to(ROOT).as_posix()] = sw135.sha256_file(path)
    summary = {}
    for variant in per_image[0]["variants"]:
        keys = [key for key in per_image[0]["variants"][variant]
                if key.endswith("_mse") or key.endswith("_fraction") or key.endswith("max_abs")]
        summary[variant] = {key: float(np.mean([row["variants"][variant][key] for row in per_image]))
                            for key in keys}
    report = {"status": "complete", "experiment": "SW0153_gamma_ordering_audit",
        "seed": SEED, "ids": ids, "pool_rows": rows.tolist(), "count": count,
        "gamma_layout": ["batch", "feature_channel=8", "spatial_patch=1024"],
        "ordered_train_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "source_core_sha256": source_sha, "source_manifest_sha256": sw135.sha256_file(manifest_path),
        "rgb_asset_validation": foundation.provenance["rgb_asset_validation"],
        "implementation_fingerprint": fingerprints, "model_state_sha256_before_after": before,
        "channel_random_seed": PERMUTATION_SEED, "train_mean_order_ascending": mean_order.tolist(),
        "per_image": per_image, "mean_change_summary": summary,
        "ground_truth_used": False, "optimizer_updates": 0,
        "interpretation_limit": "Ordering sensitivity is representational evidence only; no mask correctness or segmentation improvement is inferred."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_source(count=args.count, output=args.output, device=args.device)
    print(json.dumps({"status": result["status"], "count": result["count"],
                      "ground_truth_used": False, "output": str(args.output)}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
