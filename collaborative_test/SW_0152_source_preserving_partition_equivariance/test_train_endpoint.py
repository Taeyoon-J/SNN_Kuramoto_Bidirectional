"""Exercises the production finalizer with a small real PyTorch module/optimizer."""
import hashlib
import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from collaborative_test.SW_0152_source_preserving_partition_equivariance import train, training


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_encoder = nn.Linear(2, 2)
        self.residual = nn.Linear(2, 2)


class _Wrapped(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Linear(2, 2)
        self.scalar = nn.Parameter(torch.ones(()))


class FinalizeEndpointTests(unittest.TestCase):
    def test_actual_finalizer_writes_hash_bound_manifest_and_group_steps(self):
        suffix = f"{os.getpid()}{uuid.uuid4().hex[:4]}"
        root = HERE / "results_archive" / f"e{suffix}"
        output = root / "o"
        work = root / "w" / "c"
        work.mkdir(parents=True)
        (work / "boundary.ptx").write_bytes(b"committed-boundary-fixture")
        try:
            wrapped, encoder = _Wrapped(), _Encoder()
            groups = {
                "spatial_residual": list(encoder.residual.parameters()),
                "graph": list(wrapped.core.parameters()),
                "native_core": [wrapped.scalar],
                "integration_scalars": [nn.Parameter(torch.tensor(0.25))],
            }
            optimizer = torch.optim.Adam([
                {"name": name, "params": params, "lr": 1e-3}
                for name, params in groups.items()])
            foundation = type("Foundation", (), {"wrapped": wrapped, "encoder": encoder})()
            frozen_sha = training.state_sha(encoder.base_encoder)
            mass_sha = hashlib.sha256(b"fixed source mass").hexdigest()
            bindings = {"source_mass_sha256": mass_sha,
                        "implementation_fingerprint": {"test": "fixture"},
                        "implementation_sha256": hashlib.sha256(b"fixture").hexdigest()}
            history = [{"update": i + 1,
                        "image_ids": list(range(1000 + i * train.BATCH,
                                                 1000 + (i + 1) * train.BATCH))}
                       for i in range(train.UPDATES)]
            result = train._finalize("equivariance", foundation=foundation,
                optimizer=optimizer, family_map=groups, history=history,
                bindings=bindings, output_dir=output, work_dir=work,
                initial_hashes={name: training.parameter_hash(params)
                                for name, params in groups.items()},
                frozen_base_sha=frozen_sha)
            self.assertEqual(result["status"], "training_complete")
            self.assertEqual(result["completed_updates"], train.UPDATES)
            self.assertEqual(result["train_ids"], [x for row in history for x in row["image_ids"]])
            self.assertEqual(set(result["optimizer_step_summary"]), set(groups))
            self.assertEqual(result["frozen_base_encoder_sha256_before_after"], frozen_sha)
            for name in ("checkpoint.pt", "optimizer.pt", "history.json", "manifest.json"):
                self.assertTrue((output / name).is_file(), name)
            self.assertEqual(result["checkpoint_sha256"], train.sha(output / "checkpoint.pt"))
            self.assertEqual(result["optimizer_sha256"], train.sha(output / "optimizer.pt"))
            self.assertEqual(result["history_sha256"], train.sha(output / "history.json"))
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
