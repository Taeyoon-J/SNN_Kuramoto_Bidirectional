from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from collaborative_test.SW_0155_supervised_spike_relations import train
from collaborative_test.SW_0155_supervised_spike_relations import preflight


class _Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_generator = nn.Linear(2, 2, bias=False)
        self.native = nn.Linear(2, 2, bias=False)


class _Wrapped(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = _Core()
        self.a_d = nn.Parameter(torch.ones(4))
        self.a_m = nn.Parameter(torch.ones(4))
        self.b = nn.Parameter(torch.zeros(4))

    def project_integrations_(self):
        with torch.no_grad():
            self.a_d.clamp_(min=0)
            self.a_m.clamp_(min=0)


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x


class _Foundation:
    def __init__(self):
        self.encoder = train.SourcePreservingDeepEncoder(_Base())
        self.wrapped = _Wrapped()


class SW0155TrainingTests(unittest.TestCase):
    def test_preflight_hdf5_schema_is_snapshotted_while_handle_is_open(self):
        class Dataset:
            def __init__(self, shape):
                self.shape = shape
        class Handle:
            def __getitem__(self, key):
                if key == "image":
                    return Dataset((100000, 128, 128, 3))
                return Dataset((100000, 128, 128, 1))
        path = train.HERE / "README.md"
        record = preflight.canonical_hdf5_record(path, Handle())
        self.assertEqual(record["path"], str(path.resolve()))
        self.assertEqual(record["image_shape"], [100000, 128, 128, 3])
        self.assertEqual(record["mask_shape"], [100000, 128, 128, 1])
        with self.assertRaisesRegex(ValueError, "schema"):
            preflight.canonical_hdf5_record(path, type("Bad", (), {
                "__getitem__": lambda self, key: Dataset((1,))})())

    def test_train_preflight_admission_accepts_train_masks_and_binds_source_hdf5(self):
        ids = list(range(1000)) + list(range(1640, 2688)) + list(range(3000, 5048))
        source = {"image_ids": ids, "source_sha256": "a" * 64,
                  "manifest_path": train.HERE / "README.md"}
        assets = {"train_cache_sha256": "b" * 64}
        hdf = train.HERE / "README.md"
        stat = hdf.stat()
        families = {
            "encoder_graph": ["graph", "spatial_residual"],
            "joint": ["graph", "integration_tensors", "native_core", "spatial_residual"],
        }
        arms = {}
        for arm, names in families.items():
            arms[arm] = {
                "status": "two_actual_updates_complete",
                "source_zero_adapter_parity_exact": True,
                "ground_truth_scope": "TRAIN masks only",
                "selected_families": names,
                "family_sha256_before": {name: "c" * 64 for name in names},
                "family_sha256_after": {name: "d" * 64 for name in names},
                "updates": [{"mean_loss": 1.0, "family_gradient_norms": {name: 0.5 for name in names}
                             } for _ in range(2)],
                "frozen_base_encoder_sha256_before_after": {"before": "e" * 64, "after": "e" * 64},
            }
        report = {
            "status": "source_preflight_complete", "experiment": train.EXPERIMENT, "seed": 1,
            "training_admission": True, "ground_truth_used": True,
            "ground_truth_scope": "TRAIN masks only",
            "implementation_fingerprint": train.implementation_fingerprint(),
            "registered": {"updates": 2, "logical_batch": 16, "microbatch": 4,
                           "steps": 1024, "settle": 512, "live_tail": 64},
            "arms": arms,
            "source": {"checkpoint_sha256": source["source_sha256"],
                "manifest_sha256": train.sha256_file(source["manifest_path"]),
                "training_ids": ids, "training_ids_sha256": train.ids_sha(ids),
                "selected_train_ids": ids[:32], "selected_train_ids_sha256": train.ids_sha(ids[:32]),
                "rgb_asset_validation": assets},
            "canonical_hdf5": {"path": str(hdf.resolve()), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "mask_shape": [100000, 128, 128, 1]},
        }
        self.assertTrue(preflight.validate_admission(report, source=source, assets=assets,
            hdf5_path=hdf, hdf5_size=stat.st_size, hdf5_mtime_ns=stat.st_mtime_ns))
        bad = dict(report, ground_truth_scope="validation masks")
        self.assertFalse(preflight.validate_admission(bad, source=source, assets=assets,
            hdf5_path=hdf, hdf5_size=stat.st_size, hdf5_mtime_ns=stat.st_mtime_ns))

    def test_trainability_and_adam_groups_match_both_registered_arms(self):
        encoder_only = _Foundation()
        eg = train.configure_trainability(encoder_only, "encoder_graph")
        self.assertEqual(set(eg), {"spatial_residual", "graph"})
        self.assertFalse(encoder_only.wrapped.core.native.weight.requires_grad)
        self.assertFalse(encoder_only.wrapped.a_d.requires_grad)
        self.assertTrue(encoder_only.wrapped.core.graph_generator.weight.requires_grad)
        self.assertFalse(encoder_only.encoder.base_encoder.weight.requires_grad)
        eg_opt = train.make_optimizer(eg)
        self.assertEqual([row["name"] for row in eg_opt.param_groups],
                         ["spatial_residual", "graph"])
        self.assertEqual([row["lr"] for row in eg_opt.param_groups], [3e-6, 3e-5])

        joint_foundation = _Foundation()
        joint = train.configure_trainability(joint_foundation, "joint")
        self.assertEqual(set(joint), {"spatial_residual", "graph", "native_core", "integration_tensors"})
        self.assertEqual(sum(p.numel() for p in joint["integration_tensors"]), 12)
        joint_opt = train.make_optimizer(joint)
        self.assertEqual([row["name"] for row in joint_opt.param_groups],
                         ["spatial_residual", "graph", "native_core", "integration_tensors"])
        self.assertEqual([row["lr"] for row in joint_opt.param_groups],
                         [3e-6, 3e-5, 3e-5, 1e-3])

    def test_sorted_hdf_rows_restore_source_order_and_check_rgb(self):
        rng = np.random.default_rng(51)
        image_rows = rng.integers(0, 256, size=(10, 128, 128, 3), dtype=np.uint8)
        masks = np.zeros((10, 128, 128, 1), dtype=np.uint16)
        for row in range(10):
            masks[row, 8:40, 12:60, 0] = row + 1
        ids = [7, 2, 5]
        cache = image_rows[ids].copy()
        order = np.argsort(ids)
        rgb, aligned_masks = train.restore_source_order(
            ids, np.asarray(ids)[order], image_rows[np.asarray(ids)[order]],
            masks[np.asarray(ids)[order]], cache)
        self.assertTrue(np.array_equal(rgb.numpy(), cache))
        self.assertEqual(aligned_masks.shape, (3, 128, 128))
        self.assertEqual(aligned_masks[:, 10, 15].tolist(), [8, 3, 6])
        bad_cache = cache.copy()
        bad_cache[1, 0, 0, 0] ^= 1
        with self.assertRaisesRegex(ValueError, "do not match"):
            train.restore_source_order(
                ids, np.asarray(ids)[order], image_rows[np.asarray(ids)[order]],
                masks[np.asarray(ids)[order]], bad_cache)

    def test_reserved_ids_rejected_before_dataset_access(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            train.load_aligned_train_rows("missing.h5", [1320], np.zeros(
                (1, 128, 128, 3), dtype=np.uint8), 0, 1, expected_rows=1)

    def test_logical_update_aggregates_four_micro_losses_and_steps_once(self):
        parameter = nn.Parameter(torch.tensor([0.4, -0.2]))
        optimizer = torch.optim.Adam([{"name": "graph", "params": [parameter], "lr": 0.01}])
        class Projector:
            def project_integrations_(self):
                pass
        targets = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]),
                   torch.tensor([1.0, 1.0]), torch.tensor([-1.0, 0.5])]
        def compute(index):
            return ((parameter - targets[index]) ** 2).mean()
        before = parameter.detach().clone()
        record = train.logical_update(optimizer=optimizer, family_map={"graph": [parameter]},
                                      compute_micro=compute, wrapped=Projector())
        self.assertTrue(torch.isfinite(torch.tensor(record["mean_loss"])))
        self.assertGreater(record["family_gradient_norms"]["graph"], 0)
        self.assertFalse(torch.equal(before, parameter.detach()))
        self.assertEqual(int(optimizer.state[parameter]["step"]), 1)


if __name__ == "__main__":
    unittest.main()
