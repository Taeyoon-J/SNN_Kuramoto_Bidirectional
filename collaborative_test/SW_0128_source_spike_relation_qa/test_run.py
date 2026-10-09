import json
import uuid
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0128_source_spike_relation_qa import run


class PairSemanticsTests(unittest.TestCase):
    def test_bg_foreground_negative_and_object_merge_are_counted_correctly(self):
        labels = torch.zeros(16, 16, dtype=torch.long)
        gt = torch.zeros_like(labels)
        labels.view(-1)[0:2] = 1
        gt.view(-1)[0:2] = 7
        labels.view(-1)[3] = 1
        labels.view(-1)[5] = 2
        gt.view(-1)[3] = 7
        gt.view(-1)[5] = 9
        q = torch.zeros(256, 256)
        q.fill_(0.5)
        q[0, 1] = q[1, 0] = 0.95
        q[0, 2] = q[2, 0] = 0.05  # foreground/background is an allowed negative
        q[1, 2] = q[2, 1] = 0.05
        q[0, 3] = q[3, 0] = 0.95  # same foreground label, mid-distance bin
        q[0, 5] = q[5, 0] = 0.05  # different foreground labels, exact distance-5 boundary
        frozen = run._build_relation_masks(labels, q)
        got = run._pair_audit(frozen, gt)
        near, mid = got["distance_0_2"], got["distance_2_5"]
        self.assertEqual(near["positive_fg_relevant"], 2)  # directed anchor decisions
        self.assertEqual(near["positive_correct"], 2)
        self.assertEqual(near["negative_fg_relevant"], 2)  # BG is a valid negative neighbor
        self.assertEqual(near["negative_correct"], 2)
        self.assertEqual(mid["positive_correct"], 1)
        self.assertEqual(mid["negative_fg_fg_different"], 1)
        self.assertEqual(mid["negative_correct"], 1)
        self.assertEqual(mid["negative_fg_relevant"], 1)
        self.assertEqual(near["eligible_anchors"], 2)

    def test_q_threshold_edges_are_inclusive_and_distance_bins_are_open_closed(self):
        labels = torch.zeros(16, 16, dtype=torch.long)
        gt = torch.zeros_like(labels)
        labels.view(-1)[:3] = torch.tensor([1, 1, 0])
        gt.view(-1)[:3] = torch.tensor([2, 2, 0])
        q = torch.zeros(256, 256)
        q.fill_(0.5)
        q[0, 1] = q[1, 0] = 0.9
        q[0, 2] = q[2, 0] = 0.1
        q[1, 2] = q[2, 1] = 0.1
        got = run._build_relation_masks(labels, q)
        self.assertEqual(int(got["distance_0_2"]["positive"].sum()), 2)
        self.assertEqual(int(got["distance_0_2"]["negative"].sum()), 2)
        # Pair at distance exactly 2 is in (0,2].
        labels.view(-1)[32] = 2
        gt.view(-1)[32] = 3
        q[0, 32] = q[32, 0] = 0.1
        self.assertEqual(int(run._build_relation_masks(labels, q)["distance_0_2"]["negative"].sum()), 3)

    def test_gt_never_changes_frozen_anchor_eligibility(self):
        labels = torch.zeros(16, 16, dtype=torch.long)
        labels.view(-1)[:2] = 1
        labels.view(-1)[2] = 2
        q = torch.zeros(256, 256)
        q.fill_(0.5)
        q[0, 1] = q[1, 0] = 0.95
        q[0, 2] = q[2, 0] = 0.05
        q[1, 2] = q[2, 1] = 0.05
        frozen = run._build_relation_masks(labels, q)
        eligible = frozen["distance_0_2"]["eligible_anchor"].clone()
        gt_a = torch.zeros(16, 16, dtype=torch.long)
        gt_b = torch.ones(16, 16, dtype=torch.long)
        audit_a = run._pair_audit(frozen, gt_a)
        audit_b = run._pair_audit(frozen, gt_b)
        self.assertTrue(torch.equal(eligible, frozen["distance_0_2"]["eligible_anchor"]))
        self.assertEqual(int(eligible.sum()), 2)
        self.assertNotEqual(audit_a["distance_0_2"]["positive_correct"],
                            audit_b["distance_0_2"]["positive_correct"])


class FreezeBeforeGroundTruthTests(unittest.TestCase):
    def _fixture(self, directory):
        output = Path(directory)
        output.mkdir(parents=True)
        labels = torch.zeros(16, 16, 16, dtype=torch.long)
        q = torch.zeros(16, 256, 256)
        bundle = {"image_ids": list(run.IDS), "predictions": {
            str(seed): {"labels": labels.clone(), "q": q.clone(),
                        "relation_masks": [run._build_relation_masks(labels[i], q[i])
                                           for i in range(16)]}
            for seed in run.SEEDS}}
        pred_path = output / "frozen_predictions.pt"
        torch.save(bundle, pred_path)
        digest = run.sha256_file(pred_path)
        hash_rows = {str(seed): [
            {bin_name: {kind: run._relation_mask_hash(mask_row[bin_name][kind])
                        for kind in ("positive", "negative", "eligible_anchor")}
             for bin_name, _, _ in run.BINS}
            for mask_row in bundle["predictions"][str(seed)]["relation_masks"]]
            for seed in run.SEEDS}
        manifest = {"status": "predictions_complete", "experiment": "SW0128",
            "image_ids": [run.IDS[0], run.IDS[-1]], "ground_truth_used_for_prediction": False,
            "implementation_fingerprint": run._implementation_fingerprint(),
            "validation_gamma_sha256": "g", "validation_gamma_manifest_sha256": "gm",
            "source_provenance": {str(seed): {"source_checkpoint_sha256": f"core{seed}",
                "source_manifest_sha256": f"manifest{seed}"} for seed in run.SEEDS},
            "relation_mask_sha256_by_seed_image_bin": {str(seed): [
                {bin_name: {kind: run._relation_mask_hash(masks[bin_name][kind])
                            for kind in ("positive", "negative", "eligible_anchor")}
                 for bin_name, _, _ in run.BINS}
                for masks in bundle["predictions"][str(seed)]["relation_masks"]]
                for seed in run.SEEDS},
            "prediction_sha256": digest}
        (output / "prediction_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return output, manifest

    def test_all_seed_bundle_is_checked_before_target_callback(self):
        root = Path.cwd().parent / f"sw128_{uuid.uuid4().hex[:8]}"
        output, _manifest = self._fixture(root / "qa")
        self.addCleanup(self._cleanup, root, output)
        def fake_source(seed):
            return Path(f"source{seed}"), Path(f"manifest{seed}"), {}
        with (mock.patch.object(run, "_source_contract", side_effect=fake_source),
                  mock.patch.object(run.base, "GAMMA_VAL", Path("gamma")),
                  mock.patch.object(run.base, "GAMMA_VAL_MANIFEST", Path("gamma_manifest")),
                  mock.patch.object(run, "sha256_file", side_effect=lambda p: {
                      "gamma": "g", "gamma_manifest": "gm",
                      "source0": "core0", "source1": "core1", "source2": "core2",
                      "manifest0": "manifest0", "manifest1": "manifest1", "manifest2": "manifest2",
                  }.get(str(p), self._real_sha(p)))):
                called = []
                targets = torch.zeros(16, 16, 16, dtype=torch.long)
                result = run._score_frozen(output, lambda: called.append(True) or targets)
                self.assertEqual(called, [True])
                self.assertTrue(result["all_three_prediction_bundles_sha_verified_before_ground_truth"])

    def test_modified_frozen_artifact_refuses_ground_truth(self):
        root = Path.cwd().parent / f"sw128_{uuid.uuid4().hex[:8]}"
        output, _manifest = self._fixture(root / "qa")
        self.addCleanup(self._cleanup, root, output)
        with (mock.patch.object(run, "_source_contract", side_effect=lambda s: (Path(f"source{s}"), Path(f"manifest{s}"), {})),
                  mock.patch.object(run.base, "GAMMA_VAL", Path("gamma")),
                  mock.patch.object(run.base, "GAMMA_VAL_MANIFEST", Path("gamma_manifest")),
                  mock.patch.object(run, "sha256_file", side_effect=lambda p: {
                      "gamma": "g", "gamma_manifest": "gm", "source0": "core0", "source1": "core1",
                      "source2": "core2", "manifest0": "manifest0", "manifest1": "manifest1",
                      "manifest2": "manifest2"}.get(str(p), self._real_sha(p)))):
                with (output / "frozen_predictions.pt").open("ab") as stream:
                    stream.write(b"tamper")
                called = []
                with self.assertRaises(ValueError):
                    run._score_frozen(output, lambda: called.append(True))
                self.assertEqual(called, [])

    @staticmethod
    def _cleanup(root, output):
        for name in ("frozen_predictions.pt", "prediction_manifest.json"):
            path = output / name
            if path.exists():
                path.unlink()
        if output.exists():
            output.rmdir()
        if root.exists():
            root.rmdir()

    @staticmethod
    def _real_sha(path):
        from hashlib import sha256
        p = Path(path)
        if not p.is_file():
            return ""
        h = sha256()
        with p.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()


if __name__ == "__main__":
    unittest.main()
