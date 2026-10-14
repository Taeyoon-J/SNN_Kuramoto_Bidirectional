from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock
import uuid

import numpy
import torch
from torch import nn

from collaborative_test.SW_0160_end_to_end_source_encoder import preflight, train
from collaborative_test.SW_0160_end_to_end_source_encoder.test_preflight import _record


class _Scratch:
    def __init__(self, base):
        self.parent = Path(base) / "results_archive" / "cpu_test_scratch"
        self.path = self.parent / f"train-{uuid.uuid4().hex}"

    def __enter__(self):
        self.parent.mkdir(parents=True, exist_ok=True)
        self.path.mkdir()
        return str(self.path)

    def __exit__(self, *_):
        resolved_parent = self.parent.resolve()
        resolved_path = self.path.resolve()
        if not resolved_path.is_relative_to(resolved_parent):
            raise RuntimeError("refusing to remove a test path outside its owned scratch root")
        for child in sorted(self.path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            child.rmdir() if child.is_dir() else child.unlink()
        self.path.rmdir()


class _Wrapped:
    def __init__(self):
        self.projected = 0

    def project_integrations_(self):
        self.projected += 1


class SW0160TrainTests(unittest.TestCase):
    def test_published_score_waits_for_live_sw0159_terminal_state(self):
        from collaborative_test.SW_0159_one_shot_event_regime import (
            predict_evaluate as sw159_eval, score_queue as sw159_queue)
        names = ("control", "candidate", "source97", "SW0143_T")
        means = {name: {metric: 0.5 for metric in train.score_contract.METRICS} for name in names}
        score = {name: {metric: {"mean": 0.5, "valid_count": 320,
                    "per_image": [0.5] * 320} for metric in train.score_contract.METRICS}
                 for name in names}
        report = {"experiment": "SW0159_one_shot_event_regime", "status": "complete",
            "seed_scope": [1], "implementation_fingerprint": sw159_eval.implementation_fingerprint(),
            "scores": score, "means": means,
            "ground_truth_opened_after_all_predictions_validated": True,
            "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True}
        queue_fp = sw159_queue.implementation_fingerprint()
        score_fp = sw159_eval.implementation_fingerprint()
        with _Scratch(train.ROOT) as directory:
            score_path = Path(directory) / "evaluation_seed1.json"
            state_path = Path(directory) / "score_state.json"
            score_path.write_text(json.dumps(report), encoding="utf-8")
            state = {"experiment": "SW0159_one_shot_event_regime", "status": "running",
                "supervisor_pid": 123, "score_sha256": train.sha256_file(score_path),
                "queue_fingerprint": queue_fp, "score_fingerprint": score_fp}
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with (mock.patch.object(sw159_queue, "STATE", state_path),
                  mock.patch.object(sw159_eval, "SCORE_PATH", score_path),
                  mock.patch.object(train, "_sw159_score_supervisor_live", return_value=True)):
                with self.assertRaisesRegex(RuntimeError, "published but its live queue"):
                    train._score_dependency()
            with (mock.patch.object(sw159_queue, "STATE", state_path),
                  mock.patch.object(sw159_eval, "SCORE_PATH", score_path),
                  mock.patch.object(train, "_sw159_score_supervisor_live", return_value=False)):
                with self.assertRaisesRegex(ValueError, "terminal or has no live owner"):
                    train._score_dependency()
            state["status"] = "complete"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with (mock.patch.object(sw159_queue, "STATE", state_path),
                  mock.patch.object(sw159_eval, "SCORE_PATH", score_path)):
                bound = train._score_dependency()
            self.assertEqual(bound["score_sha256"], state["score_sha256"])
            self.assertEqual(bound["status"], "complete")

    def test_serialized_optimizer_steps_match_training_manifest_and_arm_groups(self):
        names = [name for name in train.model.LEARNING_RATES if name != "source_base"]
        groups, state = [], {}
        summary = {name: 0 for name in train.model.LEARNING_RATES}
        for index, name in enumerate(names):
            groups.append({"name": name, "lr": train.model.LEARNING_RATES[name], "params": [index]})
            state[index] = {"step": torch.tensor(train.UPDATES),
                "exp_avg": torch.ones(1), "exp_avg_sq": torch.ones(1)}
            summary[name] = train.UPDATES
        payload = {"param_groups": groups, "state": state}
        self.assertTrue(train.validate_optimizer_payload(payload, summary, train.model.ARMS[0]))
        state[0]["step"] = torch.tensor(train.UPDATES - 1)
        self.assertFalse(train.validate_optimizer_payload(payload, summary, train.model.ARMS[0]))

    def test_four_micro_logical_update_records_finite_credit_and_real_change(self):
        p = nn.Parameter(torch.tensor([1.0, -0.5]))
        families = {"source_base": [], "spatial_residual": [p]}
        optimizer = torch.optim.Adam([{"name": "spatial_residual", "params": [p], "lr": 0.01}])
        wrapped = _Wrapped()
        targets = [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]),
                   torch.tensor([0.5, -0.5]), torch.tensor([-1.0, 1.0])]
        row = train._logical_update(optimizer=optimizer, families=families, wrapped=wrapped,
            compute_micro=lambda index: (p - targets[index]).square().mean())
        self.assertEqual(wrapped.projected, 1)
        self.assertEqual(set(row["family_gradient_norms"]), set(families))
        self.assertEqual(row["family_gradient_norms"]["source_base"], 0.0)
        self.assertGreater(row["family_gradient_norms"]["spatial_residual"], 0.0)
        self.assertGreater(row["family_parameter_change_l2"]["spatial_residual"], 0.0)
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue(numpy.isfinite(row["mean_loss"]))
        self.assertTrue(numpy.isfinite(row["global_preclip_gradient_norm"]))

    def test_admission_binds_current_preflight_and_refuses_missing_or_changed_score(self):
        report = _record(admit=True)
        report["implementation_fingerprint"] = preflight.implementation_fingerprint()
        upstream = report["upstream"]
        score = {"status": "complete", "score_sha256": "s" * 64}
        with _Scratch(train.ROOT) as directory:
            path = Path(directory) / "preflight.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with (mock.patch.object(preflight, "validate_upstream", return_value=upstream),
                  mock.patch.object(train, "_score_dependency", return_value=score)):
                accepted, dependency = train.validate_preflight(path)
                self.assertIs(accepted["training_admission"], True)
                self.assertEqual(dependency, score)
                with self.assertRaisesRegex(ValueError, "changed since"):
                    train.validate_preflight(path, score_dependency={**score, "score_sha256": "x" * 64})
                rejected = _record(admit=False)
                rejected["implementation_fingerprint"] = preflight.implementation_fingerprint()
                rejected["source"]["source_base_encoder_sha256"] = \
                    rejected["arms"]["full_encoder_joint"]["base_encoder_sha256_before"]
                rejected["arms"]["full_encoder_joint"]["base_encoder_sha256_after"] = \
                    rejected["arms"]["full_encoder_joint"]["base_encoder_sha256_before"]
                rejected["arms"]["full_encoder_joint"]["base_encoder_changed"] = False
                rejected["admission_reasons"] = ["candidate_source_base_unchanged_after_two_updates"]
                path.write_text(json.dumps(rejected), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "scientifically rejected"):
                    train.validate_preflight(path)

    def test_current_train_fingerprint_resolves_checkpoint_and_source_dependencies(self):
        fp = train.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0160_end_to_end_source_encoder/train.py", fp)
        self.assertIn("collaborative_test/SW_0159_one_shot_event_regime/score_queue.py", fp)
        self.assertIn("collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py", fp)
        self.assertTrue(all(len(value) == 64 for value in fp.values()))


if __name__ == "__main__":
    unittest.main()
