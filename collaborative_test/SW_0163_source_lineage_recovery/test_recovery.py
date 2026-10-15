"""CPU regressions for the isolated SW0163 source-lineage recovery path."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import numpy as np
import torch
torch.set_num_threads(1)
import hashlib, json, shutil, unittest, uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TEST_TMP = Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
TEST_TMP.mkdir(parents=True, exist_ok=True)

class OwnedTemp:
    """Short, task-owned fixtures avoid Windows TEMP ACL failures."""
    def __enter__(self):
        self.path = TEST_TMP / ("sw163_recovery_" + uuid.uuid4().hex[:10])
        self.path.mkdir()
        return str(self.path)
    def __exit__(self, *_):
        shutil.rmtree(self.path)

from collaborative_test.SW_0161_supervised_spike_partition import cache as cache161
from collaborative_test.SW_0162_connected_signed_partition import preflight as digest162
from collaborative_test.SW_0163_source_lineage_recovery import predict_evaluate as recovery
from collaborative_test.SW_0163_source_lineage_recovery import prediction_queue, score_queue
from collaborative_test.SW_0163_relation_context_assignment import model, train, training_queue
from collaborative_test.SW_0163_relation_context_assignment import recovery_training_queue

class LineageDigestTests(unittest.TestCase):
    def foundation(self):
        return SimpleNamespace(
            wrapped=torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4)),
            encoder=torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.BatchNorm2d(2)),
            patcher=torch.nn.Identity(),
            feature_mean=torch.tensor([0.1, 0.2], dtype=torch.float32),
            feature_std=torch.tensor([1.0, 2.0], dtype=torch.float32),
            feature_clip=3.0,
        )

    def test_recovery_uses_cache_digest_and_rejects_sw0162_family(self):
        f=self.foundation()
        expected=cache161._state_digest(f)
        self.assertEqual(recovery._source_state_digest(f), expected)
        other=digest162._source_state_digest(f)
        self.assertNotEqual(other, expected)
        self.assertEqual(recovery.require_source_lineage(f, expected), expected)
        with self.assertRaisesRegex(ValueError, "differs from the SW0161 cache digest"):
            recovery.require_source_lineage(f, other)

    def test_parameter_and_feature_clip_changes_invalidate_lineage(self):
        f=self.foundation(); expected=cache161._state_digest(f)
        with torch.no_grad(): f.wrapped[0].weight[0,0].add_(0.25)
        self.assertNotEqual(cache161._state_digest(f), expected)
        with self.assertRaises(ValueError): recovery.require_source_lineage(f, expected)
        f=self.foundation(); expected=cache161._state_digest(f)
        f.feature_clip=3.25
        self.assertNotEqual(cache161._state_digest(f), expected)
        with self.assertRaises(ValueError): recovery.require_source_lineage(f, expected)

    def test_all_head_bindings_must_share_one_source_digest(self):
        sha="a"*64
        training={arm:{"bindings":{"source_model_state_sha256":sha}} for arm in model.ARMS}
        self.assertEqual(recovery._expected_training_source_state(training), sha)
        training[model.ARMS[-1]]["bindings"]["source_model_state_sha256"]="b"*64
        with self.assertRaisesRegex(ValueError, "do not share"):
            recovery._expected_training_source_state(training)

    def test_recovery_outputs_are_isolated_and_provenance_explicit(self):
        self.assertIn("source_lineage_recovery_20261015", str(recovery.RECOVERY_ROOT))
        self.assertTrue(str(recovery.RECOVERY_ROOT).startswith(str(recovery.ROOT)))
        self.assertEqual(recovery.DIGEST_FAMILY,"SW0161_supervised_spike_partition.cache._state_digest")
        self.assertEqual(recovery.recovery_provenance()["id"], recovery.RECOVERY_ID)
        self.assertEqual(recovery.PREDICTION_ROOT.parent,recovery.RECOVERY_ROOT)
        self.assertEqual(prediction_queue.command()[2],"collaborative_test.SW_0163_source_lineage_recovery.predict_evaluate")

class RecoveryQueueDependencyTests(unittest.TestCase):
    def test_prediction_waits_for_live_recovery_training_owner_then_admits(self):
        with OwnedTemp() as temp:
            path=Path(temp)/"state.json"
            pending={"status":"training_arms_queued","supervisor_pid":1234}
            path.write_text(json.dumps(pending))
            self.assertIsNone(prediction_queue.training_dependency(state_path=path,process_live=lambda pid: True))
            arms=list(train.ARMS)
            done={"status":"complete","recovery_status":"complete_recovered","supervisor_pid":1234,
                  "queue_fingerprint":"sealed-base-fp","recovery_fingerprint":{"x":"y"},
                  "results":{a:True for a in arms},"final_arms":{a:{} for a in arms}}
            path.write_text(json.dumps(done))
            with mock.patch.object(recovery_training_queue,"verify_recovery_fingerprint") as verify, \
                 mock.patch.object(training_queue,"implementation_fingerprint",return_value="sealed-base-fp"):
                result=prediction_queue.training_dependency(state_path=path)
            self.assertEqual(result["state"],done); verify.assert_called_once_with({"x":"y"})
            done["final_arms"].pop(arms[0]); path.write_text(json.dumps(done))
            with mock.patch.object(recovery_training_queue,"verify_recovery_fingerprint"), \
                 mock.patch.object(training_queue,"implementation_fingerprint",return_value="sealed-base-fp"):
                with self.assertRaisesRegex(ValueError,"inventory is incomplete"):
                    prediction_queue.training_dependency(state_path=path)

    def test_score_dependency_requires_recovery_prediction_owner_binding_and_hashes(self):
        with OwnedTemp() as temp:
            root=Path(temp); state_path=root/"state.json"; expected={}
            for arm in (*train.ARMS,"source_T_control"):
                folder=root/arm; folder.mkdir()
                (folder/"predictions.npz").write_bytes((arm+"-prediction").encode())
                (folder/"protocol.json").write_bytes((arm+"-protocol").encode())
                expected[arm]={"prediction_sha256":hashlib.sha256((folder/"predictions.npz").read_bytes()).hexdigest(),
                               "protocol_sha256":hashlib.sha256((folder/"protocol.json").read_bytes()).hexdigest()}
            state={"status":"complete","queue_fingerprint":"pred-fp","prediction_hashes":expected,
                   "training_recovery_fingerprint":{"x":"y"}}
            state_path.write_text(json.dumps(state))
            real_out=evaluation_out=score_queue.evaluation._out
            def fake_out(arm): return root/arm
            with mock.patch.object(prediction_queue,"implementation_fingerprint",return_value="pred-fp"), \
                 mock.patch.object(recovery_training_queue,"verify_recovery_fingerprint") as verify, \
                 mock.patch.object(score_queue.evaluation,"_out",side_effect=fake_out):
                result=score_queue.prediction_dependency(state_path=state_path)
            self.assertEqual(result["state"],state); verify.assert_called_once_with({"x":"y"})
            state["prediction_hashes"]["raw_only"]["prediction_sha256"]="0"*64
            state_path.write_text(json.dumps(state))
            with mock.patch.object(prediction_queue,"implementation_fingerprint",return_value="pred-fp"), \
                 mock.patch.object(recovery_training_queue,"verify_recovery_fingerprint"), \
                 mock.patch.object(score_queue.evaluation,"_out",side_effect=fake_out):
                with self.assertRaisesRegex(ValueError,"archive changed"):
                    score_queue.prediction_dependency(state_path=state_path)

    def test_queue_fingerprints_bind_recovery_entrypoints_and_owner_guard(self):
        pf=prediction_queue.implementation_fingerprint()
        sf=score_queue.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0163_source_lineage_recovery/predict_evaluate.py",pf)
        self.assertIn("collaborative_test/SW_0163_source_lineage_recovery/prediction_queue.py",sf)
        self.assertTrue(any(name.endswith("SW_0126_history_event_binding/pilot_queue.py") for name in pf))
        self.assertEqual(prediction_queue.MODULE,"collaborative_test.SW_0163_source_lineage_recovery.prediction_queue")

if __name__=="__main__": unittest.main()
