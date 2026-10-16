"""Owner/wait contracts for the optional SW0175 TRAIN8 diagnostic."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0175_critical_component_binding import train8_queue as queue


def _central_temp() -> Path:
    root = Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cleanup(root: Path, temp_root: Path) -> None:
    resolved, parent = root.resolve(), temp_root.resolve()
    if resolved.parent != parent:
        raise AssertionError(f"unsafe temporary test path: {resolved}")
    for item in sorted(resolved.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            item.rmdir()
    resolved.rmdir()


class Train8QueueTests(unittest.TestCase):
    def test_worker_command_pins_module_gpu_output_and_completed_score(self):
        score_sha = "a" * 64
        command = queue.worker_command(2, queue.REPORT, score_sha)
        self.assertEqual(command[command.index("-m") + 1], queue.MODULE)
        self.assertIn(("--physical-gpu", "2"), list(zip(command, command[1:])))
        self.assertIn(("--score-sha256", score_sha), list(zip(command, command[1:])))
        self.assertEqual(Path(command[command.index("--output") + 1]), queue.REPORT)
        with self.assertRaises(ValueError):
            queue.worker_command(2, queue.REPORT, "not-a-sha")

    def test_actual_owner_api_uses_four_argument_stable_idle(self):
        from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
        import inspect
        self.assertEqual(len(inspect.signature(owner._stable_idle).parameters), 4)

    def test_completed_score_requires_exact_artifact_hash_then_public_validation(self):
        temp_root = _central_temp()
        root = temp_root / f"sw175_train8_score_{os.getpid()}"
        root.mkdir()
        try:
            score_path = root / "score.json"
            score_path.write_text('{"status":"complete"}\n', encoding="utf-8")
            digest = hashlib.sha256(score_path.read_bytes()).hexdigest()
            state_path = root / "state.json"
            state_path.write_text(json.dumps({"experiment": "SW0175_score_queue_v3",
                "status": "complete", "score_path": str(score_path),
                "score_sha256": digest}), encoding="utf-8")
            score_module = SimpleNamespace(RUNTIME=root)
            evaluator = SimpleNamespace(SCORE_PATH=score_path,
                prepare_prediction_inputs_cpu=mock.Mock(return_value={}),
                validate_score=mock.Mock(return_value={"status": "complete"}))
            result = queue._score_complete(object(), score_module, evaluator)
            self.assertEqual(result["score_sha256"], digest)
            evaluator.validate_score.assert_called_once_with(path=score_path, prepared={})
            score_path.write_text('{"status":"tampered"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                queue._score_complete(object(), score_module, evaluator)
        finally:
            _cleanup(root, temp_root)


    def test_supervisor_happy_path_reaches_popen_with_score_bound_command(self):
        temp_root = _central_temp()
        root = temp_root / f"sw175_train8_supervise_{os.getpid()}"
        root.mkdir()
        runtime = root / "runtime"
        report_path = root / "audit.json"
        score_runtime = root / "score_runtime"
        score_runtime.mkdir()
        score_path = root / "score.json"
        score_path.write_text('{"score":true}\n', encoding="utf-8")
        score_sha = hashlib.sha256(score_path.read_bytes()).hexdigest()
        (score_runtime / "state.json").write_text(json.dumps({"status":"complete",
            "score_sha256":score_sha,"score_path":str(score_path)}), encoding="utf-8")
        lease_path = root / "lease"
        lease_path.write_bytes(b"")
        lease = lease_path.open("a+")
        proof = {"source_cache_manifest_sha256":"a"*64}
        fingerprint = {"source":"b"*64}
        scoreq = SimpleNamespace(RUNTIME=score_runtime)
        evaluator = SimpleNamespace(SCORE_PATH=score_path)
        audit = SimpleNamespace(prepare_inputs_cpu=lambda: {"admitted":{"proof":proof},
                "fingerprint":fingerprint}, implementation_fingerprint=lambda:fingerprint)
        class Child:
            pid = 43210
            returncode = 0
            def poll(self): return 0
            def wait(self): return 0
        child = Child()
        owner = SimpleNamespace(_acquire=lambda _stop:(2,"gpu-2",0,lease),
            _stable_idle=lambda gpu,gpu_uuid,stop,on_sample:True,
            gpu_guard=SimpleNamespace(_gpu_info=lambda gpu:("gpu-2",0,[])),MAX_MEMORY_MIB=512)
        calls = []
        def popen(command, **kwargs):
            calls.append(command)
            report_path.write_text(json.dumps({
                "status":"complete_non_gating_diagnostic",
                "implementation_fingerprint":fingerprint,"source_proof":proof,
                "optimizer_updates":0,"validation_data_read":False,
                "cuda_visible_devices":"2","score_sha256":score_sha}),encoding="utf-8")
            return child
        fake_fcntl = SimpleNamespace(LOCK_EX=1,LOCK_NB=2,LOCK_UN=3,flock=mock.Mock())
        with (mock.patch.dict(sys.modules,{"fcntl":fake_fcntl}),
              mock.patch.object(queue,"RUNTIME",runtime),
              mock.patch.object(queue,"REPORT",report_path),
              mock.patch.object(queue,"_imports_preserving_cuda",return_value=(audit,evaluator,scoreq,owner)),
              mock.patch.object(queue,"_score_complete",return_value={"score_sha256":score_sha,
                  "score_path":str(score_path)}),
              mock.patch.object(queue.signal,"signal"),
              mock.patch.object(queue.subprocess,"Popen",side_effect=popen)):
            try:
                result=queue.supervise(timeout_seconds=60)
            finally:
                if not lease.closed: lease.close()
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0],queue.worker_command(2,report_path,score_sha))
        self.assertEqual(result["status"],"complete_non_gating_diagnostic")
        state=json.loads((runtime/"state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"],"complete_non_gating_diagnostic")
        _cleanup(root,temp_root)

    def test_prelaunch_failure_is_terminal_and_existing_state_is_preserved(self):
        temp_root = _central_temp()
        root = temp_root / f"sw175_train8_prelaunch_{os.getpid()}"
        root.mkdir()
        runtime = root / "runtime"
        report_path = root / "audit.json"
        fake_fcntl = SimpleNamespace(LOCK_EX=1,LOCK_NB=2,LOCK_UN=3,flock=mock.Mock())
        audit = SimpleNamespace(prepare_inputs_cpu=mock.Mock(side_effect=ValueError("admission failed")))
        evaluator,scoreq,owner=object(),object(),object()
        with (mock.patch.dict(sys.modules,{"fcntl":fake_fcntl}),
              mock.patch.object(queue,"RUNTIME",runtime),
              mock.patch.object(queue,"REPORT",report_path),
              mock.patch.object(queue,"_imports_preserving_cuda",return_value=(audit,evaluator,scoreq,owner)),
              mock.patch.object(queue,"_score_complete",return_value={"score_sha256":"a"*64,
                  "score_path":"unused"}),mock.patch.object(queue.signal,"signal")):
            with self.assertRaisesRegex(ValueError,"admission failed"):
                queue.supervise(timeout_seconds=60)
        state=json.loads((runtime/"state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"],"failed")
        original=json.dumps({"preserved":True})
        (runtime/"state.json").write_text(original,encoding="utf-8")
        with (mock.patch.dict(sys.modules,{"fcntl":fake_fcntl}),
              mock.patch.object(queue,"RUNTIME",runtime),
              mock.patch.object(queue,"REPORT",report_path),
              mock.patch.object(queue.signal,"signal")):
            with self.assertRaises(FileExistsError): queue.supervise(timeout_seconds=60)
        self.assertEqual((runtime/"state.json").read_text(encoding="utf-8"),original)
        _cleanup(root,temp_root)

    def test_nonterminal_score_requires_live_exact_supervisor(self):
        temp_root = _central_temp()
        root = temp_root / f"sw175_train8_pending_{os.getpid()}"
        root.mkdir()
        try:
            (root / "state.json").write_text(json.dumps({
                "experiment": "SW0175_score_queue_v3", "status": "scoring_gt_last",
                "supervisor_pid": 1234}), encoding="utf-8")
            score_module = SimpleNamespace(RUNTIME=root)
            evaluator = SimpleNamespace(SCORE_PATH=root / "score.json")
            owner = SimpleNamespace(_proc_record=lambda pid: {"state": "S", "argv":
                ["python", "-m", queue.SCORE_MODULE]})
            self.assertIs(queue._score_complete(owner, score_module, evaluator), False)
            owner._proc_record = lambda pid: {"state": "S", "argv":
                ["python", "-m", "collaborative_test.SW_0175_critical_component_binding.score_queue"]}
            with self.assertRaises(RuntimeError):
                queue._score_complete(owner, score_module, evaluator)
        finally:
            _cleanup(root, temp_root)


if __name__ == "__main__":
    unittest.main()
