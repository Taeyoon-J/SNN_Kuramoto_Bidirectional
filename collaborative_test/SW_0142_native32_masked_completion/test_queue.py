from __future__ import annotations
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import contextlib, io, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'collaborative_test'),str(ROOT/'snn_kuramoto_bidirectional'),str(HERE)]
from collaborative_test.SW_0142_native32_masked_completion import stage_queue as q

class QueueContractTests(unittest.TestCase):
    def test_registered_dag_has_only_cache_warm_two_paired_arms_predict_and_score(self):
        plan=q.task_plan();ids=[x['task_id'] for x in plan]
        self.assertEqual(len(ids),9);self.assertEqual(len(set(ids)),9)
        self.assertEqual(q.task_groups(),[("cache_seed1",),("warm_seed1",),("preflight_encoder_graph","preflight_core_trainable"),("train_encoder_graph","train_core_trainable"),("predict_encoder_graph","predict_core_trainable"),("score_seed1",)])
        byid={x['task_id']:x for x in plan}
        self.assertEqual(byid['warm_seed1']['deps'],['cache_seed1'])
        for arm in q.run.ARMS:
            self.assertEqual(byid[f'train_{arm}']['deps'],[f'preflight_{arm}'])
            self.assertEqual(byid[f'predict_{arm}']['deps'],[f'train_{arm}'])
        self.assertEqual(set(byid['score_seed1']['deps']),{f'predict_{arm}' for arm in q.run.ARMS})

    def test_score_is_cuda_hidden_and_gpu_stages_map_physical_device(self):
        env=q.environment({'stage':'score'})
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
        env=q.environment({'stage':'train'},3,Path('/tmp/unique-triton'))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'3');self.assertTrue(env['TRITON_CACHE_DIR'].endswith('unique-triton'))
        self.assertIn('--device',q.command({'stage':'train','arm':'core_trainable'}))
        self.assertEqual(q.command({'stage':'score'})[-1],str(q.task_output({'stage':'score'})))

    def test_dryrun_is_readonly_plan(self):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):rc=q.main(['--dry-run'])
        self.assertEqual(rc,0);self.assertIn(q.run.NAME,output.getvalue());self.assertFalse(q.STATE.exists())

    def test_missing_outputs_do_not_validate(self):
        for task in q.task_plan():self.assertFalse(q.validate_result(task))

    @unittest.skipIf(q.fcntl is None,'Linux flock is used only by the server queue')
    def test_gpu_query_error_marks_own_task_failed_and_releases_lease(self):
        class BrokenOwner:
            @staticmethod
            def _nvidia_gpu_info(_gpu):raise RuntimeError('synthetic nvidia-smi failure')
        obj=q.Queue.__new__(q.Queue);obj.guard=__import__('threading').RLock();obj.state={'tasks':{'cache_seed1':{'status':'queued'}}};obj._save=lambda:None
        with tempfile.TemporaryDirectory() as td, patch.object(q,'GPU_LEASE_DIR',Path(td)), patch.object(q,'owner',BrokenOwner):
            ok=obj._execute(q.task_plan()[0])
            self.assertFalse(ok);self.assertEqual(obj.state['tasks']['cache_seed1']['status'],'failed')
            self.assertFalse(list(Path(td).glob('*.held')))

    def test_every_sibling_dependency_is_checked_before_next_stage(self):
        obj=q.Queue.__new__(q.Queue);obj.tasks=q.task_plan();obj.state={'status':'running','tasks':{t['task_id']:{'status':'queued'} for t in obj.tasks}};obj.guard=__import__('threading').RLock();obj._save=lambda:None
        executed=[];obj._execute=lambda task:executed.append(task['task_id']) or True
        checked=[]
        def validator(task):
            checked.append(task['task_id']);return task['task_id']!='preflight_core_trainable'
        with patch.object(q,'validate_result',validator):rc=obj.run()
        self.assertEqual(rc,1);self.assertIn('preflight_core_trainable',checked)
        self.assertFalse(any(x.startswith('train_') for x in executed))

if __name__=='__main__':unittest.main()
