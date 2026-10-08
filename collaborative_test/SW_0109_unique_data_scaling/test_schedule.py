import json
import unittest
import numpy as np

from run import BATCH,EXPOSURES,SIZES,SEEDS,pool_ids,row_indices,training_order,validate_gamma_manifest,validate_preflight_record
from coordinator import (task_plan,free_gpus,upstream_complete,ready_tasks,reserve_gpu,
 cleanup_owned_workers,evaluation_manifest_matches)
from summarize import paired_bootstrap,scores_from_report

class ScalingScheduleTests(unittest.TestCase):
 def test_exposure_count_map_uses_manifest_json_keys(self):
  from run import exposure_count_map
  order=np.concatenate((np.arange(2500),np.arange(2500)))
  counts=exposure_count_map(order)
  decoded=json.loads(json.dumps(counts))
  self.assertEqual(set(counts),{str(BATCH*256),str(EXPOSURES)})
  self.assertEqual(decoded[str(BATCH*256)],2500)
  self.assertEqual(decoded[str(EXPOSURES)],2500)
 def test_pools_are_nested_and_exact_unique_counts(self):
  pools={n:set(pool_ids(n).tolist()) for n in SIZES}
  self.assertEqual({n:len(x) for n,x in pools.items()},{2500:2500,10000:10000,70000:70000})
  self.assertLessEqual(pools[2500],pools[10000]);self.assertLessEqual(pools[10000],pools[70000])
  self.assertEqual(min(pools[70000]),0);self.assertEqual(max(pools[70000]),70639)
  self.assertTrue(all(x<1000 or 1640<=x<70640 for x in pools[70000]))
 def test_each_condition_has_same_exact_70k_exposures_and_no_partial_batch(self):
  for seed in SEEDS:
   for size in SIZES:
    pool,order=training_order(seed,size)
    self.assertEqual(len(order),EXPOSURES);self.assertEqual(EXPOSURES//BATCH,4375)
    self.assertEqual(len(np.unique(order)),size);self.assertEqual(set(order.tolist()),set(pool.tolist()))
    self.assertEqual(len(order)%BATCH,0)
    self.assertTrue(np.array_equal(np.sort(order[:size]),np.sort(pool)))
    if size<EXPOSURES:
     self.assertTrue(np.array_equal(np.sort(order[size:2*size]),np.sort(pool)))
     self.assertFalse(np.array_equal(order[:size],order[size:2*size]))
    rows=row_indices(order)
    self.assertEqual(rows.shape,(70000,));self.assertTrue(np.all((rows>=0)&(rows<70000)))
    self.assertTrue(np.array_equal(order,training_order(seed,size)[1]))
 def test_queue_registers_three_seed_preflights_baselines_and_nine_full_conditions(self):
  tasks=task_plan();train=[t for t in tasks if t['stage']=='train']
  evals=[t for t in tasks if t['stage']=='eval']
  self.assertEqual(len([t for t in tasks if t['stage']=='preflight']),3)
  self.assertEqual(len([t for t in evals if t['kind']=='baseline']),3)
  self.assertEqual(len(train),9);self.assertEqual(len([t for t in evals if t['kind']=='scaling']),9)
  self.assertEqual({(t['seed'],t['size']) for t in train},{(s,n) for s in SEEDS for n in SIZES})
 def test_scheduler_unblocks_only_seed_local_dependencies(self):
  tasks=task_plan();pending=list(tasks)
  ready=ready_tasks(pending,set())
  self.assertEqual({t['task_id'] for t in ready},{f'preflight_seed{s}' for s in SEEDS})
  completed={'preflight_seed0'};ready=ready_tasks(pending,completed)
  self.assertGreaterEqual(len(ready),4)
  self.assertIn('baseline_seed0',{t['task_id'] for t in ready})
  self.assertIn('train_seed0_N2500',{t['task_id'] for t in ready})
  self.assertNotIn('baseline_seed1',{t['task_id'] for t in ready})
  self.assertNotIn('eval_seed0_N2500',{t['task_id'] for t in ready})
  completed.add('train_seed0_N2500')
  self.assertIn('eval_seed0_N2500',{t['task_id'] for t in ready_tasks(pending,completed)})
 def test_gpu_selection_excludes_owned_or_over_budget_devices(self):
  owners={0:[],1:[123],2:[],3:[]};used={0:400,2:513,3:0}
  self.assertEqual(free_gpus(lambda g:owners[g],lambda g:used[g]),[0,3])
 def test_gpu_reservation_prevents_duplicate_assignment(self):
  reservations={};owner=lambda gpu:[];memory=lambda gpu:0
  first=reserve_gpu(reservations,owner,memory);second=reserve_gpu(reservations,owner,memory)
  self.assertIsNotNone(first);self.assertIsNotNone(second);self.assertNotEqual(first,second)
  self.assertEqual(set(reservations),{first,second})
 def test_failure_cleanup_stops_only_coordinator_owned_children(self):
  class FakeProcess:
   def __init__(self,pid):self.pid=pid
   def poll(self):return None
  own0=FakeProcess(111);own1=FakeProcess(222);stopped=[]
  workers={'a':{'proc':own0,'pid':111,'gpu':0},'b':{'proc':own1,'pid':222,'gpu':1}}
  cancelled=cleanup_owned_workers(workers,stop_fn=lambda proc:stopped.append(proc.pid))
  self.assertEqual(stopped,[111,222]);self.assertNotIn(999,stopped)
  self.assertEqual({x['pid'] for x in cancelled},{111,222})
 def test_evaluation_validator_binds_seed_pool_and_checkpoint(self):
  task={'seed':2,'size':10000};checkpoint=__import__('pathlib').Path('/expected/core.pt')
  manifest={'status':'complete','seed':2,'pool_size':10000,'ids':[1320,1639],'images':320,
   'ground_truth_used_for_prediction':False,'checkpoint':str(checkpoint),'checkpoint_sha256':'abc'}
  self.assertTrue(evaluation_manifest_matches(manifest,task,checkpoint,'abc'))
  bad=dict(manifest);bad['seed']=1
  self.assertFalse(evaluation_manifest_matches(bad,task,checkpoint,'abc'))
  bad=dict(manifest);bad['checkpoint_sha256']='other'
  self.assertFalse(evaluation_manifest_matches(bad,task,checkpoint,'abc'))
 def test_upstream_dependency_requires_terminal_complete_and_no_active_workers(self):
  from pathlib import Path
  from unittest.mock import patch
  p=Path('fixture-state.json')
  for state,expected in (({'status':'running','active':{}},False),
                         ({'status':'complete','active':{'pid':9}},False),
                         ({'status':'complete','active':{}},True)):
   with patch('coordinator.Path.is_file',return_value=True),patch('coordinator.read',return_value=state):
    self.assertEqual(upstream_complete(p,'sw0108'),expected)
  with patch('coordinator.Path.is_file',return_value=False):
   self.assertFalse(upstream_complete(p,'sw0108'))
 def test_gamma_manifest_accepts_registered_schema_and_rejects_wrong_segments(self):
  from unittest.mock import patch
  train={'gamma_sha256':'g','preprocessing_sha256':'p','encoder_sha256':'e',
   'training_ids':{'segments':[[0,999],[1640,70639]],'count':70000}}
  with patch('run.sha',side_effect=lambda path: {'feature_preprocessing.pt':'p','input_layer_encoder.pt':'e'}.get(path.name,'g')):
   result=validate_gamma_manifest(train,'g')
   self.assertEqual(result['segments'],[[0,999],[1640,70639]])
   invalid=json.loads(json.dumps(train));invalid['training_ids']['segments'][1][1]=70638
   with self.assertRaises(AssertionError):validate_gamma_manifest(invalid,'g')
   val={'gamma_sha256':'g','preprocessing_sha256':'p','encoder_sha256':'e','image_ids':[1320,1639]}
   self.assertEqual(validate_gamma_manifest(val,'g',True)['count'],320)
 def test_preflight_requires_four_finite_updates_for_all_three_pools(self):
  from unittest.mock import patch
  pools={str(n):{'status':'passed','pool_size':n,'unique_images_in_pool':n,'graph_bitwise_unchanged':True,
   'batches':[{'finite_gradient':True,'throwaway_adam_update':True,'gradient_norm_preclip':.1} for _ in range(4)]} for n in SIZES}
  record={'status':'passed','seed':1,'source_core_sha256':'source','pool_checks':pools,
   'encoder_sha256':'hash','feature_preprocessing_sha256':'hash','runner_sha256':'hash'}
  with patch('run.sha',return_value='hash'):
   self.assertTrue(validate_preflight_record(record,1,'source'))
   record['pool_checks']['10000']['batches'].pop()
   self.assertFalse(validate_preflight_record(record,1,'source'))
 def test_summary_recomputes_finite_metrics_and_rejects_prediction_gt(self):
  values=np.linspace(0.,1.,320).tolist();mean=float(np.mean(values))
  report={'ids':[1320,1639],'images':320,'ground_truth_used_for_prediction':False,
   'sweep':[{'scored_targets':{'our_hdf5':{'per_image':{m:values for m in ('fg_ari','foreground_iou','matched_object_iou')},
    'valid_count':{m:320 for m in ('fg_ari','foreground_iou','matched_object_iou')},
    'metrics':{m:mean for m in ('fg_ari','foreground_iou','matched_object_iou')}}}}]}
  self.assertEqual(scores_from_report(report)['fg_ari'].shape,(320,))
  bad=json.loads(json.dumps(report));bad['sweep'][0]['scored_targets']['our_hdf5']['metrics']['fg_ari']+=1e-5
  with self.assertRaises(ValueError):scores_from_report(bad)
  bad=json.loads(json.dumps(report));bad['ground_truth_used_for_prediction']=True
  with self.assertRaises(ValueError):scores_from_report(bad)
 def test_paired_image_bootstrap_is_deterministic_and_uses_seed_paired_deltas(self):
  delta=[np.linspace(-.1,.2,320),np.linspace(-.05,.1,320),np.linspace(-.03,.15,320)]
  first=paired_bootstrap(delta,seed=109,replicates=2000);second=paired_bootstrap(delta,seed=109,replicates=2000)
  self.assertEqual(first,second);self.assertAlmostEqual(first['mean_delta'],float(np.mean([x.mean() for x in delta])))
  self.assertEqual(len(first['ci95_percentile']),2)

if __name__=='__main__':unittest.main()
