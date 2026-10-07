import unittest
import tempfile
from pathlib import Path
from coordinator import eligible_gpus,transition,valid_preflight,sha,sw0105_dependency

class CoordinatorTests(unittest.TestCase):
 def test_busy_or_residual_memory_gpus_are_excluded(self):
  owners={0:[41],1:[],2:[],3:[]};used={0:100,1:513,2:24,3:2000}
  got=eligible_gpus((0,1,2,3),lambda g:owners[g],lambda g:used[g])
  self.assertEqual(got,[2])
 def test_preflight_failure_is_terminal_and_does_not_enqueue(self):
  task={'stage':'preflight','seed':0}
  status,next_tasks=transition(task,1,False)
  self.assertEqual(status,'failed');self.assertEqual(next_tasks,[])
 def test_preflight_success_enqueues_only_paired_seed0(self):
  status,tasks=transition({'stage':'preflight','seed':0},0,True)
  self.assertEqual(status,'preflight_complete')
  self.assertEqual(tasks,[{'stage':'train','arm':'candidate','seed':0},{'stage':'train','arm':'control','seed':0}])
  self.assertFalse(any(t['seed']!=0 for t in tasks))
 def test_training_success_schedules_only_its_full_eval(self):
  task={'stage':'train','arm':'candidate','seed':0}
  status,tasks=transition(task,0,True)
  self.assertEqual(status,'training_complete')
  self.assertEqual(tasks,[{'stage':'eval','arm':'candidate','seed':0}])
 def test_failed_training_or_evaluation_never_requeues(self):
  for task in ({'stage':'train','arm':'control','seed':0},{'stage':'eval','arm':'control','seed':0}):
   status,tasks=transition(task,2,False)
   self.assertEqual(status,'failed');self.assertEqual(tasks,[])
 def test_preflight_validator_requires_shared_artifact_and_all_family_guards(self):
  with tempfile.TemporaryDirectory() as td:
   artifact=Path(td)/'warm.pt';artifact.write_bytes(b'cpu test artifact')
   families={'encoder':1.,'graph_generator':2.,'kuramoto':3.}
   record={'status':'passed','seed':0,'training_ids_first4096':list(range(4096)),'shuffle_seed':117,
    'warmup_source_core_and_encoder_unchanged':True,'initial_candidate_control_hard_forward_exact':True,
    'production_labels_exact':True,'control_reconstruction_has_no_core_encoder_gradient':True,
    'throwaway_candidate_update_finite_and_changed_graph_encoder_parameters':True,
    'throwaway_decoder_optimizer_state_loaded_from_shared_32_batch_warmup':True,
    'candidate_reconstruction_gradient_norms_by_family':families,
    'lambda_calibration':[{'ratio':.25,'reconstruction_grad_norms_by_family':families} for _ in range(4)],
    'row_scramble_by_batch':[{'images':16,'mean_shuffled_minus_original_mse':.1} for _ in range(4)],
    'row_scramble_images':64,'row_scramble_delta_mean':.1,'warmup_artifact':str(artifact),
    'warmup_artifact_sha256':sha(artifact),'lambda':.25}
   path=Path(td)/'preflight.json';path.write_text(__import__('json').dumps(record))
   self.assertTrue(valid_preflight(path))
   record['lambda_calibration'][2]['reconstruction_grad_norms_by_family']['encoder']=0.
   path.write_text(__import__('json').dumps(record))
   self.assertFalse(valid_preflight(path))
 def test_waits_for_sw0105_terminal_queue_and_valid_results_before_gpu_use(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);state=root/'queue.json';out=root/'out';out.mkdir()
   state.write_text(__import__('json').dumps({'status':'running','active':{},'seed_status':{}}))
   self.assertEqual(sw0105_dependency(state,out),'waiting')
   state.write_text(__import__('json').dumps({'status':'complete','active':{'1':{'pid':42}},'seed_status':{str(i):'evaluation_complete' for i in range(3)}}))
   self.assertEqual(sw0105_dependency(state,out),'waiting')
   state.write_text(__import__('json').dumps({'status':'complete','active':{},'seed_status':{str(i):'evaluation_complete' for i in range(3)}}))
   scores={m:[.5]*320 for m in ('fg_ari','foreground_iou','matched_object_iou')}
   metrics={m:.5 for m in scores};counts={m:320 for m in scores}
   evaluation={'images':320,'ids':[1320,1639],'sweep':[{'scored_targets':{'our_hdf5':{'per_image':scores,'metrics':metrics,'valid_count':counts}}}]}
   for seed in range(3):(out/f'seed{seed}').mkdir();(out/f'seed{seed}'/'evaluation.json').write_text(__import__('json').dumps(evaluation))
   self.assertEqual(sw0105_dependency(state,out),'ready')
   (out/'seed2'/'evaluation.json').write_text('{}')
   self.assertEqual(sw0105_dependency(state,out),'failed')

if __name__=='__main__':unittest.main()
