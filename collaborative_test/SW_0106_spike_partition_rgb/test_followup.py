import unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch

import followup_coordinator as queue
from followup_policy import full_validation_reserve_gate,paired_image_bootstrap,pilot_promotion_gate

def result(value):
 per={m:np.full(320,value,dtype=np.float64) for m in ('fg_ari','foreground_iou','matched_object_iou')}
 return {'metrics':{k:float(v.mean()) for k,v in per.items()},'per_image':per}

class FollowupTests(unittest.TestCase):
 def test_completed_sw107_is_not_queued_again_after_sw106(self):
  state={'sw0107_status':'complete','pending':[]}
  queue.start_sw107(state,'seed0_expansion_gate_failed')
  self.assertEqual(state['status'],'complete');self.assertEqual(state['pending'],[])

 def test_priority_sw107_finishes_before_waiting_on_sw106(self):
  state={'phase':'sw107_eval_all3','sw107_first':True,'active':{},'pending':[],
   'completed':{f'sw107-eval::{s}':{} for s in range(3)}}
  with patch.object(queue.subprocess,'run'):
   queue.next_phase(state)
  self.assertEqual(state['phase'],'wait_seed0');self.assertEqual(state['sw0107_status'],'complete')

 def test_paired_bootstrap_is_deterministic_and_uses_same_image_delta(self):
  a=np.tile(np.linspace(.2,.8,320),(3,1));b=a-.02
  x=paired_image_bootstrap(a,b);y=paired_image_bootstrap(a,b)
  self.assertEqual(x,y);self.assertTrue(x['lower_bound_positive']);self.assertAlmostEqual(x['mean_delta'],.02)

 def test_pilot_gate_requires_all_registered_margin_and_seed_checks(self):
  c=[result(.80) for _ in range(3)];r=[result(.78) for _ in range(3)]
  self.assertEqual(pilot_promotion_gate(c,r,r)['status'],'passed')
  bad=[result(.78) for _ in range(3)]
  self.assertEqual(pilot_promotion_gate(bad,r,r)['status'],'failed')

 def test_final_reserve_gate_requires_all_three_metrics_strictly_above_slot(self):
  c=[result(.60) for _ in range(3)];control=[result(.40) for _ in range(3)];slot=[result(.50) for _ in range(3)]
  self.assertEqual(full_validation_reserve_gate(c,control,slot)['status'],'passed')
  for x in c:x['metrics']['matched_object_iou']=.50
  self.assertEqual(full_validation_reserve_gate(c,control,slot)['status'],'failed')

 def test_preflight_validator_accepts_later_registered_seed_and_seed_specific_shuffle(self):
  artifact=Path('dummy-warm.pt');families={'encoder':1.,'graph_generator':2.,'kuramoto':3.}
  row={'status':'passed','seed':2,'training_ids_first4096':list(range(4096)),'shuffle_seed':119,
   'warmup_source_core_and_encoder_unchanged':True,'initial_candidate_control_hard_forward_exact':True,'production_labels_exact':True,
   'control_reconstruction_has_no_core_encoder_gradient':True,'throwaway_candidate_update_finite_and_changed_graph_encoder_parameters':True,
   'throwaway_decoder_optimizer_state_loaded_from_shared_32_batch_warmup':True,'candidate_reconstruction_gradient_norms_by_family':families,
   'lambda_calibration':[{'ratio':.1,'reconstruction_grad_norms_by_family':families} for _ in range(4)],
   'row_scramble_by_batch':[{'images':16,'mean_shuffled_minus_original_mse':.1} for _ in range(4)],'row_scramble_images':64,
   'row_scramble_delta_mean':.1,'warmup_artifact':str(artifact),'warmup_artifact_sha256':'sha','lambda':.2}
  with (patch.object(queue,'read_json',return_value=row),patch.object(queue,'task_output',return_value=Path('dummy.json')),
        patch.object(Path,'is_file',return_value=True),patch.object(queue,'file_sha',return_value='sha')):
   self.assertTrue(queue.validate_task({'stage':'preflight','seed':2}))
  row['shuffle_seed']=117
  with patch.object(queue,'read_json',return_value=row),patch.object(queue,'task_output',return_value=Path('dummy.json')):
   self.assertFalse(queue.validate_task({'stage':'preflight','seed':2}))
  row['shuffle_seed']=119;row['lambda_calibration'][0]['reconstruction_grad_norms_by_family'].pop('kuramoto')
  with patch.object(queue,'read_json',return_value=row),patch.object(queue,'task_output',return_value=Path('dummy.json')):
   self.assertFalse(queue.validate_task({'stage':'preflight','seed':2}))

 def test_eval_overwrite_guard_checks_result_not_training_directory(self):
  out=Path('dummy-existing-training-dir')
  with patch.object(queue,'task_output',return_value=out),patch.object(Path,'exists',lambda p: str(p).endswith('evaluation.json')):
   self.assertTrue(queue.output_marker_exists({'stage':'eval'}))
  with patch.object(queue,'task_output',return_value=out),patch.object(Path,'exists',return_value=False):
   self.assertFalse(queue.output_marker_exists({'stage':'eval'}))

 def test_assignments_never_duplicate_device(self):
  active={'a':{'gpu':0},'b':{'gpu':2}}
  self.assertEqual(queue.unique_assignments([0,1,2,3],active),[1,3])

 def test_sw107_branch_runs_even_when_sw0106_gate_stops(self):
  state={'phase':'pilot_eval12','status':'running','active':{},'pending':[]}
  queue.start_sw107(state,'three_seed_pilot_gate_failed')
  self.assertEqual(state['phase'],'sw107_preflight_all3')
  self.assertEqual(state['status'],'running')
  self.assertEqual(state['pending'],[{'stage':'sw107-preflight','seed':s} for s in range(3)])
  self.assertEqual(state['sw0106_outcome'],'three_seed_pilot_gate_failed')

if __name__=='__main__':unittest.main()
