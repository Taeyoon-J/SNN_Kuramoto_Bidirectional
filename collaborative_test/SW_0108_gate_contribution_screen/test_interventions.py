import unittest
from types import SimpleNamespace

import torch

from run import ARMS,apply_model_intervention,constant_half_gate,patch_function_modules
from coordinator import screen_tasks,upstream_ready

class InterventionTests(unittest.TestCase):
 def test_constant_gate_replaces_carrier_and_membrane_gate(self):
  theta=torch.tensor([[[0.,1.,2.,3.]]])
  carrier,gate=constant_half_gate(theta)
  self.assertTrue(torch.equal(carrier,.5*torch.sin(theta)))
  self.assertTrue(torch.equal(gate,torch.full((1,1),.5)))
 def test_k_and_retention_interventions_restore_exact_values(self):
  for arm in ARMS:
   model=SimpleNamespace(kuramoto=SimpleNamespace(K=256.),
    dendric_layer=SimpleNamespace(tau_n=torch.nn.Parameter(torch.tensor([[-1.,0.,1.]]))),
    membrane_layer=SimpleNamespace(tau_m=torch.nn.Parameter(torch.tensor([-2.,0.,2.]))))
   n=model.dendric_layer.tau_n.detach().clone();m=model.membrane_layer.tau_m.detach().clone()
   restore=apply_model_intervention(model,arm)
   if arm=='kuramoto_K0':self.assertEqual(model.kuramoto.K,0.)
   if arm=='dendrite_no_retention':self.assertTrue(torch.equal(torch.sigmoid(model.dendric_layer.tau_n),torch.zeros_like(n)))
   if arm=='membrane_no_retention':self.assertTrue(torch.equal(torch.sigmoid(model.membrane_layer.tau_m),torch.zeros_like(m)))
   restore();self.assertEqual(model.kuramoto.K,256.)
   self.assertTrue(torch.equal(model.dendric_layer.tau_n,n));self.assertTrue(torch.equal(model.membrane_layer.tau_m,m))
 def test_gate_and_event_callables_are_restored_after_intervention(self):
  from run import Trace
  core=SimpleNamespace(sinusoidal_gating=lambda h,t,d,gate_mode='raw':(h[t],torch.zeros(h[t].shape[:-1])))
  membrane=SimpleNamespace(act_fun_adp=lambda x:(x>0).float())
  trace=Trace(steps=4,settle=2);hist=[torch.tensor([[[0.,1.,2.,3.]]])]
  old_gate=core.sinusoidal_gating;old_event=membrane.act_fun_adp
  restore=patch_function_modules(core,membrane,'constant_gate_half',trace)
  carrier,gate=core.sinusoidal_gating(hist,0,2,gate_mode='raw')
  self.assertTrue(torch.equal(carrier,.5*torch.sin(hist[0])))
  self.assertTrue(torch.equal(gate,torch.full((1,1),.5)));restore()
  self.assertIs(core.sinusoidal_gating,old_gate);self.assertIs(membrane.act_fun_adp,old_event)
  restore=patch_function_modules(core,membrane,'events_forced_on',trace)
  actual=membrane.act_fun_adp(torch.tensor([-1.,1.]));self.assertTrue(torch.equal(actual,torch.ones(2)));restore()
  self.assertIs(core.sinusoidal_gating,old_gate);self.assertIs(membrane.act_fun_adp,old_event)
 def test_screen_has_fixed_baseline_then_each_single_intervention(self):
  tasks=screen_tasks()
  self.assertEqual([x['arm'] for x in tasks],['all',*ARMS])
  self.assertEqual(len(set(x['arm'] for x in tasks)),7)
 def test_upstream_allows_sw0107_terminal_even_if_shared_queue_continues(self):
  state={'status':'waiting_for_seed0_pilot_terminal','sw0107_status':'complete',
   'active':{'sw0106':{'task':{'stage':'train'}}},'cpu_preparation':[]}
  import coordinator
  old=coordinator.Path.is_file
  try:
   coordinator.Path.is_file=lambda self:True
   import unittest.mock
   with unittest.mock.patch('coordinator.read_json',return_value=state):self.assertTrue(upstream_ready('mock-state'))
   state['active']['sw0107']={'task':{'stage':'sw107-train'}}
   with unittest.mock.patch('coordinator.read_json',return_value=state):self.assertFalse(upstream_ready('mock-state'))
  finally:coordinator.Path.is_file=old

if __name__=='__main__':unittest.main()
