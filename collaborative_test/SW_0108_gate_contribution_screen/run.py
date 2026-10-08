"""Local wrapper for frozen-core SW0108 one-at-a-time gate sensitivity screening."""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[2]
EVALUATOR=ROOT/'collaborative_test/SW_0040_peer_transfer/evaluate.py'
ARMS=('baseline','kuramoto_K0','constant_gate_half','dendrite_no_retention','membrane_no_retention','events_forced_on')
SOURCE_DEFAULT=ROOT/'trained_models/SW0097_graph_adaptation/seed0_positive_frozen/core.pt'

def constant_half_gate(theta):
 return .5*torch.sin(theta),torch.full(theta.shape[:-1],.5,device=theta.device,dtype=theta.dtype)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def load_evaluator():
 sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'collaborative_test'))
 spec=importlib.util.spec_from_file_location('sw0108_shared_evaluator',EVALUATOR)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module

class ScalarTrace:
 def __init__(self):self.count=0;self.total=None;self.square=None;self.minimum=None;self.maximum=None
 def add(self,value):
  x=torch.as_tensor(value).detach().float()
  if x.numel()==0:return
  total=x.sum();square=x.square().sum();minimum=x.min();maximum=x.max()
  if self.total is None:self.total=total;self.square=square;self.minimum=minimum;self.maximum=maximum
  else:
   self.total=self.total+total;self.square=self.square+square
   self.minimum=torch.minimum(self.minimum,minimum);self.maximum=torch.maximum(self.maximum,maximum)
  self.count+=x.numel()
 def result(self):
  if not self.count:return {'count':0,'mean':None,'std':None,'min':None,'max':None}
  mean=float(self.total.cpu())/self.count;moment=float(self.square.cpu())/self.count
  var=max(0.,moment-mean*mean)
  return {'count':self.count,'mean':mean,'std':math.sqrt(var),
   'min':float(self.minimum.cpu()),'max':float(self.maximum.cpu())}

class Trace:
 NAMES=('carrier','gate','actfun_input','event_before_override','dendrite_output','membrane_output','spike_output','phase_output')
 def __init__(self,steps=1024,settle=512,arm='baseline'):
  self.steps=steps;self.settle=settle;self.step_counter=0;self.handles=[]
  self.arm=arm;self.forced_spike_gate_checks=0
  self.values={k:ScalarTrace() for k in self.NAMES};self.settled={k:ScalarTrace() for k in self.NAMES}
  self.temporal_current={};self.temporal_batches={'gate':[],'event':[],'forced_event':[]}
 def add(self,name,value,step=None):
  self.values[name].add(value)
  tick=self.step_counter%self.steps if step is None else step
  if tick>=self.settle:self.settled[name].add(value)
 def add_temporal(self,name,value,step):
  if step<self.settle:return
  x=torch.as_tensor(value).detach().float()
  current=self.temporal_current.get(name)
  if current is None:self.temporal_current[name]=[x.clone(),x.square(),1]
  else:current[0].add_(x);current[1].add_(x.square());current[2]+=1
 def finish_temporal_batch(self):
  for name,(total,square,count) in self.temporal_current.items():
   mean=total/count;std=(square/count-mean.square()).clamp_min(0).sqrt()
   self.temporal_batches[name].append({'unit_count':std.numel(),'mean':float(mean.mean()),
    'temporal_std_mean':float(std.mean()),'constant_unit_fraction':float((std<=1e-8).float().mean()),
    'always_on_fraction':float((mean>=1.-1e-8).float().mean()) if name=='event' else None})
  self.temporal_current={}
 def hooks(self,model):
  self.handles.append(model.dendric_layer.register_forward_hook(lambda m,i,o:self.add('dendrite_output',o)))
  def membrane_hook(module,inputs,output):
   tick=self.step_counter%self.steps
   self.add('membrane_output',output[0],tick);self.add('spike_output',output[1],tick)
   if self.arm=='events_forced_on':
    gate=inputs[1].expand_as(output[1])
    if not torch.equal(output[1],gate):raise AssertionError('forced-on event output did not equal the actual membrane gate')
    self.forced_spike_gate_checks+=1
   self.step_counter+=1
  self.handles.append(model.membrane_layer.register_forward_hook(membrane_hook))
  self.handles.append(model.register_forward_pre_hook(lambda m,i:setattr(self,'temporal_current',{})))
  def core_hook(module,inputs,output):
   self.finish_temporal_batch()
   if isinstance(output,(tuple,list)) and len(output)>=4:
    phase=output[3];self.values['phase_output'].add(phase)
    if phase.ndim>=2:self.settled['phase_output'].add(phase[:,self.settle:])
  self.handles.append(model.register_forward_hook(core_hook))
 def summary(self):return {'all_frames':{k:v.result() for k,v in self.values.items()},
  'settled_frames':{k:v.result() for k,v in self.settled.items()},'settle':self.settle,'steps':self.steps,
  'per_unit_settled_time':{k:{'batches':len(v),'temporal_std_mean':sum(x['temporal_std_mean'] for x in v)/len(v) if v else None,
   'constant_unit_fraction_mean':sum(x['constant_unit_fraction'] for x in v)/len(v) if v else None,
   'always_on_fraction_mean':sum(x['always_on_fraction'] for x in v if x['always_on_fraction'] is not None)/len(v) if any(x['always_on_fraction'] is not None for x in v) else None}
   for k,v in self.temporal_batches.items()},'forced_on_spike_equals_gate_checks':self.forced_spike_gate_checks}
 def remove(self):
  for handle in self.handles:handle.remove()
  self.handles.clear()

def apply_model_intervention(model,arm):
 """Apply only the registered local intervention and return an exact restorer."""
 if arm not in ARMS:raise ValueError(f'unknown intervention arm {arm}')
 saved={'K':model.kuramoto.K,'tau_n':model.dendric_layer.tau_n.detach().clone(),
        'tau_m':model.membrane_layer.tau_m.detach().clone()}
 with torch.no_grad():
  if arm=='kuramoto_K0':model.kuramoto.K=0.0
  elif arm=='dendrite_no_retention':model.dendric_layer.tau_n.fill_(float('-inf'))
  elif arm=='membrane_no_retention':model.membrane_layer.tau_m.fill_(float('-inf'))
 def restore():
  with torch.no_grad():
   model.kuramoto.K=saved['K']
   model.dendric_layer.tau_n.copy_(saved['tau_n'])
   model.membrane_layer.tau_m.copy_(saved['tau_m'])
 return restore

def patch_function_modules(core_module,membrane_module,arm,trace):
 original_gate=core_module.sinusoidal_gating
 original_event=membrane_module.act_fun_adp
 def gate_proxy(theta_hist,t,phase_delay_steps,gate_mode='sigmoid'):
  gamma,gate=original_gate(theta_hist,t,phase_delay_steps,gate_mode=gate_mode)
  if arm=='constant_gate_half':
   gamma,gate=constant_half_gate(theta_hist[t])
  trace.add('carrier',gamma);trace.add('gate',gate)
  trace.add_temporal('gate',gate,trace.step_counter%trace.steps)
  return gamma,gate
 def event_proxy(inputs):
  events=original_event(inputs)
  trace.add('actfun_input',inputs);trace.add('event_before_override',events)
  trace.add_temporal('event',events,trace.step_counter%trace.steps)
  if arm=='events_forced_on':trace.add_temporal('forced_event',torch.ones_like(events),trace.step_counter%trace.steps)
  return torch.ones_like(events) if arm=='events_forced_on' else events
 core_module.sinusoidal_gating=gate_proxy
 membrane_module.act_fun_adp=event_proxy
 def restore():
  core_module.sinusoidal_gating=original_gate
  membrane_module.act_fun_adp=original_event
 return restore

def patch_actual_model_globals(model,arm,trace):
 core_module=sys.modules[type(model).__module__]
 membrane_module=sys.modules[type(model.membrane_layer).__module__]
 restore_functions=patch_function_modules(core_module,membrane_module,arm,trace)
 restore_model=apply_model_intervention(model,arm);trace.hooks(model)
 def restore():
  trace.remove();restore_functions()
  restore_model()
 return restore

def run_evaluation(arm,checkpoint,gamma_path,dataset_path,output_dir,device='cuda:0'):
 out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
 report_path=out/'evaluation.json';diag_path=out/'diagnostics_gt_free.json'
 if report_path.exists() or diag_path.exists():raise FileExistsError(f'preserve existing SW0108 result: {out}')
 evaluator=load_evaluator();trace=Trace(steps=1024,settle=512,arm=arm);restorers=[]
 original=evaluator._core
 def factory(*args,**kwargs):
  model=original(*args,**kwargs);restorers.append(patch_actual_model_globals(model,arm,trace));return model
 evaluator._core=factory
 argv=['evaluate.py','--checkpoint',str(checkpoint),'--gamma-path',str(gamma_path),
  '--gamma-global-start','1320','--dataset-path',str(dataset_path),'--output-path',str(report_path),
  '--start','1320','--count','320','--steps','1024','--settle','512','--thresholds','.50',
  '--min-group-size','2','--background','largest_component','--dendritic-projection','shared',
  '--graph-spatial-decay','.35','--geodesic-steps','3','--geodesic-radius','1.5',
  '--geodesic-contrast','2','--geodesic-temperature','.5','--geodesic-cap','16',
  '--kuramoto-backend','factorized','--gate-mode','raw','--batch-size','8','--device',device]
 old_argv=sys.argv;sys.argv=argv
 try:
  evaluator.main()
 finally:
  sys.argv=old_argv;evaluator._core=original
  for restore in reversed(restorers):restore()
 if not report_path.is_file():raise RuntimeError('shared evaluator did not write the expected result')
 report=json.loads(report_path.read_text())
 trace_record={'status':'complete','arm':arm,'ground_truth_used':False,
  'source_checkpoint':str(Path(checkpoint).resolve()),'source_checkpoint_sha256':sha(checkpoint),
  'screen_runner_sha256':sha(Path(__file__)),'shared_evaluator_sha256':sha(EVALUATOR),
  'protocol_sha256':sha(Path(__file__).with_name('protocol.json')),
  'trace':trace.summary(),'scope':'scalar summaries from real forward hooks; no activation histories retained'}
 diag_path.write_text(json.dumps(trace_record,indent=2,allow_nan=False)+'\n')
 return report_path

def smoke_all(checkpoint,gamma_path,device='cuda:0',steps=32):
 """Two B8 real rollouts per arm, exact baseline/no-op replay, and restoration checks."""
 evaluator=load_evaluator();gamma_blob=torch.load(gamma_path,map_location='cpu',weights_only=True)
 if gamma_blob.ndim!=3 or gamma_blob.shape[1:]!=(8,256) or (len(gamma_blob)!=320 and len(gamma_blob)<1640):raise ValueError('registered native gamma cache shape/range mismatch')
 results=[];baseline_outputs=None
 for arm in ARMS:
  trace=Trace(steps=steps,settle=steps//2,arm=arm);model=evaluator._core(device,str(checkpoint),steps,'shared',3,1.5,2.,.5,16.,.35,'factorized','raw')
  state={k:v.detach().clone() for k,v in model.state_dict().items()};initial_k=model.kuramoto.K
  restore=patch_actual_model_globals(model,arm,trace)
  variant_outputs=[]
  try:
   gamma_row=0 if len(gamma_blob)==320 else 1320
   for offset in (0,8):
    with torch.no_grad():result=model(gamma_blob[gamma_row+offset:gamma_row+offset+8].to(device),return_core_out=True,return_theta=True)
    if not all(torch.isfinite(x).all() for x in result if torch.is_tensor(x)):raise FloatingPointError(f'{arm}: nonfinite model output')
    groups,spikes,core_out,theta=result
    expected_components=(8,4,256,steps)
    if tuple(spikes.shape)!=(8,256,steps) or tuple(model.last_component_spikes.shape)!=expected_components:raise AssertionError(f'{arm}: unexpected spike/component shape')
    if not torch.equal(spikes,model.last_component_spikes.mean(dim=1)):raise AssertionError(f'{arm}: component fold/order changed')
    variant_outputs.append((spikes.detach().clone(),theta.detach().clone()))
   if arm=='kuramoto_K0' and model.kuramoto.K!=0:raise AssertionError('K=0 intervention not installed')
   if arm=='dendrite_no_retention' and bool((torch.sigmoid(model.dendric_layer.tau_n)!=0).any()):raise AssertionError('dendritic retention did not reach zero')
   if arm=='membrane_no_retention' and bool((torch.sigmoid(model.membrane_layer.tau_m)!=0).any()):raise AssertionError('membrane retention did not reach zero')
   if arm=='constant_gate_half':
    gate_stats=trace.settled['gate'].result()
    if gate_stats['min']!=.5 or gate_stats['max']!=.5:raise AssertionError('constant-gate intervention did not reach the actual membrane gate')
   if arm=='events_forced_on' and trace.forced_spike_gate_checks!=2*steps:raise AssertionError('forced-on event hook did not participate in each actual membrane step')
  finally:
   restore()
   if model.kuramoto.K!=initial_k:raise AssertionError(f'{arm}: K value was not restored')
   if any(not torch.equal(v,model.state_dict()[k]) for k,v in state.items()):raise AssertionError(f'{arm}: state_dict changed during frozen smoke')
  replay=[]
  for offset in (0,8):
   with torch.no_grad():result=model(gamma_blob[gamma_row+offset:gamma_row+offset+8].to(device),return_core_out=True,return_theta=True)
   replay.append((result[1],result[3]))
  if any(not torch.equal(actual[0],base[0]) or not torch.equal(actual[1],base[1]) for actual,base in zip(replay,baseline_outputs or variant_outputs)):
   raise AssertionError(f'{arm}: restoring intervention does not bitwise reproduce the baseline forward')
  if baseline_outputs is None:baseline_outputs=variant_outputs
  results.append({'arm':arm,'status':'passed','images':16,'batch_size':8,'steps':steps,
   'spike_shape':list(variant_outputs[0][0].shape),'spike_rate':float(torch.cat([x[0] for x in variant_outputs]).mean()),
   'restored_baseline_bitwise_match':True,'trace':trace.summary()})
 return {'status':'passed','source_checkpoint_sha256':sha(checkpoint),'gamma_sha256':sha(gamma_path),'device':device,'steps':steps,'arms':results,'gt_used':False}

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight','evaluate'],required=True)
 p.add_argument('--arm',choices=ARMS,default=None);p.add_argument('--checkpoint',type=Path,default=SOURCE_DEFAULT)
 p.add_argument('--gamma-path',type=Path,required=True);p.add_argument('--dataset-path',type=Path,required=True)
 p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='cuda:0')
 a=p.parse_args()
 if a.stage=='preflight':
  if a.output_dir.exists():raise FileExistsError(a.output_dir)
  a.output_dir.parent.mkdir(parents=True,exist_ok=True)
  rec=smoke_all(a.checkpoint,a.gamma_path,a.device)
  a.output_dir.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
 else:
  if a.arm is None:raise ValueError('--arm is required for evaluation')
  run_evaluation(a.arm,a.checkpoint,a.gamma_path,a.dataset_path,a.output_dir,a.device)

if __name__=='__main__':main()
