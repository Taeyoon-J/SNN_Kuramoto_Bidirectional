"""Matched SW0097 continuation with auxiliary-only event detach credit."""
import argparse, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path
import h5py
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'snn_kuramoto_bidirectional'),str(ROOT/'collaborative_test')]
from SW_0094_aligned_joint_pilot.run import GAMMA,VAL_GAMMA,DATASET,hparams,aligned_affinity,rgb_batch
from SW_0102_cannot_link_draft.calibration_preflight import expected_training_ids,make_criterion
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from snn_kuramoto_bidirectional.training.train_s2net_core import _forward_with_plv
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch,evaluate_patch_masks
OUT=ROOT/'trained_models/SW0105_event_detach_aux'
VALOUT=ROOT/'trained_models/SW0105_event_detach_aux'
SOURCE=ROOT/'trained_models/SW0095_full70k_aligned_loss'
CONTROL=ROOT/'trained_models/SW0097_graph_adaptation'
ASSETS=Path('/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002')
METRICS=('fg_ari','foreground_iou','matched_object_iou')

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def load_core(seed,device,steps,checkpoint=None,training=True):
 hp=hparams('raw'); hp.num_time_steps=steps
 core=S2NetCore(hp,device=device).to(device)
 ckpt=checkpoint or (SOURCE/f'seed{seed}/core.pt')
 core.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True),strict=True)
 core.graph_generator.requires_grad_(False)
 core._detect_object_groups=lambda out,spikes:[[] for _ in range(spikes.size(0))]
 core.train(training)
 if core.graph_generator.uses_feedback or core.kuramoto.spike_pulse_gain is not None: raise AssertionError('registered feedback/pulse must be disabled')
 return core,Path(ckpt)
def install_capture(core):
 rec={'event':[],'gate':[]}; mod=sys.modules[core.membrane_layer.__class__.__module__]; orig=mod.act_fun_adp
 def event_hook(x):
  e=orig(x); rec['event'].append(e); return e
 mod.act_fun_adp=event_hook
 def gate_hook(_m,inputs): rec['gate'].append(inputs[1])
 handle=core.membrane_layer.register_forward_pre_hook(gate_hook)
 return rec,mod,orig,[handle]
def cleanup(mod,orig,handles):
 mod.act_fun_adp=orig
 for h in handles: h.remove()
def captured_forward(core,gamma,steps,criterion,settle,trace=True):
 if not trace:
  result=_forward_with_plv(core,gamma,criterion,settle,'phase','mean')
  return result,None
 rec,mod,orig,handles=install_capture(core)
 try: result=_forward_with_plv(core,gamma,criterion,settle,'phase','mean')
 finally: cleanup(mod,orig,handles)
 b=gamma.shape[0]; d=core.osc_dim;n=core.in_dim
 if len(rec['event'])!=steps or len(rec['gate'])!=steps: raise AssertionError('capture count mismatch')
 e=torch.stack(rec['event']).reshape(steps,b,d,n).permute(1,2,3,0)
 g=torch.stack(rec['gate']).reshape(steps,b,d,n).permute(1,2,3,0)
 z=e*g
 if not torch.equal(z,core.last_component_spikes): raise AssertionError('event*gate does not equal actual core history')
 if tuple(z.shape)!=(b,d,n,steps): raise AssertionError('history shape mismatch')
 return result,{'event':e,'gate':g,'actual':z}
def affinity(z,settle): return spike_synchrony_affinity(z.mean(dim=1),z,settle=settle)
def family_grad(core):
 out={}
 for n,p in core.named_parameters():
  if p.grad is None: continue
  if not torch.isfinite(p.grad).all(): raise FloatingPointError('nonfinite '+n)
  f=n.split('.')[0]; out[f]=out.get(f,0.)+float(p.grad.detach().double().square().sum())
 return {k:math.sqrt(v) for k,v in out.items()}
def verify_contract(seed):
 m=json.loads((CONTROL/f'seed{seed}_positive_frozen/manifest.json').read_text())
 ids=expected_training_ids(seed)
 if m.get('training_ids')!=ids or m.get('steps')!=256 or m.get('batch')!=16 or m.get('train_steps')!=64 or m.get('train_settle')!=32: raise AssertionError('matched control manifest mismatch')
 source=SOURCE/f'seed{seed}/core.pt'; control=CONTROL/f'seed{seed}_positive_frozen/core.pt'
 if m.get('source_sha256')!=sha(source): raise AssertionError('matched source SHA mismatch')
 return {'ids':ids,'source':source,'control':control,'source_sha256':sha(source),'control_sha256':sha(control),'control_manifest':m}
def run_preflight(seed,device):
 contract=verify_contract(seed); gamma=torch.load(GAMMA,map_location='cpu',weights_only=True,mmap=True)
 torch.manual_seed(117+seed); ix=torch.randperm(70000,generator=torch.Generator().manual_seed(117+seed))[:4096][:16]
 batch=gamma[ix].to(device); criterion=make_criterion(); core,_=load_core(seed,device,64)
 initial={k:v.detach().clone() for k,v in core.state_dict().items()}
 # Legacy tracing-off and tracing-on exact-forward check, same initial state and batch.
 with torch.no_grad():
  off,_=captured_forward(core,batch,64,criterion,32,trace=False); offsp=core.last_component_spikes.detach().clone()
  on,cap=captured_forward(core,batch,64,criterion,32,trace=True); onsp=core.last_component_spikes.detach().clone()
 if not torch.equal(offsp,onsp) or not torch.equal(off[1],on[1]) or not torch.equal(off[3],on[3]): raise AssertionError('trace on/off forward mismatch')
 qfull=affinity(cap['actual'],32); zaux=cap['event'].detach()*cap['gate']; qaux=affinity(zaux,32)
 if not torch.equal(cap['actual'],zaux) or not torch.equal(qfull,qaux): raise AssertionError('aux detach changed forward history or affinity')
 with torch.no_grad(): primary,_=criterion(plv=on[3],theta=on[4])
 fullspike,_=criterion(plv=qfull); auxspike,_=criterion(plv=qaux)
 if not torch.equal(fullspike,auxspike): raise AssertionError('aux and full loss values differ')
 # Fresh core: direct reference auxiliary expression, finite candidate gradient/update.
 del core,cap
 core,_=load_core(seed,device,64); before={k:v.detach().clone() for k,v in core.state_dict().items()}
 core.zero_grad(set_to_none=True)
 (_,spikes,out,plv,theta),cap=captured_forward(core,batch,64,criterion,32,trace=True)
 pval,_=criterion(plv=plv,theta=theta); aux=cap['event'].detach()*cap['gate']; q=affinity(aux,32); sval,_=criterion(plv=q); loss=pval+5.*sval
 if not torch.isfinite(loss): raise FloatingPointError('preflight nonfinite loss')
 # Independently rebuild the approved direct reference expression and compare parameter gradients.
 ref_aux=cap['event'].detach()*cap['gate']; ref_q=affinity(ref_aux,32); ref_spike,_=criterion(plv=ref_q); ref_loss=pval+5.*ref_spike
 if not torch.equal(aux,ref_aux) or not torch.equal(q,ref_q) or not torch.equal(loss,ref_loss): raise AssertionError('candidate differs from direct auxiliary reference values')
 named=[p for n,p in core.named_parameters() if p.requires_grad and not n.startswith('graph_generator.')]
 candidate_grads=torch.autograd.grad(loss,named,retain_graph=True,allow_unused=True)
 reference_grads=torch.autograd.grad(ref_loss,named,retain_graph=True,allow_unused=True)
 for a,b in zip(candidate_grads,reference_grads):
  if (a is None)!=(b is None) or (a is not None and not torch.equal(a,b)): raise AssertionError('candidate gradient differs from direct reference')
 loss.backward(); norms=family_grad(core)
 eligible=[p for n,p in core.named_parameters() if p.requires_grad and not n.startswith('graph_generator.')]
 if not norms or not any(v>0 for v in norms.values()): raise AssertionError('no eligible gradient')
 if any(not torch.isfinite(p.grad).all() for p in eligible if p.grad is not None): raise FloatingPointError('nonfinite parameter gradient')
 torch.nn.utils.clip_grad_norm_(eligible,1.)
 opt=torch.optim.Adam([{'params':eligible,'lr':3e-5}]); opt.step()
 after=core.state_dict(); changed=[k for k in before if not torch.equal(before[k],after[k])]
 for prefix in ('graph_generator.','dendric_layer.','membrane_layer.'):
  if any(k.startswith(prefix) for k in changed): raise AssertionError(f'expected frozen parameters changed: {prefix}')
 if not any(k.startswith('kuramoto.') for k in changed): raise AssertionError('eligible upstream core did not update')
 return {'status':'passed','seed':seed,'device':str(device),'training_ids_first16':contract['ids'][:16],'source_sha256':contract['source_sha256'],'matched_control_sha256':contract['control_sha256'],'trace_on_off_actual_forward_exact':True,'event_gate_reconstruction_exact':True,'full_vs_aux_affinity_exact':True,'full_vs_aux_spike_loss_exact':True,'direct_reference_values_and_parameter_gradients_exact':True,'trainable_gradient_norms_by_family':norms,'updated_keys':changed,'graph_dendrite_membrane_bitwise_fixed':True,'core_key_count':len(before)}
def train(seed,device):
 c=verify_contract(seed); out=OUT/f'seed{seed}'; out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(2); torch.manual_seed(117+seed)
 gamma=torch.load(GAMMA,map_location='cpu',weights_only=True,mmap=True); criterion=make_criterion()
 core,_=load_core(seed,device,64); initial={k:v.detach().clone() for k,v in core.state_dict().items()}
 params=[p for n,p in core.named_parameters() if p.requires_grad and not n.startswith('graph_generator.')]
 opt=torch.optim.Adam([{'params':params,'lr':3e-5}]); ids=c['ids']; indices=torch.randperm(70000,generator=torch.Generator().manual_seed(117+seed))[:4096]
 hist=[]; started=time.time()
 for step in range(256):
  batch=gamma[indices[step*16:(step+1)*16]].to(device); opt.zero_grad(set_to_none=True)
  (_,spikes,coreout,plv,theta),cap=captured_forward(core,batch,64,criterion,32,trace=True)
  primary,_=criterion(plv=plv,theta=theta)
  aux=cap['event'].detach()*cap['gate']; q=affinity(aux,32); spike_loss,_=criterion(plv=q)
  loss=primary+5.*spike_loss
  if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite loss step {step}')
  loss.backward(); norms=family_grad(core)
  if not norms or not any(v>0 for v in norms.values()): raise AssertionError(f'empty gradient step {step}')
  torch.nn.utils.clip_grad_norm_(params,1.); opt.step()
  hist.append({'step':step+1,'loss':float(loss.detach()),'primary':float(primary.detach()),'aux_spike_unweighted':float(spike_loss.detach()),'gradient_norms_by_family':norms})
  if step==0 or (step+1)%32==0: write(out/'progress.json',{'seed':seed,'step':step+1,'total_steps':256,'started':started,'updated':time.time()})
 changed=[k for k in initial if not torch.equal(initial[k],core.state_dict()[k])]
 for prefix in ('graph_generator.','dendric_layer.','membrane_layer.'):
  if any(k.startswith(prefix) for k in changed): raise AssertionError(f'{prefix} unexpectedly changed')
 torch.save(core.state_dict(),out/'core.pt'); write(out/'history.json',hist)
 write(out/'manifest.json',{'status':'training_complete','seed':seed,'source_sha256':c['source_sha256'],'control_sha256':c['control_sha256'],'training_ids':ids,'shuffle_seed':117+seed,'updates':256,'batch_size':16,'train_steps':64,'settle':32,'lr':3e-5,'gradient_clip':1.,'spike_aux_coefficient':5.,'aux_expression':'event.detach() * actual_gate','actual_core_forward_unchanged':True,'graph_dendrite_membrane_bitwise_fixed':True,'changed_keys':changed,'ground_truth_used_for_training':False,'started':started,'completed':time.time(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
 return out

def evaluate_seed(seed,device):
 out=OUT/f'seed{seed}'; ck=out/'core.pt'; gamma=VAL_GAMMA; ev=out/'evaluation.json'
 cmd=[sys.executable,str(ROOT/'collaborative_test/SW_0040_peer_transfer/evaluate.py'),'--checkpoint',str(ck),'--gamma-path',str(gamma),'--gamma-global-start','1320','--gamma-manifest',str(ROOT/'data/SW_0042_hdf5_aligned/manifest.json'),'--dataset-path',str(DATASET),'--output-path',str(ev),'--start','1320','--count','320','--steps','1024','--settle','512','--membrane-vth','.06','--min-group-size','2','--background','largest_component','--thresholds','.50','--dendritic-projection','shared','--graph-spatial-decay','.35','--geodesic-steps','3','--geodesic-radius','1.5','--geodesic-contrast','2','--geodesic-temperature','.5','--geodesic-cap','16','--kuramoto-backend','factorized']
 subprocess.run(cmd,check=True,cwd=ROOT)
 return ev

def main():
 p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['preflight','train','eval'],required=True); p.add_argument('--seed',type=int,choices=[0,1,2],required=True); p.add_argument('--device',default='cuda:0'); a=p.parse_args(); torch.set_num_threads(2)
 if a.stage=='preflight': r=run_preflight(a.seed,torch.device(a.device)); write(ROOT/'collaborative_test/SW_0105_event_detach_aux/results_archive'/f'preflight_direct_reference_seed{a.seed}.json',r)
 elif a.stage=='train': r=train(a.seed,torch.device(a.device)); print(str(r),flush=True)
 else: r=evaluate_seed(a.seed,a.device); print(str(r),flush=True)
 print(json.dumps({'stage':a.stage,'seed':a.seed,'result':str(r)},default=str),flush=True)
if __name__=='__main__': main()
