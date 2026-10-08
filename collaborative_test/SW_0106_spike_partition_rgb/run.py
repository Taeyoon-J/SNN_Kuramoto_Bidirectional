"""Paired SW0106 RGB-explanation pilot; actual inference remains spike-only."""
import argparse, hashlib, json, math, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'snn_kuramoto_bidirectional'),str(ROOT/'collaborative_test')]
from SW_0094_aligned_joint_pilot.run import ASSETS,DATASET,GAMMA,VAL_GAMMA,hparams
from SW_0106_spike_partition_rgb.partition_rgb import SharedRGBDecoder,groups_to_onehot,reconstruct_one,rgb_patch_means
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.loss_function import UnsupervisedS2NetLoss
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity,spike_synchrony_components
from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_input_encoder
from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
from snn_kuramoto_bidirectional.training.train_s2net_core import _forward_with_plv

OUT=ROOT/'trained_models/SW0106_spike_partition_rgb'
ARCHIVE=ROOT/'collaborative_test/SW_0106_spike_partition_rgb/results_archive'
SOURCE_ROOT=ROOT/'trained_models/SW0095_full70k_aligned_loss'
CONTROL_ROOT=ROOT/'trained_models/SW0097_graph_adaptation'
CACHE=ROOT/'data/SW_0106_spike_partition_rgb'
TRAIN_RGB=CACHE/'train_rgb_uint8.npy'; VAL_RGB=CACHE/'validation_rgb_uint8.npy'
FEATURE_STATS=ASSETS/'feature_preprocessing.pt'
ENCODER_SOURCE=ASSETS/'input_encoder/input_layer_encoder.pt'
METRICS=('fg_ari','foreground_iou','matched_object_iou')

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def write(path,obj):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def source_path(seed): return SOURCE_ROOT/f'seed{seed}/core.pt'

def verify_contract(seed):
 inds,ids=read_pool_indices(seed)
 control=CONTROL_ROOT/f'seed{seed}_positive_frozen'
 manifest=control/'manifest.json'
 if not manifest.is_file(): raise FileNotFoundError(f'matched SW0097 control manifest missing: {manifest}')
 m=json.loads(manifest.read_text())
 if m.get('training_ids')!=ids or m.get('steps')!=256 or m.get('batch')!=16 or m.get('train_steps')!=64 or m.get('train_settle')!=32:
  raise AssertionError('SW0106 IDs/budget differ from matched SW0097 frozen control')
 source=source_path(seed)
 if not source.is_file(): raise FileNotFoundError(source)
 source_sha=sha(source)
 if m.get('source_sha256')!=source_sha: raise AssertionError('matched control source SHA differs from SW0095 core')
 train_meta=json.loads((CACHE/'train_rgb_uint8.npy.complete.json').read_text())
 val_meta=json.loads((CACHE/'validation_rgb_uint8.npy.complete.json').read_text())
 expected_train=np.concatenate((np.arange(1000,dtype='<i8'),np.arange(1640,70640,dtype='<i8')))
 expected_val=np.arange(1320,1640,dtype='<i8')
 if train_meta.get('status')!='complete' or train_meta.get('cache_shape')!=[70000,128,128,3] or train_meta.get('ids_mapping_sha256')!=hashlib.sha256(expected_train.tobytes()).hexdigest():
  raise AssertionError('training RGB cache completion/ID mapping mismatch')
 if val_meta.get('status')!='complete' or val_meta.get('cache_shape')!=[320,128,128,3] or val_meta.get('ids_mapping_sha256')!=hashlib.sha256(expected_val.tobytes()).hexdigest():
  raise AssertionError('validation RGB cache completion/ID mapping mismatch')
 return inds,ids,source,source_sha,sha(control/'core.pt')

def make_criterion():
 return UnsupervisedS2NetLoss(spike_rate_weight=0.,spike_smooth_weight=0.,
  spike_diversity_weight=0.,structural_weight=0.,plv_bimodality_weight=6.,
  plv_balance_weight=10.,plv_coherence_weight=.5,plv_collapse_weight=1.,
  plv_target_density=.867,patch_grid_size=(16,16))

def load_models(device,seed=0):
 source=source_path(seed)
 if not source.is_file(): raise FileNotFoundError(source)
 hp=hparams('raw'); hp.num_time_steps=64
 core=S2NetCore(hp.validate(),device=device).to(device)
 core.load_state_dict(torch.load(source,map_location=device,weights_only=True),strict=True)
 core._detect_object_groups=lambda out,spikes:[[] for _ in range(spikes.size(0))]
 if core.graph_generator.uses_feedback or core.kuramoto.spike_pulse_gain is not None:
  raise AssertionError('feedback and spike pulse must remain disabled')
 encoder=load_input_encoder(str(ENCODER_SOURCE),num_kernels=8,kernel_size=3,channels=3,device=device)
 encoder.requires_grad_(True); encoder.train()
 stats=torch.load(FEATURE_STATS,map_location=device,weights_only=True)
 mean,std=stats['mean'].to(device),stats['std'].to(device); clip=float(stats.get('clip',3.))
 if stats.get('mode')!='standardize' or not bool((std>0).all()): raise AssertionError('registered feature preprocessing mismatch')
 patcher=FeaturePatchGammaInitializer(grid_size=16).to(device)
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(106)
  decoder=SharedRGBDecoder().to(device)
 return core,encoder,patcher,mean,std,clip,decoder

def seed_process(seed,device):
 torch.manual_seed(117+seed)
 if device.type=='cuda': torch.cuda.manual_seed_all(117+seed)

def encode(encoder,patcher,mean,std,clip,images):
 feat=encoder(images.float()/255.)
 return patcher(((feat-mean)/std).clamp(-clip,clip))

def read_batch(cache,indices,device):
 arr=np.asarray(cache[np.asarray(indices,dtype=np.int64)]).copy()
 images=torch.from_numpy(arr).permute(0,3,1,2).to(device=device,dtype=torch.float32)
 return images

def read_pool_indices(seed=0):
 gen=torch.Generator(device='cpu').manual_seed(117+seed)
 chosen=torch.randperm(70000,generator=gen)[:4096]
 ids=[int(i) if int(i)<1000 else int(i)+640 for i in chosen]
 return chosen,ids

def affinity(core):
 comp=core.last_component_spikes
 if comp is None: raise RuntimeError('actual component spike history missing')
 return spike_synchrony_affinity(comp.mean(dim=1),comp,settle=32)

def hard_labels(spikes,components):
 groups=spike_synchrony_components(spikes.detach().cpu(),synchrony_threshold=.50,
   min_group_size=2,settle=32,components=components.detach().cpu(),background='largest_component',
   affinity_mode='spike',spatial_grid_size=16)
 labels=spatial_components_to_patch_labels(groups,16,device=spikes.device).reshape(spikes.shape[0],-1)
 hard=[groups_to_onehot(g,device=spikes.device,dtype=spikes.dtype) for g in groups]
 if any(not torch.equal(h.argmax(-1),lab) for h,lab in zip(hard,labels)): raise AssertionError('H labels differ from production classifier label conversion')
 return labels,hard,groups

def grad_norm(grads):
 sq=0.
 for g in grads:
  if g is not None:
   if not torch.isfinite(g).all(): raise FloatingPointError('nonfinite gradient')
   sq+=float(g.detach().double().square().sum())
 return math.sqrt(sq)

def family_norm(named,grads):
 accum={}
 for (name,_),g in zip(named,grads):
  if g is None: continue
  parts=name.split('.')
  family=parts[1] if parts[0]=='core' and len(parts)>1 else parts[0]
  accum[family]=accum.get(family,0.)+float(g.detach().double().square().sum())
 return {k:math.sqrt(v) for k,v in accum.items()}

def read_full320_metrics(path):
 d=json.loads(Path(path).read_text())
 if d.get('images')!=320 or d.get('ids')!=[1320,1639]: raise AssertionError(f'full320 evaluation contract mismatch: {path}')
 rows=d.get('sweep')
 if not isinstance(rows,list) or not rows: raise AssertionError(f'missing evaluator sweep rows: {path}')
 scored=rows[0].get('scored_targets',{}).get('our_hdf5')
 if scored is None or any(int(v)!=320 for v in scored.get('valid_count',{}).values()): raise AssertionError(f'invalid per-image metric counts: {path}')
 metrics=scored.get('metrics',{})
 if any(k not in metrics or not math.isfinite(float(metrics[k])) for k in METRICS): raise AssertionError(f'nonfinite metrics: {path}')
 return {k:float(metrics[k]) for k in METRICS}

def seed0_expansion_gate():
 cand=OUT/'candidate_seed0/evaluation.json';ctrl=OUT/'control_seed0/evaluation.json'
 sw97=CONTROL_ROOT/'seed0_positive_frozen/evaluation.json'
 cm=read_full320_metrics(cand);ct=read_full320_metrics(ctrl);base=read_full320_metrics(sw97)
 checks={'fg_ari_candidate_plus_0.01_vs_control':cm['fg_ari']>=ct['fg_ari']+.01,
  'fg_ari_candidate_plus_0.01_vs_sw0097':cm['fg_ari']>=base['fg_ari']+.01,
  'foreground_iou_slot_margin':cm['foreground_iou']>=.25358913,
  'matched_object_iou_slot_margin':cm['matched_object_iou']>=.25693698}
 if not all(checks.values()): raise RuntimeError(f'seed0 expansion gate failed; no seed1/2 authorized: {checks}')
 return {'candidate':cm,'joint_control':ct,'sw0097':base,'checks':checks}

def forward_batch(core,encoder,patcher,mean,std,clip,images,criterion):
 gamma=encode(encoder,patcher,mean,std,clip,images)
 result=_forward_with_plv(core,gamma,criterion,32,'phase','mean')
 groups,spikes,coreout,plv,theta=result
 components=core.last_component_spikes
 if tuple(components.shape)!=(images.shape[0],4,256,64): raise AssertionError('component history shape mismatch')
 q=affinity(core)
 labels,hard,detected=hard_labels(spikes,components)
 target=rgb_patch_means(images/255.)
 return gamma,result,q,labels,hard,detected,target

def image_losses(q,hard,gamma,target,decoder,credit):
 losses=[]; preds=[]; diagnostics=[]
 for b in range(q.shape[0]):
  hb=hard[b] if isinstance(hard,(list,tuple)) else hard[b]
  pred,loss,diag=reconstruct_one(q[b],hb,gamma[b].transpose(0,1),target[b],decoder,credit)
  preds.append(pred);losses.append(loss);diagnostics.append(diag)
 return torch.stack(preds),torch.stack(losses).mean(),diagnostics

def setup_optimizer(core,encoder,decoder):
 joint=[p for p in core.parameters() if p.requires_grad]+[p for p in encoder.parameters() if p.requires_grad]
 core_params=[p for p in core.parameters() if p.requires_grad]
 encoder_params=[p for p in encoder.parameters() if p.requires_grad]
 jo=torch.optim.Adam([{'params':core_params,'lr':3e-5},{'params':encoder_params,'lr':3e-6}])
 do=torch.optim.Adam(decoder.parameters(),lr=3e-4)
 return joint,core_params,encoder_params,jo,do

def preflight(seed,device,output):
 output=Path(output)
 output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists(): raise FileExistsError(f'preserve existing preflight record: {output}')
 if not TRAIN_RGB.is_file() or not VAL_RGB.is_file(): raise FileNotFoundError('verified RGB caches are required')
 train=np.load(TRAIN_RGB,mmap_mode='r'); val=np.load(VAL_RGB,mmap_mode='r')
 if train.shape!=(70000,128,128,3) or val.shape!=(320,128,128,3): raise AssertionError('cache shape mismatch')
 if json.loads((CACHE/'train_rgb_uint8.npy.complete.json').read_text())['status']!='complete': raise AssertionError('train RGB cache incomplete')
 if json.loads((CACHE/'validation_rgb_uint8.npy.complete.json').read_text())['status']!='complete': raise AssertionError('validation RGB cache incomplete')
 if seed>0: seed0_expansion_gate()
 inds,ids,source,source_sha,control_sha=verify_contract(seed); gamma_cache=torch.load(GAMMA,map_location='cpu',weights_only=True,mmap=True)
 if tuple(gamma_cache.shape)!=(70000,8,256): raise AssertionError('registered gamma cache shape mismatch')
 core,encoder,patcher,mean,std,clip,decoder=load_models(device,seed)
 criterion=make_criterion(); core.train(); encoder.train(); decoder.train()
 initial_core={k:v.detach().clone() for k,v in core.state_dict().items()}
 initial_encoder={k:v.detach().clone() for k,v in encoder.state_dict().items()}
 named_joint=[(f'core.{n}',p) for n,p in core.named_parameters() if p.requires_grad]+[(f'encoder.{n}',p) for n,p in encoder.named_parameters() if p.requires_grad]
 decoder_params=list(decoder.parameters())
 # Registered cache match and initial forward snapshot on actual first training batch.
 ixs=inds[:16].tolist(); images=read_batch(train,ixs,device); expected=gamma_cache[inds[:16]].to(device)
 with torch.no_grad():
  gamma=encode(encoder,patcher,mean,std,clip,images)
  gamma_diff=float((gamma-expected).abs().max())
 if gamma_diff>2e-5: raise AssertionError(f'RGB-to-registered-gamma difference {gamma_diff} exceeds 2e-5')
 # Decoder-only warmup exactly32 batches; no core/encoder optimizer exists here.
 warm_opt=torch.optim.Adam(decoder_params,lr=3e-4); warm_losses=[]; max_k=0
 core_mode,encoder_mode=core.training,encoder.training;core.eval();encoder.eval()
 try:
  for step in range(32):
   ix=inds[step*16:(step+1)*16].tolist(); batch=read_batch(train,ix,device)
   with torch.no_grad():
    gamma,res,q,labels,hard,groups,target=forward_batch(core,encoder,patcher,mean,std,clip,batch,criterion)
   warm_opt.zero_grad(set_to_none=True); _,rec,diags=image_losses(q,hard,gamma.detach(),target,decoder,False)
   if not torch.isfinite(rec): raise FloatingPointError('nonfinite decoder warmup loss')
   rec.backward(); torch.nn.utils.clip_grad_norm_(decoder_params,1.); warm_opt.step()
   warm_losses.append(float(rec.detach())); max_k=max(max_k,max(d['K'] for d in diags))
 finally:
  core.train(core_mode);encoder.train(encoder_mode)
 if any(not torch.equal(v,initial_core[k]) for k,v in core.state_dict().items()): raise AssertionError('decoder warmup changed core')
 if any(not torch.equal(v,initial_encoder[k]) for k,v in encoder.state_dict().items()): raise AssertionError('decoder warmup changed encoder')
 # Candidate/control initial hard forward and old objective are identical.
 batch=read_batch(train,inds[:16].tolist(),device)
 gamma,res,q,labels,hard,groups,target=forward_batch(core,encoder,patcher,mean,std,clip,batch,criterion)
 _,spikes,out,plv,theta=res; primary,_=criterion(plv=plv,theta=theta); spike,_=criterion(plv=q); old=primary+5.*spike
 pred_c,rec_c,_=image_losses(q,hard,gamma,target,decoder,True)
 pred_h,rec_h,_=image_losses(q.detach(),hard,gamma.detach(),target,decoder,False)
 if not torch.equal(pred_c,pred_h) or not torch.equal(rec_c,rec_h): raise AssertionError('candidate/control hard-forward reconstruction differs')
 if not torch.isfinite(old+rec_c): raise FloatingPointError('nonfinite initial loss')
 # Candidate assignment credit reaches encoder, learned graph, and upstream core.
 qgrads=torch.autograd.grad(rec_c,[p for _,p in named_joint],retain_graph=True,allow_unused=True)
 qnorms=family_norm(named_joint,qgrads)
 for family in ('encoder','graph_generator','kuramoto'):
  if qnorms.get(family,0.)<=0: raise AssertionError(f'no candidate reconstruction Q-credit for {family}')
 # Control readout uses exactly H and has no reconstruction gradient into Q/model.
 control_grads=torch.autograd.grad(rec_h,[p for _,p in named_joint],retain_graph=True,allow_unused=True)
 if any(g is not None and bool((g!=0).any()) for g in control_grads): raise AssertionError('control reconstruction unexpectedly reaches core/encoder')
 # Calibration is no-GT/no-update; λ uses the prescribed first four batches.
 ratios=[]; calibration=[]; rec_grads=[]; old_grads=[]
 for j in range(4):
  ix=inds[j*16:(j+1)*16].tolist(); im=read_batch(train,ix,device)
  _,rr,qq,ll,hh,_,tt=forward_batch(core,encoder,patcher,mean,std,clip,im,criterion)
  _,ss,oo,pp,th=rr; prim,_=criterion(plv=pp,theta=th); sl,_=criterion(plv=qq); oldj=prim+5.*sl
  _,recj,_=image_losses(qq,hh,encode(encoder,patcher,mean,std,clip,im),tt,decoder,True)
  og=torch.autograd.grad(oldj,[p for _,p in named_joint],retain_graph=True,allow_unused=True)
  rg=torch.autograd.grad(recj,[p for _,p in named_joint],retain_graph=False,allow_unused=True)
  on=grad_norm(og); rn=grad_norm(rg); rec_family=family_norm(named_joint,rg)
  if on<=0 or rn<=0: raise AssertionError(f'calibration batch {j} has inert objective gradients')
  for family in ('encoder','graph_generator','kuramoto'):
   if rec_family.get(family,0.)<=0: raise AssertionError(f'calibration batch {j} reconstruction gradient is inert for {family}')
  ratio=.25*on/rn; ratios.append(ratio); calibration.append({'batch':j,'old_joint_grad_norm':on,'reconstruction_joint_grad_norm':rn,'reconstruction_grad_norms_by_family':rec_family,'ratio':ratio})
 measured_lambda=float(np.median(np.asarray(ratios,dtype=np.float64)))
 if seed==0:
  lam=measured_lambda
 else:
  seed0_path=Path(output).parent/'preflight_seed0.json'
  if not seed0_path.is_file(): raise RuntimeError('later-seed preflight requires successful seed0 calibration')
  seed0=json.loads(seed0_path.read_text())
  if seed0.get('status')!='passed': raise RuntimeError('seed0 calibration did not pass')
  lam=float(seed0['lambda'])
 if not math.isfinite(lam) or lam<=0: raise AssertionError('invalid shared reconstruction coefficient')
 # Within-image mask row scrambling checks assignment use on the same four TRAIN batches.
 with torch.no_grad():
  diffs=[];shuffle_by_batch=[]
  for j in range(4):
   ix=inds[j*16:(j+1)*16].tolist();im=read_batch(train,ix,device)
   gj,_,qj,_,hj,_,tj=forward_batch(core,encoder,patcher,mean,std,clip,im,criterion)
   batch_diffs=[]
   for b in range(16):
    rowperm=torch.randperm(256,generator=torch.Generator().manual_seed(106+j*16+b)).to(device)
    _,orig,_=reconstruct_one(qj[b],hj[b],gj[b].transpose(0,1),tj[b],decoder,False)
    _,scrambled,_=reconstruct_one(qj[b],hj[b][rowperm],gj[b].transpose(0,1),tj[b],decoder,False)
    batch_diffs.append(float((scrambled-orig).detach()))
   shuffle_by_batch.append({'batch':j,'images':len(batch_diffs),'mean_shuffled_minus_original_mse':float(np.mean(batch_diffs))})
   diffs.extend(batch_diffs)
 if float(np.mean(diffs))<=0: raise AssertionError('mean hard-partition row scrambling did not increase reconstruction loss')
 # Freeze the paired decoder and its optimizer state before the throwaway update.
 artifact=output.parent/f'preflight_decoder_seed{seed}.pt'
 if artifact.exists(): raise FileExistsError(f'preserve existing warmup artifact: {artifact}')
 torch.save({'decoder_state_dict':decoder.state_dict(),'decoder_optimizer_state_dict':warm_opt.state_dict()},artifact)
 artifact_sha=sha(artifact)
 # One throwaway candidate update with the actual trainable parameter groups.
 joint,core_params,encoder_params,joint_opt,dec_opt=setup_optimizer(core,encoder,decoder)
 dec_opt.load_state_dict(warm_opt.state_dict())
 _,rr,qq,ll,hh,_,tt=forward_batch(core,encoder,patcher,mean,std,clip,batch,criterion)
 _,ss,oo,pp,th=rr; prim,_=criterion(plv=pp,theta=th); sl,_=criterion(plv=qq); oldj=prim+5.*sl
 _,recj,_=image_losses(qq,hh,encode(encoder,patcher,mean,std,clip,batch),tt,decoder,True)
 joint_opt.zero_grad(set_to_none=True);dec_opt.zero_grad(set_to_none=True)
 jg=torch.autograd.grad(oldj+lam*recj,joint,retain_graph=True,allow_unused=True)
 dg=torch.autograd.grad(recj,decoder_params,allow_unused=True)
 joint_preclip=grad_norm(jg);decoder_preclip=grad_norm(dg)
 if joint_preclip<=0 or decoder_preclip<=0: raise AssertionError('throwaway update has an empty trainable gradient')
 for p,g in zip(joint,jg): p.grad=None if g is None else g.detach().clone()
 for p,g in zip(decoder_params,dg): p.grad=None if g is None else g.detach().clone()
 if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in joint+decoder_params): raise FloatingPointError('nonfinite update preflight gradient')
 torch.nn.utils.clip_grad_norm_(joint,1.);torch.nn.utils.clip_grad_norm_(decoder_params,1.)
 core0={k:v.detach().clone() for k,v in core.state_dict().items()}; enc0={k:v.detach().clone() for k,v in encoder.state_dict().items()}
 graph_param0={n:p.detach().clone() for n,p in core.graph_generator.named_parameters()}
 encoder_param0={n:p.detach().clone() for n,p in encoder.named_parameters()}
 joint_opt.step();dec_opt.step()
 changed_core=[k for k,v in core0.items() if not torch.equal(v,core.state_dict()[k])]
 if not changed_core: raise AssertionError('throwaway update did not change core')
 if not any(k.startswith('graph_generator.') for k in changed_core): raise AssertionError('throwaway update did not change learned graph')
 if not any(not torch.equal(v,dict(encoder.named_parameters())[k]) for k,v in encoder_param0.items()): raise AssertionError('throwaway update did not change encoder parameters')
 if not any(not torch.equal(v,dict(core.graph_generator.named_parameters())[k]) for k,v in graph_param0.items()): raise AssertionError('throwaway update did not change learned graph parameters')
 if any(not torch.isfinite(p).all() for p in list(core.parameters())+list(encoder.parameters())+decoder_params): raise FloatingPointError('nonfinite parameter after update')
 report={'status':'passed','seed':seed,'device':str(device),'source_core_sha256':source_sha,'matched_control_core_sha256':control_sha,'input_encoder_sha256':sha(ENCODER_SOURCE),
  'training_ids_first4096':ids,'shuffle_seed':117+seed,'training_batch_size':16,'train_time_steps':64,'settle':32,
  'cached_gamma_max_abs_difference_first_batch':gamma_diff,'decoder_warmup_batches':32,'decoder_warmup_loss_first_last':[warm_losses[0],warm_losses[-1]],
  'warmup_source_core_and_encoder_unchanged':True,'initial_candidate_control_hard_forward_exact':True,
  'production_labels_exact':True,'candidate_reconstruction_gradient_norms_by_family':qnorms,
  'control_reconstruction_has_no_core_encoder_gradient':True,'lambda_calibration':calibration,'lambda':lam,
  'row_scramble_images':len(diffs),'row_scramble_delta_mean':float(np.mean(diffs)),
  'row_scramble_positive_count':sum(x>0 for x in diffs),'row_scramble_by_batch':shuffle_by_batch,
  'max_dynamic_K_seen':max_k,'throwaway_candidate_joint_grad_norm_preclip':joint_preclip,
  'throwaway_decoder_grad_norm_preclip':decoder_preclip,'throwaway_changed_core_keys':changed_core,
  'throwaway_candidate_update_finite_and_changed_graph_encoder_parameters':True,
  'throwaway_decoder_optimizer_state_loaded_from_shared_32_batch_warmup':True,
  'calibration_measured_lambda':measured_lambda,'fixed_lambda_source_seed':0,'warmup_artifact':str(artifact),'warmup_artifact_sha256':artifact_sha,
  'ground_truth_used':False,'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved() if device.type=='cuda' else 0}
 write(output,report)
 return report

def train(seed,arm,device,steps=256,batch_size=16):
 if seed not in (0,1,2) or arm not in ('candidate','control'): raise ValueError('invalid pilot pair member')
 if steps!=256 or batch_size!=16: raise ValueError('registered SW0106 pilot budget is fixed at 256x16')
 if seed>0: seed0_expansion_gate()
 out=OUT/f'{arm}_seed{seed}';out.mkdir(parents=True,exist_ok=False)
 train=np.load(TRAIN_RGB,mmap_mode='r');inds,ids,source,source_sha,control_sha=verify_contract(seed)
 preflight_path=ROOT/f'collaborative_test/SW_0106_spike_partition_rgb/results_archive/preflight_seed{seed}.json'
 if not preflight_path.is_file(): raise RuntimeError(f'preflight missing: {preflight_path}')
 preflight_record=json.loads(preflight_path.read_text())
 if preflight_record.get('status')!='passed' or preflight_record.get('seed')!=seed or preflight_record.get('training_ids_first4096')!=ids or preflight_record.get('source_core_sha256')!=source_sha or preflight_record.get('matched_control_core_sha256')!=control_sha:
  raise AssertionError('preflight source/control contract mismatch')
 artifact=Path(preflight_record['warmup_artifact'])
 if not artifact.is_file() or sha(artifact)!=preflight_record.get('warmup_artifact_sha256'):
  raise AssertionError('shared decoder warmup artifact missing or SHA mismatch')
 warm=torch.load(artifact,map_location=device,weights_only=True)
 core,encoder,patcher,mean,std,clip,decoder=load_models(device,seed)
 decoder.load_state_dict(warm['decoder_state_dict'],strict=True)
 core.train();encoder.train();decoder.train();criterion=make_criterion()
 joint,core_params,encoder_params,joint_opt,dec_opt=setup_optimizer(core,encoder,decoder)
 dec_opt.load_state_dict(warm['decoder_optimizer_state_dict'])
 lam=float(preflight_record['lambda']);calibration=preflight_record['lambda_calibration']
 history=[];started=time.time()
 initial_core={k:v.detach().clone() for k,v in core.state_dict().items()};initial_enc={k:v.detach().clone() for k,v in encoder.state_dict().items()}
 for step in range(256):
  im=read_batch(train,inds[step*16:(step+1)*16].tolist(),device)
  gamma,res,q,labels,hard,groups,target=forward_batch(core,encoder,patcher,mean,std,clip,im,criterion)
  _,spikes,out0,plv,theta=res;primary,_=criterion(plv=plv,theta=theta);spike,_=criterion(plv=q);old=primary+5.*spike
  _,rec,diags=image_losses(q,hard,gamma,target,decoder,arm=='candidate')
  total=old+(lam*rec if arm=='candidate' else 0.)
  if not torch.isfinite(total+rec): raise FloatingPointError(f'nonfinite loss at update{step}')
  joint_opt.zero_grad(set_to_none=True);dec_opt.zero_grad(set_to_none=True)
  jg=torch.autograd.grad(total,joint,retain_graph=True,allow_unused=True)
  dg=torch.autograd.grad(rec,decoder.parameters(),allow_unused=True)
  for p,g in zip(joint,jg): p.grad=None if g is None else g.detach().clone()
  for p,g in zip(decoder.parameters(),dg): p.grad=None if g is None else g.detach().clone()
  jnorm=grad_norm([p.grad for p in joint]);dnorm=grad_norm([p.grad for p in decoder.parameters()])
  if jnorm<=0 or dnorm<=0: raise AssertionError(f'empty gradient at update{step}')
  torch.nn.utils.clip_grad_norm_(joint,1.);torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.)
  joint_opt.step();dec_opt.step()
  history.append({'update':step+1,'total':float(total.detach()),'old_objective':float(old.detach()),'primary':float(primary.detach()),
   'positive_product_spike_unweighted':float(spike.detach()),'reconstruction_unweighted':float(rec.detach()),
   'joint_grad_norm_preclip':jnorm,'decoder_grad_norm_preclip':dnorm,'predicted_groups_mean':float(np.mean([len(g) for g in groups])),
   'K_max':max(d['K'] for d in diags)})
  if step==0 or (step+1)%32==0: write(out/'progress.json',{'status':'training','arm':arm,'seed':seed,'update':step+1,'total_updates':256,'updated':time.time()})
 changed_core=[k for k in initial_core if not torch.equal(initial_core[k],core.state_dict()[k])]
 changed_enc=[k for k in initial_enc if not torch.equal(initial_enc[k],encoder.state_dict()[k])]
 if not any(k.startswith('graph_generator.') for k in changed_core): raise AssertionError('learned graph did not update')
 if not any(not torch.equal(initial_enc[k],dict(encoder.named_parameters())[k]) for k,_ in encoder.named_parameters()): raise AssertionError('encoder parameters did not update')
 torch.save(core.state_dict(),out/'core.pt');torch.save(encoder.state_dict(),out/'encoder.pt');torch.save(decoder.state_dict(),out/'decoder.pt')
 optimizer_artifact=out/'optimizer_state.pt'
 torch.save({'joint_optimizer_state_dict':joint_opt.state_dict(),'decoder_optimizer_state_dict':dec_opt.state_dict(),
  'seed':seed,'arm':arm,'updates':256,'shared_lambda':lam},optimizer_artifact)
 optimizer_sha=sha(optimizer_artifact)
 write(out/'history.json',history)
 write(out/'manifest.json',{'status':'training_complete','arm':arm,'seed':seed,'source_core_sha256':source_sha,
  'matched_control_core_sha256':control_sha,
  'encoder_source_sha256':sha(ENCODER_SOURCE),'feature_preprocessing_sha256':sha(FEATURE_STATS),
  'feature_preprocessing_path':str(FEATURE_STATS),'training_ids':ids,'training_pool_indices':inds.tolist(),'shuffle_seed':117+seed,
  'updates':256,'batch_size':16,'train_steps':64,'settle':32,'old_spike_aux_weight':5.,'shared_lambda':lam,
  'lambda_calibration':calibration,'decoder_warmup_batches':32,'decoder_lr':3e-4,'joint_core_lr':3e-5,'joint_encoder_lr':3e-6,
  'gradient_clip_joint':1.,'gradient_clip_decoder':1.,'forward_prediction':'unchanged actual event*gate spike classifier',
  'ground_truth_used_for_training':False,'started':started,'completed':time.time(),
  'changed_core_keys':changed_core,'changed_encoder_keys':changed_enc,
  'optimizer_state_artifact':'optimizer_state.pt','optimizer_state_sha256':optimizer_sha,
  'joint_optimizer_steps':256,'decoder_optimizer_steps':288,
  'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved() if device.type=='cuda' else 0})
 (out/'TRAINING_COMPLETED').write_text('SW0106 fixed256-update pilot complete\n')
 return out

def evaluate(seed,arm,device):
 out=OUT/f'{arm}_seed{seed}';train=np.load(VAL_RGB,mmap_mode='r')
 core,encoder,patcher,mean,std,clip,decoder=load_models(device,seed)
 core.load_state_dict(torch.load(out/'core.pt',map_location=device,weights_only=True),strict=True)
 encoder.load_state_dict(torch.load(out/'encoder.pt',map_location=device,weights_only=True),strict=True)
 encoder.eval();core.eval();gamma=[]
 with torch.no_grad():
  for start in range(0,320,16):
   im=read_batch(train,list(range(start,min(start+16,320))),device)
   gamma.append(encode(encoder,patcher,mean,std,clip,im).cpu())
 path=out/'gamma_validation.pt';torch.save(torch.cat(gamma),path)
 gm=out/'gamma_manifest.json';write(gm,{'source':'SW0106 trained registered input encoder','image_ids':[1320,1639],
  'gamma_global_start':1320,'shape':[320,8,256],'encoder_sha256':sha(out/'encoder.pt'),'gamma_sha256':sha(path),
  'ground_truth_used_for_prediction':False})
 cmd=[sys.executable,str(ROOT/'collaborative_test/SW_0040_peer_transfer/evaluate.py'),'--checkpoint',str(out/'core.pt'),
  '--gamma-path',str(path),'--gamma-global-start','1320','--gamma-manifest',str(gm),'--dataset-path',str(DATASET),
  '--output-path',str(out/'evaluation.json'),'--start','1320','--count','320','--batch-size','8','--steps','1024','--settle','512',
  '--membrane-vth','.06','--min-group-size','2','--background','largest_component','--thresholds','.50',
  '--dendritic-projection','shared','--graph-spatial-decay','.35','--geodesic-steps','3','--geodesic-radius','1.5',
  '--geodesic-contrast','2','--geodesic-temperature','.5','--geodesic-cap','16','--kuramoto-backend','factorized','--device',str(device)]
 subprocess.run(cmd,check=True,cwd=ROOT)
 (out/'COMPLETED').write_text('SW0106 training and full320 actual-spike evaluation complete\n')
 return out/'evaluation.json'

def optimizer_steps(state_dict):
 steps=[]
 for state in state_dict.get('state',{}).values():
  step=state.get('step')
  if step is None: continue
  steps.append(int(step.item()) if torch.is_tensor(step) else int(step))
 return steps

def optimizer_step_map(state_dict):
 result={}
 for key,state in state_dict.get('state',{}).items():
  step=state.get('step')
  if step is not None:result[str(key)]=int(step.item()) if torch.is_tensor(step) else int(step)
 return result

def full_train(seed,arm,device):
 """One additional matched pass over all70k cached RGBs; never resets pilot Adam."""
 if seed not in (0,1,2) or arm not in ('candidate','control'): raise ValueError('invalid SW0106 full continuation')
 pilot=OUT/f'{arm}_seed{seed}';out=OUT/f'full70k_{arm}_seed{seed}'
 if out.exists(): raise FileExistsError(f'preserve existing full70k output; do not overwrite: {out}')
 gate_path=ARCHIVE/'full70k_promotion_gate.json'
 if not gate_path.is_file() or json.loads(gate_path.read_text()).get('status')!='passed':
  raise RuntimeError('full70k promotion gate has not passed')
 if not (pilot/'TRAINING_COMPLETED').is_file() or not (pilot/'evaluation.json').is_file():
  raise FileNotFoundError(f'completed/evaluated pilot is required: {pilot}')
 pilot_manifest=json.loads((pilot/'manifest.json').read_text())
 optimizer_path=pilot/'optimizer_state.pt'
 if not optimizer_path.is_file(): raise FileNotFoundError(f'pilot Adam artifact missing; never reset optimizer: {optimizer_path}')
 optimizer_sha=sha(optimizer_path)
 if optimizer_sha!=pilot_manifest.get('optimizer_state_sha256') or pilot_manifest.get('optimizer_state_artifact')!='optimizer_state.pt':
  raise AssertionError('pilot optimizer artifact SHA/manifest mismatch')
 if pilot_manifest.get('seed')!=seed or pilot_manifest.get('arm')!=arm or pilot_manifest.get('updates')!=256:
  raise AssertionError('pilot manifest does not match requested full continuation')
 pilot_joint_steps=pilot_manifest.get('joint_optimizer_steps')
 pilot_decoder_steps=pilot_manifest.get('decoder_optimizer_steps')
 if pilot_joint_steps!=256 or pilot_decoder_steps!=288: raise AssertionError('pilot optimizer step counts are not the registered 256/288')
 fullgate=json.loads(gate_path.read_text())
 if fullgate.get('seed0_pilot_gate',{}).get('status')!='passed': raise AssertionError('seed0 expansion gate is not recorded as passed')
 inds,ids,source,source_sha,control_sha=verify_contract(seed)
 if fullgate.get('three_seed_pilot_gate',{}).get('status')!='passed':
  raise AssertionError('three-seed pilot gate is not recorded as passed')
 train=np.load(TRAIN_RGB,mmap_mode='r')
 if train.shape!=(70000,128,128,3): raise AssertionError('exact70k RGB cache shape mismatch')
 order=torch.randperm(70000,generator=torch.Generator(device='cpu').manual_seed(117+seed)).tolist()
 if len(order)!=70000 or len(set(order))!=70000: raise AssertionError('full-pool order is not a permutation')
 ids_order=[int(i) if int(i)<1000 else int(i)+640 for i in order]
 expected_pilot=torch.randperm(70000,generator=torch.Generator(device='cpu').manual_seed(117+seed))[:4096].tolist()
 if order[:4096]!=expected_pilot or ids_order[:4096]!=pilot_manifest.get('training_ids'):
  raise AssertionError('full-pool order does not continue the registered pilot exposure order')
 torch.manual_seed(117+seed)
 if device.type=='cuda':torch.cuda.manual_seed_all(117+seed)
 core,encoder,patcher,mean,std,clip,decoder=load_models(device,seed)
 stats_sha=sha(FEATURE_STATS)
 if pilot_manifest.get('feature_preprocessing_sha256')!=stats_sha:
  raise AssertionError('registered feature preprocessing changed or was not frozen in pilot manifest')
 if pilot_manifest.get('encoder_source_sha256')!=sha(ENCODER_SOURCE):
  raise AssertionError('registered input encoder source changed since pilot')
 core.load_state_dict(torch.load(pilot/'core.pt',map_location=device,weights_only=True),strict=True)
 encoder.load_state_dict(torch.load(pilot/'encoder.pt',map_location=device,weights_only=True),strict=True)
 decoder.load_state_dict(torch.load(pilot/'decoder.pt',map_location=device,weights_only=True),strict=True)
 joint,core_params,encoder_params,joint_opt,dec_opt=setup_optimizer(core,encoder,decoder)
 saved=torch.load(optimizer_path,map_location=device,weights_only=True)
 if saved.get('seed')!=seed or saved.get('arm')!=arm or saved.get('updates')!=256 or saved.get('shared_lambda')!=pilot_manifest.get('shared_lambda'):
  raise AssertionError('optimizer state artifact contract mismatch')
 js=saved['joint_optimizer_state_dict'];ds=saved['decoder_optimizer_state_dict']
 joint_steps_before=optimizer_step_map(js);decoder_steps_before=optimizer_step_map(ds)
 if not joint_steps_before or not decoder_steps_before or max(joint_steps_before.values())!=256 or max(decoder_steps_before.values())!=288:
  raise AssertionError('optimizer artifact does not contain required 256/288 pilot Adam moment steps')
 if min(joint_steps_before.values())<1 or min(decoder_steps_before.values())<1:
  raise AssertionError('optimizer artifact contains an invalid per-parameter step')
 joint_opt.load_state_dict(js);dec_opt.load_state_dict(ds)
 lam=float(pilot_manifest['shared_lambda'])
 pf=json.loads((ARCHIVE/f'preflight_seed{seed}.json').read_text())
 if pf.get('status')!='passed' or pf.get('lambda')!=lam: raise AssertionError('seed preflight lambda mismatch')
 out.mkdir(parents=True,exist_ok=False)
 criterion=make_criterion();core.train();encoder.train();decoder.train();history=[];started=time.time()
 for step in range(4375):
  batch=read_batch(train,order[step*16:(step+1)*16],device)
  gamma,res,q,labels,hard,groups,target=forward_batch(core,encoder,patcher,mean,std,clip,batch,criterion)
  _,spikes,out0,plv,theta=res;primary,_=criterion(plv=plv,theta=theta);spike,_=criterion(plv=q);old=primary+5.*spike
  _,rec,diags=image_losses(q,hard,gamma,target,decoder,arm=='candidate')
  total=old+(lam*rec if arm=='candidate' else 0.)
  if not torch.isfinite(total+rec):raise FloatingPointError(f'nonfinite full70k loss at update{step}')
  joint_opt.zero_grad(set_to_none=True);dec_opt.zero_grad(set_to_none=True)
  jg=torch.autograd.grad(total,joint,retain_graph=True,allow_unused=True)
  dg=torch.autograd.grad(rec,decoder.parameters(),allow_unused=True)
  for p,g in zip(joint,jg):p.grad=None if g is None else g.detach().clone()
  for p,g in zip(decoder.parameters(),dg):p.grad=None if g is None else g.detach().clone()
  jnorm=grad_norm([p.grad for p in joint]);dnorm=grad_norm([p.grad for p in decoder.parameters()])
  if jnorm<=0 or dnorm<=0:raise AssertionError(f'empty full70k gradient at update{step}')
  torch.nn.utils.clip_grad_norm_(joint,1.);torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.)
  joint_opt.step();dec_opt.step()
  history.append({'update':step+1,'total':float(total.detach()),'old_objective':float(old.detach()),
   'primary':float(primary.detach()),'positive_product_spike_unweighted':float(spike.detach()),
   'reconstruction_unweighted':float(rec.detach()),'joint_grad_norm_preclip':jnorm,
   'decoder_grad_norm_preclip':dnorm,'predicted_groups_mean':float(np.mean([len(g) for g in groups])),
   'K_max':max(d['K'] for d in diags)})
  if step==0 or (step+1)%128==0:write(out/'progress.json',{'status':'training','stage':'full70k','arm':arm,'seed':seed,'update':step+1,'total_updates':4375,'updated':time.time()})
 if len(history)!=4375:raise AssertionError('full70k pass did not consume exactly 70000 images')
 joint_steps_after=optimizer_step_map(joint_opt.state_dict());decoder_steps_after=optimizer_step_map(dec_opt.state_dict())
 for name,before,after in (('joint',joint_steps_before,joint_steps_after),('decoder',decoder_steps_before,decoder_steps_after)):
  if not after or not set(before).issubset(after):raise AssertionError(f'{name} Adam state was reset or lost during full70k continuation')
  deltas=[after[k]-v for k,v in before.items()]
  if any(delta<0 or delta>4375 for delta in deltas) or max(deltas)!=4375:
   raise AssertionError(f'{name} Adam moments did not advance through the complete 4375-update pass')
 torch.save(core.state_dict(),out/'core.pt');torch.save(encoder.state_dict(),out/'encoder.pt');torch.save(decoder.state_dict(),out/'decoder.pt')
 full_optimizer=out/'optimizer_state.pt'
 torch.save({'joint_optimizer_state_dict':joint_opt.state_dict(),'decoder_optimizer_state_dict':dec_opt.state_dict(),
  'seed':seed,'arm':arm,'pilot_updates':256,'full70k_updates':4375,'joint_total_steps':4631,
  'decoder_total_steps':4663,'shared_lambda':lam},full_optimizer)
 full_optimizer_sha=sha(full_optimizer)
 write(out/'history.json',history)
 write(out/'manifest.json',{'status':'full70k_training_complete','stage':'full70k','arm':arm,'seed':seed,
  'pilot_core_sha256':sha(pilot/'core.pt'),'pilot_encoder_sha256':sha(pilot/'encoder.pt'),
  'pilot_decoder_sha256':sha(pilot/'decoder.pt'),'pilot_optimizer_sha256':optimizer_sha,
  'feature_preprocessing_path':str(FEATURE_STATS),'feature_preprocessing_sha256':stats_sha,
  'pilot_optimizer_artifact':str(optimizer_path),'pilot_optimizer_steps':{'joint':256,'decoder':288},
  'full_optimizer_state_sha256':full_optimizer_sha,'full_updates':4375,'full_images':70000,
  'cumulative_update_budgets':{'joint':4631,'decoder':4663},
  'pilot_optimizer_per_parameter_steps':{'joint_min':min(joint_steps_before.values()),'joint_max':max(joint_steps_before.values()),
   'decoder_min':min(decoder_steps_before.values()),'decoder_max':max(decoder_steps_before.values())},
  'final_optimizer_per_parameter_steps':{'joint_min':min(joint_steps_after.values()),'joint_max':max(joint_steps_after.values()),
   'decoder_min':min(decoder_steps_after.values()),'decoder_max':max(decoder_steps_after.values())},
  'batch_size':16,'train_steps':64,'settle':32,
  'pool_indices_sha256':hashlib.sha256(np.asarray(order,dtype='<i8').tobytes()).hexdigest(),
  'training_ids_sha256':hashlib.sha256(np.asarray(ids_order,dtype='<i8').tobytes()).hexdigest(),
  'training_ids_first4096_match_pilot':ids_order[:4096]==pilot_manifest.get('training_ids'),
  'shuffle_seed':117+seed,'shared_lambda':lam,'loss':'old phase + 5x positive-product spike; candidate-only lambda*RGB assignment MSE',
  'ground_truth_used_for_training':False,'started':started,'completed':time.time(),
  'runner_sha256':sha(Path(__file__)),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved() if device.type=='cuda' else 0})
 (out/'FULL70K_TRAINING_COMPLETED').write_text('SW0106 additional full70000-image continuation complete\n')
 return out

def full_evaluate(seed,arm,device):
 """Regenerate validation gamma from the full-pass encoder; preserve actual spike readout."""
 out=OUT/f'full70k_{arm}_seed{seed}'
 if not (out/'FULL70K_TRAINING_COMPLETED').is_file():raise FileNotFoundError('completed full70k model required')
 full_manifest=json.loads((out/'manifest.json').read_text())
 if full_manifest.get('feature_preprocessing_sha256')!=sha(FEATURE_STATS):
  raise AssertionError('feature preprocessing changed since full70k training')
 if (out/'evaluation.json').exists():raise FileExistsError('preserve existing full70k evaluation; do not overwrite')
 val=np.load(VAL_RGB,mmap_mode='r')
 core,encoder,patcher,mean,std,clip,decoder=load_models(device,seed)
 core.load_state_dict(torch.load(out/'core.pt',map_location=device,weights_only=True),strict=True)
 encoder.load_state_dict(torch.load(out/'encoder.pt',map_location=device,weights_only=True),strict=True)
 encoder.eval();core.eval();gamma=[]
 with torch.no_grad():
  for start in range(0,320,16):
   im=read_batch(val,list(range(start,min(start+16,320))),device)
   gamma.append(encode(encoder,patcher,mean,std,clip,im).cpu())
 path=out/'gamma_validation.pt';torch.save(torch.cat(gamma),path)
 gm=out/'gamma_manifest.json';write(gm,{'source':'SW0106 full70k trained registered input encoder','image_ids':[1320,1639],
  'gamma_global_start':1320,'shape':[320,8,256],'encoder_sha256':sha(out/'encoder.pt'),'gamma_sha256':sha(path),
  'ground_truth_used_for_prediction':False})
 cmd=[sys.executable,str(ROOT/'collaborative_test/SW_0040_peer_transfer/evaluate.py'),'--checkpoint',str(out/'core.pt'),
  '--gamma-path',str(path),'--gamma-global-start','1320','--gamma-manifest',str(gm),'--dataset-path',str(DATASET),
  '--output-path',str(out/'evaluation.json'),'--start','1320','--count','320','--batch-size','8','--steps','1024','--settle','512',
  '--membrane-vth','.06','--min-group-size','2','--background','largest_component','--thresholds','.50',
  '--dendritic-projection','shared','--graph-spatial-decay','.35','--geodesic-steps','3','--geodesic-radius','1.5',
  '--geodesic-contrast','2','--geodesic-temperature','.5','--geodesic-cap','16','--kuramoto-backend','factorized','--device',str(device)]
 subprocess.run(cmd,check=True,cwd=ROOT)
 read_full320_metrics(out/'evaluation.json')
 (out/'COMPLETED').write_text('SW0106 full70k training and full320 actual-spike evaluation complete\n')
 return out/'evaluation.json'

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight','train','eval','full-train','full-eval'],required=True)
 p.add_argument('--seed',type=int,choices=[0,1,2],default=0);p.add_argument('--arm',choices=['candidate','control'],default='candidate')
 p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path,default=None)
 a=p.parse_args();torch.set_num_threads(2);seed_process(a.seed,torch.device(a.device))
 if a.stage=='preflight':
  output=a.output or ROOT/f'collaborative_test/SW_0106_spike_partition_rgb/results_archive/preflight_seed{a.seed}.json'
  result=preflight(a.seed,torch.device(a.device),output)
 elif a.stage=='train': result=train(a.seed,a.arm,torch.device(a.device))
 elif a.stage=='eval': result=evaluate(a.seed,a.arm,torch.device(a.device))
 elif a.stage=='full-train': result=full_train(a.seed,a.arm,torch.device(a.device))
 else: result=full_evaluate(a.seed,a.arm,torch.device(a.device))
 print(json.dumps({'stage':a.stage,'result':str(result)},default=str),flush=True)
if __name__=='__main__': main()
