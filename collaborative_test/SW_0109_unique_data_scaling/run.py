"""Matched-compute, nested unique-image scaling from the audited 2,500-image cores."""
import argparse,hashlib,json,math,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'collaborative_test')]
from SW_0094_aligned_joint_pilot.run import ASSETS,hparams
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.loss_function import UnsupervisedS2NetLoss
from snn_kuramoto_bidirectional.training.train_s2net_core import _forward_with_plv
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity

ARCHIVE=HERE/'results_archive';OUT=ROOT/'trained_models/SW0109_unique_data_scaling'
ENCODER=ASSETS/'input_encoder/input_layer_encoder.pt';FEATURE_STATS=ASSETS/'feature_preprocessing.pt'
AUDIT=HERE/'source_audit.json';PROTOCOL=HERE/'protocol.json'
SIZES=(2500,10000,70000);SEEDS=(0,1,2);BATCH=16;UPDATES=4375;EXPOSURES=BATCH*UPDATES
METRICS=('fg_ari','foreground_iou','matched_object_iou')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def write(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def pool_ids(size,extra_seed=108):
 if size not in SIZES:raise ValueError(f'unregistered unique-image pool size {size}')
 original=np.concatenate((np.arange(1000,dtype=np.int64),np.arange(1640,3140,dtype=np.int64)))
 remaining=np.arange(3140,70640,dtype=np.int64)
 extra=np.random.default_rng(extra_seed).permutation(remaining)[:size-len(original)]
 ids=np.concatenate((original,extra))
 if len(np.unique(ids))!=size:raise AssertionError('pool contains duplicate image IDs')
 return ids
def row_indices(global_ids):
 ids=np.asarray(global_ids,dtype=np.int64)
 if np.any((ids<0)|(ids>=1000)&((ids<1640)|(ids>=70640))):raise ValueError('ID outside the registered source pool')
 return np.where(ids<1000,ids,1000+ids-1640).astype(np.int64)
def training_order(seed,size):
 if seed not in SEEDS:raise ValueError('seed must be registered 0,1,2')
 pool=pool_ids(size)
 reps,rem=divmod(EXPOSURES,size)
 if rem:raise AssertionError('registered pool sizes must tile 70,000 exposures without a partial pool cycle')
 rng=np.random.default_rng(117+seed)
 order=np.concatenate([rng.permutation(pool) for _ in range(reps)])
 if len(order)!=EXPOSURES or len(order)%BATCH:raise AssertionError('training order is not exact full-batch exposure budget')
 return pool,order
def nested_pool_summary():
 pools={n:set(pool_ids(n).tolist()) for n in SIZES}
 if not pools[2500]<=pools[10000]<=pools[70000]:raise AssertionError('data pools are not nested')
 return {str(n):{'unique_ids':len(pools[n]),'pool_sha256':hashlib.sha256(np.asarray(sorted(pools[n]),dtype='<i8').tobytes()).hexdigest()} for n in SIZES}
def source_entry(seed):
 audit=json.loads(AUDIT.read_text())
 if (audit.get('status')!='source_provenance_verified' or audit.get('source_training_ids_match_original2500') is not True
  or audit.get('original_training_segments')!=[[0,999],[1640,3139]]):raise AssertionError('source_audit is not verified for the registered nested pool')
 match=[x for x in audit.get('sources',[]) if x.get('seed')==seed]
 if len(match)!=1:raise AssertionError(f'missing unique source audit entry for seed{seed}')
 item=match[0];core=Path(item['core_path']);manifest=Path(item['manifest_path'])
 if item.get('downstream_unique_images')!=2500:raise AssertionError(f'seed{seed} is not an audited 2500-unique-image initialization')
 if seed==0 and item.get('recorded_exposures')!=25000:raise AssertionError('seed0 pre-large-data exposure record mismatch')
 if seed>0 and item.get('training_completed_marker') is not True:raise AssertionError(f'seed{seed} source completion marker is not recorded')
 if not core.is_file() or sha(core)!=item.get('core_sha256'):raise AssertionError(f'seed{seed} source core SHA mismatch')
 if not manifest.is_file() or sha(manifest)!=item.get('manifest_sha256'):raise AssertionError(f'seed{seed} source manifest SHA mismatch')
 marker=core.parent/'TRAINING_COMPLETED'
 if item.get('training_completed_marker') is True and not marker.is_file():raise FileNotFoundError(marker)
 return item,core,manifest
def validate_gamma_manifest(m,gamma_sha,validation=False):
 if m.get('gamma_sha256')!=gamma_sha:raise AssertionError('gamma manifest SHA mismatch')
 preprocess=m.get('preprocessing_sha256',m.get('feature_stats_sha256',m.get('feature_preprocessing_sha256')))
 if preprocess!=sha(FEATURE_STATS):raise AssertionError('gamma cache does not use common frozen registered preprocessing')
 encoder=m.get('encoder_sha256',m.get('input_encoder_sha256'))
 if encoder!=sha(ENCODER):raise AssertionError('gamma cache does not use common frozen registered encoder')
 if validation:
  if m.get('image_ids')!=[1320,1639]:raise AssertionError('validation gamma manifest IDs mismatch')
  return {'kind':'validation','image_ids':[1320,1639],'count':320}
 ids=m.get('training_ids');segments=[[0,999],[1640,70639]]
 if not isinstance(ids,dict) or ids.get('segments')!=segments or ids.get('count')!=70000:
  raise AssertionError('70k gamma manifest training_ids must exactly declare registered inclusive segments and count')
 expected=np.concatenate((np.arange(1000,dtype='<i8'),np.arange(1640,70640,dtype='<i8')))
 mapping_sha=hashlib.sha256(expected.tobytes()).hexdigest()
 declared=ids.get('mapping_sha256',ids.get('ids_sha256'))
 if declared is not None and declared!=mapping_sha:raise AssertionError('70k gamma explicit ID mapping SHA mismatch')
 return {'kind':'training','segments':segments,'count':70000,'mapping_sha256':mapping_sha}
def gamma_contract(gamma_path,manifest_path,validation=False):
 gpath=Path(gamma_path);mpath=Path(manifest_path)
 if not gpath.is_file() or not mpath.is_file():raise FileNotFoundError(gpath if not gpath.is_file() else mpath)
 m=json.loads(mpath.read_text());blob=torch.load(gpath,map_location='cpu',weights_only=True,mmap=True)
 expected=(320,8,256) if validation else (70000,8,256)
 if tuple(blob.shape)!=expected:raise AssertionError(f'gamma shape {tuple(blob.shape)} != {expected}')
 validate_gamma_manifest(m,sha(gpath),validation)
 return blob,m
def criterion():
 return UnsupervisedS2NetLoss(spike_rate_weight=0.,spike_smooth_weight=0.,spike_diversity_weight=0.,
  structural_weight=0.,plv_bimodality_weight=6.,plv_balance_weight=10.,plv_coherence_weight=.5,
  plv_collapse_weight=1.,plv_target_density=.867,patch_grid_size=(16,16))
def load_core(seed,checkpoint,device):
 hp=hparams('raw');hp.num_time_steps=64;core=S2NetCore(hp.validate(),device=device).to(device)
 core.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True),strict=True)
 core._detect_object_groups=lambda out,spikes:[[] for _ in range(spikes.size(0))]
 core.graph_generator.requires_grad_(False)
 if core.graph_generator.uses_feedback or core.kuramoto.spike_pulse_gain is not None:raise AssertionError('graph feedback/pulse differs from registered fixed source')
 return core
def affinity(core):
 c=core.last_component_spikes
 if c is None or tuple(c.shape[1:])!=(4,256,64):raise AssertionError('actual component spike history shape mismatch')
 return spike_synchrony_affinity(c.mean(dim=1),c,settle=32)
def primary_loss(core,gamma,lossfn):
 _,_,_,plv,theta=_forward_with_plv(core,gamma,lossfn,32,'phase','mean')
 q=affinity(core);primary,_=lossfn(plv=plv,theta=theta);spike,_=lossfn(plv=q)
 return primary+5.*spike,primary,spike
def validate_data_paths(train_gamma,train_manifest,val_gamma,val_manifest):
 tg,_=gamma_contract(train_gamma,train_manifest,False);vg,_=gamma_contract(val_gamma,val_manifest,True)
 return tg,vg
def validate_source(seed):return source_entry(seed)
def validate_preflight_record(pf,seed,source_sha,train_gamma_sha=None,train_manifest_sha=None):
 if (pf.get('status')!='passed' or pf.get('seed')!=seed or pf.get('source_core_sha256')!=source_sha
  or set(pf.get('pool_checks',{}))!={str(n) for n in SIZES}
  or pf.get('encoder_sha256')!=sha(ENCODER) or pf.get('feature_preprocessing_sha256')!=sha(FEATURE_STATS)
  or pf.get('runner_sha256')!=sha(Path(__file__))):return False
 if train_gamma_sha is not None and pf.get('train_gamma_sha256')!=train_gamma_sha:return False
 if train_manifest_sha is not None and pf.get('train_gamma_manifest_sha256')!=train_manifest_sha:return False
 for size in SIZES:
  check=pf['pool_checks'][str(size)];batches=check.get('batches',[])
  if (check.get('status')!='passed' or check.get('pool_size')!=size or check.get('unique_images_in_pool')!=size
   or check.get('graph_bitwise_unchanged') is not True or len(batches)!=4):return False
  for batch in batches:
   if (batch.get('finite_gradient') is not True or batch.get('throwaway_adam_update') is not True
    or not math.isfinite(float(batch.get('gradient_norm_preclip',float('nan')))) or float(batch['gradient_norm_preclip'])<=0):return False
 return True
def preflight(seed,device,train_gamma,train_manifest,output):
 if Path(output).exists():raise FileExistsError(f'preserve existing preflight attempt {output}')
 entry,source,_=source_entry(seed);gamma,_=gamma_contract(train_gamma,train_manifest)
 pool_checks={}
 for size in SIZES:
  pool,order=training_order(seed,size);torch.manual_seed(117+seed)
  if device.type=='cuda':torch.cuda.manual_seed_all(117+seed)
  core=load_core(seed,source,device);params=[p for p in core.parameters() if p.requires_grad]
  if not params or any(p.requires_grad for p in core.graph_generator.parameters()):raise AssertionError('trainable/frozen module contract mismatch')
  graph={k:v.detach().clone() for k,v in core.graph_generator.state_dict().items()}
  initial={k:v.detach().clone() for k,v in core.state_dict().items()}
  opt=torch.optim.Adam(params,lr=3e-5);lossfn=criterion();core.train();core.graph_generator.eval();batches=[]
  for batch_index in range(4):
   ids=order[batch_index*BATCH:(batch_index+1)*BATCH];idx=row_indices(ids)
   loss,primary,spike=primary_loss(core,gamma[torch.as_tensor(idx)].to(device),lossfn)
   if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite B16 scaling preflight loss size={size} batch={batch_index}')
   opt.zero_grad(set_to_none=True);loss.backward();grads=[p.grad for p in params]
   if any(g is not None and not torch.isfinite(g).all() for g in grads):raise FloatingPointError(f'nonfinite scaling preflight gradient size={size} batch={batch_index}')
   norm=math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads if g is not None))
   if not math.isfinite(norm) or norm<=0:raise AssertionError(f'empty scaling preflight gradient size={size} batch={batch_index}')
   torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
   batches.append({'batch':batch_index,'global_ids':ids.tolist(),'gamma_rows':idx.tolist(),
    'loss':float(loss.detach()),'primary_loss':float(primary.detach()),'positive_actual_spike_product_unweighted':float(spike.detach()),
    'gradient_norm_preclip':norm,'finite_gradient':True,'throwaway_adam_update':True})
  changed=[k for k,v in initial.items() if not torch.equal(v,core.state_dict()[k])]
  if not any(not k.startswith('graph_generator.') for k in changed):raise AssertionError(f'preflight changed no trainable core parameter size={size}')
  if any(not torch.equal(v,core.graph_generator.state_dict()[k]) for k,v in graph.items()):raise AssertionError(f'preflight changed fixed learned graph size={size}')
  pool_checks[str(size)]={'status':'passed','pool_size':size,'unique_images_in_pool':int(np.unique(pool).size),
   'pool_sha256':hashlib.sha256(np.asarray(sorted(pool),dtype='<i8').tobytes()).hexdigest(),
   'batches':batches,'changed_core_keys':changed,'graph_bitwise_unchanged':True}
 if sha(source)!=entry['core_sha256']:raise AssertionError('source checkpoint changed during preflight')
 rec={'status':'passed','seed':seed,'source_core_path':str(source),'source_core_sha256':entry['core_sha256'],
  'source_manifest_sha256':entry['manifest_sha256'],'train_gamma_sha256':sha(train_gamma),'train_gamma_manifest_sha256':sha(train_manifest),
  'encoder_sha256':sha(ENCODER),'feature_preprocessing_sha256':sha(FEATURE_STATS),'pool_checks':pool_checks,
  'batch_size':BATCH,'steps':64,'settle':32,'ground_truth_used':False,
  'runner_sha256':sha(Path(__file__))}
 write(output,rec);return rec
def train(seed,size,device,train_gamma,train_manifest,output):
 out=Path(output)
 if out.exists():raise FileExistsError(f'preserve prior scaling output without overwrite: {out}')
 entry,source,_=source_entry(seed);pf=json.loads((ARCHIVE/f'preflight_seed{seed}.json').read_text())
 if not validate_preflight_record(pf,seed,entry['core_sha256'],sha(train_gamma),sha(train_manifest)):
  raise AssertionError('source/cache-matched all-pool B16 preflight required')
 gamma,gm=gamma_contract(train_gamma,train_manifest);pool,order=training_order(seed,size);rows=row_indices(order)
 torch.manual_seed(117+seed)
 if device.type=='cuda':torch.cuda.manual_seed_all(117+seed)
 core=load_core(seed,source,device);params=[p for p in core.parameters() if p.requires_grad]
 opt=torch.optim.Adam(params,lr=3e-5);lossfn=criterion();graph_initial={k:v.detach().clone() for k,v in core.graph_generator.state_dict().items()}
 hist=[];started=time.time();out.mkdir(parents=True,exist_ok=False);core.train();core.graph_generator.eval()
 for step in range(UPDATES):
  ix=rows[step*BATCH:(step+1)*BATCH];g=gamma[torch.as_tensor(ix)].to(device)
  total,primary,spike=primary_loss(core,g,lossfn)
  if not torch.isfinite(total):raise FloatingPointError(f'nonfinite scaling loss step={step+1} size={size} seed={seed}')
  opt.zero_grad(set_to_none=True);total.backward()
  norm=torch.nn.utils.clip_grad_norm_(params,1.)
  if not torch.isfinite(norm) or float(norm)<=0:raise FloatingPointError(f'nonfinite/empty scaling gradient step={step+1}')
  opt.step();hist.append({'update':step+1,'loss':float(total.detach()),'primary':float(primary.detach()),
   'positive_actual_spike_product_unweighted':float(spike.detach()),'gradient_norm_preclip':float(norm)})
  if step==255:torch.save(core.state_dict(),out/'prefix_256_core.pt')
  if step==0 or (step+1)%128==0:write(out/'progress.json',{'status':'training','seed':seed,'unique_images':size,'update':step+1,'total_updates':UPDATES,'updated':time.time()})
 if len(hist)!=UPDATES:raise AssertionError('wrong update count')
 if any(not torch.equal(v,core.graph_generator.state_dict()[k]) for k,v in graph_initial.items()):raise AssertionError('frozen graph changed')
 if any(not torch.isfinite(p).all() for p in core.parameters()):raise FloatingPointError('nonfinite final parameter')
 torch.save(core.state_dict(),out/'core.pt');write(out/'history.json',hist)
 exposure_counts={n:int(np.unique(order[:n]).size) for n in (BATCH*256,EXPOSURES)}
 write(out/'manifest.json',{'status':'training_complete','seed':seed,'unique_image_count':size,
  'pool_unique_sha256':hashlib.sha256(np.asarray(sorted(pool),dtype='<i8').tobytes()).hexdigest(),
  'training_order_sha256':hashlib.sha256(np.asarray(order,dtype='<i8').tobytes()).hexdigest(),
  'training_image_ids_sha256':hashlib.sha256(np.asarray(order,dtype='<i8').tobytes()).hexdigest(),
  'exposures':EXPOSURES,'updates':UPDATES,'batch_size':BATCH,'train_steps':64,'settle':32,
  'prefix_checkpoint_update':256,'prefix_unique_images':exposure_counts[str(BATCH*256)],
  'full_unique_images_seen':exposure_counts[str(EXPOSURES)],'pool_cycles':EXPOSURES//size,
  'pool_segment_contract':[[0,999],[1640,70639]],'remaining_pool_selection_seed':108,
  'training_order_seed':117+seed,'learning_rate':3e-5,'gradient_clip':1.,'optimizer':'fresh Adam per condition',
  'cycle_shuffle':'independent without-replacement permutation of the selected pool for each complete cycle',
  'objective':'phase primary + 5x positive-product actual-spike affinity','graph_frozen':True,'encoder_frozen':True,
  'source_core_path':str(source),'source_core_sha256':entry['core_sha256'],'source_manifest_sha256':entry['manifest_sha256'],
  'train_gamma_sha256':sha(train_gamma),'train_gamma_manifest_sha256':sha(train_manifest),
  'encoder_sha256':sha(ENCODER),'feature_preprocessing_sha256':sha(FEATURE_STATS),
  'ground_truth_used_for_training':False,'best_epoch_selection':False,'started':started,'completed':time.time(),
  'core_sha256':sha(out/'core.pt'),'runner_sha256':sha(Path(__file__))})
 (out/'TRAINING_COMPLETED').write_text('fixed 70,000-exposure unique-image scaling condition complete\n')
 return out
def evaluate(seed,size,checkpoint,device,val_gamma,val_manifest,dataset,output):
 out=Path(output)
 if out.exists():raise FileExistsError(f'preserve prior evaluation output: {out}')
 gamma,gm=gamma_contract(val_gamma,val_manifest,validation=True)
 if not Path(checkpoint).is_file():raise FileNotFoundError(checkpoint)
 report=out/'evaluation.json';out.mkdir(parents=True,exist_ok=False)
 cmd=[sys.executable,str(ROOT/'collaborative_test/SW_0040_peer_transfer/evaluate.py'),'--checkpoint',str(checkpoint),
  '--gamma-path',str(val_gamma),'--gamma-global-start','1320','--gamma-manifest',str(val_manifest),'--dataset-path',str(dataset),
  '--output-path',str(report),'--start','1320','--count','320','--batch-size','8','--steps','1024','--settle','512',
  '--membrane-vth','.06','--min-group-size','2','--background','largest_component','--thresholds','.50',
  '--dendritic-projection','shared','--graph-spatial-decay','.35','--geodesic-steps','3','--geodesic-radius','1.5',
  '--geodesic-contrast','2','--geodesic-temperature','.5','--geodesic-cap','16','--kuramoto-backend','factorized','--gate-mode','raw','--device',str(device)]
 subprocess.run(cmd,cwd=ROOT,check=True)
 d=json.loads(report.read_text());score=d['sweep'][0]['scored_targets']['our_hdf5']
 if d.get('ids')!=[1320,1639] or d.get('images')!=320 or d.get('ground_truth_used_for_prediction') is not False:raise AssertionError('evaluation contract mismatch')
 for key in METRICS:
  vals=score.get('per_image',{}).get(key,[])
  if score.get('valid_count',{}).get(key)!=320 or len(vals)!=320 or not all(math.isfinite(float(x)) for x in vals):raise AssertionError(f'invalid per-image {key}')
  if not math.isfinite(float(score.get('metrics',{}).get(key,float('nan')))):raise AssertionError(f'invalid metric mean {key}')
 write(out/'evaluation_manifest.json',{'status':'complete','seed':seed,'pool_size':size,'checkpoint':str(Path(checkpoint).resolve()),
  'checkpoint_sha256':sha(checkpoint),'validation_gamma_sha256':sha(val_gamma),'validation_gamma_manifest_sha256':sha(val_manifest),
  'dataset_path':str(Path(dataset).resolve()),'ids':[1320,1639],'images':320,'ground_truth_used_for_prediction':False,
  'metrics':{k:float(score['metrics'][k]) for k in METRICS},'valid_count':{k:int(score['valid_count'][k]) for k in METRICS},
  'runner_sha256':sha(Path(__file__))})
 (out/'COMPLETED').write_text('same native320 full validation complete\n')
 return report
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight','train','eval'],required=True)
 p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--size',type=int,choices=SIZES,default=2500)
 p.add_argument('--device',default='cuda:0');p.add_argument('--train-gamma',type=Path,required=True);p.add_argument('--train-manifest',type=Path,required=True)
 p.add_argument('--val-gamma',type=Path,required=True);p.add_argument('--val-manifest',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
 p.add_argument('--checkpoint',type=Path,default=None);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(2);dev=torch.device(a.device)
 if a.stage=='preflight':res=preflight(a.seed,dev,a.train_gamma,a.train_manifest,a.output)
 elif a.stage=='train':res=train(a.seed,a.size,dev,a.train_gamma,a.train_manifest,a.output)
 else:
  checkpoint=a.checkpoint
  if checkpoint is None:checkpoint=OUT/f'seed{a.seed}_N{a.size}/core.pt'
  res=evaluate(a.seed,a.size,checkpoint,dev,a.val_gamma,a.val_manifest,a.dataset,a.output)
 print(json.dumps({'stage':a.stage,'seed':a.seed,'size':a.size,'result':str(res)},default=str),flush=True)
if __name__=='__main__':main()
