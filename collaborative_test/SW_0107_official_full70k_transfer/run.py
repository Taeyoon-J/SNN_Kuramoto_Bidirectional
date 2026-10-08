"""One additional official CLEVR70k pass from matched SW0097 frozen checkpoints."""
import argparse,hashlib,json,math,subprocess,sys,time
from pathlib import Path
import h5py,numpy as np,torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'collaborative_test')]
from SW_0094_aligned_joint_pilot.run import ASSETS,DATASET,hparams
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.loss_function import UnsupervisedS2NetLoss
from snn_kuramoto_bidirectional.training.train_s2net_core import _forward_with_plv
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from snn_kuramoto_bidirectional.gamma_initializer import feature_maps_to_patch_gamma
from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_input_encoder

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results_archive';OUT=ROOT/'trained_models/SW0107_official_full70k_transfer'
PREP=HERE/'prepare_official.py';RGB=ROOT/'data/SW_0107_official_full70k_transfer/official_rgb_float16.h5'
GAMMA=ROOT/'data/SW_0107_official_full70k_transfer/official_gamma.pt'
POOL=ROOT/'data/SW_0107_official_full70k_transfer'
CONTROL=ROOT/'trained_models/SW0097_graph_adaptation'
STATS=ASSETS/'feature_preprocessing.pt';ENC=ASSETS/'input_encoder/input_layer_encoder.pt'
METRICS=('fg_ari','foreground_iou','matched_object_iou')

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def source(seed):return CONTROL/f'seed{seed}_positive_frozen/core.pt'
def criterion():return UnsupervisedS2NetLoss(spike_rate_weight=0.,spike_smooth_weight=0.,spike_diversity_weight=0.,
 structural_weight=0.,plv_bimodality_weight=6.,plv_balance_weight=10.,plv_coherence_weight=.5,
 plv_collapse_weight=1.,plv_target_density=.867,patch_grid_size=(16,16))
def affinity(core):
 c=core.last_component_spikes
 return spike_synchrony_affinity(c.mean(dim=1),c,settle=32)
def load_core(seed,device):
 hp=hparams('raw');hp.num_time_steps=64
 core=S2NetCore(hp.validate(),device=device).to(device);core.load_state_dict(torch.load(source(seed),map_location=device,weights_only=True),strict=True)
 core._detect_object_groups=lambda out,spikes:[[] for _ in range(spikes.size(0))]
 if core.graph_generator.uses_feedback or core.kuramoto.spike_pulse_gain is not None:raise AssertionError('SW0107 feedback/pulse changed')
 core.graph_generator.requires_grad_(False)
 return core
def pool_indices(seed):return torch.randperm(70000,generator=torch.Generator(device='cpu').manual_seed(117+seed))
def preflight(seed,device):
 if seed not in (0,1,2):raise ValueError('bad SW0107 preflight seed')
 report_path=HERE/'results_archive'/f'preflight_seed{seed}.json'
 if report_path.exists():raise FileExistsError(f'preserve existing SW0107 preflight: {report_path}')
 source_path=source(seed);source_sha=sha(source_path);exp=json.loads((POOL/'official_export_manifest.json').read_text())
 gm=json.loads((POOL/'official_gamma_manifest.json').read_text());frozen=json.loads((POOL/'frozen_sources.json').read_text())
 if exp.get('images')!=70000 or exp.get('source_indices')!=[0,69999] or exp.get('filter')!='none' or exp.get('skip')!=0:raise AssertionError('exact full70k export required')
 if gm.get('gamma_sha256')!=sha(GAMMA) or gm.get('rgb_export_sha256')!=sha(RGB):raise AssertionError('official gamma provenance mismatch')
 frozen_core={int(x['seed']):x['sha256'] for x in frozen.get('cores',[])}
 if (frozen.get('status')!='frozen_before_official_export' or frozen_core.get(seed)!=source_sha or
   frozen.get('encoder',{}).get('sha256')!=sha(ENC) or frozen.get('feature_stats',{}).get('sha256')!=sha(STATS) or
   exp.get('frozen_sources_sha256')!=sha(POOL/'frozen_sources.json') or gm.get('encoder_sha256')!=sha(ENC) or gm.get('feature_stats_sha256')!=sha(STATS)):
  raise AssertionError('seed source/encoder/stat differs from frozen pre-export contract')
 gamma=torch.load(GAMMA,map_location='cpu',weights_only=True,mmap=True)
 if tuple(gamma.shape)!=(70000,8,256):raise AssertionError('gamma cache shape mismatch')
 torch.manual_seed(117+seed)
 if device.type=='cuda':torch.cuda.manual_seed_all(117+seed)
 order=pool_indices(seed);core=load_core(seed,device);params=[p for p in core.parameters() if p.requires_grad]
 initial={k:v.detach().cpu().clone() for k,v in core.state_dict().items()};graph={k:v.detach().cpu().clone() for k,v in core.graph_generator.state_dict().items()}
 opt=torch.optim.Adam(params,lr=3e-5);lf=criterion();core.train();core.graph_generator.eval();g=gamma[order[:16]].to(device)
 _,_,_,plv,theta=_forward_with_plv(core,g,lf,32,'phase','mean');q=affinity(core);primary,_=lf(plv=plv,theta=theta);spike,_=lf(plv=q);loss=primary+5.*spike
 if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite preflight loss')
 opt.zero_grad(set_to_none=True);loss.backward();grads=[p.grad for p in params]
 if any(x is not None and not bool(torch.isfinite(x).all()) for x in grads):raise FloatingPointError('nonfinite preflight gradient')
 norm=math.sqrt(sum(float(x.detach().double().square().sum()) for x in grads if x is not None))
 if not math.isfinite(norm) or norm<=0:raise AssertionError('empty or nonfinite preflight gradient')
 torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
 if any(not bool(torch.isfinite(p).all()) for p in core.parameters()):raise FloatingPointError('nonfinite core parameter after SW0107 throwaway update')
 changed=[k for k,v in initial.items() if not torch.equal(v,core.state_dict()[k].detach().cpu())]
 graph_unchanged=all(torch.equal(v,core.graph_generator.state_dict()[k].detach().cpu()) for k,v in graph.items())
 if not graph_unchanged:raise AssertionError('frozen graph changed in SW0107 throwaway update')
 if not any(not k.startswith('graph_generator.') for k in changed):raise AssertionError('throwaway update changed no trainable core parameter')
 if sha(source_path)!=source_sha:raise AssertionError('source checkpoint changed during preflight')
 out={'status':'passed','seed':seed,'device':str(device),'source_core_sha256':source_sha,'official_export_sha256':sha(RGB),
  'official_gamma_sha256':sha(GAMMA),'first_batch_pool_indices':order[:16].tolist(),'shuffle_seed':117+seed,
  'batch_size':16,'time_steps':64,'settle':32,'loss':'phase primary + 5x positive-product actual-spike affinity',
  'loss_value':float(loss.detach()),'gradient_norm_preclip':norm,'gradient_finite':True,'throwaway_optimizer_step':True,
  'changed_core_keys':changed,'graph_bitwise_unchanged':True,'source_hash_unchanged':True,
  'encoder_stats_frozen_sha256':{'encoder':sha(ENC),'feature_stats':sha(STATS)},'ground_truth_used':False,'runner_sha256':sha(Path(__file__))}
 write(report_path,out);return out
def train(seed,device):
 if seed not in (0,1,2):raise ValueError('bad SW0107 seed')
 out=OUT/f'seed{seed}'
 if out.exists():raise FileExistsError(f'preserve existing SW0107 attempt: {out}')
 source_path=source(seed);source_sha=sha(source_path)
 pf=json.loads((HERE/'results_archive'/f'preflight_seed{seed}.json').read_text())
 if pf.get('status')!='passed' or pf.get('seed')!=seed or pf.get('source_core_sha256')!=source_sha or pf.get('shuffle_seed')!=117+seed or pf.get('graph_bitwise_unchanged') is not True:
  raise AssertionError('matching passed SW0107 B16 preflight required before training')
 exp=json.loads((POOL/'official_export_manifest.json').read_text());gm=json.loads((POOL/'official_gamma_manifest.json').read_text())
 if exp.get('images')!=70000 or exp.get('source_indices')!=[0,69999] or exp.get('filter')!='none' or exp.get('skip')!=0:raise AssertionError('SW0107 export is not exact unfiltered first70k')
 if gm.get('gamma_sha256')!=sha(GAMMA) or gm.get('rgb_export_sha256')!=sha(RGB):raise AssertionError('official gamma cache provenance mismatch')
 if gm.get('encoder_sha256')!=sha(ENC) or gm.get('feature_stats_sha256')!=sha(STATS):raise AssertionError('registered frozen encoder/stats changed')
 frozen=json.loads((POOL/'frozen_sources.json').read_text())
 frozen_core={int(x['seed']):x['sha256'] for x in frozen.get('cores',[])}
 if frozen.get('status')!='frozen_before_official_export' or frozen_core.get(seed)!=source_sha or frozen.get('encoder',{}).get('sha256')!=sha(ENC) or frozen.get('feature_stats',{}).get('sha256')!=sha(STATS):
  raise AssertionError('SW0097 source/encoder/stat artifacts differ from pre-export freeze')
 if exp.get('frozen_sources_sha256')!=sha(POOL/'frozen_sources.json'):
  raise AssertionError('official export does not reference exact frozen-source manifest')
 gamma=torch.load(GAMMA,map_location='cpu',weights_only=True,mmap=True)
 if tuple(gamma.shape)!=(70000,8,256):raise AssertionError('official gamma shape mismatch')
 torch.manual_seed(117+seed)
 if device.type=='cuda':torch.cuda.manual_seed_all(117+seed)
 core=load_core(seed,device);params=[p for p in core.parameters() if p.requires_grad]
 if not params or any(p.requires_grad for p in core.graph_generator.parameters()):raise AssertionError('core/graph trainability contract mismatch')
 opt=torch.optim.Adam(params,lr=3e-5);lossfn=criterion();order=pool_indices(seed)
 inds=np.arange(70000,dtype='<i8');targetids=inds.tolist()
 out.mkdir(parents=True,exist_ok=False);started=time.time();hist=[]
 initial_graph={k:v.detach().cpu().clone() for k,v in core.graph_generator.state_dict().items()}
 core.train();core.graph_generator.eval()
 for step in range(4375):
  ix=order[step*16:(step+1)*16];g=gamma[ix].to(device)
  _,spikes,_,plv,theta=_forward_with_plv(core,g,lossfn,32,'phase','mean')
  q=affinity(core);primary,_=lossfn(plv=plv,theta=theta);spike,_=lossfn(plv=q);total=primary+5.*spike
  if not bool(torch.isfinite(total)):raise FloatingPointError(f'nonfinite SW0107 loss step{step}')
  opt.zero_grad(set_to_none=True);total.backward()
  norm=torch.nn.utils.clip_grad_norm_(params,1.)
  if not bool(torch.isfinite(norm)) or float(norm)<=0:raise FloatingPointError(f'invalid SW0107 gradient step{step}')
  opt.step();hist.append({'update':step+1,'loss':float(total.detach()),'primary':float(primary.detach()),
   'positive_product_spike':float(spike.detach()),'grad_norm_preclip':float(norm)})
  if step==0 or (step+1)%128==0:write(out/'progress.json',{'status':'training','step':step+1,'total':4375,'updated':time.time()})
 if any(not torch.equal(v,core.graph_generator.state_dict()[k].detach().cpu()) for k,v in initial_graph.items()):raise AssertionError('frozen SW0097 graph changed')
 torch.save(core.state_dict(),out/'core.pt');write(out/'history.json',hist)
 write(out/'manifest.json',{'status':'training_complete','seed':seed,'source':'SW0097 positive_frozen whole completed checkpoint',
  'source_core':str(source_path),'source_core_sha256':source_sha,'registered_encoder_sha256':sha(ENC),'feature_preprocessing_sha256':sha(STATS),
  'official_rgb_export_sha256':sha(RGB),'official_gamma_sha256':sha(GAMMA),'tfds_source_indices':[0,69999],
  'training_unique_images':70000,'updates':4375,'batch_size':16,'learning_rate':3e-5,'gradient_clip':1.,
  'train_time_steps':64,'settle':32,'shuffle_seed':117+seed,'training_order_sha256':hashlib.sha256(np.asarray(order,dtype='<i8').tobytes()).hexdigest(),
  'training_ids_sha256':hashlib.sha256(np.asarray(targetids,dtype='<i8').tobytes()).hexdigest(),'objective':'phase primary + 5x positive-product actual-spike affinity',
  'encoder_frozen':True,'graph_frozen':True,'optimizer_initialization':'fresh Adam; SW0097 weights continued, no optimizer moment artifact claimed',
  'ground_truth_used_for_training':False,'started':started,'completed':time.time(),'runner_sha256':sha(Path(__file__))})
 (out/'TRAINING_COMPLETED').write_text('one additional exact70000 official TFDS pass complete\n')
 return out

def native_val_gamma(seed,device):
 stats=torch.load(STATS,map_location=device,weights_only=True);mean,std=stats['mean'].to(device),stats['std'].to(device);clip=float(stats.get('clip',3.))
 encoder=load_input_encoder(str(ENC),num_kernels=8,kernel_size=3,channels=3,device=device).eval();rows=[]
 with h5py.File(DATASET,'r') as h,torch.no_grad():
  for start in range(1320,1640,16):
   x=torch.from_numpy(np.asarray(h['image'][start:start+16]).copy()).permute(0,3,1,2).to(device,dtype=torch.float32)
   f=encoder(x/255.);rows.append(feature_maps_to_patch_gamma(((f-mean)/std).clamp(-clip,clip),grid_size=16,device=device).float().cpu())
 return torch.cat(rows)
def evaluate(seed,device):
 out=OUT/f'seed{seed}'
 if not (out/'TRAINING_COMPLETED').is_file():raise FileNotFoundError('SW0107 trained checkpoint required')
 if (out/'evaluation.json').exists():raise FileExistsError('preserve existing SW0107 evaluation')
 m=json.loads((out/'manifest.json').read_text())
 if sha(source(seed))!=m['source_core_sha256'] or sha(ENC)!=m['registered_encoder_sha256'] or sha(STATS)!=m['feature_preprocessing_sha256']:
  raise AssertionError('frozen SW0097 source or encoder preprocessing changed')
 core=load_core(seed,device);core.load_state_dict(torch.load(out/'core.pt',map_location=device,weights_only=True),strict=True);core.eval()
 g=native_val_gamma(seed,device);gp=out/'gamma_validation.pt';torch.save(g,gp)
 gm=out/'gamma_validation_manifest.json';write(gm,{'source':'frozen SW0097 registered encoder on native CLEVR validation','ids':[1320,1639],
  'shape':list(g.shape),'encoder_sha256':sha(ENC),'feature_preprocessing_sha256':sha(STATS),'gamma_sha256':sha(gp),'ground_truth_used_for_prediction':False})
 cmd=[sys.executable,str(ROOT/'collaborative_test/SW_0040_peer_transfer/evaluate.py'),'--checkpoint',str(out/'core.pt'),
  '--gamma-path',str(gp),'--gamma-global-start','1320','--gamma-manifest',str(gm),'--dataset-path',str(DATASET),
  '--output-path',str(out/'evaluation.json'),'--start','1320','--count','320','--batch-size','8','--steps','1024','--settle','512',
  '--membrane-vth','.06','--min-group-size','2','--background','largest_component','--thresholds','.50','--dendritic-projection','shared',
  '--graph-spatial-decay','.35','--geodesic-steps','3','--geodesic-radius','1.5','--geodesic-contrast','2','--geodesic-temperature','.5',
  '--geodesic-cap','16','--kuramoto-backend','factorized','--gate-mode','raw','--device',str(device)]
 subprocess.run(cmd,cwd=ROOT,check=True)
 d=json.loads((out/'evaluation.json').read_text());row=d['sweep'][0]['scored_targets']['our_hdf5']
 if d.get('ids')!=[1320,1639] or d.get('ground_truth_used_for_prediction') is not False or any(row['valid_count'].get(k)!=320 for k in METRICS):raise AssertionError('invalid SW0107 full320 output')
 (out/'COMPLETED').write_text('SW0107 full320 actual-spike evaluation complete\n')
 return out/'evaluation.json'
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight','train','eval'],required=True);p.add_argument('--seed',type=int,choices=[0,1,2],required=True);p.add_argument('--device',default='cuda:0')
 a=p.parse_args();torch.set_num_threads(2);d=torch.device(a.device)
 result=preflight(a.seed,d) if a.stage=='preflight' else train(a.seed,d) if a.stage=='train' else evaluate(a.seed,d)
 print(json.dumps({'stage':a.stage,'seed':a.seed,'result':str(result)}),flush=True)
if __name__=='__main__':main()
