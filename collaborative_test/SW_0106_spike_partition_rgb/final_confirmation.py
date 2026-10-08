"""Freeze SW0106 and matched Slot assets, then score only reserved90640-90959."""
import argparse,csv,hashlib,json,math,os,subprocess,sys
from pathlib import Path

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results_archive'
OUT=ROOT/'trained_models/SW0106_spike_partition_rgb'
DATASET=Path('/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5')
SLOT_ROOT=Path('/Data0/kevinswk/patch_v2_sw/trained_models')
SLOT_MODEL=Path('/Data0/kevinswk/patch_v2/trained_models/slot_attention_official_patch_eval_20261002/model.py')
SLOT_PREDICT=ROOT/'collaborative_test/SW_0092_cross_dataset_training/slot_predict_gpu.py'
SLOT_SCORE=ROOT/'collaborative_test/SW_0092_cross_dataset_training/score_predictions.py'
TF_PY='/Data0/kevinswk/envs/slot_attention_gpu_tf215/bin/python'
RUNNER=HERE/'run.py'
EVALUATOR=ROOT/'collaborative_test/SW_0040_peer_transfer/evaluate.py'
FINAL_DIR=ARCHIVE/'final_reserve'
FREEZE=FINAL_DIR/'frozen_models_manifest.json'
START=90640;COUNT=320;END=START+COUNT-1
METRICS=('fg_ari','foreground_iou','matched_object_iou')
BOOTSTRAP_REPLICATES=10000;BOOTSTRAP_SEED=1060106

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def readj(path):return json.loads(Path(path).read_text())
def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def canonical_sha(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def full_metric(path,ids=(1320,1639)):
 d=readj(path)
 if d.get('images')!=320 or d.get('ids')!=list(ids) or d.get('ground_truth_used_for_prediction') is not False:
  raise ValueError(f'invalid no-GT full320 evaluation: {path}')
 row=d.get('sweep',[{}])[0].get('scored_targets',{}).get('our_hdf5')
 if row is None:raise ValueError(f'missing our_hdf5 result: {path}')
 result={}
 for m in METRICS:
  x=np.asarray(row.get('per_image',{}).get(m,[]),dtype=np.float64)
  if x.shape!=(320,) or int(row.get('valid_count',{}).get(m,-1))!=320 or not np.isfinite(x).all():
   raise ValueError(f'invalid {m} per-image vector: {path}')
  avg=float(x.mean());recorded=float(row.get('metrics',{}).get(m,float('nan')))
  if not math.isfinite(recorded) or abs(avg-recorded)>1e-12:raise ValueError(f'{m} aggregate mismatch: {path}')
  result[m]={'mean':recorded,'per_image':x}
 return d,result
def slot_folder(seed):return SLOT_ROOT/(f'SW0092_slot_our70000_s{seed}_final' if seed==0 else f'SW0092_slot_our70000_s{seed}')
def verify_freeze_payload(d):
 expected=d.get('frozen_payload_sha256');body=dict(d);body.pop('frozen_payload_sha256',None)
 if not expected or canonical_sha(body)!=expected:raise AssertionError('frozen reserve manifest digest mismatch')
 if d.get('reserve_read_started') is not False or d.get('reserve_ids')!=[START,END]:raise AssertionError('freeze manifest has invalid reserve state')
 for group in ('sw0106_full_models','slot_checkpoints','protocol_files','code'):
  for entry in d.get(group,[]):
   if sha(entry['path'])!=entry['sha256']:raise AssertionError(f"frozen input changed after preregistration: {entry['path']}")
 return d

def freeze():
 if FREEZE.exists():raise FileExistsError(f'preserve prior freeze manifest: {FREEZE}')
 gate=readj(ARCHIVE/'full70k_reserve_gate.json')
 if gate.get('status')!='passed':raise RuntimeError('full70k reserve unlock gate has not passed')
 files={'sw0106_full_models':[],'slot_checkpoints':[],'protocol_files':[],'code':[]}
 sys.path.insert(0,str(ROOT/'collaborative_test'))
 from SW_0094_aligned_joint_pilot.run import ASSETS
 stats=ASSETS/'feature_preprocessing.pt'
 stats_sha=sha(stats)
 for arm in ('candidate','control'):
  for seed in range(3):
   folder=OUT/f'full70k_{arm}_seed{seed}'
   m=readj(folder/'manifest.json')
   if m.get('status')!='full70k_training_complete' or m.get('full_updates')!=4375 or m.get('full_images')!=70000:
    raise AssertionError(f'full70k model incomplete before reserve freeze: {folder}')
   if m.get('feature_preprocessing_sha256')!=stats_sha:raise AssertionError('full model did not retain the registered preprocessing SHA')
   for name in ('core.pt','encoder.pt'):
    p=folder/name;files['sw0106_full_models'].append({'arm':arm,'seed':seed,'kind':name,'path':str(p),'sha256':sha(p)})
   files['sw0106_full_models'].append({'arm':arm,'seed':seed,'kind':'registered_feature_preprocessing','path':str(stats),'sha256':stats_sha})
   full_metric(folder/'evaluation.json')
 for seed in range(3):
  folder=slot_folder(seed);prefix=folder/'checkpoint'/'ckpt-21880'
  if not Path(str(prefix)+'.index').is_file():raise FileNotFoundError(f'frozen Slot checkpoint index missing: {prefix}.index')
  shards=sorted(prefix.parent.glob(prefix.name+'.*'))
  if not shards:raise FileNotFoundError(f'Slot checkpoint shards missing: {prefix}')
  for p in shards:files['slot_checkpoints'].append({'seed':seed,'path':str(p),'sha256':sha(p)})
  protocol=folder/'training_protocol.json';pd=readj(protocol)
  if pd.get('seed')!=seed or pd.get('unique_training_images')!=70000 or pd.get('ground_truth_used_for_training') is not False or pd.get('epochs')!=10 or pd.get('training_id_segments_inclusive')!=[[0,999],[1640,70639]]:
   raise AssertionError(f'Slot training protocol mismatch for seed{seed}')
  files['protocol_files'].append({'seed':seed,'kind':'slot_training_protocol','path':str(protocol),'sha256':sha(protocol)})
 # Inspect only HDF5 metadata; no image or mask row/payload is read before freeze.
 st=DATASET.stat()
 with h5py.File(DATASET,'r') as h5:
  root_keys=sorted(list(h5.keys()))
  datasets={}
  for name in ('image','mask'):
   ds=h5[name];datasets[name]={'shape':list(ds.shape),'dtype':str(ds.dtype),'chunks':list(ds.chunks) if ds.chunks else None,'compression':ds.compression}
 data_meta={'path':str(DATASET),'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns,
  'root_keys':root_keys,'datasets':datasets,'metadata_only_no_slice_read':True}
 files['code']=[{'kind':name,'path':str(path),'sha256':sha(path)} for name,path in (
  ('sw0106_runner',RUNNER),('sw0106_evaluator',EVALUATOR),('slot_predictor',SLOT_PREDICT),
  ('slot_scorer',SLOT_SCORE),('slot_model',SLOT_MODEL),('final_confirmation',Path(__file__)))]
 manifest={'status':'frozen','created_unix':__import__('time').time(),'reserve_ids':[START,END],
  'reserve_read_started':False,'validation_gate_sha256':sha(ARCHIVE/'full70k_reserve_gate.json'),
  'sw0106_full_models':files['sw0106_full_models'],'slot_checkpoints':files['slot_checkpoints'],
  'protocol_files':files['protocol_files'],'code':files['code'],'dataset_metadata':data_meta,
  'dataset_metadata_sha256':canonical_sha(data_meta),'dataset_scope':'metadata/stat identity only; no reserved image, mask, or gamma payload read before this freeze',
  'inference_protocol':{'snn':{'batch_size':8,'steps':1024,'settle':512,'threshold':.50,'min_group_size':2,'background':'largest_component','prediction':'actual event*gate spikes'},
   'slot':{'batch_size':1,'num_slots':11,'iterations':3,'inference_seed':0,'normalization':'RGB /127.5 - 1','crop':False,'background':'perimeter majority, smallest slot wins ties'},
   'paired_bootstrap':{'replicates':BOOTSTRAP_REPLICATES,'seed':BOOTSTRAP_SEED,'unit':'same reserved image IDs; per-image paired difference across seeds'}}}
 manifest['frozen_payload_sha256']=canonical_sha(manifest)
 FINAL_DIR.mkdir(parents=True,exist_ok=True);write(FREEZE,manifest)
 verify_freeze_payload(readj(FREEZE))
 print(json.dumps({'status':'frozen','path':str(FREEZE),'frozen_payload_sha256':manifest['frozen_payload_sha256'],'reserve_read_started':False}),flush=True)

def check_frozen():
 if not FREEZE.is_file():raise FileNotFoundError('reserve freeze manifest is required before reading reserved data')
 d=verify_freeze_payload(readj(FREEZE))
 st=DATASET.stat()
 with h5py.File(DATASET,'r') as h:
  current={'path':str(DATASET),'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'root_keys':sorted(list(h.keys())),
   'datasets':{name:{'shape':list(h[name].shape),'dtype':str(h[name].dtype),'chunks':list(h[name].chunks) if h[name].chunks else None,
     'compression':h[name].compression} for name in ('image','mask')},'metadata_only_no_slice_read':True}
 if canonical_sha(current)!=d.get('dataset_metadata_sha256'):
  raise AssertionError('dataset identity/stat/shape changed since freeze; reserved rows remain locked')
 if sha(FREEZE)!=file_sha_recorded_from_queue():
  # Queue records the freeze-file digest after freeze; this catches replaced manifests.
  raise AssertionError('freeze manifest file SHA differs from queue record')
 return d
def file_sha_recorded_from_queue():
 # Follow-up coordinator stores the exact freeze-file SHA in state; accept the
 # payload digest only for direct standalone CPU validation before queue launch.
 state_path=ARCHIVE/'followup_queue_state.json'
 if state_path.is_file():
  q=readj(state_path);expected=q.get('reserve_freeze_manifest_sha256')
  if expected:return expected
 return sha(FREEZE)

def run_reserve_ours(seed,arm):
 if arm not in ('candidate','control') or seed not in (0,1,2):raise ValueError('invalid reserve model selector')
 frozen=check_frozen();frozen_models=frozen['sw0106_full_models']
 folder=OUT/f'full70k_{arm}_seed{seed}';out=FINAL_DIR/f'ours_{arm}_seed{seed}'
 if out.exists():raise FileExistsError(f'preserve previous reserve attempt; no overwrite: {out}')
 expected={(x['kind']):x['sha256'] for x in frozen_models if x['arm']==arm and x['seed']==seed}
 for name in ('core.pt','encoder.pt'):
  if sha(folder/name)!=expected[name]:raise AssertionError(f'{name} changed after frozen preregistration')
 stats=next(x for x in frozen_models if x['arm']==arm and x['seed']==seed and x['kind']=='registered_feature_preprocessing')
 if sha(stats['path'])!=stats['sha256']:raise AssertionError('registered preprocessing changed after freeze')
 import torch
 sys.path.insert(0,str(HERE))
 import run as runner
 device=torch.device('cuda:0');torch.set_num_threads(2)
 core,encoder,patcher,mean,std,clip,decoder=runner.load_models(device,seed)
 core.load_state_dict(torch.load(folder/'core.pt',map_location=device,weights_only=True),strict=True)
 encoder.load_state_dict(torch.load(folder/'encoder.pt',map_location=device,weights_only=True),strict=True)
 core.eval();encoder.eval();gamma=[]
 with h5py.File(DATASET,'r') as h5,torch.no_grad():
  for offset in range(0,COUNT,16):
   rgb=h5['image'][START+offset:min(START+offset+16,START+COUNT)]
   images=torch.from_numpy(np.asarray(rgb).copy()).permute(0,3,1,2).to(device=device,dtype=torch.float32)
   gamma.append(runner.encode(encoder,patcher,mean,std,clip,images).cpu())
 out.mkdir(parents=True,exist_ok=False);gpath=out/'gamma_reserve.pt';torch.save(torch.cat(gamma),gpath)
 gm=out/'gamma_manifest.json';runner.write(gm,{'source':'SW0106 full70k trained registered encoder','image_ids':[START,END],
  'gamma_global_start':START,'shape':[COUNT,8,256],'encoder_sha256':sha(folder/'encoder.pt'),
  'feature_preprocessing_sha256':sha(stats['path']),'gamma_sha256':sha(gpath),
  'ground_truth_used_for_prediction':False,'reserve_freeze_manifest_sha256':sha(FREEZE)})
 cmd=[sys.executable,str(EVALUATOR),'--checkpoint',str(folder/'core.pt'),'--gamma-path',str(gpath),
  '--gamma-global-start',str(START),'--gamma-manifest',str(gm),'--dataset-path',str(DATASET),
  '--output-path',str(out/'evaluation.json'),'--start',str(START),'--count',str(COUNT),'--batch-size','8',
  '--steps','1024','--settle','512','--membrane-vth','.06','--min-group-size','2','--background','largest_component',
  '--thresholds','.50','--dendritic-projection','shared','--graph-spatial-decay','.35','--geodesic-steps','3',
  '--geodesic-radius','1.5','--geodesic-contrast','2','--geodesic-temperature','.5','--geodesic-cap','16',
  '--kuramoto-backend','factorized','--device','cuda:0']
 subprocess.run(cmd,cwd=ROOT,check=True)
 full_metric(out/'evaluation.json',ids=(START,END))
 (out/'RESERVE_COMPLETED').write_text('frozen reserve320 actual-spike evaluation complete\n')

def run_reserve_slot(seed):
 if seed not in (0,1,2):raise ValueError('invalid Slot seed')
 check_frozen();folder=slot_folder(seed);out=FINAL_DIR/f'slot_seed{seed}'
 if out.exists():raise FileExistsError(f'preserve previous Slot reserve attempt; no overwrite: {out}')
 import os
 env=dict(os.environ,SW0092_ALLOW_GPU='1',TF_FORCE_GPU_ALLOW_GROWTH='true',OMP_NUM_THREADS='2')
 cmd=[TF_PY,str(SLOT_PREDICT),'--checkpoint-dir',str(folder/'checkpoint'),'--checkpoint-prefix','ckpt-21880',
  '--checkpoint-source','SW0092 frozen70k epoch10 final reserve90640-90959','--training-seed',str(seed),
  '--inference-seed','0','--training-protocol',str(folder/'training_protocol.json'),'--tf-intra-threads','2',
  '--tf-inter-threads','1','--model-py',str(SLOT_MODEL),'--dataset',str(DATASET),'--start',str(START),
  '--count',str(COUNT),'--output-dir',str(out)]
 subprocess.run(cmd,cwd=ROOT,env=env,check=True)
 proto=readj(out/'protocol.json')
 if proto.get('image_ids')!=[START,END] or proto.get('checkpoint_sha256') is None:raise AssertionError('Slot reserve protocol mismatch')
 score=[sys.executable,str(SLOT_SCORE),'--predictions',str(out/'predictions.npz'),'--protocol',str(out/'protocol.json'),
  '--dataset',str(DATASET),'--start',str(START),'--count',str(COUNT),'--output-dir',str(out)]
 subprocess.run(score,cwd=ROOT,check=True)

def slot_csv_metrics(path):
 import csv
 with open(path,newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 if len(rows)!=COUNT or [int(r['image_id']) for r in rows]!=list(range(START,END+1)):raise ValueError(f'Slot reserve CSV IDs/count invalid: {path}')
 values={m:np.asarray([float(r[m]) for r in rows],dtype=np.float64) for m in METRICS}
 if any(not np.isfinite(v).all() for v in values.values()):raise ValueError(f'nonfinite Slot reserve metric: {path}')
 return values
def paired_ci(c,r):
 delta=(np.asarray(c)-np.asarray(r)).mean(axis=0);rng=np.random.default_rng(BOOTSTRAP_SEED)
 idx=rng.integers(0,COUNT,size=(BOOTSTRAP_REPLICATES,COUNT));v=delta[idx].mean(axis=1)
 lo,hi=np.quantile(v,[.025,.975],method='linear')
 return {'mean_delta':float(delta.mean()),'ci95':[float(lo),float(hi)],'lower_positive':bool(lo>0)}
def summarize():
 check_frozen();records={'candidate':[],'control':[],'slot':[]}
 arrays={k:[] for k in records}
 for arm in ('candidate','control'):
  for seed in range(3):
   path=FINAL_DIR/f'ours_{arm}_seed{seed}'/'evaluation.json';d,vals=full_metric(path,ids=(START,END))
   records[arm].append({'seed':seed,'metrics':{m:vals[m]['mean'] for m in METRICS}})
   arrays[arm].append({m:vals[m]['per_image'] for m in METRICS})
 for seed in range(3):
  folder=FINAL_DIR/f'slot_seed{seed}';d=readj(folder/'evaluation_summary.json')
  if d.get('protocol',{}).get('image_ids')!=[START,END] or d.get('ground_truth_used_for_prediction') is not False:raise ValueError('Slot reserve result protocol invalid')
  vals=slot_csv_metrics(folder/'per_image.csv');means={m:float(vals[m].mean()) for m in METRICS}
  stored=d.get('scores',{}).get('mean');counts=d.get('scores',{}).get('valid_count')
  if not isinstance(stored,dict) or not isinstance(counts,dict):raise ValueError('Slot reserve scorer summary is missing aggregate/count fields')
  if any(int(counts.get(m,-1))!=320 or abs(means[m]-float(stored[m]))>1e-12 for m in METRICS):raise ValueError('Slot per-image aggregate/count mismatch')
  records['slot'].append({'seed':seed,'metrics':means});arrays['slot'].append(vals)
 summary={'status':'complete_reserved_evaluation','role':'independent frozen reserve confirmation','ids':[START,END],
  'images':COUNT,'metrics':{},'per_seed':records,
  'bootstrap':{'replicates':BOOTSTRAP_REPLICATES,'seed':BOOTSTRAP_SEED,'unit':'reserved image; paired seed means'},
  'exposure_context':{'ours_pilot_unique_images':4096,'ours_additional_full_pass_unique_images':70000,
   'ours_pilot_plus_full_continuation':74096,'slot_unique_images_per_seed':70000,'slot_epochs':10,'slot_batch_size':32,
   'note':'Earlier SW0095 core/encoder pretraining also differs; this is not equal total training budget.'},
  'ground_truth_used_for_prediction':False,'reserve_frozen_manifest_sha256':sha(FREEZE),
  'full_reserve_gate':{'candidate_mean_strictly_above_slot_all3_metrics':None,'goal_complete':False}}
 for model in ('candidate','control','slot'):
  summary['metrics'][model]={m:float(np.mean([x['metrics'][m] for x in records[model]])) for m in METRICS}
 for competitor in ('control','slot'):
  summary['metrics'][f'candidate_minus_{competitor}']={}
  for m in METRICS:
   diffs=[arrays['candidate'][s][m]-arrays[competitor][s][m] for s in range(3)]
   summary['metrics'][f'candidate_minus_{competitor}'][m]={'mean_delta':float(np.mean([x.mean() for x in diffs])),
    'positive_seed_count':sum(float(x.mean())>0 for x in diffs),
    'paired_image_bootstrap':paired_ci(np.stack([arrays['candidate'][s][m] for s in range(3)]),
      np.stack([arrays[competitor][s][m] for s in range(3)]))}
 summary['full_reserve_gate']['candidate_mean_strictly_above_slot_all3_metrics']=all(
  summary['metrics']['candidate'][m]>summary['metrics']['slot'][m] for m in METRICS)
 summary['goal_complete']=bool(summary['full_reserve_gate']['candidate_mean_strictly_above_slot_all3_metrics'])
 summary['full_reserve_gate']['goal_complete']=summary['goal_complete']
 write(FINAL_DIR/'summary.json',summary)
 print(json.dumps({'status':summary['status'],'goal_complete':summary['goal_complete'],'means':summary['metrics']},indent=2),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['freeze','reserve-ours','reserve-slot','summarize'],required=True)
 p.add_argument('--seed',type=int,default=0);p.add_argument('--arm',choices=['candidate','control'],default='candidate')
 a=p.parse_args()
 if a.stage=='freeze':freeze()
 elif a.stage=='reserve-ours':run_reserve_ours(a.seed,a.arm)
 elif a.stage=='reserve-slot':run_reserve_slot(a.seed)
 else:summarize()
if __name__=='__main__':main()
