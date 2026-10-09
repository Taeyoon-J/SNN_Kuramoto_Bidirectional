"""Strict post-evaluation gate for the preregistered SW0129 paired replication."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OLD_ARCHIVE=ROOT/'collaborative_test/SW_0106_spike_partition_rgb/results_archive'
OLD_OUT=ROOT/'trained_models/SW0106_spike_partition_rgb'
from collaborative_test.SW_0129_shared_rgb_seed_replication import coordinator,run
METRICS=('fg_ari','foreground_iou','matched_object_iou')
SLOT_FLOORS={'foreground_iou':0.25358913115224261,'matched_object_iou':0.25693697915214362}
HISTORICAL_SHA={'candidate_seed0_evaluation.json':'124fed288ff7c31d817b3eae346438e2b23f088a5adcf89a4cfd8723a80bcdc3',
 'control_seed0_evaluation.json':'39f5f0e61516743687c36179a6d75def898bc5264c2f8cce2c16ccb4e0141e58',
 'source_seed0_evaluation.json':'009da533e0f3e7fa86d9840a85d87640d9424acf660512f64232ad1669fe61e0'}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def extract(path):
 d=json.loads(Path(path).read_text(encoding='utf-8-sig'))
 if d.get('ids')!=[1320,1639] or d.get('images')!=320 or d.get('ground_truth_used_for_prediction') is not False:raise ValueError(f'evaluation contract mismatch: {path}')
 inf=d.get('inference',{})
 if (inf.get('steps')!=1024 or inf.get('settle')!=512 or inf.get('min_group_size')!=2
  or inf.get('synchrony_thresholds')!=[0.5] or inf.get('affinity_modes')!=['spike']
  or d.get('background')!='largest_component'):raise ValueError(f'native evaluation readout differs from frozen SW0129 contract: {path}')
 score=d['sweep'][0]['scored_targets']['our_hdf5'];out={}
 for metric in METRICS:
  vals=np.asarray(score['per_image'][metric],dtype=np.float64)
  mean=float(score['metrics'][metric])
  if vals.shape!=(320,) or not np.isfinite(vals).all() or not math.isfinite(mean) or int(score['valid_count'][metric])!=320:
   raise ValueError(f'invalid per-image evaluation values for {metric}: {path}')
  if abs(float(vals.mean())-mean)>1e-10:raise ValueError(f'mean does not match per-image values: {metric} {path}')
  out[metric]=vals
 return out

def paired_bootstrap_ci(differences,replicates=10000,seed=129):
 arr=np.asarray(differences,dtype=np.float64)
 if arr.shape!=(3,320) or not np.isfinite(arr).all():raise ValueError('bootstrap requires three matched 320-image difference arrays')
 rng=np.random.RandomState(seed);indices=rng.randint(0,320,size=(replicates,320))
 # Every replicate samples the same image indices across all three seeds.
 sampled=arr[:,indices].mean(axis=(0,2))
 return [float(x) for x in np.quantile(sampled,[.025,.975])],float(arr.mean())

def historical_path(name):
 if name=='candidate':return OLD_OUT/'candidate_seed0'/'evaluation.json'
 if name=='control':return OLD_OUT/'control_seed0'/'evaluation.json'
 if name=='source':return run.CONTROL_ROOT/'seed0_positive_frozen'/'evaluation.json'
 raise ValueError(name)

def _historical(name):
 path=historical_path(name)
 expected=HISTORICAL_SHA['candidate_seed0_evaluation.json' if name=='candidate' else 'control_seed0_evaluation.json' if name=='control' else 'source_seed0_evaluation.json']
 if not path.is_file() or sha(path)!=expected:raise ValueError(f'historical seed0 {name} report missing or SHA changed: {path}')
 raw=json.loads(path.read_text(encoding='utf-8-sig'))
 expected_checkpoint=(OLD_OUT/f'{name}_seed0'/'core.pt') if name in ('candidate','control') else (run.CONTROL_ROOT/'seed0_positive_frozen'/'core.pt')
 if raw.get('checkpoint')!=str(expected_checkpoint):raise ValueError(f'historical seed0 report checkpoint path mismatch: {path}')
 return extract(path),path

def validate_historical_seed0():
 reports={}
 for name in ('candidate','control','source'):
  values,path=_historical(name);reports[name]={'path':str(path),'sha256':sha(path),'metrics':{m:float(values[m].mean()) for m in METRICS}}
 return reports

def collect_all():
 cand={};ctrl={};source={};provenance={'historical_seed0':validate_historical_seed0(),'source97_reports':{},'new_training_and_evaluation':{}}
 cand[0],_= _historical('candidate')
 ctrl[0],_= _historical('control')
 source[0],_= _historical('source')
 for seed in (1,2):
  for arm,dest in (('candidate',cand),('control',ctrl)):
   task={'stage':'evaluate','seed':seed,'arm':arm}
   if not coordinator.valid_result(task):raise RuntimeError(f'SW0129 results incomplete or invalid; preserve and inspect: seed{seed}/{arm}')
   folder=run.OUT/f'{arm}_seed{seed}'
   report=folder/'evaluation.json';manifest=folder/'manifest.json'
   m=json.loads(manifest.read_text())
   if m.get('source_core_sha256')!=run.verify_contract(seed)[3] or m.get('shared_lambda')!=run.SEED0_LAMBDA:
    raise ValueError(f'training source/lambda binding mismatch: {manifest}')
   dest[seed]=extract(report)
   provenance['new_training_and_evaluation'][f'seed{seed}_{arm}']={'training_manifest_sha256':sha(manifest),'evaluation_sha256':sha(report),'checkpoint_sha256':sha(folder/'core.pt'),'encoder_sha256':sha(folder/'encoder.pt')}
  path=run.CONTROL_ROOT/f'seed{seed}_positive_frozen'/'evaluation.json'
  if not path.is_file():raise FileNotFoundError(path)
  source_report=json.loads(path.read_text(encoding='utf-8-sig'))
  expected_checkpoint=run.CONTROL_ROOT/f'seed{seed}_positive_frozen'/'core.pt'
  if source_report.get('checkpoint')!=str(expected_checkpoint):raise ValueError(f'SW0097 source report checkpoint path mismatch: {path}')
  source[seed]=extract(path)
  provenance['source97_reports'][str(seed)]={'evaluation_sha256':sha(path),'source95_core_sha256':run.verify_contract(seed)[3]}
 return cand,ctrl,source,provenance

def summarize(cand,ctrl,source,provenance=None):
 if any(set(x)!={0,1,2} for x in (cand,ctrl,source)):raise ValueError('all seed0 historical and new seed1/2 result rows are required')
 rows={};means={};deltas={}
 for label,arms in (('candidate',cand),('control',ctrl),('source97',source)):
  rows[label]={m:{str(seed):arms[seed][m].tolist() for seed in (0,1,2)} for m in METRICS}
  means[label]={m:float(np.stack([arms[s][m] for s in (0,1,2)]).mean()) for m in METRICS}
 deltas['candidate_minus_control']={m:np.stack([cand[s][m]-ctrl[s][m] for s in (0,1,2)]) for m in METRICS}
 deltas['candidate_minus_source97']={m:np.stack([cand[s][m]-source[s][m] for s in (0,1,2)]) for m in METRICS}
 cis={name:{'mean_delta':paired_bootstrap_ci(values)[1],'ci95':paired_bootstrap_ci(values)[0]} for name,values in ((label,deltas[label]['fg_ari']) for label in deltas)}
 seed_gains={name:{str(s):float(deltas[name]['fg_ari'][s].mean()) for s in (0,1,2)} for name in deltas}
 checks={'candidate_mean_fg_above_control':means['candidate']['fg_ari']>means['control']['fg_ari'],
  'candidate_mean_fg_above_source97':means['candidate']['fg_ari']>means['source97']['fg_ari'],
  'at_least_two_seed_fg_gains_vs_control':sum(v>0 for v in seed_gains['candidate_minus_control'].values())>=2,
  'at_least_two_seed_fg_gains_vs_source97':sum(v>0 for v in seed_gains['candidate_minus_source97'].values())>=2,
  'shared_image_ci_lower_positive_vs_control':cis['candidate_minus_control']['ci95'][0]>0,
  'shared_image_ci_lower_positive_vs_source97':cis['candidate_minus_source97']['ci95'][0]>0,
  'foreground_iou_above_slot_plus_0.05':means['candidate']['foreground_iou']>SLOT_FLOORS['foreground_iou'],
  'matched_object_iou_above_slot_plus_0.05':means['candidate']['matched_object_iou']>SLOT_FLOORS['matched_object_iou']}
 return {'experiment':'SW0129','status':'complete','promotion_status':'passed' if all(checks.values()) else 'failed',
  'historical_seed0_reused':True,'historical_seed0_reports':provenance.get('historical_seed0') if provenance else None,
  'source97_report_provenance':provenance.get('source97_reports') if provenance else None,
  'new_training_and_evaluation_provenance':provenance.get('new_training_and_evaluation') if provenance else None,'new_training_seeds':[1,2],'per_seed_per_image_metrics':rows,'three_seed_means':means,
  'per_seed_fg_gain':seed_gains,'candidate_fg_paired_bootstrap':cis,'promotion_checks':checks,
  'bootstrap':{'replicates':10000,'seed':129,'resampling':'same sampled 320 image indices shared across all three seeds; NumPy RandomState MT19937 seed 129'},
  'slot_iou_floors':SLOT_FLOORS,'interpretation_limit':'A pass supports this shared RGB assignment-credit recipe; it does not establish gating contributions or data scaling.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'results_archive'/'three_seed_summary.json');a=p.parse_args()
 if a.output.exists():raise FileExistsError(f'preserving existing summary: {a.output}')
 cand,ctrl,source,provenance=collect_all();result=summarize(cand,ctrl,source,provenance);a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':result['promotion_status'],'output':str(a.output),'means':result['three_seed_means']},allow_nan=False),flush=True)
if __name__=='__main__':main()
