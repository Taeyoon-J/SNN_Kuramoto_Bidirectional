import json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'trained_models/SW0107_official_full70k_transfer'
SLOT=ROOT/'collaborative_test/SW_0092_cross_dataset_training/results/slot_our70000/summary.json'
METRICS=('fg_ari','foreground_iou','matched_object_iou')
def load(path):
 d=json.loads(Path(path).read_text());r=d['sweep'][0]['scored_targets']['our_hdf5'];res={}
 if d.get('images')!=320 or d.get('ids')!=[1320,1639] or d.get('ground_truth_used_for_prediction') is not False:raise ValueError(f'invalid eval {path}')
 for m in METRICS:
  x=np.asarray(r['per_image'][m],dtype=np.float64)
  if x.shape!=(320,) or int(r['valid_count'][m])!=320 or not np.isfinite(x).all() or abs(float(x.mean())-float(r['metrics'][m]))>1e-12:raise ValueError(f'invalid metric {m} at {path}')
  res[m]={'mean':float(x.mean()),'per_image':x.tolist()}
 return res
def load_mean(path):
 d=json.loads(Path(path).read_text());r=d.get('sweep',[{}])[0].get('scored_targets',{}).get('our_hdf5')
 if d.get('images')!=320 or d.get('ids')!=[1320,1639] or d.get('ground_truth_used_for_prediction') is not False or not isinstance(r,dict):raise ValueError(f'invalid comparison evaluation {path}')
 out={}
 for k in METRICS:
  x=np.asarray(r.get('per_image',{}).get(k,[]),dtype=np.float64);v=float(r.get('metrics',{}).get(k,float('nan')))
  if x.shape!=(320,) or int(r.get('valid_count',{}).get(k,-1))!=320 or not np.isfinite(x).all() or not np.isfinite(v) or abs(float(x.mean())-v)>1e-12:raise ValueError(f'invalid comparison metric {k} in {path}')
  out[k]=v
 return out
def main():
 seeds=[]
 for s in range(3):
  folder=OUT/f'seed{s}';m=json.loads((folder/'manifest.json').read_text());
  ev=load(folder/'evaluation.json')
  br=load_mean(ROOT/f'trained_models/SW0097_graph_adaptation/seed{s}_positive_frozen/evaluation.json')
  seeds.append({'seed':s,'source_sha256':m['source_core_sha256'],'metrics':{k:ev[k]['mean'] for k in METRICS},
   'sw0097_metrics':br,'delta_vs_sw0097':{k:ev[k]['mean']-br[k] for k in METRICS}})
 slot=json.loads(SLOT.read_text())['epochs']['10']
 means={k:float(np.mean([x['metrics'][k] for x in seeds])) for k in METRICS}
 slotmeans={k:float(slot['mean'][k]) for k in METRICS}
 base={k:float(np.mean([x['sw0097_metrics'][k] for x in seeds])) for k in METRICS}
 result={'status':'complete','experiment':'SW0107 additional official CLEVR full70000 transfer from SW0097 positive_frozen seeds',
  'ids':[1320,1639],'images':320,'per_seed':seeds,'mean':means,'mean_sw0097':base,'mean_slot_epoch10':slotmeans,
  'delta_vs_sw0097':{k:means[k]-base[k] for k in METRICS},'delta_vs_slot_epoch10':{k:means[k]-slotmeans[k] for k in METRICS},
  'goal_complete':False,'ground_truth_used_for_prediction':False,
  'interpretation':'Additional official-domain continuation from already pretrained SW0097 cores; not a from-scratch or equal-exposure causal comparison.'}
 p=HERE/'results_archive'/'summary.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':'complete','mean':means,'delta_vs_slot':result['delta_vs_slot_epoch10']},indent=2),flush=True)
if __name__=='__main__':main()
