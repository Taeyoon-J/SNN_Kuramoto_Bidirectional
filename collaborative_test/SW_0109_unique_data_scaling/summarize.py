"""Prespecified CPU-only summary of SW0109 matched per-image scaling endpoints."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from run import HERE,OUT,SEEDS,SIZES,METRICS,sha

def scores_from_report(d,source='<in-memory>'):
 if d.get('ids')!=[1320,1639] or d.get('images')!=320 or d.get('ground_truth_used_for_prediction') is not False:
  raise ValueError(f'nonregistered validation split or GT prediction in {source}')
 scored=d['sweep'][0]['scored_targets']['our_hdf5'];result={}
 for metric in METRICS:
  values=np.asarray(scored['per_image'][metric],dtype=np.float64)
  if len(values)!=320 or scored['valid_count'].get(metric)!=320 or not np.isfinite(values).all():raise ValueError(f'invalid per-image values {metric}: {source}')
  reported=float(scored['metrics'][metric])
  if not np.isfinite(reported) or abs(float(values.mean())-reported)>1e-12:raise ValueError(f'inconsistent finite aggregate {metric}: {source}')
  result[metric]=values
 return result
def read_scores(path):return scores_from_report(json.loads(Path(path).read_text()),path)
def paired_bootstrap(deltas,seed=109,replicates=10000):
 arr=np.stack(deltas);rng=np.random.default_rng(seed);nseed,nimage=arr.shape
 samples=np.empty(replicates,dtype=np.float64)
 # Use identical sampled image IDs across all source seeds on each replicate.
 for i in range(replicates):
  indices=rng.integers(0,nimage,size=nimage)
  samples[i]=arr[:,indices].mean(axis=1).mean()
 return {'replicates':replicates,'seed':seed,'mean_delta':float(arr.mean()),
  'ci95_percentile':[float(x) for x in np.quantile(samples,[.025,.975])],
  'positive_lower_bound':bool(np.quantile(samples,.025)>0),
  'resampling':'same paired image indices across all three trained seeds per bootstrap replicate',
  'interpretation':'conditional on these three trained cores; not a training-seed population interval'}
def build_summary(root=OUT,replicates=10000):
 root=Path(root);scores={}
 for seed in SEEDS:
  scores[(seed,'baseline')]=read_scores(root/f'baseline_seed{seed}'/'evaluation.json')
  for size in SIZES:scores[(seed,size)]=read_scores(root/f'seed{seed}_N{size}_evaluation'/'evaluation.json')
 by_size={}
 for size in SIZES:
  perseed={str(seed):{m:float(scores[(seed,size)][m].mean()) for m in METRICS} for seed in SEEDS}
  by_size[str(size)]={'per_seed_means':perseed,'mean':{m:float(np.mean([perseed[str(s)][m] for s in SEEDS])) for m in METRICS},
   'source_core_sha256':{str(s):json.loads((root/f'seed{s}_N{size}'/'manifest.json').read_text())['source_core_sha256'] for s in SEEDS}}
 deltas={m:[scores[(s,70000)][m]-scores[(s,2500)][m] for s in SEEDS] for m in METRICS}
 fg=deltas['fg_ari'];bootstrap=paired_bootstrap(fg,seed=109,replicates=replicates)
 checks={'fg_ari_mean_delta_ge_0.01':float(np.mean(fg))>=.01,
  'fg_ari_positive_seed_deltas_at_least_2':sum(float(x.mean())>0 for x in fg)>=2,
  'paired_image_bootstrap_lower_bound_gt_0':bootstrap['positive_lower_bound']}
 fgmeans=[by_size[str(n)]['mean']['fg_ari'] for n in SIZES]
 report={'status':'complete','validation_ids':[1320,1639],'images_per_checkpoint':320,
  'metrics':list(METRICS),'baseline_per_seed':{str(s):{m:float(scores[(s,'baseline')][m].mean()) for m in METRICS} for s in SEEDS},
  'by_unique_pool_size':by_size,'fg_ari_deltas_70000_minus_2500_by_seed':{str(s):float(fg[s].mean()) for s in SEEDS},
  'paired_image_bootstrap_fg_ari':bootstrap,'registered_improvement_checks':checks,
  'registered_data_scaling_gate_passed':all(checks.values()),
  'fg_ari_means_non_decreasing_2500_10000_70000':bool(fgmeans[0]<=fgmeans[1]<=fgmeans[2]),
  'all_results_retained':True,'best_epoch_selection':False,'ground_truth_used_for_training':False,
  'protocol_sha256':sha(HERE/'protocol.json'),'source_audit_sha256':sha(HERE/'source_audit.json')}
 return report
def main():
 p=argparse.ArgumentParser();p.add_argument('--replicates',type=int,default=10000);p.add_argument('--output',type=Path,default=OUT/'scaling_summary.json');a=p.parse_args()
 if a.replicates<1000:raise ValueError('registered bootstrap requires at least1000 replicates')
 report=build_summary(replicates=a.replicates);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':report['status'],'registered_data_scaling_gate_passed':report['registered_data_scaling_gate_passed'],'output':str(a.output)},indent=2),flush=True)
if __name__=='__main__':main()
