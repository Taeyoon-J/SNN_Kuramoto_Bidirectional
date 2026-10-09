"""Dependency adapter and strict validators for the SW0129 seed replication."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'collaborative_test')]
from collaborative_test.SW_0129_shared_rgb_seed_replication import run
METRICS=('fg_ari','foreground_iou','matched_object_iou')

def task_plan():
 rows=[]
 for seed in (1,2): rows.append({'experiment':'SW0129','task_id':f'sw0129_preflight_s{seed}','stage':'preflight','seed':seed,'arm':'paired','depends_on':[]})
 for seed in (1,2):
  pf=f'sw0129_preflight_s{seed}'
  for arm in ('control','candidate'):
   train=f'sw0129_train_s{seed}_{arm}'
   rows.append({'experiment':'SW0129','task_id':train,'stage':'train','seed':seed,'arm':arm,'depends_on':[pf]})
   rows.append({'experiment':'SW0129','task_id':f'sw0129_eval_s{seed}_{arm}','stage':'evaluate','seed':seed,'arm':arm,'depends_on':[train]})
 return rows

def output_dir(task): return run.OUT/f"{task['arm']}_seed{int(task['seed'])}"
def artifact_path(task):
 stage=task['stage'];seed=int(task['seed'])
 if stage=='preflight': return run.ARCHIVE/f'preflight_seed{seed}.json'
 folder=output_dir(task)
 if stage=='train': return folder/'manifest.json'
 if stage=='evaluate': return folder/'evaluation.json'
 raise ValueError(f'unknown SW0129 stage {stage!r}')

def command(task,device='cuda:0'):
 seed=int(task['seed']);stage=task['stage']
 if stage=='preflight': return [sys.executable,str(run.RUNNER),'--stage','preflight','--seed',str(seed),'--device',device,'--output',str(artifact_path(task))]
 if stage=='train': return [sys.executable,str(run.RUNNER),'--stage','train','--seed',str(seed),'--arm',task['arm'],'--device',device,'--output-dir',str(output_dir(task))]
 return [sys.executable,str(run.RUNNER),'--stage','eval','--seed',str(seed),'--arm',task['arm'],'--device',device,'--output-dir',str(output_dir(task))]

def _valid_score(row):
 score=row['sweep'][0]['scored_targets']['our_hdf5']
 for metric in METRICS:
  vals=score['per_image'][metric];mean=float(score['metrics'][metric])
  if int(score['valid_count'][metric])!=320 or len(vals)!=320 or not all(math.isfinite(float(v)) for v in vals): return False
  if not math.isfinite(mean) or abs(sum(map(float,vals))/320-mean)>1e-10: return False
 return True

def valid_result(task):
 try:
  stage=task['stage'];seed=int(task['seed']);arm=task['arm'];path=artifact_path(task)
  if not path.is_file(): return False
  row=json.loads(path.read_text(encoding='utf-8'))
  inds,ids,source,source_sha,control_sha=run.verify_contract(seed)
  fp=run.implementation_fingerprint()
  if stage=='preflight':
   ref=run.validate_seed0_reference()
   return (row.get('status')=='passed' and row.get('experiment')=='SW0129' and row.get('seed')==seed
    and row.get('implementation_fingerprint')==fp and row.get('lambda_reference')==ref and row.get('lambda')==run.SEED0_LAMBDA
    and row.get('training_ids_first4096')==ids and row.get('source_core_sha256')==source_sha
    and row.get('matched_control_core_sha256')==control_sha and row.get('ground_truth_used') is False
    and Path(row.get('warmup_artifact','')).is_file() and run.sha(row['warmup_artifact'])==row.get('warmup_artifact_sha256')
    and row.get('decoder_warmup_batches')==32 and row.get('warmup_source_core_and_encoder_unchanged') is True
    and row.get('initial_candidate_control_hard_forward_exact') is True
    and row.get('throwaway_candidate_update_finite_and_changed_graph_encoder_parameters') is True
    and len(row.get('lambda_calibration',[]))==4 and math.isfinite(float(row.get('calibration_measured_lambda',float('nan'))))
    and all(math.isfinite(float(x.get('ratio',float('nan')))) and float(x['ratio'])>0
      and all(math.isfinite(float(x.get('reconstruction_grad_norms_by_family',{}).get(f,float('nan'))))
       and float(x['reconstruction_grad_norms_by_family'][f])>0 for f in ('encoder','graph_generator','kuramoto'))
      for x in row['lambda_calibration'])
    and all(math.isfinite(float(row.get('candidate_reconstruction_gradient_norms_by_family',{}).get(f,float('nan'))))
      and float(row['candidate_reconstruction_gradient_norms_by_family'][f])>0 for f in ('encoder','graph_generator','kuramoto'))
    and row.get('row_scramble_images')==64 and row.get('row_scramble_positive_count',0)>0
    and math.isfinite(float(row.get('row_scramble_delta_mean',float('nan')))) and float(row['row_scramble_delta_mean'])>0)
  if stage=='train':
   folder=path.parent;manifest=row;history=json.loads((folder/'history.json').read_text())
   required=('core.pt','encoder.pt','decoder.pt','optimizer_state.pt','history.json','TRAINING_COMPLETED')
   if any(not (folder/name).is_file() for name in required): return False
   pfpath=run.ARCHIVE/f'preflight_seed{seed}.json';pf=json.loads(pfpath.read_text())
   return (manifest.get('status')=='training_complete' and manifest.get('experiment')=='SW0129' and manifest.get('seed')==seed and manifest.get('arm')==arm
    and manifest.get('implementation_fingerprint')==fp and manifest.get('updates')==256 and manifest.get('batch_size')==16
    and manifest.get('train_steps')==64 and manifest.get('settle')==32 and manifest.get('shared_lambda')==run.SEED0_LAMBDA
    and manifest.get('source_core_sha256')==source_sha and manifest.get('matched_control_core_sha256')==control_sha
    and manifest.get('training_ids')==ids and manifest.get('shuffle_seed')==117+seed
    and manifest.get('preflight_sha256')==run.sha(pfpath)
    and len(history)==256 and [int(x.get('update',-1)) for x in history]==list(range(1,257))
    and all(math.isfinite(float(x[k])) for x in history for k in ('total','old_objective','primary','positive_product_spike_unweighted','reconstruction_unweighted','joint_grad_norm_preclip','decoder_grad_norm_preclip'))
    and all(float(x['joint_grad_norm_preclip'])>0 and float(x['decoder_grad_norm_preclip'])>0 for x in history)
    and all(manifest.get(key+'_sha256')==run.sha(folder/name) for key,name in (('core','core.pt'),('encoder','encoder.pt'),('decoder','decoder.pt'),('optimizer_state','optimizer_state.pt'),('history','history.json')))
    and (folder/'TRAINING_COMPLETED').is_file())
  folder=path.parent;sidepath=folder/'evaluation_manifest.json';gamma=folder/'gamma_validation.pt';gmeta=folder/'gamma_manifest.json'
  if not all(p.is_file() for p in (sidepath,gamma,gmeta,(folder/'COMPLETED'))): return False
  side=json.loads(sidepath.read_text());meta=json.loads((folder/'manifest.json').read_text())
  gamma_meta=json.loads(gmeta.read_text())
  _inds,ids,source,source_sha,_control_sha=run.verify_contract(seed)
  return (meta.get('status')=='training_complete' and meta.get('experiment')=='SW0129' and meta.get('seed')==seed and meta.get('arm')==arm
   and meta.get('implementation_fingerprint')==fp and meta.get('shared_lambda')==run.SEED0_LAMBDA
   and meta.get('source_core_sha256')==source_sha and meta.get('training_ids')==ids
   and meta.get('core_sha256')==run.sha(folder/'core.pt') and meta.get('encoder_sha256')==run.sha(folder/'encoder.pt')
   and row.get('ids')==[1320,1639] and row.get('images')==320 and row.get('ground_truth_used_for_prediction') is False and _valid_score(row)
   and side.get('status')=='complete' and side.get('experiment')=='SW0129' and side.get('seed')==seed and side.get('arm')==arm
   and side.get('implementation_fingerprint')==fp and side.get('training_manifest_sha256')==run.sha(folder/'manifest.json')
   and side.get('checkpoint_sha256')==run.sha(folder/'core.pt') and side.get('encoder_checkpoint_sha256')==run.sha(folder/'encoder.pt')
   and side.get('gamma_sha256')==run.sha(gamma) and side.get('gamma_manifest_sha256')==run.sha(gmeta)
   and gamma_meta.get('image_ids')==[1320,1639] and gamma_meta.get('shape')==[320,8,256]
   and gamma_meta.get('encoder_sha256')==run.sha(folder/'encoder.pt') and gamma_meta.get('gamma_sha256')==run.sha(gamma)
   and gamma_meta.get('ground_truth_used_for_prediction') is False
   and side.get('evaluation_sha256')==run.sha(path) and side.get('training_runner_sha256')==run.sha(run.RUNNER)
   and side.get('evaluation_runner_sha256')==run.sha(ROOT/'collaborative_test/SW_0040_peer_transfer/evaluate.py')
   and side.get('ground_truth_used_for_prediction') is False and side.get('metrics')==row['sweep'][0]['scored_targets']['our_hdf5']['metrics']
   and row.get('inference',{}).get('steps')==1024 and row.get('inference',{}).get('settle')==512
   and row.get('inference',{}).get('synchrony_thresholds')==[0.5] and row.get('background')=='largest_component')
 except (OSError,KeyError,IndexError,TypeError,ValueError,AssertionError,RuntimeError): return False
