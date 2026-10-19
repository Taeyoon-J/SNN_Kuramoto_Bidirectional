
import ctypes,json,os,subprocess,sys,time
from pathlib import Path
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
root=Path('/Data0/kevinswk/patch_v2_sw');os.chdir(root)
a=root/'collaborative_test/SW_0245_native64_precision_three_seed64/results_archive';out=a/'completion_audit_v1'

def write(name,value):
 p=out/(name+'.tmp');p.write_bytes((json.dumps(value,indent=2)+'\n').encode());p.replace(out/name)

def state(phase,**fields):write('state.json',dict(status=phase,pid=os.getpid(),updated_unix=time.time(),**fields))
try:
 done=set();deadline=time.monotonic()+7*86400
 while time.monotonic()<deadline:
  for seed in (0,1,2):
   for arm in ('control','within16'):
    folder=a/arm/('seed'+str(seed));key=(seed,arm);qp=folder/'queue/state.json'
    if qp.exists():
     q=json.loads(qp.read_bytes())
     if q['status']=='failed':raise RuntimeError('preserved queue failed: '+str(key)+' '+str(q.get('error')))
    if key in done or not (folder/'training_report.json').exists():continue
    state('auditing_actual_CPU_checkpoint',seed=seed,arm=arm,completed_audits=len(done))
    name='seed'+str(seed)+'_'+arm
    with (out/(name+'.stdout')).open('xb') as stdout,(out/(name+'.stderr')).open('xb') as stderr:
     subprocess.run([sys.executable,'-B','-m','collaborative_test.SW_0245_native64_precision_three_seed64.run','--arm',arm,'--seed',str(seed),'--audit-training'],cwd=root,env=os.environ,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,check=True,timeout=1200)
    done.add(key)
  if len(done)==6 and (a/'score_v1/summary.json').exists():break
  state('waiting_for_registered_jobs',completed_audits=len(done));time.sleep(45)
 else:raise TimeoutError('completion audit observation deadline; experiment jobs unchanged')
 import numpy as np
 from collaborative_test.SW_0245_native64_precision_three_seed64 import run
 from collaborative_test.SW_0244_native64_within_precision import run as previous
 run.activate('control')
 score_raw=(a/'score_v1/summary.json').read_bytes();score=json.loads(score_raw)
 assert score['fingerprint']==run.family_fingerprint()
 bundles={arm:{seed:run.validated_seed(seed,arm) for seed in run.SEEDS} for arm in run.ARMS}
 slots=run.reference_score.load_slot_masks();dataset=bundles['control'][0]['binding']['dataset'];stat=Path(dataset['path']).stat()
 assert (stat.st_size,stat.st_mtime_ns)==(dataset['size'],dataset['mtime_ns'])
 for seed in run.SEEDS:
  b=bundles['control'][seed]['binding']
  for arm in run.ARMS:
   bundle=bundles[arm][seed]
   assert bundle['binding']['dataset']==dataset
   for key in ('TRAIN_ids','TRAIN_RGB_sha256','TRAIN_labels_sha256','teacher_arrays_sha256','initial_model_sha256','frozen_core_sha256','input_statistics_sha256','DEV_RGB_sha256','parent_control_sha256','dataset'):assert bundle['binding'][key]==b[key]
   assert np.array_equal(bundle['parent'],bundles['control'][seed]['parent'])
 gt=run.reference.source.source.full.read_gt(dict(dataset=dataset))[list(run.OFFSETS)]
 private=json.loads((a/'score_v1/evaluation_private.json').read_bytes());vectors={k:[] for k in ('control','within16','parent237','slot')};maximum=0.
 for seed in run.SEEDS:
  arrays={arm:bundles[arm][seed]['labels'] for arm in run.ARMS};arrays.update(parent237=bundles['control'][seed]['parent'],slot=slots[seed]['labels64'][list(run.OFFSETS)])
  for name,labels in arrays.items():
   values,error=run.reference.source.source.full.metrics.measured_vectors(labels,gt);maximum=max(maximum,float(error));vectors[name].append(values)
 vectors={k:np.stack(v) for k,v in vectors.items()};means={k:v.mean(axis=(0,1)) for k,v in vectors.items()}
 keys=run.reference_score.METRICS
 for name,values in vectors.items():
  maximum=max(maximum,float(np.abs(values-np.asarray(private['vectors'][name])).max()))
  maximum=max(maximum,float(np.abs(means[name]-np.array([score['scores'][name][k] for k in keys])).max()))
  for seed in run.SEEDS:maximum=max(maximum,float(np.abs(values[seed].mean(0)-np.array([score['per_seed'][name][str(seed)][k] for k in keys])).max()))
 for key,ref in (('within16_minus_control','control'),('within16_minus_parent237','parent237')):
  maximum=max(maximum,float(np.abs(means['within16']-means[ref]-np.array([score[key][k] for k in keys])).max()))
 assert maximum<1e-12
 positive=bool(means['within16'][0]-means['control'][0]>=.01 and means['within16'][0]-means['parent237'][0]>=.01 and np.all(means['within16'][1:]>means['slot'][1:]))
 assert positive==score['within16_promising_for_separate_full320_confirmation']
 assert score['seed_scope']==[0,1,2] and score['image_count_per_seed']==64 and score['full320_improvement_proven'] is False
 audits={arm:{str(seed):json.loads((run.archive_for(seed,arm)/'training_audit.json').read_bytes()) for seed in run.SEEDS} for arm in run.ARMS}
 losses={arm:{str(seed):{k:bundles[arm][seed]['report'][k] for k in ('initial_TRAIN_loss','final_TRAIN_loss')} for seed in run.SEEDS} for arm in run.ARMS}
 replay={}
 for arm in run.ARMS:
  old=previous.validated_seed(1,arm);new=bundles[arm][1];same_model=new['report']['final_model_sha256']==old['report']['final_model_sha256'];matches=[]
  for i,offset in enumerate(previous.OFFSETS):
   j=run.OFFSETS.index(offset);matches.append(bool(np.array_equal(old['labels'][i],new['labels'][j])))
  replay[arm]=dict(same_final_model_bitwise=same_model,original_eight_masks_bitwise_equal=all(matches),matched_mask_count=sum(matches))
 result=dict(status='completed_six_actual_CPU_checkpoint_audits_and_raw_mask_rescoring',score_sha256=run.sha(score_raw),fingerprint=run.family_fingerprint(),training_audits=audits,TRAIN_losses=losses,seed1_screen_replay=replay,raw_masks_independently_rescored=768,independent_metrics_per_seed_means_and_deltas_maximum_abs_error=maximum,all_prediction_protocols_validated_before_evaluation_GT=True,optimizer_updates=0,additional_native_forwards=0,all_GPU_jobs_untouched=True)
 write('summary.json',result);state('complete',raw_masks_independently_rescored=768)
except BaseException as exc:state('failed_preserved_for_review',error=repr(exc),all_GPU_jobs_untouched=True);raise
