"""Exclusive owner-aware SW0106 seed0 preflight/pair queue."""
import argparse, hashlib, json, math, os, signal, subprocess, sys, time
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results_archive'
OUT=ROOT/'trained_models/SW0106_spike_partition_rgb'
RUNNER=HERE/'run.py'
CONTROL=ROOT/'trained_models/SW0097_graph_adaptation'
SOURCE=ROOT/'trained_models/SW0095_full70k_aligned_loss'
SW0105_STATE=ROOT/'collaborative_test/SW_0105_event_detach_aux/results_archive/queue_state.json'
SW0105_OUT=ROOT/'trained_models/SW0105_event_detach_aux'
GPUS=(0,1,2,3)
MAX_CONCURRENT_ARMS=2
MAX_USED_MIB=512
ACTIVE_OWNER_POLL_SECONDS=5
STATE=ARCHIVE/'gpu_queue_state.json'
LOCK=OUT/'gpu_queue.lock'
METRICS=('fg_ari','foreground_iou','matched_object_iou')
HISTORICAL_RUNNER_SHA256='27de2cb8e3852b66e247467dd2ccd702eff3f7f36e7fd60da22c3f61014849a8'

def read_json(path): return json.loads(Path(path).read_text())
def write_atomic(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def gpu_owners(gpu):
 r=subprocess.run(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
 return [int(x.strip()) for x in r.stdout.splitlines() if x.strip().isdigit()]
def gpu_used_mib(gpu):
 r=subprocess.run(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 vals=[int(x.strip()) for x in r.stdout.splitlines() if x.strip().isdigit()]
 if len(vals)!=1: raise RuntimeError(f'unexpected memory query for GPU{gpu}: {r.stdout!r}')
 return vals[0]
def eligible_gpus(gpus=GPUS,owners_fn=gpu_owners,used_fn=gpu_used_mib):
 """Only GPUs with no compute owner and at most 512MiB used are eligible."""
 available=[]
 for gpu in gpus:
  if owners_fn(gpu): continue
  if used_fn(gpu)>MAX_USED_MIB: continue
  available.append(gpu)
 return available
def process_descendants(roots):
 r=subprocess.run(['ps','-eo','pid=,ppid='],capture_output=True,text=True,check=True)
 parents={}
 for line in r.stdout.splitlines():
  cells=line.split()
  if len(cells)==2:
   try: parents[int(cells[0])]=int(cells[1])
   except ValueError: pass
 keep=set(roots);changed=True
 while changed:
  changed=False
  for pid,ppid in parents.items():
   if ppid in keep and pid not in keep:keep.add(pid);changed=True
 return keep
def terminate_owned(proc):
 try:os.killpg(proc.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:proc.wait(timeout=10)
 except subprocess.TimeoutExpired:
  try:os.killpg(proc.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  proc.wait()
def valid_preflight(path):
 d=read_json(path)
 if d.get('status')!='passed' or d.get('seed')!=0:return False
 if len(d.get('training_ids_first4096',[]))!=4096 or d.get('shuffle_seed')!=117:return False
 for flag in ('warmup_source_core_and_encoder_unchanged','initial_candidate_control_hard_forward_exact','production_labels_exact','control_reconstruction_has_no_core_encoder_gradient','throwaway_candidate_update_finite_and_changed_graph_encoder_parameters','throwaway_decoder_optimizer_state_loaded_from_shared_32_batch_warmup'):
  if d.get(flag) is not True:return False
 for family in ('encoder','graph_generator','kuramoto'):
  if float(d.get('candidate_reconstruction_gradient_norms_by_family',{}).get(family,0.))<=0:return False
 calibration=d.get('lambda_calibration',[])
 if len(calibration)!=4:return False
 for row in calibration:
  if not math.isfinite(float(row.get('ratio',float('nan')))) or float(row['ratio'])<=0:return False
  fam=row.get('reconstruction_grad_norms_by_family',{})
  if any(not math.isfinite(float(fam.get(k,0.))) or float(fam.get(k,0.))<=0 for k in ('encoder','graph_generator','kuramoto')):return False
 shuffle=d.get('row_scramble_by_batch',[])
 if len(shuffle)!=4 or d.get('row_scramble_images')!=64 or float(d.get('row_scramble_delta_mean',0.))<=0:return False
 if any(row.get('images')!=16 or not math.isfinite(float(row.get('mean_shuffled_minus_original_mse',float('nan')))) for row in shuffle):return False
 artifact=Path(d.get('warmup_artifact',''))
 if not artifact.is_file() or sha(artifact)!=d.get('warmup_artifact_sha256'):return False
 if not math.isfinite(float(d.get('lambda',float('nan')))) or float(d['lambda'])<=0:return False
 return True
def valid_full320(path):
 d=read_json(path)
 if d.get('images')!=320 or d.get('ids')!=[1320,1639]:return False
 rows=d.get('sweep')
 if not isinstance(rows,list) or not rows:return False
 scored=rows[0].get('scored_targets',{}).get('our_hdf5')
 if scored is None:return False
 for metric in METRICS:
  if scored.get('valid_count',{}).get(metric)!=320:return False
  vals=scored.get('per_image',{}).get(metric,[])
  if len(vals)!=320 or not all(math.isfinite(float(x)) for x in vals):return False
  if not math.isfinite(float(scored.get('metrics',{}).get(metric,float('nan')))):return False
 return True
def sw0105_dependency(state_path=SW0105_STATE,output_root=SW0105_OUT):
 """Do not contend with the existing SW0105 evaluation queue."""
 state_path=Path(state_path);output_root=Path(output_root)
 if not state_path.is_file():return 'waiting'
 try: state=read_json(state_path)
 except (OSError,json.JSONDecodeError):return 'waiting'
 if state.get('status') in ('error','failed'):return 'failed'
 if state.get('status')!='complete' or state.get('active'):return 'waiting'
 statuses=state.get('seed_status',{})
 for seed in range(3):
  if statuses.get(str(seed)) not in ('evaluation_complete','evaluation_already_complete'):return 'waiting'
  if not valid_full320(output_root/f'seed{seed}'/'evaluation.json'):return 'failed'
 return 'ready'
def valid_training(seed,arm,preflight,folder=None,reviewed_warmup_sha=None):
 folder=Path(folder) if folder is not None else OUT/f'{arm}_seed{seed}'
 m=read_json(folder/'manifest.json');hist=read_json(folder/'history.json')
 cm=read_json(CONTROL/f'seed{seed}_positive_frozen/manifest.json')
 if not (folder/'TRAINING_COMPLETED').is_file():return False
 if m.get('status')!='training_complete' or m.get('seed')!=seed or m.get('arm')!=arm:return False
 if m.get('updates')!=256 or m.get('batch_size')!=16 or len(hist)!=256 or m.get('ground_truth_used_for_training') is not False:return False
 if m.get('training_ids')!=cm.get('training_ids') or m.get('source_core_sha256')!=cm.get('source_sha256'):return False
 if m.get('shared_lambda')!=preflight.get('lambda'):return False
 artifact=Path(preflight.get('warmup_artifact',''))
 actual_warmup=preflight.get('warmup_artifact_sha256')
 if not artifact.is_file() or sha(artifact)!=actual_warmup:return False
 declared=m.get('shared_decoder_warmup_artifact_sha256')
 if not warmup_provenance_ok(m,artifact,actual_warmup,sha(ARCHIVE/'preflight_seed0.json'),reviewed=reviewed_warmup_sha is not None):return False
 if reviewed_warmup_sha is not None and reviewed_warmup_sha!=actual_warmup:return False
 for f in ('core.pt','encoder.pt','decoder.pt','optimizer_state.pt'):
  if not (folder/f).is_file():return False
 if m.get('optimizer_state_sha256')!=sha(folder/'optimizer_state.pt'):return False
 if m.get('joint_optimizer_steps')!=256 or m.get('decoder_optimizer_steps')!=288:return False
 for row in hist:
  for k in ('total','old_objective','primary','positive_product_spike_unweighted','reconstruction_unweighted','joint_grad_norm_preclip','decoder_grad_norm_preclip'):
   if not math.isfinite(float(row[k])):return False
  if float(row['joint_grad_norm_preclip'])<=0 or float(row['decoder_grad_norm_preclip'])<=0:return False
 return True

def warmup_provenance_ok(manifest,artifact,artifact_sha,preflight_sha,reviewed=False):
 """New runs require all provenance; reviewed legacy runs may omit these fields only."""
 expected_path=str(Path(artifact).resolve())
 fields=(('shared_decoder_warmup_artifact',expected_path),
         ('shared_decoder_warmup_artifact_sha256',artifact_sha),
         ('shared_decoder_warmup_preflight_sha256',preflight_sha))
 for key,expected in fields:
  actual=manifest.get(key)
  if actual is None:
   if not reviewed:return False
  elif actual!=expected:return False
 return True

def directory_sha256(folder):
 """Hash an archived partial tree by sorted relative path and each file SHA; symlinks are rejected."""
 root=Path(folder)
 if not root.is_dir():raise FileNotFoundError(root)
 h=hashlib.sha256()
 files=sorted(root.rglob('*'),key=lambda p:p.relative_to(root).as_posix())
 for p in files:
  if p.is_symlink():raise ValueError(f'symlink in recovery archive: {p}')
  if p.is_file():
   h.update(p.relative_to(root).as_posix().encode('utf-8'));h.update(b'\0');h.update(bytes.fromhex(sha(p)))
 return h.hexdigest()

def validate_reviewed_resume_sidecar(sidecar_path):
 """Validate a root-reviewed historical control exception and archived partial candidate."""
 s=read_json(sidecar_path)
 if s.get('schema_version')!=1 or s.get('status')!='reviewed' or not str(s.get('reviewed_by','')).strip():
  raise ValueError('resume sidecar must be schema1 and explicitly reviewed')
 if s.get('seed')!=0 or s.get('arm')!='control' or s.get('historical_runner_sha256')!=HISTORICAL_RUNNER_SHA256:
  raise ValueError('resume sidecar is not for the reviewed historical seed0 control run')
 resume_id=s.get('resume_id','')
 if not isinstance(resume_id,str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,64}',resume_id):raise ValueError('invalid recovery identifier')
 pf=Path(s.get('preflight_path',''))
 if pf.resolve()!= (ARCHIVE/'preflight_seed0.json').resolve() or sha(pf)!=s.get('preflight_sha256') or not valid_preflight(pf):
  raise ValueError('reviewed preflight path/hash/validator mismatch')
 preflight=read_json(pf);warm=Path(s.get('warmup_artifact_path',''))
 if warm.resolve()!=Path(preflight['warmup_artifact']).resolve() or s.get('warmup_artifact_sha256')!=preflight.get('warmup_artifact_sha256') or sha(warm)!=s.get('warmup_artifact_sha256'):
  raise ValueError('reviewed shared warmup path or SHA mismatch')
 control=OUT/'control_seed0';c=s.get('control_artifacts',{})
 if Path(c.get('directory','')).resolve()!=control.resolve():raise ValueError('sidecar control directory mismatch')
 expected={'manifest':'manifest.json','history':'history.json','training_marker':'TRAINING_COMPLETED',
  'core':'core.pt','encoder':'encoder.pt','decoder':'decoder.pt','optimizer_state':'optimizer_state.pt'}
 if set(c.get('sha256',{}))!=set(expected):raise ValueError('sidecar must hash all control outputs')
 for key,name in expected.items():
  p=control/name
  if not p.is_file() or sha(p)!=c['sha256'][key]:raise ValueError(f'historical control artifact SHA mismatch: {p}')
 manifest=read_json(control/'manifest.json')
 for key,expected in (('shared_decoder_warmup_artifact',str(warm.resolve())),
                      ('shared_decoder_warmup_artifact_sha256',s['warmup_artifact_sha256']),
                      ('shared_decoder_warmup_preflight_sha256',s['preflight_sha256'])):
  if manifest.get(key) is not None and manifest.get(key)!=expected:
   raise ValueError(f'historical control has mismatched warmup provenance field {key}')
 if not valid_training(0,'control',preflight,folder=control,reviewed_warmup_sha=s['warmup_artifact_sha256']):
  raise ValueError('historical control fails checks beyond explicitly missing warmup provenance fields')
 hist=Path(s.get('historical_runner_path',''))
 if not hist.is_file() or sha(hist)!=HISTORICAL_RUNNER_SHA256:raise ValueError('archived historical runner SHA mismatch')
 q=Path(s.get('queue_state_path',''))
 if ARCHIVE.resolve() not in q.resolve().parents or not q.is_file() or sha(q)!=s.get('queue_state_sha256'):raise ValueError('historical queue state must be an archived path with matching SHA')
 qs=read_json(q);failed=qs.get('failed',{})
 if qs.get('status')!='failed' or failed.get('task')!={'stage':'train','arm':'control','seed':0} or failed.get('return_code')!=0 or failed.get('validation_passed') is not False:
  raise ValueError('historical queue state does not show the reviewed validator-only control rejection')
 assert_reviewed_state_workers_gone(qs)
 log=Path(s.get('control_train_log_path',''))
 if not log.is_file() or sha(log)!=s.get('control_train_log_sha256') or failed.get('log')!=str(log):raise ValueError('control train log SHA/path mismatch')
 partial=Path(s.get('candidate_partial_archive_path',''));original=OUT/'candidate_seed0'
 recovery_root=OUT/'recovery'
 if original.exists() or not partial.is_dir() or recovery_root.resolve() not in partial.resolve().parents:
  raise ValueError('candidate partial must first be archived under the recovery directory; canonical output must be absent')
 if directory_sha256(partial)!=s.get('candidate_partial_tree_sha256'):raise ValueError('archived candidate partial tree hash mismatch')
 new_output=Path(s.get('candidate_output_dir',''))
 if new_output.resolve()!=original.resolve() or new_output.exists():raise ValueError('resume must use the now-vacant canonical candidate output only')
 return s

def assert_reviewed_state_workers_gone(state,kill_fn=os.kill):
 """Archived failed snapshots may retain stale active records, but no worker may still live."""
 for job in state.get('active',{}).values():
  pid=job.get('pid')
  if not isinstance(pid,int) or pid<=0:raise ValueError('archived failed state has invalid active PID')
  try:kill_fn(pid,0)
  except ProcessLookupError:continue
  except PermissionError:raise ValueError(f'cannot prove archived queue worker PID {pid} is gone')
  else:raise ValueError(f'archived queue worker PID {pid} is still live')

def reviewed_resume_tasks(sidecar,control_dir=None,candidate_dir=None):
 """Reuse validated preflight/control; evaluate control only when its endpoint is absent."""
 control=Path(control_dir) if control_dir is not None else OUT/'control_seed0'
 candidate=Path(candidate_dir) if candidate_dir is not None else OUT/'candidate_seed0'
 tasks=[]
 ev=control/'evaluation.json';marker=control/'COMPLETED'
 if ev.exists() or marker.exists():
  if not (ev.is_file() and marker.is_file() and valid_full320(ev)):
   raise ValueError('existing control evaluation is partial/invalid; preserve it rather than overwrite')
 else:
  tasks.append({'stage':'eval','arm':'control','seed':0,'output_dir':str(control)})
 tasks.append({'stage':'train','arm':'candidate','seed':0,'output_dir':str(candidate)})
 return tasks

def resume_task_plan(control_dir,candidate_dir,control_evaluation_valid=False):
 """Pure selector used after the reviewed sidecar has authenticated prior artifacts."""
 control=Path(control_dir);candidate=Path(candidate_dir)
 if candidate.exists():raise ValueError('candidate output must be vacant; preserve partials in recovery archive')
 tasks=[]
 ev=control/'evaluation.json';marker=control/'COMPLETED'
 if ev.exists() or marker.exists():
  if not (ev.is_file() and marker.is_file() and control_evaluation_valid):
   raise ValueError('existing control evaluation is partial/invalid; preserve it rather than overwrite')
 else:tasks.append({'stage':'eval','arm':'control','seed':0,'output_dir':str(control)})
 tasks.extend(({'stage':'train','arm':'candidate','seed':0,'output_dir':str(candidate)},
               {'stage':'eval','arm':'candidate','seed':0,'output_dir':str(candidate)}))
 return tasks
def transition(task,returncode,valid):
 """Pure stage transition; failures never requeue or overwrite attempts."""
 if returncode!=0 or not valid:return 'failed',[]
 if task['stage']=='preflight':return 'preflight_complete',[{'stage':'train','arm':'candidate','seed':0},{'stage':'train','arm':'control','seed':0}]
 if task['stage']=='train':return 'training_complete',[{'stage':'eval','arm':task['arm'],'seed':task['seed']}]
 if task['stage']=='eval':return 'complete',[]
 raise ValueError(task['stage'])
def worker_command(task):
 if task['stage']=='preflight':
  return [sys.executable,str(RUNNER),'--stage','preflight','--seed','0','--device','cuda:0','--output',str(ARCHIVE/'preflight_seed0.json')]
 cmd=[sys.executable,str(RUNNER),'--stage',task['stage'],'--seed',str(task['seed']),'--arm',task['arm'],'--device','cuda:0']
 if task.get('output_dir'):cmd.extend(['--output-dir',str(task['output_dir'])])
 return cmd
def save_state(state):
 snapshot={k:v for k,v in state.items() if k!='active'}
 snapshot['active']={key:{k:v for k,v in job.items() if k in ('gpu','pid','stage','arm','seed','log_path','owners_before','started')} for key,job in state.get('active',{}).items()}
 write_atomic(STATE,snapshot)
def key_for(task):return f"{task['stage']}:{task.get('arm','')}:seed{task['seed']}"

def run_reviewed_resume(sidecar_path,poll_seconds=10):
 """Explicit single-worker recovery; original queue state and attempt directories are immutable."""
 sidecar=validate_reviewed_resume_sidecar(sidecar_path)
 recovery_id=sidecar['resume_id'];state_path=ARCHIVE/f'reviewed_resume_{recovery_id}_queue_state.json'
 lock_path=OUT/'queue'/f'reviewed_resume_{recovery_id}.lock'
 if state_path.exists():raise FileExistsError(f'preserve prior recovery state: {state_path}')
 control=OUT/'control_seed0';candidate=OUT/'candidate_seed0'
 control_eval_valid=(control/'evaluation.json').is_file() and valid_full320(control/'evaluation.json')
 tasks=resume_task_plan(control,candidate,control_evaluation_valid=control_eval_valid)
 lock_path.parent.mkdir(parents=True,exist_ok=True)
 fd=os.open(lock_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 state={'status':'waiting_for_free_gpu','mode':'explicit_reviewed_resume','resume_id':recovery_id,
  'sidecar_path':str(Path(sidecar_path).resolve()),'sidecar_sha256':sha(sidecar_path),
  'queue_pid':os.getpid(),'created':time.time(),'historical_state_preserved':str(STATE),
  'pending':tasks,'active':{},'attempts':[],'failed':None,'training_restarted':False}
 save_atomic=lambda:write_atomic(state_path,state)
 save_atomic()
 proc=None
 try:
  for index,task in enumerate(tasks):
   output=Path(task.get('output_dir',control))
   if task['stage']=='train' and output.exists():raise FileExistsError(f'refusing to overwrite recovery output {output}')
   if task['stage']=='eval' and (output/'evaluation.json').exists():raise FileExistsError(f'refusing to overwrite evaluation {output}')
   logpath=OUT/'queue'/f'reviewed_resume_{recovery_id}_{index}_{task["stage"]}_{task["arm"]}.log'
   if logpath.exists():raise FileExistsError(f'preserve prior recovery log {logpath}')
   state['status']='waiting_for_free_gpu';state['current_task']=task;state['pending']=tasks[index:];save_atomic()
   gpu=None
   while gpu is None:
    available=eligible_gpus()
    for candidate_gpu in available:
     if not gpu_owners(candidate_gpu) and gpu_used_mib(candidate_gpu)<=MAX_USED_MIB:
      gpu=candidate_gpu;break
    if gpu is None:time.sleep(poll_seconds)
   owners_before={str(g):gpu_owners(g) for g in GPUS}
   with logpath.open('x') as log:
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    proc=subprocess.Popen(worker_command(task),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    state['active']={key_for(task):{'pid':proc.pid,'gpu':gpu,'task':task,'log':str(logpath)}};state['status']='running';save_atomic()
    while proc.poll() is None:
     owners=gpu_owners(gpu);desc=process_descendants([proc.pid]);foreign=[pid for pid in owners if pid not in desc]
     if foreign:
      terminate_owned(proc);raise RuntimeError(f'recovery worker encountered foreign GPU owner(s) {foreign}; log preserved at {logpath}')
     time.sleep(ACTIVE_OWNER_POLL_SECONDS)
    rc=proc.returncode
   state['active']={}
   valid=False
   if rc==0:
    if task['stage']=='train':
     pf=read_json(ARCHIVE/'preflight_seed0.json')
     valid=valid_training(0,'candidate',pf,folder=output)
    else:valid=(output/'COMPLETED').is_file() and valid_full320(output/'evaluation.json')
   attempt={'task':task,'gpu':gpu,'pid':proc.pid,'owners_before':owners_before,'return_code':rc,
    'validation_passed':valid,'log':str(logpath),'completed':time.time()}
   state['attempts'].append(attempt)
   if not valid:
    state['status']='failed';state['failed']=attempt;state['pending']=[];save_atomic()
    raise RuntimeError(f'reviewed recovery task failed validation; artifacts/log preserved: {logpath}')
   state['pending']=tasks[index+1:];save_atomic()
  state['status']='complete';state['pending']=[];state['completed']=time.time();save_atomic()
  # Downstream SW0107 uses the canonical queue sentinel. The immutable failed snapshot
  # remains byte-for-byte available at the archived path authenticated by the sidecar.
  if sha(STATE)!=sidecar['queue_state_sha256']:
   raise RuntimeError('canonical historical queue state changed after sidecar review; refusing to publish completion')
  write_atomic(STATE,{**state,'canonicalized_from_reviewed_resume':recovery_id,
   'historical_failed_state_archive':str(Path(sidecar['queue_state_path']).resolve()),
   'historical_failed_state_sha256':sidecar['queue_state_sha256']})
  return state
 except Exception as exc:
  if state.get('status') not in ('failed','complete'):
   state['status']='error';state['error']=f'{type(exc).__name__}: {exc}';state['pending']=[];save_atomic()
  raise
 finally:
  if proc is not None and proc.poll() is None:terminate_owned(proc)
  try:lock_path.unlink()
  except FileNotFoundError:pass

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--poll-seconds',type=int,default=60)
 p.add_argument('--resume-reviewed',type=Path,default=None);a=p.parse_args()
 if a.resume_reviewed is not None:
  if not a.run:raise SystemExit('Reviewed recovery requires --run; default remains readiness-only.')
  if a.poll_seconds<5:raise ValueError('poll interval must be at least5 seconds')
  result=run_reviewed_resume(a.resume_reviewed,a.poll_seconds)
  print(json.dumps({'status':result['status'],'resume_id':result['resume_id'],'attempts':result['attempts']},indent=2),flush=True)
  return
 if not a.run:raise SystemExit('Readiness only by default. Add --run for the approved exclusive SW0106 queue.')
 if a.poll_seconds<5:raise ValueError('poll interval must be at least5 seconds')
 ARCHIVE.mkdir(parents=True,exist_ok=True);(OUT/'queue').mkdir(parents=True,exist_ok=True)
 if STATE.exists():raise FileExistsError(f'preserve existing queue state and inspect before restart: {STATE}')
 fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 preflight_path=ARCHIVE/'preflight_seed0.json'
 if preflight_path.exists():
  LOCK.unlink();raise FileExistsError(f'preflight attempt already exists; preserve and inspect: {preflight_path}')
 for arm in ('candidate','control'):
  folder=OUT/f'{arm}_seed0'
  if folder.exists():
   LOCK.unlink();raise FileExistsError(f'preserve existing partial pilot output and inspect: {folder}')
 state={'status':'waiting_for_free_gpu','mode':'exclusive_no_foreign_gpu_sharing','queue_pid':os.getpid(),'created':time.time(),
  'eligible_gpus':list(GPUS),'max_concurrent_arms':MAX_CONCURRENT_ARMS,'max_idle_gpu_used_mib':MAX_USED_MIB,
  'wait_for_SW0105_evaluation_queue_complete':str(SW0105_STATE),
  'poll_seconds':a.poll_seconds,'current_stage':'preflight','preflight_status':'pending',
  'arm_status':{'candidate':'blocked_on_preflight','control':'blocked_on_preflight'},'pending':[{'stage':'preflight','seed':0}],
  'active':{},'attempts':[],'failed':None,'training_restarted':False}
 save_state(state)
 try:
  while state['pending'] or state['active']:
   dependency=sw0105_dependency()
   state['upstream_SW0105_status']=dependency
   if dependency=='failed':
    state['status']='failed';state['failed']={'dependency':'SW0105_full320_evaluations','status':dependency};state['pending']=[];save_state(state)
    raise RuntimeError('SW0105 evaluation queue failed or a required full320 result is invalid; preserving SW0106 without GPU launch')
   if dependency=='waiting':
    state['status']='waiting_for_SW0105_evaluations';save_state(state);time.sleep(a.poll_seconds);continue
   if state['status']=='waiting_for_SW0105_evaluations':state['status']='waiting_for_free_gpu'
   for key,job in list(state['active'].items()):
    proc=job['proc'];rc=proc.poll()
    if rc is not None:
     job['log'].close();task=job['task'];stage_valid=False
     if rc==0:
      if task['stage']=='preflight':stage_valid=valid_preflight(preflight_path)
      elif task['stage']=='train':
       pf=read_json(preflight_path);stage_valid=valid_training(0,task['arm'],pf)
      else:stage_valid=(OUT/f"{task['arm']}_seed0/COMPLETED").is_file() and valid_full320(OUT/f"{task['arm']}_seed0/evaluation.json")
     status,next_tasks=transition(task,rc,stage_valid)
     state['attempts'].append({'task':task,'gpu':job['gpu'],'pid':proc.pid,'return_code':rc,'log':job['log_path'],'terminal_status':status,'completed':time.time()})
     del state['active'][key]
     if status=='failed':
      state['failed']={'task':task,'return_code':rc,'log':job['log_path'],'validation_passed':stage_valid};state['status']='failed';state['pending']=[];save_state(state);raise RuntimeError(f"SW0106 {task['stage']} failed; preserved log {job['log_path']}")
     if task['stage']=='preflight':
      state['preflight_status']='passed';state['current_stage']='paired_seed0_pilots'
      state['arm_status']={'candidate':'pending','control':'pending'}
     elif task['stage']=='train':state['arm_status'][task['arm']]='training_complete_eval_pending'
     elif task['stage']=='eval':state['arm_status'][task['arm']]='complete'
     state['pending'][0:0]=next_tasks
     save_state(state)
    else:
     owners=gpu_owners(job['gpu']);desc=process_descendants([proc.pid]);foreign=[pid for pid in owners if pid not in desc]
     if foreign:
      terminate_owned(proc);job['log'].close();task=job['task'];attempt={'task':task,'gpu':job['gpu'],'pid':proc.pid,'foreign_gpu_pids':foreign,'log':job['log_path'],'status':'stopped_on_foreign_gpu_owner'}
      state['attempts'].append(attempt);state['failed']=attempt;state['status']='failed';state['pending']=[];del state['active'][key];save_state(state)
      raise RuntimeError(f"stopped SW0106 worker after foreign GPU owner appeared; preserved {job['log_path']}")
   active_arms=sum(1 for job in state['active'].values() if job['task']['stage'] in ('train','eval'))
   if state['pending'] and (state['pending'][0]['stage']=='preflight' or active_arms<MAX_CONCURRENT_ARMS):
    for gpu in eligible_gpus():
     if not state['pending']:break
     task=state['pending'][0]
     if task['stage']=='preflight' and state['active']:break
     if task['stage']!='preflight' and active_arms>=MAX_CONCURRENT_ARMS:break
     if gpu_owners(gpu) or gpu_used_mib(gpu)>MAX_USED_MIB:
      state['attempts'].append({'task':task,'gpu':gpu,'status':'launch_skipped_gpu_became_busy','observed_at':time.time()});continue
     if task['stage']=='train' and (OUT/f"{task['arm']}_seed0").exists():raise FileExistsError('pilot output exists; do not overwrite')
     if task['stage']=='eval' and (OUT/f"{task['arm']}_seed0/evaluation.json").exists():raise FileExistsError('evaluation exists; do not overwrite')
     logpath=OUT/'queue'/f"{task['stage']}_{task.get('arm','preflight')}_attempt1_gpu{gpu}.log"
     if logpath.exists():raise FileExistsError(f'preserve prior attempt log: {logpath}')
     log=logpath.open('w');owners_before={str(g):gpu_owners(g) for g in GPUS}
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
     proc=subprocess.Popen(worker_command(task),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
     time.sleep(1)
     if proc.poll() is None:
      owners_now=gpu_owners(gpu);desc=process_descendants([proc.pid]);foreign=[pid for pid in owners_now if pid not in desc]
      if foreign:
       terminate_owned(proc);log.close();raise RuntimeError(f'launch race GPU{gpu}; foreign owners {foreign}; worker stopped')
     key=key_for(task);state['pending'].pop(0)
     state['active'][key]={'task':task,'gpu':gpu,'pid':proc.pid,'proc':proc,'log':log,'log_path':str(logpath),'owners_before':owners_before,'started':time.time()}
     state['attempts'].append({'task':task,'gpu':gpu,'pid':proc.pid,'owners_before':owners_before,'log':str(logpath),'status':'running'});state['status']='running';state['current_stage']=task['stage'];save_state(state)
     if task['stage']=='preflight':break
     active_arms+=1
   if state['pending'] or state['active']:
    if not state['active']:state['status']='waiting_for_free_gpu'
    save_state(state);time.sleep(ACTIVE_OWNER_POLL_SECONDS if state['active'] else a.poll_seconds)
  state['status']='complete';state['completed']=time.time();state['current_stage']='complete';save_state(state)
 except Exception as exc:
  if state.get('status')!='failed':
   state['status']='error';state['error']=f'{type(exc).__name__}: {exc}';state['failed_at']=time.time();save_state(state)
  raise
 finally:
  for job in state['active'].values():terminate_owned(job['proc'])
  try:LOCK.unlink()
  except FileNotFoundError:pass
 print(json.dumps({'status':state['status'],'preflight_status':state['preflight_status'],'arm_status':state['arm_status'],'attempts':state['attempts']},indent=2),flush=True)
if __name__=='__main__':main()
