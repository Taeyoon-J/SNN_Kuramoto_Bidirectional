"""Exclusive owner-aware SW0106 seed0 preflight/pair queue."""
import argparse, hashlib, json, math, os, signal, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results_archive'
OUT=ROOT/'trained_models/SW0106_spike_partition_rgb'
RUNNER=HERE/'run.py'
CONTROL=ROOT/'trained_models/SW0097_graph_adaptation'
SOURCE=ROOT/'trained_models/SW0095_full70k_aligned_loss'
GPUS=(0,1,2,3)
MAX_CONCURRENT_ARMS=2
MAX_USED_MIB=512
ACTIVE_OWNER_POLL_SECONDS=5
STATE=ARCHIVE/'gpu_queue_state.json'
LOCK=OUT/'gpu_queue.lock'
METRICS=('fg_ari','foreground_iou','matched_object_iou')

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
def valid_training(seed,arm,preflight):
 folder=OUT/f'{arm}_seed{seed}';m=read_json(folder/'manifest.json');hist=read_json(folder/'history.json')
 cm=read_json(CONTROL/f'seed{seed}_positive_frozen/manifest.json')
 if not (folder/'TRAINING_COMPLETED').is_file():return False
 if m.get('status')!='training_complete' or m.get('seed')!=seed or m.get('arm')!=arm:return False
 if m.get('updates')!=256 or m.get('batch_size')!=16 or len(hist)!=256 or m.get('ground_truth_used_for_training') is not False:return False
 if m.get('training_ids')!=cm.get('training_ids') or m.get('source_core_sha256')!=cm.get('source_sha256'):return False
 if m.get('shared_lambda')!=preflight.get('lambda') or m.get('shared_decoder_warmup_artifact_sha256')!=preflight.get('warmup_artifact_sha256'):return False
 for f in ('core.pt','encoder.pt','decoder.pt'):
  if not (folder/f).is_file():return False
 for row in hist:
  for k in ('total','old_objective','primary','positive_product_spike_unweighted','reconstruction_unweighted','joint_grad_norm_preclip','decoder_grad_norm_preclip'):
   if not math.isfinite(float(row[k])):return False
  if float(row['joint_grad_norm_preclip'])<=0 or float(row['decoder_grad_norm_preclip'])<=0:return False
 return True
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
 return [sys.executable,str(RUNNER),'--stage',task['stage'],'--seed',str(task['seed']),'--arm',task['arm'],'--device','cuda:0']
def save_state(state):
 snapshot={k:v for k,v in state.items() if k!='active'}
 snapshot['active']={key:{k:v for k,v in job.items() if k in ('gpu','pid','stage','arm','seed','log_path','owners_before','started')} for key,job in state.get('active',{}).items()}
 write_atomic(STATE,snapshot)
def key_for(task):return f"{task['stage']}:{task.get('arm','')}:seed{task['seed']}"
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--poll-seconds',type=int,default=60);a=p.parse_args()
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
  'poll_seconds':a.poll_seconds,'current_stage':'preflight','preflight_status':'pending',
  'arm_status':{'candidate':'blocked_on_preflight','control':'blocked_on_preflight'},'pending':[{'stage':'preflight','seed':0}],
  'active':{},'attempts':[],'failed':None,'training_restarted':False}
 save_state(state)
 try:
  while state['pending'] or state['active']:
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
