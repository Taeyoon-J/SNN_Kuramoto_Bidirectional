"""Exclusive owner-aware SW0109 preflight, baseline, and 9-condition queue."""
import argparse,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
from run import ROOT,HERE,OUT,ARCHIVE,SEEDS,SIZES,METRICS,ENCODER,FEATURE_STATS,sha,source_entry,validate_preflight_record

RUNNER=HERE/'run.py';SUMMARIZER=HERE/'summarize.py';EVALUATOR=ROOT/'collaborative_test/SW_0040_peer_transfer/evaluate.py'
GPUS=(0,1,2,3);MAX_USED_MIB=512

def read(path):return json.loads(Path(path).read_text())
def write(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def owners(g):
 r=subprocess.run(['nvidia-smi',f'--id={g}','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
 return [int(x.strip()) for x in r.stdout.splitlines() if x.strip().isdigit()]
def memory(g):
 r=subprocess.run(['nvidia-smi',f'--id={g}','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 return int(r.stdout.strip().splitlines()[0])
def free_gpus(owner_fn=owners,memory_fn=memory):return [g for g in GPUS if not owner_fn(g) and memory_fn(g)<=MAX_USED_MIB]
def reserve_gpu(reservations,owner_fn=owners,memory_fn=memory):
 for gpu in GPUS:
  if gpu in reservations:continue
  if owner_fn(gpu):continue
  if memory_fn(gpu)>MAX_USED_MIB:continue
  reservations[gpu]='reserving'
  return gpu
 return None
def descendants(root):
 r=subprocess.run(['ps','-eo','pid=,ppid='],capture_output=True,text=True,check=True);parents={}
 for line in r.stdout.splitlines():
  c=line.split()
  if len(c)==2:
   try:parents[int(c[0])]=int(c[1])
   except ValueError:pass
 keep={root};changed=True
 while changed:
  changed=False
  for pid,ppid in parents.items():
   if ppid in keep and pid not in keep:keep.add(pid);changed=True
 return keep
def stop_owned(proc):
 try:os.killpg(proc.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:proc.wait(timeout=10)
 except subprocess.TimeoutExpired:
  try:os.killpg(proc.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  proc.wait()
def cleanup_owned_workers(workers,stop_fn=stop_owned):
 stopped=[]
 for task_id,record in list(workers.items()):
  proc=record['proc']
  if proc.poll() is None:stop_fn(proc)
  stopped.append({'task_id':task_id,'pid':record['pid'],'gpu':record['gpu'],'return_code':proc.poll()})
 return stopped
def pool_schedule(seed):
 from run import training_order
 return {n:training_order(seed,n) for n in SIZES}
def task_plan():
 tasks=[]
 for seed in SEEDS:
  tasks.append({'task_id':f'preflight_seed{seed}','stage':'preflight','seed':seed,'depends_on':[]})
 for seed in SEEDS:
  tasks.append({'task_id':f'baseline_seed{seed}','stage':'eval','kind':'baseline','seed':seed,'size':2500,
   'depends_on':[f'preflight_seed{seed}']})
 for size in SIZES:
  for seed in SEEDS:
   train_id=f'train_seed{seed}_N{size}';eval_id=f'eval_seed{seed}_N{size}'
   tasks.append({'task_id':train_id,'stage':'train','kind':'scaling','seed':seed,'size':size,
    'depends_on':[f'preflight_seed{seed}']})
   tasks.append({'task_id':eval_id,'stage':'eval','kind':'scaling','seed':seed,'size':size,'depends_on':[train_id]})
 return tasks
def ready_tasks(pending,completed):
 done=set(completed)
 return [t for t in pending if set(t.get('depends_on',())).issubset(done)]
def evaluation_manifest_matches(manifest,task,checkpoint,checkpoint_sha):
 return (manifest.get('status')=='complete' and manifest.get('seed')==task['seed']
  and manifest.get('pool_size')==task['size'] and manifest.get('ids')==[1320,1639]
  and manifest.get('images')==320 and manifest.get('ground_truth_used_for_prediction') is False
  and manifest.get('checkpoint_sha256')==checkpoint_sha
  and Path(manifest.get('checkpoint','')).resolve()==Path(checkpoint).resolve())
def upstream_complete(path,kind):
 p=Path(path)
 if not p.is_file():return False
 d=read(p)
 if kind=='sw0108':return d.get('status')=='complete' and not d.get('active')
 return d.get('status')=='complete' and not d.get('active')
def output_for(t):
 if t['stage']=='preflight':return ARCHIVE/f'preflight_seed{t["seed"]}.json'
 if t['stage']=='train':return OUT/f'seed{t["seed"]}_N{t["size"]}'
 if t['kind']=='baseline':return OUT/f'baseline_seed{t["seed"]}'
 return OUT/f'seed{t["seed"]}_N{t["size"]}_evaluation'
def command(t,args):
 common=['--seed',str(t['seed']),'--train-gamma',str(args.train_gamma),'--train-manifest',str(args.train_manifest),
  '--val-gamma',str(args.val_gamma),'--val-manifest',str(args.val_manifest),'--dataset',str(args.dataset),'--device','cuda:0']
 if t['stage']=='preflight':return [sys.executable,str(RUNNER),'--stage','preflight','--output',str(output_for(t)),*common]
 if t['stage']=='train':return [sys.executable,str(RUNNER),'--stage','train','--size',str(t['size']),'--output',str(output_for(t)),*common]
 model=(source_entry(t['seed'])[1] if t['kind']=='baseline' else OUT/f'seed{t["seed"]}_N{t["size"]}/core.pt')
 return [sys.executable,str(RUNNER),'--stage','eval','--size',str(t['size']),'--checkpoint',str(model),'--output',str(output_for(t)),*common]
def valid(t):
 out=output_for(t)
 if t['stage']=='preflight':
  if not out.is_file():return False
  d=read(out);source=source_entry(t['seed'])[0]
  return (d.get('ground_truth_used') is False and validate_preflight_record(d,t['seed'],source['core_sha256']))
 if t['stage']=='train':
  if not (out/'TRAINING_COMPLETED').is_file() or not (out/'core.pt').is_file():return False
  m=read(out/'manifest.json');hist=read(out/'history.json')
  source=source_entry(t['seed'])[0]
  return (m.get('status')=='training_complete' and m.get('seed')==t['seed'] and m.get('unique_image_count')==t['size']
   and m.get('updates')==4375 and m.get('exposures')==70000 and m.get('full_unique_images_seen')==t['size']
   and m.get('graph_frozen') is True and m.get('encoder_frozen') is True and m.get('ground_truth_used_for_training') is False
   and m.get('source_core_sha256')==source['core_sha256'] and m.get('source_manifest_sha256')==source['manifest_sha256']
   and len(hist)==4375 and m.get('core_sha256')==sha(out/'core.pt'))
 if not (out/'COMPLETED').is_file() or not (out/'evaluation.json').is_file():return False
 m=read(out/'evaluation_manifest.json');d=read(out/'evaluation.json')
 checkpoint=source_entry(t['seed'])[1] if t['kind']=='baseline' else OUT/f'seed{t["seed"]}_N{t["size"]}/core.pt'
 if not evaluation_manifest_matches(m,t,checkpoint,sha(checkpoint)):return False
 if d.get('ids')!=[1320,1639] or d.get('images')!=320 or d.get('ground_truth_used_for_prediction') is not False:return False
 try:
  from summarize import scores_from_report
  scores_from_report(d,out/'evaluation.json')
 except (ValueError,KeyError,TypeError):return False
 return True
def run_queue(args):
 state_path=ARCHIVE/'queue_state.json';lock=OUT/'queue.lock'
 if state_path.exists():raise FileExistsError(f'preserve existing scaling queue state {state_path}')
 if (ARCHIVE/'scaling_summary.json').exists():raise FileExistsError('preserve existing scaling summary')
 for p in (args.train_gamma,args.train_manifest,args.val_gamma,args.val_manifest,args.dataset,args.sw0108_state,args.sw0106_state):
  if not p.is_file():raise FileNotFoundError(p)
 tasks=task_plan()
 for task in tasks:
  out=output_for(task)
  if out.exists():raise FileExistsError(f'preserve existing partial/complete output; no automatic resume: {out}')
  log=OUT/'queue_logs'/f'{task["task_id"]}.log'
  if log.exists():raise FileExistsError(f'preserve existing log {log}')
 OUT.mkdir(parents=True,exist_ok=True);ARCHIVE.mkdir(parents=True,exist_ok=True)
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 ds=args.dataset.stat()
 state={'status':'waiting_for_SW0108_and_SW0106_complete','queue_pid':os.getpid(),'mode':'exclusive_no_foreign_gpu_sharing',
  'protocol_sha256':sha(HERE/'protocol.json'),'source_audit_sha256':sha(HERE/'source_audit.json'),
  'runner_sha256':sha(RUNNER),'summary_runner_sha256':sha(SUMMARIZER),'shared_evaluator_sha256':sha(EVALUATOR),
  'training_gamma_sha256':sha(args.train_gamma),'training_gamma_manifest_sha256':sha(args.train_manifest),
  'registered_encoder_sha256':sha(ENCODER),'feature_preprocessing_sha256':sha(FEATURE_STATS),
  'validation_gamma_sha256':sha(args.val_gamma),'validation_gamma_manifest_sha256':sha(args.val_manifest),
  'dataset_path':str(args.dataset.resolve()),'dataset_identity':{'size_bytes':ds.st_size,'mtime_ns':ds.st_mtime_ns},
  'sw0108_state':str(args.sw0108_state.resolve()),'sw0106_state':str(args.sw0106_state.resolve()),
  'eligible_gpus':list(GPUS),'max_used_mib':MAX_USED_MIB,'pending':tasks.copy(),'active':{},'attempts':[],'failed':None,
  'scheduler':'dependency-aware concurrent workers; one local reservation per GPU; no automatic resume'}
 pending={t['task_id']:t for t in tasks};workers={};reservations={};completed=set();state['completed_task_ids']=[];failure=None
 def update_active():
  state['active']={key:{k:v for k,v in record.items() if k!='proc'} for key,record in workers.items()}
 def refresh_pending():state['pending']=list(pending.values())
 write(state_path,state)
 try:
  while not(upstream_complete(args.sw0108_state,'sw0108') and upstream_complete(args.sw0106_state,'sw0106')):
   state['status']='waiting_for_SW0108_and_SW0106_complete';write(state_path,state);time.sleep(args.poll_seconds)
  state['upstream_status']='complete'
  state['status']='running';write(state_path,state)
  while pending or workers:
   progress=False
   for task_id,record in list(workers.items()):
    proc=record['proc'];rc=proc.poll()
    if rc is None:
     foreign=[pid for pid in owners(record['gpu']) if pid not in descendants(record['pid'])]
     if foreign:
      stop_owned(proc);rc=proc.poll();record['return_code']=rc
      attempt={'task':record['task'],'task_id':task_id,'pid':record['pid'],'gpu':record['gpu'],
       'owners_before':record['owners_before'],'foreign_owners_during':foreign,'return_code':rc,
       'validated':False,'log':record['log'],'finished':time.time(),'failure':'foreign_owner_appeared'}
      state['attempts'].append(attempt);failure=attempt;raise RuntimeError(f'foreign GPU owner appeared on GPU{record["gpu"]}; stopped only SW0109 worker {record["pid"]}')
     continue
    ok=rc==0 and valid(record['task'])
    attempt={'task':record['task'],'task_id':task_id,'pid':record['pid'],'gpu':record['gpu'],
     'owners_before':record['owners_before'],'return_code':rc,'validated':ok,'log':record['log'],'finished':time.time()}
    state['attempts'].append(attempt);workers.pop(task_id);reservations.pop(record['gpu'],None);progress=True
    if not ok:
     failure=attempt;raise RuntimeError(f'SW0109 task failed validation; preserve output/log {output_for(record["task"])} {record["log"]}')
    completed.add(task_id);state['completed_task_ids']=sorted(completed)
   refresh_pending();update_active();write(state_path,state)
   launchable=ready_tasks(list(pending.values()),completed)
   for task in launchable:
    gpu=reserve_gpu(reservations)
    if gpu is None:break
    task_id=task['task_id'];out=output_for(task);log=OUT/'queue_logs'/f'{task_id}.log'
    reservations[gpu]=task_id
    # Recheck after the local reservation and immediately before process creation.
    if owners(gpu) or memory(gpu)>MAX_USED_MIB:
     reservations.pop(gpu,None);continue
    owners_before={str(g):owners(g) for g in GPUS};log.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    try:
     with log.open('x') as stream:
      proc=subprocess.Popen(command(task,args),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    except Exception:
     reservations.pop(gpu,None);raise
    workers[task_id]={'proc':proc,'pid':proc.pid,'gpu':gpu,'task':task,'log':str(log),
     'owners_before':owners_before,'started':time.time()}
    pending.pop(task_id);progress=True
   refresh_pending();update_active()
   if pending and not workers and not ready_tasks(list(pending.values()),completed):
    raise RuntimeError('dependency deadlock: pending conditions have no active or validated prerequisites')
   state['status']='running' if workers else 'waiting_for_free_gpu';write(state_path,state)
   if not progress:time.sleep(min(args.poll_seconds,5))
  subprocess.run([sys.executable,str(SUMMARIZER),'--output',str(ARCHIVE/'scaling_summary.json')],cwd=ROOT,check=True)
  if not (ARCHIVE/'scaling_summary.json').is_file():raise RuntimeError('scaling summary was not written')
  state['summary_path']=str(ARCHIVE/'scaling_summary.json');state['summary_sha256']=sha(ARCHIVE/'scaling_summary.json')
  state['status']='complete';state['completed']=time.time();state['pending']=[];write(state_path,state);return state
 except Exception as exc:
  cancelled=cleanup_owned_workers(workers)
  state.setdefault('interrupted_active',[]).extend(cancelled)
  workers.clear();reservations.clear();refresh_pending();update_active()
  state.update({'status':'failed','failure':failure,'error':f'{type(exc).__name__}: {exc}','failed':failure})
  write(state_path,state)
  raise
 finally:
  try:lock.unlink()
  except FileNotFoundError:pass
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--poll-seconds',type=int,default=30)
 p.add_argument('--sw0108-state',type=Path,required=True);p.add_argument('--sw0106-state',type=Path,required=True)
 p.add_argument('--train-gamma',type=Path,required=True);p.add_argument('--train-manifest',type=Path,required=True)
 p.add_argument('--val-gamma',type=Path,required=True);p.add_argument('--val-manifest',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
 a=p.parse_args()
 if not a.run:raise SystemExit('Readiness only. Add --run after review to queue registered preflights and scaling tasks.')
 if a.poll_seconds<5:raise ValueError('poll interval must be at least5 seconds')
 r=run_queue(a);print(json.dumps({'status':r['status'],'attempts':r['attempts']},indent=2),flush=True)
if __name__=='__main__':main()
