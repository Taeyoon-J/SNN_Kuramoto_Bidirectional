import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'trained_models/SW0105_event_detach_aux'
ARCHIVE=ROOT/'collaborative_test/SW_0105_event_detach_aux/results_archive'
RUNNER=ROOT/'collaborative_test/SW_0105_event_detach_aux/run.py'
CONTROL=ROOT/'trained_models/SW0097_graph_adaptation'
SOURCE=ROOT/'trained_models/SW0095_full70k_aligned_loss'
GPUS=(0,1,2,3)

def read_json(p): return json.loads(p.read_text())
def write_atomic(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); tmp.replace(p)
def gpu_processes(gpu):
 r=subprocess.run(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
 return [int(x.strip()) for x in r.stdout.splitlines() if x.strip().isdigit()]
def free_gpus(): return [g for g in GPUS if not gpu_processes(g)]
def process_descendants(roots):
 r=subprocess.run(['ps','-eo','pid=,ppid='],capture_output=True,text=True,check=True)
 parents={}
 for line in r.stdout.splitlines():
  cells=line.split()
  if len(cells)==2:
   try: parents[int(cells[0])]=int(cells[1])
   except ValueError: pass
 keep=set(roots); changed=True
 while changed:
  changed=False
  for pid,ppid in parents.items():
   if ppid in keep and pid not in keep: keep.add(pid); changed=True
 return keep
def verify_trained(seed):
 folder=OUT/f'seed{seed}'; m=read_json(folder/'manifest.json'); hist=read_json(folder/'history.json')
 c=read_json(CONTROL/f'seed{seed}_positive_frozen/manifest.json')
 if m.get('status')!='training_complete' or len(hist)!=256 or m.get('updates')!=256: raise AssertionError(f'seed{seed} training not complete')
 if m.get('training_ids')!=c.get('training_ids') or len(m['training_ids'])!=4096: raise AssertionError(f'seed{seed} IDs mismatch')
 if m.get('source_sha256')!=c.get('source_sha256') or m.get('control_sha256') is None: raise AssertionError(f'seed{seed} source contract mismatch')
 if not (folder/'core.pt').is_file(): raise FileNotFoundError(folder/'core.pt')
 ev=folder/'evaluation.json'
 if ev.exists():
  data=read_json(ev)
  if data.get('images')!=320 or data.get('ids')!=[1320,1639]: raise AssertionError(f'existing seed{seed} evaluation incomplete; preserve and inspect')
 return folder

def command(seed):
 return [sys.executable,str(RUNNER),'--stage','eval','--seed',str(seed),'--device','cuda:0']
def save_state(state):
 snapshot=dict(state)
 snapshot['active']={seed:{k:v for k,v in job.items() if k in ('gpu','pid','log_path','owners_before','started')} for seed,job in state['active'].items()}
 write_atomic(ARCHIVE/'queue_state.json',snapshot)
def terminate_owned(proc):
 try: os.killpg(proc.pid,signal.SIGTERM)
 except ProcessLookupError: pass

def main():
 p=argparse.ArgumentParser(); p.add_argument('--run-eval',action='store_true'); p.add_argument('--poll-seconds',type=int,default=60); a=p.parse_args()
 if not a.run_eval: raise SystemExit('Readiness only by default. Use --run-eval to start the approved evaluation-only queue.')
 ARCHIVE.mkdir(parents=True,exist_ok=True); (OUT/'queue').mkdir(parents=True,exist_ok=True)
 lockpath=OUT/'queue/coordinator.lock'; fd=os.open(lockpath,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 os.write(fd,str(os.getpid()).encode()); os.close(fd)
 state={'status':'running','mode':'evaluation_only_completed_seed_checkpoints','created':time.time(),'eligible_gpus':list(GPUS),'poll_seconds':a.poll_seconds,'seed_status':{},'active':{},'attempts':[],'foreign_overlap_stopped':[],'training_restarted':False}
 pending=[]
 for seed in range(3):
  folder=verify_trained(seed); ev=folder/'evaluation.json'
  if ev.exists(): state['seed_status'][str(seed)]='evaluation_already_complete'; continue
  pending.append(seed); state['seed_status'][str(seed)]='awaiting_evaluation'
 save_state(state)
 try:
  while pending or state['active']:
   # Revalidate workers and resource isolation before scheduling.
   for seed,job in list(state['active'].items()):
    proc=job['proc']; dev=job['gpu']; owners=gpu_processes(dev); descendants=process_descendants([proc.pid])
    foreign=[pid for pid in owners if pid not in descendants]
    if foreign:
     terminate_owned(proc); rc=proc.wait(); job['log'].close()
     attempt={'seed':int(seed),'gpu':dev,'pid':proc.pid,'return_code':rc,'foreign_gpu_pids':foreign,'log':job['log_path'],'status':'stopped_on_foreign_gpu_owner'}
     state['foreign_overlap_stopped'].append(attempt); state['attempts'].append(attempt); del state['active'][seed]
     existing=(OUT/f'seed{seed}'/'evaluation.json')
     if existing.exists() and read_json(existing).get('images')==320: state['seed_status'][seed]='evaluation_complete'
     else: pending.insert(0,int(seed))
     save_state(state)
   for gpu in free_gpus():
    if not pending: break
    current_owners=gpu_processes(gpu)
    if current_owners:
     state['attempts'].append({'gpu':gpu,'owner_pids_before_launch':current_owners,'status':'launch_skipped_gpu_became_busy'})
     save_state(state); continue
    seed=pending.pop(0); folder=verify_trained(seed)
    if (folder/'evaluation.json').exists():
     data=read_json(folder/'evaluation.json')
     if data.get('images')==320 and data.get('ids')==[1320,1639]: state['seed_status'][str(seed)]='evaluation_complete'; save_state(state); continue
     raise AssertionError(f'existing seed{seed} evaluation is incomplete; inspect instead of overwriting')
    logpath=OUT/'queue'/f'eval_seed{seed}_attempt{sum(1 for x in state["attempts"] if x["seed"]==seed)+1}_gpu{gpu}.log'
    log=logpath.open('w'); env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2')
    before={str(g):gpu_processes(g) for g in GPUS}
    proc=subprocess.Popen(command(seed),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    time.sleep(1)
    owners_now=gpu_processes(gpu); descendants=process_descendants([proc.pid]); foreign=[pid for pid in owners_now if pid not in descendants]
    if foreign:
     terminate_owned(proc); rc=proc.wait(); log.close(); attempt={'seed':seed,'gpu':gpu,'pid':proc.pid,'return_code':rc,'foreign_gpu_pids':foreign,'owners_before':before,'log':str(logpath),'status':'launch_race_stopped'}
     state['foreign_overlap_stopped'].append(attempt); state['attempts'].append(attempt)
     existing=(OUT/f'seed{seed}'/'evaluation.json')
     if existing.exists() and read_json(existing).get('images')==320: state['seed_status'][str(seed)]='evaluation_complete'
     else: pending.insert(0,seed)
     save_state(state); continue
    state['active'][str(seed)]={'gpu':gpu,'pid':proc.pid,'proc':proc,'log':log,'log_path':str(logpath),'owners_before':before,'started':time.time()}
    state['seed_status'][str(seed)]='evaluating'; state['attempts'].append({'seed':seed,'gpu':gpu,'pid':proc.pid,'owners_before':before,'launch_owners':owners_now,'log':str(logpath),'status':'running'}); save_state(state)
   for seed,job in list(state['active'].items()):
    rc=job['proc'].poll()
    if rc is None: continue
    job['log'].close(); folder=verify_trained(int(seed))
    if rc!=0: raise RuntimeError(f'eval seed{seed} failed rc={rc}; inspect {job["log_path"]}')
    data=read_json(folder/'evaluation.json')
    if data.get('images')!=320 or data.get('ids')!=[1320,1639]: raise AssertionError(f'eval seed{seed} invalid 320-image evaluation')
    state['seed_status'][seed]='evaluation_complete'; state['attempts'].append({'seed':int(seed),'gpu':job['gpu'],'pid':job['proc'].pid,'return_code':0,'images':320,'status':'complete'}); del state['active'][seed]; save_state(state)
   if pending or state['active']: time.sleep(a.poll_seconds)
  state['status']='complete'; state['completed']=time.time(); save_state(state)
 finally:
  for job in state['active'].values(): terminate_owned(job['proc'])
  try: lockpath.unlink()
  except FileNotFoundError: pass
 print(json.dumps({'status':state['status'],'seed_status':state['seed_status'],'attempts':state['attempts'],'foreign_overlap_stopped':state['foreign_overlap_stopped']},indent=2),flush=True)
if __name__=='__main__': main()
