"""Exclusive seed0 sensitivity-screen queue; readiness only unless --run is supplied."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from run import ARMS,BASELINE_REFERENCE,BASELINE_REFERENCE_SHA256,ROOT,SOURCE_DEFAULT,baseline_match,sha

HERE=Path(__file__).resolve().parent;OUT=ROOT/'trained_models/SW0108_gate_contribution_screen'
ARCHIVE=HERE/'results_archive';EVAL=HERE/'run.py';GPUS=(0,1,2,3);MAX_USED_MIB=512

def read_json(path):return json.loads(Path(path).read_text())
def write(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def owners(gpu):
 r=subprocess.run(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
 return [int(x.strip()) for x in r.stdout.splitlines() if x.strip().isdigit()]
def memory(gpu):
 r=subprocess.run(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 return int(r.stdout.strip().splitlines()[0])
def available_gpus(owner_fn=owners,memory_fn=memory):return [g for g in GPUS if not owner_fn(g) and memory_fn(g)<=MAX_USED_MIB]
def descendants(root_pid):
 r=subprocess.run(['ps','-eo','pid=,ppid='],capture_output=True,text=True,check=True);parents={}
 for line in r.stdout.splitlines():
  c=line.split()
  if len(c)==2:
   try:parents[int(c[0])]=int(c[1])
   except ValueError:pass
 keep={root_pid};changed=True
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
def upstream_ready(path):
 if not Path(path).is_file():return False
 d=read_json(path)
 if d.get('status')=='complete' and not d.get('active'):return True
 # In --sw107-first mode the shared queue can proceed into SW0106 after SW0107;
 # permit the screen once the registered SW0107 phase is terminal and no SW0107 worker remains.
 if d.get('sw0107_status')!='complete':return False
 for job in d.get('active',{}).values():
  stage=job.get('task',{}).get('stage','')
  if str(stage).startswith('sw107-'):return False
 return not any(row.get('status')=='running' and str(row.get('task',{}).get('stage','')).startswith('sw107-')
                for row in d.get('cpu_preparation',[]))
def screen_tasks():return [{'stage':'preflight','arm':'all'}]+[{'stage':'evaluate','arm':a} for a in ARMS]
def valid_result(arm,folder):
 folder=Path(folder)
 if arm=='all':
  p=ARCHIVE/'preflight_seed0.json'
  if not p.is_file():return False
  d=read_json(p)
  return d.get('status')=='passed' and [x.get('arm') for x in d.get('arms',[])]==list(ARMS) and all(x.get('status')=='passed' for x in d['arms'])
 p=folder/'evaluation.json';dpath=folder/'diagnostics_gt_free.json'
 if not p.is_file() or not dpath.is_file():return False
 d=read_json(p)
 if d.get('ids')!=[1320,1639] or d.get('images')!=320 or d.get('ground_truth_used_for_prediction') is not False:return False
 rows=d.get('sweep',[])
 if len(rows)!=1 or rows[0].get('synchrony_threshold')!=.5:return False
 q=rows[0].get('scored_targets',{}).get('our_hdf5',{})
 for metric in ('fg_ari','foreground_iou','matched_object_iou'):
  values=q.get('per_image',{}).get(metric,[])
  if q.get('valid_count',{}).get(metric)!=320 or len(values)!=320:return False
  if not all(__import__('math').isfinite(float(v)) for v in values):return False
  if not __import__('math').isfinite(float(q.get('metrics',{}).get(metric,float('nan')))):return False
 trace=read_json(dpath)
 if trace.get('arm')!=arm or trace.get('ground_truth_used') is not False:return False
 if arm=='baseline':return baseline_match(d)['passed'] and trace.get('registered_SW0097_baseline_match',{}).get('passed') is True
 return True
def worker_cmd(task,source,gamma,dataset,device):
 cmd=[sys.executable,str(EVAL),'--stage',task['stage'],'--checkpoint',str(source),'--gamma-path',str(gamma),
  '--dataset-path',str(dataset),'--output-dir',str(ARCHIVE/'preflight_seed0.json' if task['stage']=='preflight' else OUT/task['arm']),
  '--device',device]
 if task['stage']=='evaluate':cmd.extend(['--arm',task['arm']])
 return cmd
def run_queue(sw107_state,source,gamma,dataset,poll=30):
 state_path=ARCHIVE/'queue_state.json';lock=OUT/'screen.lock'
 for p in (source,gamma,dataset):
  if not Path(p).is_file():raise FileNotFoundError(p)
 if state_path.exists():raise FileExistsError(f'preserve existing SW0108 state {state_path}')
 if OUT.exists():
  for arm in ARMS:
   if (OUT/arm).exists():raise FileExistsError(f'preserve existing SW0108 arm output {OUT/arm}')
 if (ARCHIVE/'preflight_seed0.json').exists():raise FileExistsError('preserve existing SW0108 preflight')
 OUT.mkdir(parents=True,exist_ok=True);ARCHIVE.mkdir(parents=True,exist_ok=True)
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 tasks=screen_tasks();state={'status':'waiting_for_SW0107_terminal','queue_pid':os.getpid(),'mode':'exclusive_no_foreign_sharing',
  'source_checkpoint':str(Path(source).resolve()),'source_sha256':sha(source),'gamma_path':str(Path(gamma).resolve()),'gamma_sha256':sha(gamma),
  'runner_sha256':sha(EVAL),'shared_evaluator_sha256':sha(ROOT/'collaborative_test/SW_0040_peer_transfer/evaluate.py'),
  'protocol_sha256':sha(HERE/'protocol.json'),
  'registered_baseline_reference':str(BASELINE_REFERENCE.resolve()),
  'registered_baseline_reference_sha256':sha(BASELINE_REFERENCE),
  'registered_baseline_expected_sha256':BASELINE_REFERENCE_SHA256,
  'dataset_path':str(Path(dataset).resolve()),'upstream_SW0107_state':str(Path(sw107_state).resolve()),'pending':tasks,'active':{},'attempts':[],'failed':None}
 st=Path(dataset).stat();state['dataset_identity']={'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
 write(state_path,state);proc=None
 try:
  while not upstream_ready(sw107_state):write(state_path,state);time.sleep(poll)
  state['upstream_status']='complete'
  for index,task in enumerate(tasks):
   output=ARCHIVE/'preflight_seed0.json' if task['stage']=='preflight' else OUT/task['arm']
   log=OUT/'queue'/f'{index:02d}_{task["stage"]}_{task["arm"]}.log';log.parent.mkdir(parents=True,exist_ok=True)
   if log.exists():raise FileExistsError(f'preserve old attempt log {log}')
   state['status']='waiting_for_free_gpu';state['current_task']=task;state['pending']=tasks[index:];write(state_path,state)
   gpu=None
   while gpu is None:
    for g in available_gpus():
     if not owners(g) and memory(g)<=MAX_USED_MIB:gpu=g;break
    if gpu is None:time.sleep(poll)
   if task['stage']=='evaluate' and output.exists():raise FileExistsError(f'preserve existing arm output {output}')
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
   owners_before={str(g):owners(g) for g in GPUS}
   with log.open('x') as stream:
    proc=subprocess.Popen(worker_cmd(task,source,gamma,dataset,'cuda:0'),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    state['status']='running';state['active']={'pid':proc.pid,'gpu':gpu,'task':task,'log':str(log)};write(state_path,state)
    while proc.poll() is None:
     foreign=[p for p in owners(gpu) if p not in descendants(proc.pid)]
     if foreign:stop_owned(proc);raise RuntimeError(f'foreign GPU owner appeared on GPU{gpu}; stopped only SW0108 worker')
     time.sleep(5)
   rc=proc.returncode;valid=rc==0 and valid_result(task['arm'],output)
   state['attempts'].append({'task':task,'pid':proc.pid,'gpu':gpu,'owners_before':owners_before,'log':str(log),
    'return_code':rc,'valid':valid,'finished':time.time()});state['active']={}
   if not valid:state['status']='failed';state['failed']=state['attempts'][-1];state['pending']=[];write(state_path,state);raise RuntimeError(f'SW0108 {task} failed; preserving outputs/log')
   state['pending']=tasks[index+1:];write(state_path,state)
  state['status']='complete';state['completed']=time.time();state['pending']=[];write(state_path,state);return state
 except Exception as exc:
  if state.get('status') not in ('failed','complete'):
   state.update({'status':'error','error':f'{type(exc).__name__}: {exc}','pending':[]});write(state_path,state)
  raise
 finally:
  if proc is not None and proc.poll() is None:stop_owned(proc)
  try:lock.unlink()
  except FileNotFoundError:pass
def main():
 global GPUS
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--sw107-state',type=Path,required=True)
 p.add_argument('--gpus',type=int,nargs='+',choices=(0,1,2,3),default=GPUS)
 p.add_argument('--checkpoint',type=Path,default=SOURCE_DEFAULT);p.add_argument('--gamma-path',type=Path,required=True);p.add_argument('--dataset-path',type=Path,required=True);p.add_argument('--poll-seconds',type=int,default=30)
 a=p.parse_args()
 GPUS=tuple(dict.fromkeys(a.gpus))
 if not a.run:raise SystemExit('Readiness only. Add --run after review to queue the seed0 screen.')
 if a.poll_seconds<5:raise ValueError('poll interval must be >=5 seconds')
 state=run_queue(a.sw107_state,a.checkpoint,a.gamma_path,a.dataset_path,a.poll_seconds)
 print(json.dumps({'status':state['status'],'attempts':state['attempts']},indent=2),flush=True)
if __name__=='__main__':main()
