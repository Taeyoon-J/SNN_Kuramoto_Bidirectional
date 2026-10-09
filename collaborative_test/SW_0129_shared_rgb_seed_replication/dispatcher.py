"""Owner-aware no-retry dispatcher for the SW0129 paired seed replication."""
from __future__ import annotations
import concurrent.futures,json,os,pathlib,signal,subprocess,sys,threading,time
try: import fcntl
except ImportError: fcntl=None
ROOT=pathlib.Path(__file__).resolve().parents[2];HERE=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'collaborative_test')]
from collaborative_test.SW_0129_shared_rgb_seed_replication import coordinator, summarize
from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
ARCHIVE=HERE/'results_archive'; STATE=ARCHIVE/'dispatcher_state.json'; QUEUE_LOCK=ARCHIVE/'dispatcher.lock'
GPU_LEASE_DIR=pathlib.Path('/tmp/kevinswk_sw0113_gpu_leases'); GPU_MEMORY_LIMIT_MIB=512; MAX_PARALLEL=4

def atomic_json(path,value):
 path=pathlib.Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)

def protected_paths(task):
 p=coordinator.artifact_path(task);stage=task['stage']
 if stage=='preflight':return (p,coordinator.run.ARCHIVE/f"preflight_decoder_seed{task['seed']}.pt")
 if stage=='train':return (p.parent,)
 folder=p.parent
 return tuple(folder/name for name in ('evaluation.json','evaluation_manifest.json','gamma_validation.pt','gamma_manifest.json','COMPLETED'))

class Dispatcher:
 def __init__(self):
  if fcntl is None:raise RuntimeError('SW0129 requires Linux flock GPU leases')
  ARCHIVE.mkdir(parents=True,exist_ok=True);self.queue_lock=QUEUE_LOCK.open('a+')
  try:fcntl.flock(self.queue_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError as exc:self.queue_lock.close();raise RuntimeError('another SW0129 dispatcher owns the queue') from exc
  if STATE.exists():self.queue_lock.close();raise FileExistsError(f'preserving previous dispatcher state: {STATE}')
  coordinator.run.validate_seed0_reference()
  for seed in (1,2): coordinator.run.verify_contract(seed)
  summarize.validate_historical_seed0()
  self.lock=threading.Lock();self.tasks=coordinator.task_plan()
  self.state={'experiment':'SW0129','status':'running','supervisor_pid':os.getpid(),'started':time.time(),
   'task_count':len(self.tasks),'runner_fingerprint':coordinator.run.implementation_fingerprint(),
   'coordinator_sha256':coordinator.run.sha(HERE/'coordinator.py'),'tasks':{t['task_id']:{'task':t,'status':'queued'} for t in self.tasks}}
  self._save()
 def _save(self):
  with self.lock:atomic_json(STATE,self.state)
 def _update(self,task_id,**fields):
  with self.lock:self.state['tasks'][task_id].update(fields);atomic_json(STATE,self.state)
 def _acquire(self,task_id):
  while True:
   for gpu in range(4):
    path=GPU_LEASE_DIR/f'gpu{gpu}.lock';path.parent.mkdir(parents=True,exist_ok=True);stream=path.open('a+')
    try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:stream.close();continue
    try:
     uuid,memory,owners=owner._nvidia_gpu_info(gpu)
     if not owners and int(memory)<=GPU_MEMORY_LIMIT_MIB:
      self._update(task_id,status='reserved',gpu=gpu,gpu_uuid=uuid,lease_path=str(path),used_before_mib=memory,reserved=time.time())
      return gpu,stream,uuid
    except BaseException:fcntl.flock(stream.fileno(),fcntl.LOCK_UN);stream.close();raise
    fcntl.flock(stream.fileno(),fcntl.LOCK_UN);stream.close()
   self._update(task_id,status='waiting_for_exclusive_gpu',last_wait=time.time());time.sleep(5)
 @staticmethod
 def _terminate_own(proc):
  if proc.poll() is not None:return
  try:os.killpg(proc.pid,signal.SIGTERM)
  except ProcessLookupError:return
  try:proc.wait(timeout=15)
  except subprocess.TimeoutExpired:
   try:os.killpg(proc.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   proc.wait()
 def _execute(self,task):
  tid=task['task_id']
  if coordinator.valid_result(task):self._update(tid,status='reused_verified',verified_at=time.time());return 'reused_verified'
  protected=protected_paths(task);present=[str(p) for p in protected if p.exists()]
  if present:self._update(tid,status='failed_existing_artifact',existing=present,finished=time.time());return 'failed_existing_artifact'
  gpu,lease,uuid=self._acquire(tid);logpath=ARCHIVE/f'{tid}.log';proc=None
  try:
   if logpath.exists():raise FileExistsError(f'preserve prior log {logpath}')
   triton=pathlib.Path(f'/tmp/kevinswk_sw0129_triton_{tid}_{int(time.time()*1000)}');triton.mkdir(mode=0o700,parents=False,exist_ok=False)
   argv=coordinator.command(task,device='cuda:0');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TRITON_CACHE_DIR=str(triton))
   self._update(tid,status='launching',gpu=gpu,gpu_uuid=uuid,argv=argv,log_path=str(logpath),triton_cache=str(triton),launched=time.time())
   with logpath.open('x',encoding='utf-8') as log:
    proc=subprocess.Popen(argv,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
    self._update(tid,status='running',child_pid=proc.pid,started=time.time())
    while proc.poll() is None:
     time.sleep(5)
     if proc.poll() is not None:break
     _u,_m,owners=owner._nvidia_gpu_info(gpu);desc=owner._process_tree(proc.pid)
     foreign=owner.foreign_owner_pids(owners,desc)
     if foreign:
      self._terminate_own(proc);self._update(tid,status='interrupted_foreign_gpu_owner',foreign_owner_pids=foreign,finished=time.time());return 'interrupted_foreign_gpu_owner'
   rc=int(proc.returncode);valid=rc==0 and coordinator.valid_result(task);status='passed' if valid else 'failed'
   self._update(tid,status=status,returncode=rc,artifact_valid=bool(valid),finished=time.time());return status
  except BaseException as exc:
   if proc is not None:self._terminate_own(proc)
   self._update(tid,status='failed',error=repr(exc),finished=time.time());return 'failed'
  finally:fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close()
 def _phase(self,rows):
  with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
   futures={pool.submit(self._execute,row):row for row in rows}
   return {futures[f]['task_id']:f.result() for f in futures}
 def _finish(self,status):
  with self.lock:self.state.update(status=status,finished=time.time());atomic_json(STATE,self.state)
 def run(self):
  phases={s:[t for t in self.tasks if t['stage']==s] for s in ('preflight','train','evaluate')}
  p=self._phase(phases['preflight'])
  if any(v not in ('passed','reused_verified') for v in p.values()):
   for t in phases['train']+phases['evaluate']:self._update(t['task_id'],status='blocked_preflight_failure')
   self._finish('preflight_failed');return
  train=self._phase(phases['train']);ready=[]
  for t in phases['evaluate']:
   if train.get(t['depends_on'][0]) in ('passed','reused_verified'):ready.append(t)
   else:self._update(t['task_id'],status='blocked_training_failure')
  if ready:self._phase(ready)
  done=all(self.state['tasks'][t['task_id']]['status'] in ('passed','reused_verified') for t in phases['evaluate'])
  self._finish('complete' if done else 'incomplete')

def main():
 if '--dry-run' in sys.argv:print(json.dumps({'experiment':'SW0129','tasks':coordinator.task_plan()},indent=2));return
 Dispatcher().run()
if __name__=='__main__':main()
