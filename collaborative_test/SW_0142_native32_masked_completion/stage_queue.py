"""Owner-aware SW0142 no-retry stage queue. No training is launched unless explicitly run."""
from __future__ import annotations
import concurrent.futures, json, os, signal, subprocess, sys, threading, time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for path in (ROOT,ROOT/'collaborative_test',ROOT/'snn_kuramoto_bidirectional',HERE):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from collaborative_test.SW_0142_native32_masked_completion import run
try:
    import fcntl
except ImportError:
    fcntl=None
try:
    from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
except ImportError:
    owner=None
GPU_LEASE_DIR=Path('/tmp/kevinswk_sw0113_gpu_leases'); GPU_LIMIT_MIB=512
ARCHIVE=run.HERE/'queue_runtime'; STATE=ARCHIVE/'queue_state.json'; LOCK=ARCHIVE/'queue.lock'


def task_plan():
    return [
      {'task_id':'cache_seed1','stage':'cache','deps':[]},
      {'task_id':'warm_seed1','stage':'warm','deps':['cache_seed1']},
      *[{'task_id':f'preflight_{arm}','stage':'preflight','arm':arm,'deps':['warm_seed1']} for arm in run.ARMS],
      *[{'task_id':f'train_{arm}','stage':'train','arm':arm,'deps':[f'preflight_{arm}']} for arm in run.ARMS],
      *[{'task_id':f'predict_{arm}','stage':'predict','arm':arm,'deps':[f'train_{arm}']} for arm in run.ARMS],
      {'task_id':'score_seed1','stage':'score','deps':[f'predict_{arm}' for arm in run.ARMS]},
    ]

def task_groups(): return [('cache_seed1',),('warm_seed1',),('preflight_encoder_graph','preflight_core_trainable'),('train_encoder_graph','train_core_trainable'),('predict_encoder_graph','predict_core_trainable'),('score_seed1',)]

def task_output(task):
    s=task['stage'];a=task.get('arm')
    if s=='cache':return run.CACHE
    if s=='warm':return run.WARM
    if s=='preflight':return run.TRAIN/f"preflight_{a}_seed1.json"
    if s=='train':return run.TRAIN/f'{a}_seed1'
    if s=='predict':return run.RESULTS/a
    if s=='score':return run.HERE/'results_archive'/'score_seed1.json'
    raise ValueError(s)

def command(task,python=sys.executable):
    stage=task['stage'];argv=[python,str(run.HERE/'run.py'),'--stage',stage]
    if stage in ('preflight','train','predict'):argv += ['--arm',task['arm']]
    if stage!='score':argv += ['--device','cuda:0']
    else:argv += ['--output',str(task_output(task))]
    return argv

def environment(task,gpu=None,triton_cache=None):
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PATCH_V2_REPO=str(run.ROOT))
    if task['stage']=='score':
        env['CUDA_VISIBLE_DEVICES']='';env.pop('TRITON_CACHE_DIR',None)
    else:
        if gpu is None:raise ValueError('GPU task needs a physical GPU lease')
        env['CUDA_VISIBLE_DEVICES']=str(gpu)
        if triton_cache:env['TRITON_CACHE_DIR']=str(triton_cache)
    return env

def validate_result(task):
    try:
        stage=task['stage'];p=task_output(task)
        if stage=='cache':
            m=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
            return m.get('status')=='complete' and m.get('experiment')==run.NAME and m.get('seed')==1 and m.get('shape')==[run.TRAIN_COUNT,4,run.N,512] and m.get('dtype')=='float16' and m.get('ground_truth_used') is False and m.get('optimizer_updates')==0 and len(m.get('chunk_sha256',[]))==run.TRAIN_COUNT//run.MICRO and all((p/name).is_file() for name in ('spikes.f16','rgb.u8','tiles.i16'))
        if stage=='warm':
            m=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
            return m.get('status')=='warm_complete' and m.get('experiment')==run.NAME and m.get('updates')==256 and m.get('ground_truth_used') is False and all((p/name).is_file() for name in ('warm.pt','warm_optimizer.pt','history.json'))
        if stage=='preflight':
            r=json.loads(p.read_text(encoding='utf-8'));return r.get('status')=='preflight_complete' and r.get('arm')==task['arm'] and r.get('seed')==1 and r.get('optimizer_updates')==1 and r.get('ground_truth_used') is False and r.get('implementation_fingerprint')==run.fingerprint()
        if stage=='train':return run.validate_training(task['arm'])['manifest'].get('status')=='training_complete'
        if stage=='predict':
            return run._own_prediction(task['arm'])['npz'].is_file()
        if stage=='score':
            r=json.loads(p.read_text(encoding='utf-8'));return r.get('status')=='scored_complete' and r.get('ground_truth_used_for_prediction') is False and r.get('ground_truth_used_for_scoring') is True and r.get('experiment')==run.NAME
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,EOFError):return False
    return False

class Queue:
    def __init__(self):
        if fcntl is None or owner is None:raise RuntimeError('SW0142 queue requires Linux flock and SW0123 owner helpers')
        if ARCHIVE.exists() or STATE.exists() or LOCK.exists():raise FileExistsError('preserve prior SW0142 queue state')
        existing=[task_output(t) for t in task_plan() if task_output(t).exists()]
        if existing:raise FileExistsError('preserve existing SW0142 stage outputs: '+', '.join(map(str,existing)))
        ARCHIVE.mkdir(parents=True,exist_ok=False);self.lock=LOCK.open('x+');fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.guard=threading.RLock();self.tasks=task_plan();self.state={'status':'running','experiment':run.NAME,'registered':True,'supervisor_pid':os.getpid(),'started':time.time(),'implementation_fingerprint':run.fingerprint(),'queue_sha256':run.sha256(Path(__file__)),'owner_dispatcher_sha256':run.sha256(owner.__file__),'tasks':{t['task_id']:{'task':t,'status':'queued'} for t in self.tasks}}
        self._save()
    def _save(self):
        with self.guard:
            tmp=STATE.with_name(STATE.name+f'.{os.getpid()}.tmp')
            with tmp.open('x',encoding='utf-8',newline='\n') as f:json.dump(self.state,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
            os.replace(tmp,STATE)
    def _update(self,task_id,**fields):
        with self.guard:self.state['tasks'][task_id].update(fields);self._save()
    def _reserve(self,tid):
        while True:
            for gpu in range(4):
                lp=GPU_LEASE_DIR/f'gpu{gpu}.lock';lp.parent.mkdir(parents=True,exist_ok=True);lease=lp.open('a+')
                try:fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:lease.close();continue
                try:
                    uuid,memory,owners=owner._nvidia_gpu_info(gpu)
                    if not owners and memory<=GPU_LIMIT_MIB:return gpu,lease,uuid,memory
                except BaseException:fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();raise
                fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close()
            self._update(tid,status='waiting_for_exclusive_gpu',last_wait=time.time());time.sleep(5)
    @staticmethod
    def _terminate(proc):
        if proc is None or proc.poll() is not None:return
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()
    def _execute(self,task):
        tid=task['task_id'];gpu=lease=None;proc=None;ok=False;cache=None
        try:
            if task['stage']=='score':env=environment(task);cache=None
            else:
                gpu,lease,uuid,memory=self._reserve(tid);cache=Path('/tmp')/f'kevinswk_sw0142_triton_{os.getpid()}_{tid}';cache.mkdir(exist_ok=False);env=environment(task,gpu,cache)
                self._update(tid,status='reserved',gpu=gpu,gpu_uuid=uuid,memory_before_mib=memory,lease_path=str(GPU_LEASE_DIR/f'gpu{gpu}.lock'))
            log=ARCHIVE/f'{tid}.log';self._update(tid,status='running',child_started=time.time(),log_path=str(log),argv=command(task),cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'))
            with log.open('x',encoding='utf-8',newline='\n') as out:
                kwargs={'env':env,'stdout':out,'stderr':subprocess.STDOUT,'start_new_session':True,'cwd':str(run.ROOT)}
                if lease is not None:kwargs['pass_fds']=(lease.fileno(),)
                proc=subprocess.Popen(command(task),**kwargs)
                self._update(tid,child_pid=proc.pid)
                while proc.poll() is None:
                    if gpu is not None:
                        _,_,owners=owner._nvidia_gpu_info(gpu);owned=owner._process_tree(proc.pid);foreign=owner.foreign_owner_pids(owners,owned)
                        if foreign:
                            self._terminate(proc);self._update(tid,status='interrupted_foreign_gpu_owner',foreign_owner_pids=foreign,returncode=proc.returncode);return False
                    time.sleep(5)
                rc=proc.wait()
            ok=rc==0 and validate_result(task);self._update(tid,status='passed' if ok else 'failed',returncode=rc,artifact_valid=bool(ok),completed=time.time());return ok
        except BaseException as exc:
            if proc is not None:self._terminate(proc)
            self._update(tid,status='failed',error=f'{type(exc).__name__}: {exc}',returncode=proc.returncode if proc else None);return False
        finally:
            if proc is not None and proc.poll() is None:self._terminate(proc)
            if lease is not None:
                try:fcntl.flock(lease.fileno(),fcntl.LOCK_UN)
                finally:lease.close()
            if cache is not None and cache.exists():
                # Triton cache is isolated operational scratch; retain on abnormal termination.
                if ok:
                    import shutil;shutil.rmtree(cache)
    def run(self):
        failed=False
        for group in task_groups():
            tasks=[next(t for t in self.tasks if t['task_id']==tid) for tid in group]
            dependencies_ok=True
            for task in tasks:
                for dependency_id in task['deps']:
                    dependency=next(row for row in self.tasks if row['task_id']==dependency_id)
                    if not validate_result(dependency):dependencies_ok=False;break
                if not dependencies_ok:break
            if not dependencies_ok:
                failed=True;break
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as pool:
                results=list(pool.map(self._execute,tasks))
            if not all(results):failed=True;break
        self.state['status']='failed' if failed else 'complete';self.state['completed']=time.time();self._save();return 1 if failed else 0

def main(argv=None):
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');a=p.parse_args(argv)
    if a.dry_run:print(json.dumps({'experiment':run.NAME,'tasks':task_plan(),'groups':task_groups()},indent=2));return 0
    return Queue().run()
if __name__=='__main__':raise SystemExit(main())
