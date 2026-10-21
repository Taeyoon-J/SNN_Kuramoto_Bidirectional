"""Twelve immediately registered independent jobs; shared GPUs, no foreign yielding."""
import argparse,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from . import scope,factory
from collaborative_test.SW_0240_native64_70000_cached_io import shared_gpu_policy as policy

MODULE='collaborative_test.SW_0249_four_model_35000'

def alive(pid):return policy.alive(pid)

def command(model,seed,stage,gpu=None):
    scope.require(model,seed)
    if stage not in ('admit','worker','reproduce'):raise ValueError('invalid stage')
    cmd=[sys.executable,'-u','-B','-m',MODULE+'.run','--model',model,'--seed',str(seed),'--'+stage]
    if gpu is not None:
        if type(gpu) is not int or gpu not in range(4):raise ValueError('invalid GPU')
        cmd+=['--gpu',str(gpu)]
    return cmd

def supervise(model,seed):
    import ctypes,fcntl
    if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('THP disable failed')
    from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner
    a=factory.archive(model,seed);q=a/'queue';q.mkdir(parents=True,exist_ok=True)
    lock=(q/'queue.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (q/'state.json').exists():raise FileExistsError('preserve existing job; no duplicate supervisor')
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    def record(status,**fields):
        factory.old.write_json(q/'state.json',dict(status=status,model=model,seed=seed,supervisor_pid=os.getpid(),updated_unix=time.time(),resource_policy=policy.NAME,**fields),replace=True)
    child=lease=None
    def gpu_stage(stage,attempt):
        nonlocal child,lease
        record('waiting_for_sufficient_free_memory',stage=stage)
        gpu,uuid,_,lease=policy.acquire(owner,stop)
        if not policy.stable_capacity(owner,gpu,uuid,stop):
            fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None;return False
        path=q/f'{stage}_attempt{attempt:03d}.log'
        with path.open('xb') as log:
            child=subprocess.Popen(command(model,seed,stage,gpu),cwd=factory.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
            record('running_worker',stage=stage,physical_gpu=gpu,gpu_uuid=uuid,child_pid=child.pid,log=str(path))
            while child.poll() is None and not stop.wait(5):pass
            if stop.is_set() and child.poll() is None:owner._stop_owned(child)
            rc=child.wait();child=None
        fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None
        if rc:
            tail=path.read_text(errors='replace')[-4000:]
            if 'out of memory' in tail.lower() and not stop.is_set():
                record('retry_after_shared_memory_oom',stage=stage,log=str(path));stop.wait(30);return False
            raise RuntimeError(f'preserved {stage} failure rc={rc}; see {path}')
        return True
    try:
        attempt=0
        if factory.need_reproduction(model,seed):
            while factory.need_reproduction(model,seed) and not stop.is_set():
                attempt+=1;gpu_stage('reproduce',attempt)
        if stop.is_set():record('stopped_by_user_signal');return
        record('actual_CPU_admission')
        with (q/'CPU_admission.stdout').open('xb') as out,(q/'CPU_admission.stderr').open('xb') as err:
            r=subprocess.run(command(model,seed,'admit'),cwd=factory.ROOT,env=os.environ,stdout=out,stderr=err,timeout=1200)
        if r.returncode:raise RuntimeError('actual CPU source/gradient admission failed; preserved stderr')
        while not (a/'protocol.json').exists() and not stop.is_set():
            attempt+=1;gpu_stage('worker',attempt)
        record('complete' if (a/'protocol.json').exists() else 'stopped_by_user_signal')
    except BaseException as exc:
        record('failed_preserved_for_review',error=repr(exc));raise
    finally:
        if child is not None and child.poll() is None:owner._stop_owned(child)
        if lease is not None:fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close()
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()

def coordinate():
    import fcntl
    a=factory.ARCHIVE;a.mkdir(parents=True,exist_ok=True)
    lock=(a/'coordinator.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=a/'registration.json'
    if path.exists():raise FileExistsError('all jobs already registered; never duplicate')
    record=dict(status='registering_twelve_jobs',pid=os.getpid(),specification=scope.specification(),jobs=[],registered_unix=time.time())
    factory.old.write_json(path,record)
    for model in scope.MODELS:
        for seed in scope.SEEDS:
            folder=factory.archive(model,seed);folder.mkdir(parents=True,exist_ok=True)
            with (folder/'supervisor.stdout').open('xb') as out,(folder/'supervisor.stderr').open('xb') as err:
                p=subprocess.Popen([sys.executable,'-u','-B','-m',MODULE+'.queue','--model',model,'--seed',str(seed)],cwd=factory.ROOT,env=os.environ,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            record['jobs'].append(dict(model=model,seed=seed,supervisor_pid=p.pid));factory.old.write_json(path,record,replace=True)
    record['status']='all_twelve_jobs_queued';factory.old.write_json(path,record,replace=True)
    while True:
        states=[]
        for job in record['jobs']:
            q=factory.archive(job['model'],job['seed'])/'queue/state.json'
            if q.exists():
                state=json.loads(q.read_bytes());states.append(state)
                if state['status']=='failed_preserved_for_review':
                    factory.old.write_json(a/'state.json',dict(status='job_failure_preserved_other_jobs_continue',job=job,state=state),replace=True)
                    raise RuntimeError('one job failed; all other independent queues left untouched')
            elif not alive(job['supervisor_pid']):raise RuntimeError('registered supervisor exited before first state')
        if len(states)==12 and all(s['status']=='complete' for s in states):break
        factory.old.write_json(a/'state.json',dict(status='training_and_evaluation',completed_jobs=sum(s['status']=='complete' for s in states),jobs=states,updated_unix=time.time()),replace=True)
        time.sleep(30)
    factory.old.write_json(a/'state.json',dict(status='all_predictions_complete_CPU_scoring_and_audit',updated_unix=time.time()),replace=True)
    with (a/'completion.stdout').open('xb') as out,(a/'completion.stderr').open('xb') as err:
        r=subprocess.run([sys.executable,'-B','-m',MODULE+'.evaluate'],cwd=factory.ROOT,env=os.environ,stdout=out,stderr=err,timeout=12*3600)
    if r.returncode:raise RuntimeError('CPU score/audit failed; predictions/checkpoints preserved')
    factory.old.write_json(a/'state.json',dict(status='complete_final_comparison_no_followup',updated_unix=time.time()),replace=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--coordinate',action='store_true');parser.add_argument('--model',choices=scope.MODELS);parser.add_argument('--seed',type=int,choices=scope.SEEDS)
    args=parser.parse_args()
    if args.coordinate:coordinate()
    elif args.model is not None and args.seed is not None:supervise(args.model,args.seed)
    else:parser.error('coordinator or model/seed required')
