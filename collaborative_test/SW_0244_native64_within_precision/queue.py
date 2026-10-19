"""Independent native-source queues with exclusive leases and per-row recovery."""
import argparse,importlib,json,os,signal,subprocess,sys,threading,time
if sys.platform=='linux':
    import ctypes
    if ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('per-process THP disable failed')
if '--worker' not in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'


def load(name):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    try:return importlib.import_module(name)
    finally:
        if visible is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=visible


api=load('collaborative_test.SW_0244_native64_within_precision.run')
MODULE='collaborative_test.SW_0244_native64_within_precision.queue';ROOT=api.ROOT


def worker_command(seed,gpu):
    api.require_seed(seed)
    if type(gpu) is not int or gpu not in range(4):raise ValueError('exclusive physical GPU required')
    return [sys.executable,'-u','-m',MODULE,'--worker','--seed',str(seed),'--physical-gpu',str(gpu),'--arm',api.require_arm(api.ARM)]


def maybe_score():
    import fcntl
    if not all((api.archive_for(s,arm)/'protocol.json').exists() for arm in api.ARMS for s in api.SEEDS):return
    with (api.ROOT_ARCHIVE/'score.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if not (api.ROOT_ARCHIVE/'score_v1/summary.json').exists():api.score()


def supervise(seed):
    import fcntl
    api.require_seed(seed);stop=threading.Event()
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,lambda *_:stop.set())
    runtime=api.archive_for(seed)/'queue';runtime.mkdir(parents=True,exist_ok=True)
    lock=(runtime/'queue.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=runtime/'state.json'
    if state.exists():raise FileExistsError('preserve existing native queue; do not duplicate')
    admission=json.loads((api.archive_for(seed)/'admission.json').read_bytes())
    if admission['cuda_initialized'] or admission['binding']['fingerprint']!=api.fingerprint():raise ValueError('CPU admission/source mismatch')
    owner=load('collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4');child=lease=None;attempt=0
    def record(status,**fields):api.write_json(state,dict(status=status,seed=seed,supervisor_pid=os.getpid(),updated_unix=time.time(),**fields),replace=True)
    try:
        record('waiting_for_exclusive_gpu')
        while not stop.is_set():
            gpu,uuid,_,lease=owner._acquire(stop)
            if not owner._stable_idle(gpu,uuid,stop,lambda *_:None):
                fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None;continue
            if api.fingerprint()!=admission['binding']['fingerprint']:raise RuntimeError('native source changed while queued')
            attempt+=1;command=worker_command(seed,gpu)
            with (runtime/f'attempt{attempt:02d}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
                with (runtime/'gpu_attempts.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(attempt=attempt,physical_gpu=gpu,gpu_uuid=uuid,child_pid=child.pid,child_argv=command,observed_unix=time.time()))+'\n');stream.flush();os.fsync(stream.fileno())
                record('running_worker',physical_gpu=gpu,gpu_uuid=uuid,child_pid=child.pid,child_argv=command);yielded=False;deadline=time.monotonic()+3600
                while child.poll() is None and not stop.wait(5):
                    if time.monotonic()>deadline:raise TimeoutError('native active worker exceeded8h')
                    live_uuid,_,owners=owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None:break
                    if live_uuid!=uuid or owner.gpu_guard._foreign_owners(owners,owner.gpu_guard._process_tree(child.pid)):
                        owner._stop_owned(child);yielded=True;break
                if stop.is_set() and child.poll() is None:owner._stop_owned(child)
                rc=child.wait()
            if (api.archive_for(seed)/'protocol.json').exists() and rc==0:break
            if not yielded or stop.is_set():raise RuntimeError(f'native worker exited{rc}; inspect preserved log')
            child=None;fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None;record('waiting_for_exclusive_gpu',last_yielded_attempt=attempt)
        if lease is not None:fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None
        if stop.is_set():record('stopped_by_signal');return
        api.validated_seed(seed)
        record('complete');maybe_score()
    except BaseException as exc:record('failed',error=repr(exc));raise
    finally:
        if child is not None and child.poll() is None:owner._stop_owned(child)
        if lease is not None:fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close()
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--seed',type=int,choices=api.SEEDS,required=True);p.add_argument('--physical-gpu',type=int,choices=range(4));p.add_argument('--arm',choices=tuple(api.ARMS),required=True);args=p.parse_args();api.activate(args.arm)
    if args.worker:
        if args.physical_gpu is None or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.physical_gpu):raise RuntimeError('wrong assigned GPU visibility')
        api.predict(args.seed)
    else:supervise(args.seed)
