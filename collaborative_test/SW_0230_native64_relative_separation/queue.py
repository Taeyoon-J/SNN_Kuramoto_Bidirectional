"""Three resumable teacher-denoising jobs, exclusive shared idle GPU0..3 leases, one CPU scorer."""
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


api=load('collaborative_test.SW_0230_native64_relative_separation.train')
MODULE='collaborative_test.SW_0230_native64_relative_separation.queue'


def confirmation_first_attempts_started(archive=None):
    from pathlib import Path
    base=Path(archive) if archive is not None else api.ROOT/'collaborative_test/SW_0227_native64_joint_projection_full320/results_archive'
    try:
        for seed in (0,1,2):
            lines=(base/f'seed{seed}/queue/gpu_attempts.jsonl').read_text().splitlines()
            if not lines:return False
            row=json.loads(lines[-1])
            if row['attempt']<1 or row['physical_gpu'] not in range(4) or row['child_pid']<=0:return False
    except (OSError,KeyError,ValueError,TypeError):return False
    return True


def worker_command(seed,count,gpu):
    api.require_job(seed,count)
    if type(gpu) is not int or gpu not in range(4):raise ValueError('exclusive physical GPU0..3 required')
    return [sys.executable,'-u','-m',MODULE,'--seed',str(seed),'--count',str(count),'--worker','--physical-gpu',str(gpu)]


def try_score():
    import fcntl
    lock=(api.ARCHIVE/'score.lock').open('a+')
    try:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        if (api.ARCHIVE/'score_v1/summary.json').exists():return
        if not all((api.archive_for(s,n)/'protocol.json').exists() for s in api.SEEDS for n in api.COUNTS):return
        evaluation=load('collaborative_test.SW_0230_native64_relative_separation.evaluate')
        evaluation.score()
    finally:lock.close()


def supervise(seed,count):
    import fcntl
    api.require_job(seed,count);stop=threading.Event()
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,lambda *_:stop.set())
    a=api.archive_for(seed,count);runtime=a/'queue';runtime.mkdir(parents=True,exist_ok=True)
    lock=(runtime/'queue.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=runtime/'state.json'
    if state.exists():raise FileExistsError('existing registered queue; inspect live PID before recovery')
    admission=json.loads((a/'admission.json').read_bytes())
    if admission['CUDA_initialized'] or admission['binding']['fingerprint']!=api.fingerprint():raise ValueError('CPU admission/source mismatch')
    owner=load('collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4');child=lease=None;attempt=0
    def record(status,**fields):api.write_json(state,dict(status=status,seed=seed,TRAIN_count=count,supervisor_pid=os.getpid(),updated_unix=time.time(),**fields),replace=True)
    try:
        record('waiting_for_full320_first_attempts')
        while not stop.is_set() and not confirmation_first_attempts_started():stop.wait(15)
        if stop.is_set():record('stopped_by_signal');return
        record('waiting_for_exclusive_gpu')
        while not stop.is_set():
            gpu,uuid,_,lease=owner._acquire(stop)
            if not owner._stable_idle(gpu,uuid,stop,lambda *_:None):
                fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None;continue
            if admission['binding']['fingerprint']!=api.fingerprint():raise ValueError('queued source changed')
            attempt+=1;command=worker_command(seed,count,gpu)
            with (runtime/f'attempt{attempt:03d}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=api.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
                with (runtime/'gpu_attempts.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(attempt=attempt,physical_gpu=gpu,gpu_uuid=uuid,child_pid=child.pid,child_argv=command,observed_unix=time.time()))+'\n');stream.flush();os.fsync(stream.fileno())
                record('running_worker',child_pid=child.pid,child_argv=command,physical_gpu=gpu,gpu_uuid=uuid);yielded=False;deadline=time.monotonic()+4*3600
                while child.poll() is None and not stop.wait(5):
                    if time.monotonic()>deadline:raise TimeoutError('bounded mini job exceeded4h')
                    actual,_,owners=owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None:break
                    if actual!=uuid or owner.gpu_guard._foreign_owners(owners,owner.gpu_guard._process_tree(child.pid)):
                        owner._stop_owned(child);yielded=True;break
                if stop.is_set() and child.poll() is None:owner._stop_owned(child)
                rc=child.wait()
            if (a/'protocol.json').exists() and rc==0:break
            if not yielded or stop.is_set():raise RuntimeError(f'worker exited{rc}; inspect preserved log')
            child=None;fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None;record('waiting_for_exclusive_gpu',last_yielded_attempt=attempt)
        if lease is not None:fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None
        if stop.is_set():record('stopped_by_signal');return
        record('complete',protocol_sha256=api.sha((a/'protocol.json').read_bytes()));try_score()
    except BaseException as exc:record('failed',error=repr(exc));raise
    finally:
        if child is not None and child.poll() is None:owner._stop_owned(child)
        if lease is not None:fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close()
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()


def worker(seed,count,gpu):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('assigned GPU visibility mismatch')
    prepared=api.prepare_cpu(seed,count);admission=json.loads((api.archive_for(seed,count)/'admission.json').read_bytes())
    if prepared['binding']!=admission['binding']:raise ValueError('worker source differs from CPU admission')
    device=api.torch.device('cuda:0');api.torch.cuda.set_device(device);api.torch.cuda.set_per_process_memory_fraction(.8,0)
    api.source.source.source.model64.move_to_device(prepared['found'],device)
    runtime=api.resume_policy.compatible_runtime(api.checkpoint.runtime_signature(device))
    report=api.fit(prepared,device,runtime)
    load('collaborative_test.SW_0230_native64_relative_separation.evaluate').predict(prepared,report,device)
    print(json.dumps(dict(status='complete',seed=seed,TRAIN_count=count)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=api.SEEDS,required=True);parser.add_argument('--count',type=int,choices=api.COUNTS,required=True);parser.add_argument('--worker',action='store_true');parser.add_argument('--physical-gpu',type=int,choices=range(4));args=parser.parse_args()
    if args.worker:
        if args.physical_gpu is None:parser.error('worker GPU required')
        worker(args.seed,args.count,args.physical_gpu)
    else:supervise(args.seed,args.count)
