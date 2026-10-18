"""One bounded read-only diagnosis on the next exclusive idle GPU0..3."""
import argparse,importlib,json,os,signal,subprocess,sys,threading,time
if '--worker' not in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'


def load(name):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    try:return importlib.import_module(name)
    finally:
        if visible is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=visible


api=load('collaborative_test.SW_0220_native64_Adam_state_audit.run')
MODULE='collaborative_test.SW_0220_native64_Adam_state_audit.queue'


def worker_command(gpu):
    if type(gpu) is not int or gpu not in range(4):raise ValueError('exclusive physical GPU0..3 required')
    return [sys.executable,'-u','-m',MODULE,'--worker','--physical-gpu',str(gpu)]


def supervise():
    import fcntl
    stop=threading.Event()
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,lambda *_:stop.set())
    runtime=api.ARCHIVE/'queue';runtime.mkdir(parents=True,exist_ok=True)
    lock=(runtime/'queue.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=runtime/'state.json'
    if state.exists():raise FileExistsError('preserve registered diagnosis; do not duplicate')
    admission=json.loads((api.ARCHIVE/'cpu_v1/admission.json').read_bytes())
    if admission['CUDA_initialized'] or admission['binding']['fingerprint']!=api.fingerprint():raise ValueError('CPU admission/source mismatch')
    owner=load('collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4');child=lease=None;attempt=0
    def record(status,**fields):api.write_json(state,dict(status=status,supervisor_pid=os.getpid(),updated_unix=time.time(),**fields),replace=True)
    try:
        record('waiting_for_exclusive_gpu')
        while not stop.is_set():
            gpu,uuid,_,lease=owner._acquire(stop)
            if not owner._stable_idle(gpu,uuid,stop,lambda *_:None):
                fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None;continue
            if admission['binding']['fingerprint']!=api.fingerprint():raise ValueError('queued source changed')
            attempt+=1;command=worker_command(gpu)
            with (runtime/f'attempt{attempt:02d}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=api.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
                with (runtime/'gpu_attempts.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(attempt=attempt,physical_gpu=gpu,gpu_uuid=uuid,child_pid=child.pid,child_argv=command,observed_unix=time.time()))+'\n');stream.flush();os.fsync(stream.fileno())
                record('running_worker',child_pid=child.pid,child_argv=command,physical_gpu=gpu,gpu_uuid=uuid);yielded=False;deadline=time.monotonic()+4*3600
                while child.poll() is None and not stop.wait(5):
                    if time.monotonic()>deadline:raise TimeoutError('bounded diagnosis exceeded4h')
                    actual,_,owners=owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None:break
                    if actual!=uuid or owner.gpu_guard._foreign_owners(owners,owner.gpu_guard._process_tree(child.pid)):
                        owner._stop_owned(child);yielded=True;break
                if stop.is_set() and child.poll() is None:owner._stop_owned(child)
                rc=child.wait()
            if (api.ARCHIVE/'report.json').exists():api.validate_report();break
            if not yielded or stop.is_set():raise RuntimeError(f'diagnosis worker exited{rc}; inspect preserved log')
            child=None;fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None;record('waiting_for_exclusive_gpu',last_yielded_attempt=attempt)
        if lease is not None:fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None
        if stop.is_set():record('stopped_by_signal');return
        record('complete',report_sha256=api.sha((api.ARCHIVE/'report.json').read_bytes()))
    except BaseException as exc:record('failed',error=repr(exc));raise
    finally:
        if child is not None and child.poll() is None:owner._stop_owned(child)
        if lease is not None:fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close()
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--physical-gpu',type=int,choices=range(4));args=parser.parse_args()
    if args.worker:
        if args.physical_gpu is None or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.physical_gpu):raise ValueError('assigned GPU visibility mismatch')
        prepared=api.prepare_cpu();admission=json.loads((api.ARCHIVE/'cpu_v1/admission.json').read_bytes())
        if prepared['audit_binding']!=admission['binding']:raise ValueError('worker source differs from CPU admission')
        api.torch.cuda.set_per_process_memory_fraction(.8,0)
        result=api.run_audit(prepared);print(json.dumps(dict(status=result['status'],decision=result['decision'])),flush=True)
    else:supervise()
