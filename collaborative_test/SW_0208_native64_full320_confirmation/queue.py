"""Four independently resumable exclusive-GPU prediction shards; GT-last score."""
import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

WORKER='--worker' in sys.argv
if not WORKER:os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'

def load(name):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    try:return importlib.import_module(name)
    finally:
        if visible is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=visible

api=load('collaborative_test.SW_0208_native64_full320_confirmation.run')
MODULE='collaborative_test.SW_0208_native64_full320_confirmation.queue'
ROOT=api.HERE.parents[1]

def worker_command(shard,gpu):
    api.offsets(shard)
    if type(gpu) is not int or gpu not in range(4):raise ValueError('invalid physical GPU')
    return [sys.executable,'-u','-m',MODULE,'--worker','--shard',str(shard),'--physical-gpu',str(gpu)]

def shard_done(shard):
    return all((api.ARCHIVE/arm/f'shard{shard}.json').is_file() for arm in api.ARMS)

def maybe_score():
    import fcntl
    if not all(shard_done(s) for s in range(4)):return
    with (api.ARCHIVE/'score.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if not (api.ARCHIVE/'score_v1/summary.json').exists():api.score()

def supervise(shard):
    import fcntl
    api.offsets(shard);stop=threading.Event()
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,lambda *_:stop.set())
    runtime=api.ARCHIVE/'queue'/f'shard{shard}';runtime.mkdir(parents=True,exist_ok=True)
    lock=(runtime/'queue.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=runtime/'state.json'
    if state.exists():raise FileExistsError('preserve existing shard supervisor state; never duplicate')
    owner=load('collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4')
    admission=json.loads((api.ARCHIVE/'registration_v1/admission.json').read_bytes())
    if admission['cuda_initialized'] or admission['binding']['fingerprint']!=api.fingerprint():raise ValueError('CPU admission invalid')
    child=lease=None;attempt=0
    def record(status,**fields):api.write_json(state,dict(status=status,shard=shard,supervisor_pid=os.getpid(),updated_unix=time.time(),**fields),replace=True)
    try:
        record('waiting_for_exclusive_gpu')
        while not stop.is_set():
            gpu,uuid,_,lease=owner._acquire(stop)
            if not owner._stable_idle(gpu,uuid,stop,lambda *_:None):
                fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None;continue
            if api.fingerprint()!=admission['binding']['fingerprint']:raise RuntimeError('source changed while queued')
            attempt+=1;command=worker_command(shard,gpu)
            with (runtime/f'attempt{attempt:02d}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
                with (runtime/'gpu_attempts.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(attempt=attempt,physical_gpu=gpu,gpu_uuid=uuid,child_pid=child.pid,child_argv=command,observed_unix=time.time()))+'\n');stream.flush();os.fsync(stream.fileno())
                record('running_worker',physical_gpu=gpu,gpu_uuid=uuid,child_pid=child.pid,child_argv=command)
                yielded=False
                while child.poll() is None and not stop.wait(5):
                    live_uuid,_,owners=owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None:break
                    if live_uuid!=uuid or owner.gpu_guard._foreign_owners(owners,owner.gpu_guard._process_tree(child.pid)):
                        owner._stop_owned(child);yielded=True;break
                if stop.is_set() and child.poll() is None:owner._stop_owned(child)
                rc=child.wait()
            if shard_done(shard):break
            if not yielded or stop.is_set():raise RuntimeError(f'shard worker exited {rc}; inspect preserved log')
            child=None;fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None
            record('waiting_for_exclusive_gpu',last_yielded_attempt=attempt)
        if lease is not None:fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None
        if stop.is_set():record('stopped_by_signal');return
        record('prediction_shard_complete');maybe_score()
    except BaseException as exc:record('failed',error=repr(exc));raise
    finally:
        if child is not None and child.poll() is None:owner._stop_owned(child)
        if lease is not None:fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close()
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--shard',required=True,type=int,choices=range(4));parser.add_argument('--physical-gpu',type=int,choices=range(4));args=parser.parse_args()
    if args.worker:
        if args.physical_gpu is None or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.physical_gpu):raise RuntimeError('wrong assigned GPU visibility')
        api.predict(args.shard)
    else:supervise(args.shard)

if __name__=='__main__':main()
