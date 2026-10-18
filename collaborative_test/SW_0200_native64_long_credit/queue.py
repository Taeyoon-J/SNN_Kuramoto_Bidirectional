"""Exclusive GPU TRAIN64 continuation and candidate-only inference, then CPU score."""
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
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
def load(name):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    try:return importlib.import_module(name)
    finally:
        if visible is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=visible
api=load('collaborative_test.SW_0200_native64_long_credit.train')
MODULE='collaborative_test.SW_0200_native64_long_credit.queue'
ROOT=api.HERE.parents[1];RUNTIME=api.ARCHIVE/'queue'
STOP=threading.Event()


def worker_command(gpu):
    if type(gpu) is not int or gpu not in range(4):raise ValueError('invalid physical GPU')
    return [sys.executable,'-u','-m',MODULE,'--worker','--physical-gpu',str(gpu)]


def supervise():
    import fcntl
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,lambda *_:STOP.set())
    RUNTIME.mkdir(parents=True,exist_ok=True);lock=(RUNTIME/'queue.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=RUNTIME/'state.json'
    if state.exists():raise FileExistsError('preserve existing TRAIN64 queue')
    owner=load('collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4')
    admission=json.loads((api.ARCHIVE/'cpu_v1/admission.json').read_bytes())
    bindings=admission['bindings']
    if admission['status']!='SW0200_exact_native64_weights_and_Adam_CPU_admitted' or admission['cuda_initialized'] or bindings['fingerprint']!=api.fingerprint():raise ValueError('CPU admission invalid')
    child=lease=None;attempt=0
    def record(status,**fields):api.write_json(state,dict(status=status,supervisor_pid=os.getpid(),updated_unix=time.time(),**fields),replace=True)
    try:
        record('waiting_for_exclusive_gpu')
        while not STOP.is_set():
            gpu,uuid,_memory,lease=owner._acquire(STOP)
            if not owner._stable_idle(gpu,uuid,STOP,lambda *_:None):
                fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None;continue
            if api.fingerprint()!=bindings['fingerprint']:raise RuntimeError('TRAIN64 source changed while queued')
            attempt+=1;command=worker_command(gpu)
            with (RUNTIME/f'attempt{attempt:02d}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
                with (RUNTIME/'gpu_attempts.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(attempt=attempt,physical_gpu=gpu,gpu_uuid=uuid,child_pid=child.pid,child_argv=command,observed_unix=time.time()))+'\n');stream.flush();os.fsync(stream.fileno())
                record('running_worker',child_pid=child.pid,child_argv=command,physical_gpu=gpu,gpu_uuid=uuid)
                yielded=False;deadline=time.monotonic()+8*3600
                while child.poll() is None and not STOP.wait(3):
                    if time.monotonic()>deadline:raise TimeoutError('TRAIN64 active worker exceeded eight hours')
                    live_uuid,_memory,owners=owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None:break
                    if live_uuid!=uuid or owner.gpu_guard._foreign_owners(owners,owner.gpu_guard._process_tree(child.pid)):
                        owner._stop_owned(child);yielded=True;break
                if STOP.is_set() and child.poll() is None:owner._stop_owned(child)
                rc=child.wait()
            if (api.ARCHIVE/'gradient_admission/rejected.json').exists():
                record('complete_rejected_gradient_admission');return
            if (api.ARCHIVE/'candidate/protocol.json').exists():
                from .evaluate import validated_candidate
                validated_candidate();break
            if not yielded or STOP.is_set():raise RuntimeError(f'TRAIN64 worker exited {rc}; inspect its preserved log')
            child=None;fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None
            record('waiting_for_exclusive_gpu',last_yielded_attempt=attempt)
        if lease is not None:fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None
        if STOP.is_set():record('stopped_by_signal');return
        record('scoring_frozen_candidate')
        from .evaluate import score_candidate
        summary=score_candidate()
        record('complete',summary_sha256=api.sha((api.ARCHIVE/'score_v1/summary.json').read_bytes()),screen_gate_passed=summary['screen_gate_passed'])
    except BaseException as exc:record('failed',error=repr(exc));raise
    finally:
        if child is not None and child.poll() is None:owner._stop_owned(child)
        if lease is not None:fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close()
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--physical-gpu',type=int,choices=range(4));args=parser.parse_args()
    if args.worker:
        if args.physical_gpu is None or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.physical_gpu):raise RuntimeError('wrong assigned GPU visibility')
        prepared=api.prepare_inputs_cpu();admission=json.loads((api.ARCHIVE/'cpu_v1/admission.json').read_bytes())
        if prepared['bindings']!=admission['bindings']:raise ValueError('worker differs from admitted exact model/Adam lineage')
        from .evaluate import gradient_admission
        if not gradient_admission(prepared):return
        report=api.run_training(prepared)
        from .evaluate import predict_candidate
        predict_candidate(prepared,report)
    else:supervise()

if __name__=='__main__':main()
