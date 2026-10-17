"""Wait for actual SW0192 admission, then exclusive-GPU prediction and CPU score."""
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
api=load('collaborative_test.SW_0193_native64_probe_screen.run')
MODULE='collaborative_test.SW_0193_native64_probe_screen.queue'
ROOT=api.HERE.parents[1]
RUNTIME=api.ARCHIVE/'screen_queue'
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
    if state.exists():raise FileExistsError('preserve existing screen queue')
    owner=load('collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4')
    child=lease=None;attempt=0
    def record(status,**fields):api.write_json(state,dict(status=status,supervisor_pid=os.getpid(),updated_unix=time.time(),**fields),replace=True)
    try:
        while not STOP.is_set():
            parent=json.loads((api.probe.ARCHIVE/'probe_queue/state.json').read_bytes())
            if parent['status']=='complete':break
            if parent['status']=='complete_rejected':record('complete_skipped_parent_scientific_rejection');return
            if parent['status']=='failed':raise RuntimeError('parent probe failed; inspect its preserved log')
            proc=owner._proc_record(parent['supervisor_pid'])
            if proc is None or proc['state']=='Z':raise RuntimeError('parent probe missing without terminal state')
            argv=proc.get('argv',[])
            if '-m' not in argv or argv[argv.index('-m')+1]!='collaborative_test.SW_0192_native64_incumbent_mini.queue':raise RuntimeError('parent supervisor PID identity changed')
            record('waiting_for_parent_admission',parent_pid=parent['supervisor_pid']);STOP.wait(15)
        if STOP.is_set():raise RuntimeError('screen queue stopped')
        prepared=api.prepare_inputs_cpu();bindings=prepared['bindings'];del prepared
        api.write_json(api.ARCHIVE/'cpu_v1/admission.json',dict(status='SW0193_CPU_parent_and_baseline_verified',bindings=bindings,cuda_initialized=api.torch.cuda.is_initialized()))
        record('waiting_for_exclusive_gpu')
        while not STOP.is_set():
            gpu,uuid,_memory,lease=owner._acquire(STOP)
            if not owner._stable_idle(gpu,uuid,STOP,lambda *_:None):
                fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None;continue
            if api.fingerprint()!=bindings['fingerprint']:raise RuntimeError('screen source changed while queued')
            attempt+=1;command=worker_command(gpu)
            with (RUNTIME/f'attempt{attempt:02d}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
                with (RUNTIME/'gpu_attempts.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(attempt=attempt,physical_gpu=gpu,gpu_uuid=uuid,child_pid=child.pid,child_argv=command))+'\n');stream.flush();os.fsync(stream.fileno())
                record('running_worker',child_pid=child.pid,child_argv=command,physical_gpu=gpu,gpu_uuid=uuid)
                yielded=False;deadline=time.monotonic()+4*3600
                while child.poll() is None and not STOP.wait(3):
                    if time.monotonic()>deadline:raise TimeoutError('native64 screen exceeded four active hours')
                    live_uuid,_memory,owners=owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None:break
                    if live_uuid!=uuid or owner.gpu_guard._foreign_owners(owners,owner.gpu_guard._process_tree(child.pid)):
                        owner._stop_owned(child);yielded=True;break
                if STOP.is_set() and child.poll() is None:owner._stop_owned(child)
                rc=child.wait()
            if all((api.ARCHIVE/arm/'protocol.json').exists() for arm in api.ARMS):
                from .score import validated_pair
                validated_pair();break
            if not yielded or STOP.is_set():raise RuntimeError(f'screen worker exited {rc}; inspect attempt log')
            child=None;fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None
            record('waiting_for_exclusive_gpu')
        # Release the inference GPU before CPU-only metric computation.
        if lease is not None:fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None
        record('scoring_frozen_pair')
        from .score import score_pair
        summary=score_pair()
        record('complete',summary_sha256=api.sha((api.ARCHIVE/'score_v1/summary.json').read_bytes()),screen_gate_passed=summary['screen_gate_passed'])
    except BaseException as exc:record('failed',error=repr(exc));raise
    finally:
        if child is not None and child.poll() is None:owner._stop_owned(child)
        if lease is not None:fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close()
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--physical-gpu',type=int,choices=range(4));args=parser.parse_args()
    if args.worker:
        if args.physical_gpu is None or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.physical_gpu):raise RuntimeError('wrong leased GPU visibility')
        prepared=api.prepare_inputs_cpu();admission=json.loads((api.ARCHIVE/'cpu_v1/admission.json').read_bytes())
        if prepared['bindings']!=admission['bindings']:raise ValueError('worker differs from admitted screen lineage')
        api.predict_pair(prepared)
    else:supervise()

if __name__=='__main__':main()
