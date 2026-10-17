"""One exclusive idle GPU; resume the same probe after a foreign-owner yield."""
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
ASSIGNED=os.environ.get('CUDA_VISIBLE_DEVICES')
if not WORKER:os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'

def load(name):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    try:return importlib.import_module(name)
    finally:
        if visible is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=visible

api=load('collaborative_test.SW_0192_native64_incumbent_mini.run')
MODULE='collaborative_test.SW_0192_native64_incumbent_mini.queue'
ROOT=api.HERE.parents[1]
RUNTIME=api.ARCHIVE/'probe_queue'
OUTPUT=api.ARCHIVE/'probe_seed0/report.json'
STOP=threading.Event()


def worker_command(gpu):
    if type(gpu) is not int or gpu not in range(4):raise ValueError('invalid physical GPU')
    return [sys.executable,'-u','-m',MODULE,'--worker','--physical-gpu',str(gpu)]


def validate_report(report,bindings):
    if report.get('bindings')!=bindings or report.get('validation_data_read') is not False:raise ValueError('probe bindings/scope changed')
    status=report.get('status','')
    if not (status=='complete_native64_training_admission' or status.startswith('complete_rejected_')):raise ValueError('nonterminal probe report')
    history=report.get('history',[])
    if not history or len(history)>api.PROPOSALS or any(row['proposal']!=i+1 for i,row in enumerate(history)):raise ValueError('probe history invalid')
    if status=='complete_native64_training_admission':
        if len(history)!=api.PROPOSALS or report['accepted_updates']<1 or not report['source_CNN_changed']:raise ValueError('probe did not commit learning')
        if any(row['credit_failures'] for row in history):raise ValueError('successful probe has missing credit')


def supervise():
    import fcntl
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,lambda *_:STOP.set())
    RUNTIME.mkdir(parents=True,exist_ok=True)
    lock=(RUNTIME/'queue.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=RUNTIME/'state.json'
    if state.exists() or OUTPUT.exists():raise FileExistsError('preserve existing probe queue')
    admission=json.loads((api.ARCHIVE/'cpu_v1/admission.json').read_bytes())
    bindings=admission['bindings']
    if admission['status']!='native64_exact_incumbent_CPU_admitted' or admission['cuda_initialized'] or bindings['fingerprint']!=api.fingerprint():raise ValueError('stale/missing CPU admission')
    owner=load('collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4')
    child=lease=None;attempt=0
    def record(status,**fields):
        api.write_json(state,dict(status=status,supervisor_pid=os.getpid(),updated_unix=time.time(),validation_data_read=False,**fields),replace=True)
    try:
        record('waiting_for_exclusive_gpu')
        while not STOP.is_set():
            gpu,uuid,_memory,lease=owner._acquire(STOP)
            if not owner._stable_idle(gpu,uuid,STOP,lambda elapsed,memory:record('waiting_for_stable_gpu',physical_gpu=gpu)):
                fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None;continue
            if api.fingerprint()!=bindings['fingerprint']:raise RuntimeError('probe source changed while queued')
            attempt+=1;command=worker_command(gpu)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
            with (RUNTIME/f'attempt{attempt:02d}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lease.fileno(),))
                with (RUNTIME/'gpu_attempts.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(attempt=attempt,physical_gpu=gpu,gpu_uuid=uuid,
                        child_pid=child.pid,child_argv=command,observed_unix=time.time()))+'\n')
                    stream.flush();os.fsync(stream.fileno())
                record('running_worker',child_pid=child.pid,child_argv=command,physical_gpu=gpu,gpu_uuid=uuid)
                deadline=time.monotonic()+4*3600;yielded=False
                while child.poll() is None and not STOP.wait(3):
                    if time.monotonic()>deadline:raise TimeoutError('two-proposal native64 probe exceeded four active hours')
                    live_uuid,_memory,owners=owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None:break
                    foreign=owner.gpu_guard._foreign_owners(owners,owner.gpu_guard._process_tree(child.pid))
                    if live_uuid!=uuid or foreign:
                        owner._stop_owned(child);yielded=True;break
                if STOP.is_set() and child.poll() is None:owner._stop_owned(child)
                rc=child.wait()
            if OUTPUT.is_file():
                report=json.loads(OUTPUT.read_bytes());validate_report(report,bindings)
                record('complete' if report['status']=='complete_native64_training_admission' else 'complete_rejected',
                    report_status=report['status'],report_sha256=api.sha(OUTPUT.read_bytes()))
                return
            if not yielded or STOP.is_set():raise RuntimeError(f'probe exited {rc}; inspect attempt{attempt:02d}.log')
            record('waiting_for_exclusive_gpu',yielded_attempt=attempt)
            child=None;fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();lease=None
    except BaseException as exc:
        record('failed',error=repr(exc));raise
    finally:
        if child is not None and child.poll() is None:owner._stop_owned(child)
        if lease is not None:fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close()
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true')
    parser.add_argument('--physical-gpu',type=int,choices=range(4));args=parser.parse_args()
    if args.worker:
        if args.physical_gpu is None or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.physical_gpu):raise RuntimeError('worker GPU visibility mismatch')
        prepared=api.prepare_inputs_cpu()
        admission=json.loads((api.ARCHIVE/'cpu_v1/admission.json').read_bytes())
        if prepared['bindings']!=admission['bindings']:raise ValueError('worker CPU initialization differs from admitted model')
        api.run_probe(prepared,OUTPUT,api.ARCHIVE/'probe_work_seed0')
    else:supervise()

if __name__=='__main__':main()
