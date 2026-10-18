"""Independent seed baseline, ready on the first exclusive idle GPU among0..3."""
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

WORKER = '--worker' in sys.argv
SEED = int(sys.argv[sys.argv.index('--seed')+1]) if '--seed' in sys.argv else 2
if not WORKER: os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def load(name):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    try: return importlib.import_module(name)
    finally:
        if visible is None: os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else: os.environ['CUDA_VISIBLE_DEVICES'] = visible


api = load('collaborative_test.SW_0207_spike_foreground_classifier.run')
MODULE = 'collaborative_test.SW_0207_spike_foreground_classifier.queue'
api.configure(SEED)
ROOT = api.HERE.parents[1]
RUNTIME = api.ARCHIVE/'queue'
STOP = threading.Event()


def worker_command(gpu):
    if type(gpu) is not int or gpu not in range(4): raise ValueError('invalid physical GPU')
    command = [sys.executable, '-u', '-m', MODULE, '--worker', '--physical-gpu', str(gpu)]
    return command + ['--seed',str(SEED)]


def supervise():
    import fcntl
    for signum in (signal.SIGINT, signal.SIGTERM): signal.signal(signum, lambda *_: STOP.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME/'queue.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (RUNTIME/'state.json').exists(): raise FileExistsError('preserve existing207 queue')
    fingerprint = api.fingerprint(); child = lease = None
    owner = load('collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4')
    def record(status, **fields):
        api.train.write_json(RUNTIME/'state.json', dict(status=status, supervisor_pid=os.getpid(),
            updated_unix=time.time(), **fields), replace=True)
    try:
        if STOP.is_set(): record('stopped_by_signal'); return
        if api.fingerprint() != fingerprint: raise RuntimeError('queued diagnostic source changed')
        record('CPU_admitting_completed_models')
        _found, _states, _rgb, binding = api.prepare_cpu()
        del _found, _states, _rgb
        api.train.write_json(api.ARCHIVE/'cpu_v1/admission.json', dict(binding=binding, cuda_initialized=api.torch.cuda.is_initialized()))
        attempt = 0
        record('waiting_for_exclusive_gpu')
        while not STOP.is_set():
            gpu, uuid, _memory, lease = owner._acquire(STOP)
            if not owner._stable_idle(gpu, uuid, STOP, lambda *_: None):
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None; continue
            if api.fingerprint() != fingerprint: raise RuntimeError('source changed after CPU admission')
            attempt += 1; command = worker_command(gpu)
            with (RUNTIME/f'attempt{attempt:02d}.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(lease.fileno(),))
                with (RUNTIME/'gpu_attempts.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(child_pid=child.pid, argv=command, physical_gpu=gpu,
                        gpu_uuid=uuid, attempt=attempt, observed_unix=time.time()))+'\n')
                    stream.flush(); os.fsync(stream.fileno())
                record('running_worker', child_pid=child.pid, physical_gpu=gpu, gpu_uuid=uuid)
                yielded = False; deadline = time.monotonic()+4*3600
                while child.poll() is None and not STOP.wait(3):
                    if time.monotonic() > deadline: raise TimeoutError('207 active worker exceeded4h')
                    actual_uuid, _memory, owners = owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None: break
                    if actual_uuid != uuid or owner.gpu_guard._foreign_owners(owners, owner.gpu_guard._process_tree(child.pid)):
                        owner._stop_owned(child); yielded = True; break
                if STOP.is_set() and child.poll() is None: owner._stop_owned(child)
                rc = child.wait()
            if all((api.ARCHIVE/arm/'protocol.json').exists() for arm in api.ARMS):
                api.validated_pair(); break
            if not yielded or STOP.is_set(): raise RuntimeError(f'207 exited{rc}; inspect preserved log')
            child = None; fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
            record('waiting_for_exclusive_gpu', last_yielded_attempt=attempt)
        if STOP.is_set(): record('stopped_by_signal'); return
        if lease is not None: fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
        record('scoring_frozen_predictions')
        api.score()
        record('complete', summary_sha256=api.train.sha((api.ARCHIVE/'score_v1/summary.json').read_bytes()))
    except BaseException as exc:
        record('failed', error=repr(exc)); raise
    finally:
        if child is not None and child.poll() is None: owner._stop_owned(child)
        if lease is not None: fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--worker', action='store_true')
    parser.add_argument('--seed', type=int, choices=(2,), required=True)
    parser.add_argument('--physical-gpu', type=int, choices=range(4)); args = parser.parse_args()
    if args.worker:
        if args.physical_gpu is None or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.physical_gpu):
            raise RuntimeError('wrong assigned GPU visibility')
        admitted = json.loads((api.ARCHIVE/'cpu_v1/admission.json').read_bytes())
        if admitted['cuda_initialized'] or admitted['binding']['fingerprint'] != api.fingerprint():
            raise ValueError('CPU admission changed')
        api.predict(expected_binding=admitted['binding'])
    else: supervise()


if __name__ == '__main__': main()
