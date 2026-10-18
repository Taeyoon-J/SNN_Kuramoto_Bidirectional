"""Wait for195, then acquire an exclusive GPU for read-only numerical benchmarking."""
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
if not WORKER: os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def load(name):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    try: return importlib.import_module(name)
    finally:
        if visible is None: os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else: os.environ['CUDA_VISIBLE_DEVICES'] = visible


api = load('collaborative_test.SW_0196_geodesic_matmul.run')
MODULE = 'collaborative_test.SW_0196_geodesic_matmul.queue'
ROOT = api.HERE.parents[1]
RUNTIME = api.ARCHIVE/'queue'
STOP = threading.Event()


def worker_command(gpu):
    if type(gpu) is not int or gpu not in range(4): raise ValueError('invalid physical GPU')
    return [sys.executable, '-u', '-m', MODULE, '--worker', '--physical-gpu', str(gpu)]


def supervise():
    import fcntl
    for signum in (signal.SIGINT, signal.SIGTERM): signal.signal(signum, lambda *_: STOP.set())
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME/'queue.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (RUNTIME/'state.json').exists(): raise FileExistsError('preserve existing196 queue')
    fingerprint = api.fingerprint(); child = lease = None
    owner = load('collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4')
    def record(status, **fields):
        api.train.write_json(RUNTIME/'state.json', dict(status=status, supervisor_pid=os.getpid(),
            updated_unix=time.time(), **fields), replace=True)
    try:
        record('waiting_for_SW0195_completion')
        while not STOP.is_set():
            state = json.loads((api.PREDECESSOR_ARCHIVE/'queue/state.json').read_bytes())
            if state['status'] == 'complete': break
            if state['status'] == 'failed': raise RuntimeError('195 failed; inspect dependency without restarting')
            # Verify a specific live dependency, not just an old queue file.
            pid = state['supervisor_pid']; proc = Path('/proc')/str(pid)
            try:
                argv = (proc/'cmdline').read_bytes().split(b'\0')
                proc_status = (proc/'stat').read_text().split(') ',1)[1].split()[0]
            except FileNotFoundError:
                if json.loads((api.PREDECESSOR_ARCHIVE/'queue/state.json').read_bytes())['status'] == 'complete': break
                raise RuntimeError('195 supervisor missing; preserve existing work')
            if b'collaborative_test.SW_0195_native64_generalization_gap.queue' not in argv:
                raise RuntimeError('195 supervisor missing or different; preserve existing work')
            if proc_status == 'Z':
                if json.loads((api.PREDECESSOR_ARCHIVE/'queue/state.json').read_bytes())['status'] == 'complete': break
                raise RuntimeError('195 supervisor is terminal')
            STOP.wait(30)
        if STOP.is_set(): record('stopped_by_signal'); return
        if api.fingerprint() != fingerprint: raise RuntimeError('queued diagnostic source changed')
        record('CPU_admitting_completed_models')
        _found, _states, _rgb, binding = api.prepare_cpu()
        del _found, _states, _rgb
        api.train.write_json(api.ARCHIVE/'cpu_v1/admission.json', dict(binding=binding, cuda_initialized=False))
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
                    if time.monotonic() > deadline: raise TimeoutError('196 active worker exceeded4h')
                    actual_uuid, _memory, owners = owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None: break
                    if actual_uuid != uuid or owner.gpu_guard._foreign_owners(owners, owner.gpu_guard._process_tree(child.pid)):
                        owner._stop_owned(child); yielded = True; break
                if STOP.is_set() and child.poll() is None: owner._stop_owned(child)
                rc = child.wait()
            if all((api.ARCHIVE/arm/'protocol.json').exists() for arm in api.ARMS):
                api.validated_pair(); break
            if not yielded or STOP.is_set(): raise RuntimeError(f'196 exited{rc}; inspect preserved log')
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
