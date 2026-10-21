"""Shared-memory admission; original worker/training fingerprint stays unchanged."""
import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from . import shared_gpu_policy as policy

if sys.platform == 'linux':
    import ctypes
    if ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) != 0:
        raise RuntimeError('per-process THP disable failed')
if '--worker' not in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def load(name):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        return importlib.import_module(name)
    finally:
        if visible is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = visible


api = load('collaborative_test.SW_0240_native64_70000_cached_io.train')
MODULE = 'collaborative_test.SW_0240_native64_70000_cached_io.queue'


def worker_command(seed, gpu):
    api.require_seed(seed)
    if type(gpu) is not int or gpu not in range(4):
        raise ValueError('one exclusive physical GPU required')
    return [sys.executable, '-u', '-B', '-m', MODULE, '--worker', '--seed', str(seed), '--physical-gpu', str(gpu)]


def worker(seed, gpu):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('worker assigned GPU visibility differs')
    p = api.prepare_cpu(seed)
    if json.loads((api.archive_for(seed) / 'admission.json').read_bytes())['binding'] != p['binding']:
        raise ValueError('worker source differs from CPU admission')
    device = api.torch.device('cuda:0'); api.torch.cuda.set_device(device)
    api.torch.cuda.set_per_process_memory_fraction(.8, 0)
    api.source.source.source.model64.move_to_device(p['found'], device)
    runtime = api.resume_policy.compatible_runtime(api.checkpoint.runtime_signature(device))
    # Same device class/runtime as the selected actual trained source is mandatory.
    if runtime != p['binding']['source']['expected_runtime']:
        raise ValueError('GPU runtime differs from accepted source')
    api.gpu_preflight(p, device)
    report = api.fit(p, device, runtime)
    load('collaborative_test.SW_0240_native64_70000_cached_io.evaluate').predict(p, report, device)
    print(json.dumps(dict(status='complete_actual_TRAIN70000_and_fixed320_prediction', seed=seed)), flush=True)


def supervise():
    import fcntl
    api.bootstrap.confirm_old_owner_stopped()
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    runtime = api.ARCHIVE / 'queue'; runtime.mkdir(parents=True, exist_ok=True)
    lock = (runtime / 'queue.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = runtime / 'state.json'
    if state.exists():
        previous = json.loads(state.read_bytes())
        policy.validate_predecessor(previous)
        (runtime / ('pre_shared_state_' + str(previous['supervisor_pid']) + '_' + str(time.time_ns()) + '.json')).write_bytes(state.read_bytes())
    for seed in api.SEEDS:
        admission = json.loads((api.archive_for(seed) / 'admission.json').read_bytes())
        if admission['CUDA_initialized'] or admission['binding']['fingerprint'] != api.fingerprint():
            raise ValueError('all three CPU admissions required before one-GPU queue')
    owner = load('collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4')
    child = lease = None; attempt = 0
    def record(status, **fields):
        api.write_json(state, dict(status=status, supervisor_pid=os.getpid(), updated_unix=time.time(),
            maximum_concurrent_GPU_workers=1, resource_policy=policy.NAME, **fields), replace=True)
    try:
        for seed in api.SEEDS:
            while not (api.archive_for(seed) / 'protocol.json').exists() and not stop.is_set():
                record('waiting_for_sufficient_free_memory', seed=seed)
                gpu, uuid, _, lease = policy.acquire(owner, stop)
                if not policy.stable_capacity(owner, gpu, uuid, stop):
                    fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                    continue
                if api.fingerprint() != json.loads((api.archive_for(seed) / 'admission.json').read_bytes())['binding']['fingerprint']:
                    raise RuntimeError('source changed while queued')
                attempt += 1; command = worker_command(seed, gpu)
                with (runtime / f'seed{seed}_attempt{attempt:03d}.log').open('x') as log:
                    child = subprocess.Popen(command, cwd=api.ROOT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lease.fileno(),))
                    record('running_worker', seed=seed, physical_gpu=gpu, gpu_uuid=uuid,
                        child_pid=child.pid, child_argv=command)
                    yielded = False; deadline = time.monotonic() + 7 * 24 * 3600
                    while child.poll() is None and not stop.wait(5):
                        if time.monotonic() > deadline:
                            raise TimeoutError('large-data worker exceeded registered 7-day bound')
                        actual, _, owners = owner.gpu_guard._gpu_info(gpu)
                        if child.poll() is not None:
                            break
                        if policy.must_stop(actual, uuid, owners):
                            owner._stop_owned(child); yielded = True; break
                    if stop.is_set() and child.poll() is None:
                        owner._stop_owned(child)
                    rc = child.wait()
                child = None
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                if rc == 0 and (api.archive_for(seed) / 'protocol.json').exists():
                    break
                if not yielded or stop.is_set():
                    raise RuntimeError(f'owned worker exited {rc}; preserved log needs diagnosis')
            if stop.is_set():
                record('stopped_by_signal', seed=seed); return
            load('collaborative_test.SW_0240_native64_70000_cached_io.evaluate').validated_seed(seed)
        load('collaborative_test.SW_0240_native64_70000_cached_io.evaluate').score()
        record('complete')
    except BaseException as exc:
        record('failed', error=repr(exc)); raise
    finally:
        if child is not None and child.poll() is None:
            owner._stop_owned(child)
        if lease is not None:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--seed', type=int, choices=api.SEEDS)
    parser.add_argument('--physical-gpu', type=int, choices=range(4))
    args = parser.parse_args()
    if args.worker:
        if args.seed is None or args.physical_gpu is None:
            parser.error('assigned seed and GPU required')
        worker(args.seed, args.physical_gpu)
    else:
        supervise()
