"""Resume frozen inference with per-process THP disabled; no scientific edits."""
import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

RECOVERY_FOLDER = 'queue_memory_recovery_v2'


def disable_thp():
    if sys.platform != 'linux':
        raise RuntimeError('Linux-only memory recovery')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0) != 0 or libc.prctl(42, 0, 0, 0, 0) != 1:
        raise OSError(ctypes.get_errno(), 'per-process THP disable failed')


def original_terminal(state, alive):
    return state.get('status') == 'failed' and not any(
        alive(pid) for pid in (state.get('child_pid'), state.get('supervisor_pid')) if pid)


def worker(api, seed):
    import numpy as np
    import torch
    p = api.prepare_cpu(seed)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.8, 0)
    api.source.source.source.model64.move_to_device(p['found'], device)
    actual = api.scope.model.resume_policy.compatible_runtime(api.scope.model.checkpoint.runtime_signature(device))
    if actual != p['binding']['candidate']['expected_runtime']:
        raise ValueError('different recovery inference runtime')
    labels, frequency = api.forward(p['found'], p['rgb'][0], device, 0)
    if not np.array_equal(labels, p['mini'][0]) or not np.array_equal(frequency, p['mini_frequency'][0]):
        raise ValueError('memory-policy replay differs from frozen223')
    api.write_json(api.archive_for(seed) / RECOVERY_FOLDER / 'replay.json', dict(
        binding=p['binding'], mask_bitwise_equal=True, frequency_bitwise_equal=True,
        per_process_THP_disabled=True, optimizer_updates=0, additional_native_forwards=1))
    # The production predictor deliberately requires a fresh CPU admission before
    # CUDA initialization. Replace this interpreter after replay, retaining the
    # same PID, exclusive lease FD, CUDA visibility and per-process THP policy.
    command = [sys.executable, '-B', '-u', '-m', api.__package__ + '.queue',
               '--worker', '--seed', str(seed), '--physical-gpu', os.environ['CUDA_VISIBLE_DEVICES']]
    os.execv(sys.executable, command)
    raise RuntimeError('exec unexpectedly returned')


def supervise(api, seed):
    import fcntl
    from . import queue
    a = api.archive_for(seed)
    original = json.loads((a / 'queue/state.json').read_bytes())
    if not original_terminal(original, lambda pid: Path(f'/proc/{pid}').exists()):
        raise ValueError('original queue must be verifiably terminal before recovery')
    if (a / 'protocol.json').exists():
        raise ValueError('already completed: do not duplicate inference')
    runtime = a / RECOVERY_FOLDER
    runtime.mkdir(parents=True, exist_ok=True)
    lock = (runtime / 'queue.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = runtime / 'state.json'
    if state.exists():
        raise FileExistsError('inspect registered recovery; do not duplicate')
    owner = queue.load('collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4')
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    def record(status, **kw):
        api.write_json(state, dict(status=status, seed=seed, supervisor_pid=os.getpid(),
            original_terminal_state_sha256=api.sha((a / 'queue/state.json').read_bytes()),
            per_process_THP_disabled=True, updated_unix=time.time(), **kw), replace=True)
    child = lease = None
    try:
        record('waiting_for_exclusive_gpu')
        attempt = 0
        while not stop.is_set():
            gpu, uuid, _, lease = owner._acquire(stop)
            if not owner._stable_idle(gpu, uuid, stop, lambda *_: None):
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
                continue
            attempt += 1
            command = [sys.executable, '-B', '-u', '-m', __package__ + '.recover_memory', '--worker', '--seed', str(seed)]
            with (runtime / f'attempt{attempt:02d}.log').open('x') as log:
                child = subprocess.Popen(command, cwd=api.ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(lease.fileno(),))
                record('running_worker', child_pid=child.pid, physical_gpu=gpu, gpu_uuid=uuid)
                yielded = False
                deadline = time.monotonic() + 8 * 3600
                while child.poll() is None and not stop.wait(5):
                    if time.monotonic() > deadline:
                        raise TimeoutError('recovery exceeded8h')
                    live_uuid, _, owners = owner.gpu_guard._gpu_info(gpu)
                    if child.poll() is not None:
                        break
                    if live_uuid != uuid or owner.gpu_guard._foreign_owners(owners, owner.gpu_guard._process_tree(child.pid)):
                        owner._stop_owned(child); yielded = True; break
                if stop.is_set() and child.poll() is None:
                    owner._stop_owned(child)
                rc = child.wait()
            if rc == 0 and (a / 'protocol.json').exists():
                break
            if not yielded or stop.is_set():
                raise RuntimeError(f'recovery worker exited{rc}; inspect preserved log')
            child = None
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
            record('waiting_for_exclusive_gpu', yielded_attempt=attempt)
        if stop.is_set():
            record('stopped_by_signal'); return
        api.validated_seed(seed)
        record('complete', protocol_sha256=api.sha((a / 'protocol.json').read_bytes()))
        if lease is not None:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close(); lease = None
        queue.maybe_score()
    except BaseException as exc:
        record('failed', error=repr(exc)); raise
    finally:
        if child is not None and child.poll() is None:
            owner._stop_owned(child)
        if lease is not None:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    disable_thp()  # Before importing NumPy, Torch or any scientific factory.
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if not args.worker:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif not os.environ.get('CUDA_VISIBLE_DEVICES', '').isdigit():
        raise ValueError('exclusive physical GPU required')
    from . import run
    if args.worker:
        worker(run, args.seed)
    else:
        supervise(run, args.seed)


if __name__ == '__main__':
    main()
