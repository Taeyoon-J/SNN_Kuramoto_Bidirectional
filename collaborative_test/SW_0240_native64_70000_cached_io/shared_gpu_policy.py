"""Only resource admission changes; never stop another user's process."""
import math
import os
from pathlib import Path
import subprocess
import time

NAME = 'shared_memory_admission_no_foreign_process_yield_v2'
HEADROOM_MIB = 512

def required_free_mib(total):
    # Unchanged original worker's PyTorch allocation cap is 80% of device memory.
    if total <= 0: raise ValueError('positive GPU total memory required')
    return math.ceil(total * .8) + HEADROOM_MIB

def eligible(total, used):
    return total > 0 and 0 <= used <= total and total-used >= required_free_mib(total)

def must_stop(actual_uuid, expected_uuid, foreign_pids):
    # Device identity remains mandatory. Process arrivals do not trigger yielding.
    return actual_uuid != expected_uuid

def snapshot(gpu):
    r=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.total,memory.used',
                      '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=12)
    uuid,total,used=[v.strip() for v in r.stdout.strip().split(',')]
    return uuid,int(total),int(used)

def alive(pid):
    try:
        raw=Path('/proc',str(pid),'stat').read_text()
        return raw[raw.rfind(')')+2:].split()[0] not in ('Z','X')
    except FileNotFoundError:return False

def validate_predecessor(state):
    if state.get('status') not in ('waiting_for_exclusive_gpu','stopped_by_signal'):
        raise RuntimeError('preserve unexpected predecessor state')
    for key in ('supervisor_pid','child_pid'):
        pid=state.get(key)
        if pid and (pid==os.getpid() or alive(pid)):
            raise RuntimeError('predecessor supervisor/worker must be stopped before takeover')

def acquire(owner, stop):
    import fcntl
    while not stop.is_set():
        candidates=[]
        for gpu in range(4):
            try:
                uuid,total,used=snapshot(gpu)
                if eligible(total,used):candidates.append((used,gpu,uuid))
            except (OSError,ValueError,subprocess.SubprocessError):continue
        for _,gpu,expected in sorted(candidates):
            owner.GPU_LEASE_DIR.mkdir(parents=True,exist_ok=True)
            lease=(owner.GPU_LEASE_DIR/f'gpu{gpu}.lock').open('a+')
            try:fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:lease.close();continue
            try:
                uuid,total,used=snapshot(gpu)
                if uuid==expected and eligible(total,used):return gpu,uuid,used,lease
            except (OSError,ValueError,subprocess.SubprocessError):pass
            fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close()
        stop.wait(5)
    raise RuntimeError('queue stopped while waiting for GPU capacity')

def stable_capacity(owner,gpu,expected,stop):
    for i in range(2):
        if stop.is_set():return False
        try:
            uuid,total,used=snapshot(gpu)
            if uuid!=expected or not eligible(total,used):return False
        except (OSError,ValueError,subprocess.SubprocessError):return False
        if i==0 and stop.wait(5):return False
    return True
