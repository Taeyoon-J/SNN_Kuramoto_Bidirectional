"""Resume only the already-passed seed-1 preflight from a failed SW0129 queue.

This is a new, one-shot state namespace. It never rewrites the original queue
receipt or changes the frozen SW0129 runner/coordinator/preflight.
"""
from __future__ import annotations
import concurrent.futures, json, os, pathlib, signal, subprocess, sys, threading, time
try:
    import fcntl
except ImportError:  # pragma: no cover - server-only launch path
    fcntl = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'collaborative_test')]
from collaborative_test.SW_0129_shared_rgb_seed_replication import coordinator, dispatcher as original_dispatcher
from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner

ARCHIVE = HERE / 'results_archive'
PARENT_STATE = ARCHIVE / 'dispatcher_state.json'
STATE = ARCHIVE / 'resume_seed1_dispatcher_state.json'
LOCK = ARCHIVE / 'resume_seed1_dispatcher.lock'
LEASE_DIR = pathlib.Path('/tmp/kevinswk_sw0113_gpu_leases')
MAX_MEMORY_MIB = 512


def resume_tasks():
    return [row for row in coordinator.task_plan()
            if row['seed'] == 1 and row['stage'] in ('train', 'evaluate')]


def validate_parent_state(state, pid_gone, log_reader):
    """Fail closed unless the old run ended exactly at seed-2 preflight."""
    if state.get('experiment') != 'SW0129' or state.get('status') != 'preflight_failed':
        raise ValueError('original dispatcher is not the recorded preflight_failed attempt')
    supervisor = state.get('supervisor_pid')
    if supervisor is not None and not pid_gone(int(supervisor)):
        raise RuntimeError(f'original dispatcher supervisor is still live/unknown: {supervisor}')
    rows = state.get('tasks')
    if not isinstance(rows, dict) or set(rows) != {t['task_id'] for t in coordinator.task_plan()}:
        raise ValueError('original dispatcher task inventory differs from registered SW0129 plan')
    for row in rows.values():
        pid = row.get('child_pid')
        if pid is not None and not pid_gone(int(pid)):
            raise RuntimeError(f'original dispatcher still has a live/unknown worker PID {pid}')
    s1 = rows['sw0129_preflight_s1']
    s2 = rows['sw0129_preflight_s2']
    if s1.get('status') != 'passed' or s1.get('artifact_valid') is not True:
        raise ValueError('seed-1 preflight was not recorded as passed')
    if s2.get('status') != 'failed' or s2.get('artifact_valid') is not False:
        raise ValueError('seed-2 preflight failure is not the terminal recorded result')
    expected_log = ARCHIVE / 'sw0129_preflight_s2.log'
    if (int(s2.get('returncode', -1)) != 1
            or pathlib.Path(s2.get('log_path', '')).as_posix() != expected_log.as_posix()
            or 'AssertionError: mean hard-partition row scrambling did not increase reconstruction loss'
            not in log_reader(expected_log)):
        raise ValueError('seed-2 record is not the registered row-scramble scientific guard failure')
    for seed in (1, 2):
        for arm in ('control', 'candidate'):
            for stage in ('train', 'evaluate'):
                prefix = 'train' if stage == 'train' else 'eval'
                row = rows[f'sw0129_{prefix}_s{seed}_{arm}']
                if not str(row.get('status', '')).startswith('blocked_'):
                    raise ValueError('original queue advanced beyond its preflight barrier')
    return True


def _atomic_json(path, value):
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


class Seed1Resume:
    def __init__(self):
        if fcntl is None:
            raise RuntimeError('seed-1 resume requires Linux flock GPU leases')
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        if STATE.exists() or LOCK.exists():
            raise FileExistsError('preserving an existing seed-1 resume attempt/state/lock')
        parent_bytes = PARENT_STATE.read_bytes()
        parent = json.loads(parent_bytes.decode('utf-8'))
        validate_parent_state(parent, owner._pid_is_confirmed_gone,
                              lambda path: pathlib.Path(path).read_text(encoding='utf-8', errors='replace'))
        if parent.get('runner_fingerprint') != coordinator.run.implementation_fingerprint():
            raise ValueError('original SW0129 runner fingerprint changed since the passed preflight')
        if parent.get('coordinator_sha256') != coordinator.run.sha(HERE / 'coordinator.py'):
            raise ValueError('original SW0129 coordinator changed since the passed preflight')
        original_dispatcher.coordinator.run.validate_seed0_reference()
        rows = {r['task_id']: r for r in coordinator.task_plan()}
        preflight = rows['sw0129_preflight_s1']
        if not coordinator.valid_result(preflight):
            raise ValueError('the original seed-1 preflight artifact no longer validates')
        coordinator.run.verify_contract(1)
        for task in resume_tasks():
            protected = original_dispatcher.protected_paths(task)
            existing = [str(p) for p in protected if p.exists()]
            if existing:
                raise FileExistsError(f'refusing to overwrite existing seed-1 outputs: {existing}')
        self.lock = LOCK.open('x+'); fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.tasks = resume_tasks()
        self.state_lock = threading.RLock()
        self.state = {
            'experiment': 'SW0129_seed1_partial_resume', 'status': 'running',
            'parent_state_path': str(PARENT_STATE),
            'parent_state_sha256': __import__('hashlib').sha256(parent_bytes).hexdigest(),
            'seed2_failure_log_sha256': coordinator.run.sha(ARCHIVE / 'sw0129_preflight_s2.log'),
            'parent_status': parent['status'], 'parent_seed1_preflight_reused': True,
            'parent_seed2_scientific_failure_preserved': True,
            'supervisor_pid': os.getpid(), 'started': time.time(),
            'runner_fingerprint': coordinator.run.implementation_fingerprint(),
            'coordinator_sha256': coordinator.run.sha(HERE / 'coordinator.py'),
            'tasks': {t['task_id']: {'task': t, 'status': 'queued'} for t in self.tasks},
        }
        _atomic_json(STATE, self.state)

    def _update(self, task_id, **fields):
        with self.state_lock:
            self.state['tasks'][task_id].update(fields)
            _atomic_json(STATE, self.state)

    def _acquire(self, task_id):
        while True:
            for gpu in range(4):
                path = LEASE_DIR / f'gpu{gpu}.lock'
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = path.open('a+')
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close(); continue
                try:
                    uuid, memory, owners = owner._nvidia_gpu_info(gpu)
                    if not owners and int(memory) <= MAX_MEMORY_MIB:
                        self._update(task_id, status='reserved', gpu=gpu, gpu_uuid=uuid,
                                     lease_path=str(path), used_before_mib=memory)
                        return gpu, stream, uuid
                except BaseException:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN); stream.close(); raise
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN); stream.close()
            time.sleep(5)

    @staticmethod
    def _terminate_own(proc):
        if proc.poll() is not None: return
        try: os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError: return
        try: proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            proc.wait()

    def _execute(self, task):
        tid = task['task_id']
        if coordinator.valid_result(task):
            self._update(tid, status='reused_verified', verified_at=time.time()); return 'reused_verified'
        present = [str(p) for p in original_dispatcher.protected_paths(task) if p.exists()]
        if present:
            self._update(tid, status='failed_existing_artifact', existing=present); return 'failed_existing_artifact'
        gpu, lease, uuid = self._acquire(tid)
        proc = None
        logpath = ARCHIVE / f'resume_seed1_{tid}.log'
        triton = pathlib.Path(f'/tmp/kevinswk_sw0129_resume_{tid}_{int(time.time()*1000)}')
        try:
            if logpath.exists() or triton.exists(): raise FileExistsError('preserving prior resume log/cache')
            triton.mkdir(mode=0o700)
            argv = coordinator.command(task, device='cuda:0')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', TRITON_CACHE_DIR=str(triton))
            self._update(tid, status='launching', gpu=gpu, gpu_uuid=uuid, argv=argv,
                         log_path=str(logpath), triton_cache=str(triton), launched=time.time())
            with logpath.open('x', encoding='utf-8') as log:
                proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True,
                                        pass_fds=(lease.fileno(),))
                self._update(tid, status='running', child_pid=proc.pid, started=time.time())
                while proc.poll() is None:
                    time.sleep(5)
                    if proc.poll() is not None: break
                    _u, _m, owners = owner._nvidia_gpu_info(gpu)
                    foreign = owner.foreign_owner_pids(owners, owner._process_tree(proc.pid))
                    if foreign:
                        self._terminate_own(proc)
                        self._update(tid, status='interrupted_foreign_gpu_owner', foreign_owner_pids=foreign)
                        return 'interrupted_foreign_gpu_owner'
            rc = int(proc.returncode); valid = rc == 0 and coordinator.valid_result(task)
            status = 'passed' if valid else 'failed'
            self._update(tid, status=status, returncode=rc, artifact_valid=bool(valid), finished=time.time())
            return status
        except BaseException as exc:
            if proc is not None: self._terminate_own(proc)
            self._update(tid, status='failed', error=repr(exc), finished=time.time())
            return 'failed'
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN); lease.close()

    def run(self):
        outcomes = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(self._execute, task): task for task in self.tasks if task['stage'] == 'train'}
            outcomes.update({futures[f]['task_id']: f.result() for f in futures})
        evals = [t for t in self.tasks if t['stage'] == 'evaluate'
                 and outcomes.get(t['depends_on'][0]) in ('passed', 'reused_verified')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(self._execute, task): task for task in evals}
            outcomes.update({futures[f]['task_id']: f.result() for f in futures})
        for task in self.tasks:
            if task['stage'] == 'evaluate' and task['task_id'] not in outcomes:
                self._update(task['task_id'], status='blocked_training_failure')
        complete = all(self.state['tasks'][t['task_id']]['status'] in ('passed', 'reused_verified')
                       for t in self.tasks)
        with self.state_lock:
            self.state.update(status='complete' if complete else 'incomplete', finished=time.time())
            _atomic_json(STATE, self.state)
        return self.state['status']


def main():
    if '--dry-run' in sys.argv:
        print(json.dumps({'parent_state': str(PARENT_STATE), 'new_state': str(STATE),
                          'tasks': resume_tasks()}, indent=2)); return
    print(json.dumps({'status': Seed1Resume().run()}, allow_nan=False))


if __name__ == '__main__': main()
