"""One supervisor must finish and reap each worker before launching the next seed."""
import json
import sys
from types import SimpleNamespace
from collaborative_test.SW_0239_native64_best_70000 import queue


def test_worker_route_and_invalid_GPU():
    command = queue.worker_command(2, 3)
    assert command[-4:] == ['--seed', '2', '--physical-gpu', '3']
    for value in (-1, 4, True):
        try:
            queue.worker_command(0, value)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid physical GPU accepted')


def test_single_GPU_sequential_seed_supervisor(tmp_path, monkeypatch):
    monkeypatch.setattr(queue.api, 'ARCHIVE', tmp_path)
    monkeypatch.setattr(queue.api, 'ROOT', tmp_path)
    monkeypatch.setattr(queue.api, 'fingerprint', lambda: {'fixed': True})
    for seed in queue.api.SEEDS:
        queue.api.write_json(queue.api.archive_for(seed) / 'admission.json',
            dict(CUDA_initialized=False, binding=dict(fingerprint={'fixed': True})))
    monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(LOCK_EX=1, LOCK_NB=2, LOCK_UN=3, flock=lambda *args: None))
    class Event:
        def is_set(self): return False
        def wait(self, seconds): return False
        def set(self): raise AssertionError('unexpected stop')
    monkeypatch.setattr(queue.threading, 'Event', Event)
    monkeypatch.setattr(queue.signal, 'signal', lambda *args: None)
    active = []; launched = []; checked = []
    class Lease:
        def fileno(self): return 99
        def close(self): pass
    owner = SimpleNamespace(_acquire=lambda stop: (1, 'fixture-GPU', [], Lease()),
        _stable_idle=lambda *args: True, _stop_owned=lambda child: None)
    def score():
        assert checked == [0, 1, 2] and not active
    evaluator = SimpleNamespace(validated_seed=lambda seed: checked.append(seed), score=score)
    monkeypatch.setattr(queue, 'load', lambda name: owner if 'stage_queue_v4' in name else evaluator)
    class Child:
        def __init__(self, command, **kwargs):
            assert not active
            self.seed = int(command[command.index('--seed') + 1]); self.pid = 100 + self.seed
            active.append(self.seed); launched.append(self.seed)
            queue.api.write_json(queue.api.archive_for(self.seed) / 'protocol.json', {'fixture': True})
        def poll(self): return 0
        def wait(self):
            assert active == [self.seed]; active.clear(); return 0
    monkeypatch.setattr(queue.subprocess, 'Popen', Child)
    queue.supervise()
    assert launched == checked == [0, 1, 2] and not active
    state = json.loads((tmp_path / 'queue/state.json').read_bytes())
    assert state['status'] == 'complete' and state['maximum_concurrent_GPU_workers'] == 1
