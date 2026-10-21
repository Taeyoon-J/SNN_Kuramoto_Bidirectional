import pytest
from . import shared_gpu_policy as policy

def test_busy_device_admitted_only_with_full_worker_budget():
    assert policy.eligible(24576,2490)
    assert policy.eligible(24576,3323)
    assert not policy.eligible(24576,6000)
    assert not policy.eligible(24576,-1)
    assert not policy.eligible(24576,25000)
    needed=policy.required_free_mib(24576)
    assert policy.eligible(24576,24576-needed)
    assert not policy.eligible(24576,24576-needed+1)

def test_other_processes_never_force_yield():
    assert not policy.must_stop('gpu-A','gpu-A',[17,42])
    assert not policy.must_stop('gpu-A','gpu-A',[])
    assert policy.must_stop('gpu-B','gpu-A',[17])

def test_live_predecessor_rejected(monkeypatch):
    monkeypatch.setattr(policy,'alive',lambda pid:True)
    with pytest.raises(RuntimeError):policy.validate_predecessor(dict(status='stopped_by_signal',supervisor_pid=123))

def test_failed_predecessor_never_silently_adopted(monkeypatch):
    monkeypatch.setattr(policy,'alive',lambda pid:False)
    with pytest.raises(RuntimeError):policy.validate_predecessor(dict(status='failed',supervisor_pid=123))
    policy.validate_predecessor(dict(status='stopped_by_signal',supervisor_pid=123))

def test_live_worker_rejected_even_if_supervisor_dead(monkeypatch):
    monkeypatch.setattr(policy,'alive',lambda pid:pid==456)
    with pytest.raises(RuntimeError):policy.validate_predecessor(dict(status='stopped_by_signal',supervisor_pid=123,child_pid=456))

def test_successor_preserves_single_worker_sequential_seeds(tmp_path,monkeypatch):
    from . import queue_shared_v2 as successor,test_queue as original_test
    monkeypatch.setattr(original_test,'queue',successor)
    monkeypatch.setattr(policy,'acquire',lambda owner,stop:owner._acquire(stop))
    monkeypatch.setattr(policy,'stable_capacity',lambda *args:True)
    original_test.test_single_GPU_sequential_seed_supervisor(tmp_path,monkeypatch)
