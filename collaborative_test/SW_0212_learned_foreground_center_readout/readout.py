"""Fixed learned foreground; reuse the preregistered nonlocal center decoder."""
import numpy as np
from collaborative_test.SW_0211_spike_anchor_center_assignment.center_head import decode


def learned_foreground(logits):
    value=np.asarray(logits)
    if value.ndim!=2 or value.shape[0]!=value.shape[1] or not np.isfinite(value).all():
        raise ValueError('finite square foreground logits required')
    return (value>=0).astype(np.int64)


def assign(foreground_logits,heat,offset):
    return decode(heat,offset,learned_foreground(foreground_logits))


def require_replay(actual,expected):
    if not np.array_equal(actual,expected):
        raise ValueError('CPU primary128 replay differs; do not score new readout')
