"""Undo the constant log-odds shift induced by normalized positive/negative BCE."""
import numpy as np


def training_log_odds_shift(targets,weights):
    t=np.asarray(targets);w=np.asarray(weights)
    if t.ndim!=1 or w.shape!=t.shape or not np.isfinite(t).all() or not np.isfinite(w).all() or (t<0).any() or (t>1).any() or (w<=0).any():raise ValueError('finite TRAIN targets and positive base weights required')
    positive=float(np.sum(t.astype(np.float64)*w.astype(np.float64)))
    negative=float(np.sum((1-t.astype(np.float64))*w.astype(np.float64)))
    if positive<=0 or negative<=0:raise ValueError('both relation classes need TRAIN mass')
    return float(np.log(negative/positive)),dict(positive_mass=positive,negative_mass=negative,weighted_positive_prior=positive/(positive+negative))


def correct(logits,shift):
    value=np.asarray(logits)
    if not np.isfinite(value).all() or not np.isfinite(shift):raise ValueError('finite frozen logits/shift required')
    return value.astype(np.float64)-shift
