"""Identical signed-energy arithmetic in cache-friendly row blocks; no model change."""
import math
import types
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
from snn_kuramoto_bidirectional.sw240 import partition as source
from snn_kuramoto_bidirectional.sw240.signed_energy import NATIVE_NODES,Q_EPS


def signed_pair_energy(q,adjacency,coupling_gain,*,block=64,workers=4):
    q=np.asarray(q);adjacency=np.asarray(adjacency)
    if q.ndim!=2 or q.shape[0]!=q.shape[1] or q.shape[0]<1:raise ValueError('Q must be a nonempty square matrix')
    if adjacency.shape!=q.shape:raise ValueError('source adjacency must match Q')
    if not np.isscalar(coupling_gain) or not math.isfinite(float(coupling_gain)) or float(coupling_gain)<=0:
        raise ValueError('source coupling gain must be finite and positive')
    n=len(q);energy=np.empty((n,n),dtype=np.float64)
    def fill(start):
        stop=min(start+block,n)
        qr=q[start:stop];qt=np.ascontiguousarray(q[:,start:stop].T)
        ar=adjacency[start:stop];at=np.ascontiguousarray(adjacency[:,start:stop].T)
        if not np.isfinite(qr).all() or not np.isfinite(ar).all() or np.any(qr < -1e-7) or np.any(qr > 1+1e-7):
            raise ValueError('Q and source adjacency must be finite; Q must lie in [0,1]')
        if not np.allclose(qr,qt,rtol=0,atol=2e-6):raise ValueError('Q must be symmetric')
        # Same sequence, dtype, parentheses, constants and log as the reference.
        sym=.5*(qr.astype(np.float64,copy=False)+qt.astype(np.float64,copy=False))
        clipped=np.clip(sym,Q_EPS,1.-Q_EPS)
        logit=np.log(clipped/(1.-clipped))
        energy[start:stop]=(1.+NATIVE_NODES*(ar.astype(np.float64,copy=False)+at.astype(np.float64,copy=False))/(2.*float(coupling_gain)))*logit
    # A second destination is necessary: do not symmetrize in place using rows
    # already changed by a previous block.
    symmetric=np.empty_like(energy)
    def symmetrize(start):
        stop=min(start+block,n)
        symmetric[start:stop]=.5*(energy[start:stop]+np.ascontiguousarray(energy[:,start:stop].T))
    # Only independent rows run concurrently; there is no floating reduction
    # across threads. Join the entire first phase before reading transposed E.
    starts=range(0,n,block)
    if workers==1:
        for start in starts:fill(start)
        for start in starts:symmetrize(start)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fill,starts));list(pool.map(symmetrize,starts))
    np.fill_diagonal(symmetric,0.)
    if not np.isfinite(symmetric).all():raise ValueError('signed pair energy became nonfinite')
    return symmetric


# Run the original heap/merge bytecode with a private energy dependency. No
# global monkeypatch, approximate graph, changed threshold or changed tie rule.
foreground_induced_merge=types.FunctionType(source.foreground_induced_merge.__code__,
    dict(source.foreground_induced_merge.__globals__,signed=SimpleNamespace(signed_pair_energy=signed_pair_energy)),
    name='foreground_induced_merge',argdefs=source.foreground_induced_merge.__defaults__,closure=source.foreground_induced_merge.__closure__)
foreground_induced_merge.__kwdefaults__=source.foreground_induced_merge.__kwdefaults__.copy()
