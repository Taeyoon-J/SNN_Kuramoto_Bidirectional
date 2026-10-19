"""Quantify binary-event information in actual emitted spikes, without GT or training.

The native forward emits S=event*gate. Recover events only where gate>0;
never infer a binary event at gate==0. This is an observation, not an ablation.
"""
import numpy as np


def summarize(spikes, gates):
    s=np.asarray(spikes);g=np.asarray(gates)
    if s.ndim!=4 or s.shape[-1]<2 or not np.issubdtype(s.dtype,np.floating):
        raise ValueError('floating [batch,component,node,time>=2] spikes required')
    if g.ndim==3:g=g[:,None,:,:]
    try:g=np.broadcast_to(g,s.shape)
    except ValueError as exc:raise ValueError('gate shape must broadcast to component spikes') from exc
    if not np.isfinite(s).all() or not np.isfinite(g).all() or np.any(g<0) or np.any(g>1):
        raise ValueError('finite spikes and gates in [0,1] required')
    if np.any(s<0) or np.any(s>1) or np.any(s[g==0]!=0):
        raise ValueError('emitted spikes must be zero at zero gates and stay in [0,1]')
    tolerance=8*np.finfo(s.dtype).eps
    on=g>0
    # Compare to the two forward values directly; division is unsafe at tiny gates.
    is_zero=s==0
    is_one=(s==g) | ((np.abs(s-g)<=tolerance*np.abs(g)) & ~is_zero)
    if np.any(on & ~(is_zero|is_one)):
        raise ValueError('spikes are not binary-events multiplied by the supplied gates')
    observed_events=on & is_one
    paired=on[...,1:] & on[...,:-1]
    changed=paired & (observed_events[...,1:]!=observed_events[...,:-1])
    def ratio(n,d):return float(n/d) if d else None
    rows=[]
    for c in range(s.shape[1]):
        count=int(on[:,c].sum());pairs=int(paired[:,c].sum())
        rows.append(dict(component=c,observable_event_count=count,
            event_one_count=int(observed_events[:,c].sum()),
            event_occupancy_given_positive_gate=ratio(observed_events[:,c].sum(),count),
            observable_transition_pair_count=pairs,
            event_transition_count=int(changed[:,c].sum()),
            transition_rate_given_both_gates_positive=ratio(changed[:,c].sum(),pairs)))
    return dict(shape=list(s.shape),sample_count=int(s.size),
        zero_gate_count=int((~on).sum()),
        events_at_zero_gates_are_unobserved=True,
        positive_gate_event_occupancy=ratio(observed_events.sum(),on.sum()),
        gate_copy_bitwise_equal=bool(np.array_equal(s,g)),
        maximum_abs_spike_minus_gate=float(np.abs(s-g).max()),
        fraction_of_positive_gate_samples_suppressed_by_event=ratio((on & is_zero).sum(),on.sum()),
        gates_identical_across_components=bool(np.array_equal(g,np.broadcast_to(g[:,:1],s.shape))),
        components=rows,
        establishes_gate_or_membrane_causal_performance_contribution=False,
        scope='Only the supplied actual traces; not a population, performance or counterfactual claim')
