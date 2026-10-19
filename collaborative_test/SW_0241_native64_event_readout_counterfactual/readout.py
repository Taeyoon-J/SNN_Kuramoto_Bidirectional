"""Explicit downstream event bypass, preserving the observed upstream trajectory."""
import numpy as np
import torch


def gate_only_trace(trace):
    spikes=trace['component_spikes'];gates=trace['component_gates']
    if spikes.ndim!=4 or spikes.shape[1]!=4:
        raise ValueError('native [B,4,N,T] spikes required')
    if gates.ndim==3:gates=gates[:,None]
    if gates.shape not in (spikes.shape,(spikes.shape[0],1,spikes.shape[2],spikes.shape[3])):
        raise ValueError('native component gate geometry required')
    if not bool(torch.isfinite(gates).all()) or bool(((gates<0)|(gates>1)).any()):
        raise ValueError('finite native gates in [0,1] required')
    components=gates.expand_as(spikes).clone()
    return dict(trace,component_spikes=components,spikes=components.mean(dim=1))


def partition_signature(labels):
    value=np.asarray(labels)
    if value.ndim!=2 or value.dtype.kind not in 'iu' or np.any(value<0):
        raise ValueError('nonnegative integer spatial labels required')
    mapping={0:0};result=np.empty(value.size,dtype=np.int64)
    for i,label in enumerate(value.flat):
        label=int(label)
        if label not in mapping:mapping[label]=len(mapping)
        result[i]=mapping[label]
    return result.reshape(value.shape)


def compare_labels(actual,counterfactual):
    a=np.asarray(actual);b=np.asarray(counterfactual)
    if a.shape!=b.shape:raise ValueError('matching mask geometry required')
    sa=partition_signature(a);sb=partition_signature(b)
    return dict(labels_bitwise_equal=bool(np.array_equal(a,b)),
        partition_equal_up_to_foreground_ID_permutation=bool(np.array_equal(sa,sb)),
        foreground_changed_patch_count=int(np.count_nonzero((a>0)!=(b>0))),
        actual_object_count=int(len(np.unique(a[a>0]))),
        gate_only_object_count=int(len(np.unique(b[b>0]))))


def predict_pair(found,trace,adj,actual_q,production_readout,correlate):
    replacement=gate_only_trace(trace)
    _,q=correlate(replacement['spikes'],replacement['component_spikes'],settle=512)
    actual=np.asarray(production_readout(found,trace,adj,actual_q),dtype=np.int64)
    gate_only=np.asarray(production_readout(found,replacement,adj,q),dtype=np.int64)
    return actual,gate_only,compare_labels(actual,gate_only)
