"""Recompute event statistics from retained traces and validate paired mask comparisons."""
import json
from pathlib import Path
import numpy as np
import torch
from collaborative_test.SW_0238_native64_spike_event_activity.spike_gate_activity import summarize
from . import run


def checked_activity(spikes,gates,prior):
    value=summarize(np.asarray(spikes)[...,512:],np.asarray(gates)[...,512:])
    if value!=prior:
        raise ValueError('retained actual trace does not reproduce prior238 scalar diagnostic')
    return value


def audit():
    if torch.cuda.is_initialized():raise ValueError('CPU-only archive audit required')
    raw=(run.ARCHIVE/'score_v1/summary.json').read_bytes();score=json.loads(raw)
    if score['fingerprint']!=run.fingerprint() or score['total_cases']!=6 or score['GT_used'] or score['optimizer_updates'] or score['changes_upstream_dynamics']:
        raise ValueError('completed diagnostic scope differs')
    cases=[];byte_count=0
    for seed in run.SEEDS:
        prior=run.base.validated_seed(seed)
        for index,row in enumerate(run.validated_seed(seed)):
            old=prior[index];folder=run.archive_for(seed)/'rows'/f'{index:04d}'
            if row['image_id']!=old['image_id'] or row['inference_seed']!=old['inference_seed'] or row['binding']['candidate']!=old['binding']['candidate']:
                raise ValueError('baseline source/image/RNG differs from238')
            with np.load(folder/'raw_trace.npz',allow_pickle=False) as data:
                spikes=data['component_spikes'];gates=data['component_gates'];adj=data['adjacency']
                if spikes.shape!=(1,4,4096,1024) or gates.shape!=spikes.shape or adj.shape!=(1,4096,4096) or not np.isfinite(adj).all():
                    raise ValueError('actual native64 retained geometry differs')
                activity=checked_activity(spikes,gates,old['activity'])
                byte_count+=spikes.nbytes+gates.nbytes+adj.nbytes
            cases.append(dict(seed=seed,case_index=index,comparison=row['comparison'],
                raw_event_occupancy=activity['positive_gate_event_occupancy'],
                prior238_scalar_diagnostic_exactly_reproduced=True))
    if score['cases']!=[{k:r[k] for k in ('seed','case_index','comparison')} for r in cases]:
        raise ValueError('score cases differ from retained paired masks')
    same=sum(r['comparison']['partition_equal_up_to_foreground_ID_permutation'] for r in cases)
    if score['identical_partition_case_count']!=same:raise ValueError('aggregate partition count differs')
    result=dict(status='complete_raw_event_statistics_and_paired_mask_consistency_audited',
        score_sha256=run.sha(raw),audit_source_sha256=run.sha(Path(__file__).read_bytes()),
        total_cases=6,identical_partition_case_count=same,cases=cases,retained_trace_uncompressed_bytes=byte_count,
        independent_raw_event_statistics_recomputation=True,paired_mask_comparisons_recomputed=True,
        independent_readout_reexecution=False,changes_upstream_dynamics=False,
        full_membrane_causal_necessity_proven=False,performance_improvement_claim=False,
        GT_used=False,optimizer_updates=0,GPU_used=False,full320_incumbent_unchanged=True)
    path=run.ARCHIVE/'completion_audit_v1/summary.json'
    if path.exists():
        if json.loads(path.read_bytes())!=result:raise ValueError('existing completion audit differs')
    else:run.write_json(path,result)
    return result


if __name__=='__main__':
    import ctypes,sys
    if sys.platform=='linux' and ctypes.CDLL(None).prctl(41,1,0,0,0)!=0:raise RuntimeError('THP disable failed')
    print(json.dumps(audit()))
