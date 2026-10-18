"""Independent zero-update mask/CI audit; never rerun the physical model."""
import json
from pathlib import Path
import numpy as np
from . import run,evaluate


def audit():
    a=run.ARCHIVE;score_raw=(a/'score_v1/summary.json').read_bytes();score=json.loads(score_raw);private=json.loads((a/'score_v1/evaluation_private.json').read_bytes())
    if score['fingerprint']!=run.fingerprint() or private['summary']!=score:raise ValueError('frozen alignment score changed')
    bundle=evaluate.validated_all();datasets=[bundle[s]['protocol']['binding']['parent']['dataset'] for s in run.SEEDS]
    if any(v!=datasets[0] for v in datasets):raise ValueError('different data')
    gt=run.source.source.full.read_gt(dict(dataset=datasets[0]))[list(run.OFFSETS)];vectors={};maximum=0.;keys=run.source.source.full.metrics.independent.METRICS
    for seed in run.SEEDS:
        state=json.loads((run.archive_for(seed)/'queue/state.json').read_bytes())
        if state['status']!='complete':raise ValueError('existing source queue not complete')
        for arm,key in (('baseline','baseline'),('aligned','labels')):
            name=f'{arm}_seed{seed}';value=np.stack([run.source.source.full.metrics.independent.image_metrics(p,t) for p,t in zip(bundle[seed][key],gt)]);vectors[name]=value
            maximum=max(maximum,float(np.max(np.abs(value-np.asarray(private['vectors'][name])))),float(np.max(np.abs(value.mean(0)-[score['per_seed_scores'][name][k] for k in keys]))))
    groups={arm:np.stack([vectors[f'{arm}_seed{s}'] for s in run.SEEDS]) for arm in ('baseline','aligned')}
    for arm,value in groups.items():maximum=max(maximum,float(np.max(np.abs(value.mean((0,1))-[score['scores'][arm][k] for k in keys]))))
    delta=(groups['aligned']-groups['baseline']).mean(0);draw=np.random.RandomState(221).randint(0,16,(10000,16));ci=np.percentile(delta[draw].mean(1),[2.5,97.5],axis=0)
    for i,key in enumerate(keys):
        expected=score['paired_aligned_minus_baseline'][key];maximum=max(maximum,abs(expected['mean']-float(delta[:,i].mean())),float(np.max(np.abs(ci[:,i]-expected['ci95']))))
    gate=bool(delta[:,0].mean()>=.01 and ci[0,0]>0);balanced=bool(gate and all(delta[:,i].mean()>=0 for i in (1,2)))
    if maximum>1e-12 or gate!=score['FG_track_candidate_gate'] or balanced!=score['balanced_candidate_gate']:raise ValueError('independent metric/CI/gates mismatch')
    result=dict(status='complete_actual_alignment_source_checkpoint_and96_mask_audit',score_sha256=run.sha(score_raw),diagnostic_source_sha256=run.sha(Path(__file__).read_bytes()),
        source_references={str(s):dict(checkpoint_sha256=bundle[s]['protocol']['binding']['source_checkpoint_sha256'],parameter_sha256=bundle[s]['protocol']['binding']['source_parameter_sha256'],
            adapter_code_sha256=bundle[s]['protocol']['adapter_code_sha256'],protocol_sha256=run.sha((run.archive_for(s)/'protocol.json').read_bytes())) for s in run.SEEDS},
        raw_masks_independently_rescored=96,independent_metrics_means_and_bootstrap_maximum_abs_error=maximum,optimizer_updates=0,
        FG_track_candidate_gate=gate,balanced_candidate_gate=balanced,Object_IoU_track_retained_but_uncertain=True,
        geometry=run.alignment.geometry(),RGB_GT_resize_or_crop=False,all_three_frozen_protocols_checked_before_DEV_GT=True,
        additional_model_forwards=0,additional_training_updates=0,full320_incumbent_unchanged=True)
    run.write_json(a/'completion_audit_v1/summary.json',result);return result


if __name__=='__main__':print(json.dumps(audit()))
