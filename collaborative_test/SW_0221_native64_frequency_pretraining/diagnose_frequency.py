"""Post-score, read-only check of whether input improvements survive spike readout.

This separate diagnostic never changes the frozen six-arm training/evaluation
fingerprint, runs a model, or chooses a checkpoint from DEV ground truth.
"""
import json
from pathlib import Path
import numpy as np
import torch
from . import train,evaluate
from .frequency_loss import make_targets,frequency_objective


def frequency_statistics(frequency,labels):
    x=np.asarray(frequency,dtype=np.float64);labels=np.asarray(labels)
    if labels.ndim!=3 or x.ndim!=3 or x.shape[:2]!=(len(labels),int(np.prod(labels.shape[1:]))) or not np.isfinite(x).all():raise ValueError('aligned finite native frequencies and patch labels required')
    rows=[]
    for value,grid in zip(x,labels):
        ids=np.unique(grid);flat=grid.reshape(-1);means=np.stack([value[flat==k].mean(0) for k in ids])
        within=float(np.mean([((value[flat==k]-mean)**2).mean() for k,mean in zip(ids,means)]))
        index=np.searchsorted(ids,grid);pairs=set()
        for left,right in ((index[:,:-1],index[:,1:]),(index[:-1,:],index[1:,:])):
            pairs.update((int(min(a,b)),int(max(a,b))) for a,b in zip(left.ravel(),right.ravel()) if a!=b)
        distances=[float(((means[a]-means[b])**2).mean()) for a,b in pairs]
        hinge=float(np.mean([max(0.,1.-d)**2 for d in distances])) if distances else 0.
        global_variance=float(((value-value.mean(0))**2).mean())
        rows.append([within,hinge,float(np.mean(distances)) if distances else 0.,global_variance])
        # Independent stable-centered NumPy calculation vs actual training loss.
        actual,parts=frequency_objective(torch.from_numpy(value[None]),make_targets(grid[None]))
        np.testing.assert_allclose([within,hinge,within+hinge],[float(parts['within']),float(parts['adjacent_separation']),float(actual)],rtol=1e-10,atol=1e-12)
    return np.asarray(rows)


def run():
    score_path=train.ARCHIVE/'score_v1/summary.json';score_raw=score_path.read_bytes();score=json.loads(score_raw)
    if score['status']!='complete_native64_encoder_frequency_six_arm_mini' or score['fingerprint']!=train.fingerprint():raise ValueError('all six final scored arms required before this diagnostic')
    bundle=evaluate.validated_all()
    before={};after={}
    for (seed,count),job in bundle.items():
        a=train.archive_for(seed,count);path=a/'reference_frequency.npz'
        if train.sha(path.read_bytes())!=job['report']['reference_frequency_file_sha256']:raise ValueError('frequency reference changed')
        with np.load(path,allow_pickle=False) as values:before[seed,count]=values['frequency'].copy()
        after[seed,count]=np.stack([evaluate.read_row(a/'rows'/f'{i:04d}',job['protocol']['binding'],i)[1] for i in range(16)])
    # DEV GT is still read only after all six actual models and mask rows validate.
    _,ref=train.source.source.full.validated_all();gt=train.source.source.full.read_gt(ref)[list(evaluate.OFFSETS)]
    keys=('within_object_variance','adjacent_margin_hinge','adjacent_class_mean_squared_distance','global_frequency_variance')
    values={};private={}
    for (seed,count),job in bundle.items():
        vectors,_=train.source.source.full.metrics.measured_vectors(job['labels'],gt)
        expected=score['per_seed_scores'][f'TRAIN{count}_seed{seed}']
        np.testing.assert_allclose(vectors.mean(0),[expected[k] for k in evaluate.METRICS],rtol=0,atol=1e-12)
        pre=frequency_statistics(before[seed,count],gt);post=frequency_statistics(after[seed,count],gt)
        values[f'seed{seed}_TRAIN{count}']=dict(before=dict(zip(keys,pre.mean(0).tolist())),after=dict(zip(keys,post.mean(0).tolist())),
            change=dict(zip(keys,(post-pre).mean(0).tolist())),mask_scores=expected)
        private[f'{seed}_{count}']=dict(before=pre.tolist(),after=post.tolist())
    if train.sha(score_path.read_bytes())!=train.sha(score_raw):raise ValueError('scored experiment changed during diagnostic')
    summary=dict(status='complete_read_only_frequency_to_mask_diagnostic',score_sha256=train.sha(score_raw),
        diagnostic_source_sha256=train.sha(Path(__file__).read_bytes()),training_fingerprint=train.fingerprint(),
        per_arm=values,GT_access_after_all_six_frozen_models_validated=True,additional_encoder_updates=0,additional_core_forwards=0,
        input_frequency_is_not_recurrent_phase_or_gating_causal_evidence=True,frequency_improvement_alone_is_not_segmentation_improvement=True,
        no_checkpoint_or_hyperparameter_selection=True)
    train.write_json(train.ARCHIVE/'frequency_diagnostic_v1/evaluation_private.json',dict(summary=summary,imagewise=private))
    train.write_json(train.ARCHIVE/'frequency_diagnostic_v1/summary.json',summary);return summary


if __name__=='__main__':print(json.dumps(run()))
