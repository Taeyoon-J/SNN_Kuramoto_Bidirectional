"""Posthoc GT-assisted evidence audit; never a predictor or threshold selector.

Outside the frozen inference fingerprint. Read GT only after every prediction
has been validated. Reuse private Q/A; perform no core forward or update.
"""
import json
import numpy as np


def auc(scores,positive):
    """Exact tie-aware pair-ranking AUC, None when either class is absent."""
    scores=np.asarray(scores,dtype=np.float64);positive=np.asarray(positive,dtype=bool)
    if scores.ndim!=1 or positive.shape!=scores.shape or not np.isfinite(scores).all():
        raise ValueError('finite score vector and matching labels required')
    p=int(positive.sum());n=len(scores)-p
    if not p or not n:return None
    _,inverse=np.unique(scores,return_inverse=True)
    pos=np.bincount(inverse,weights=positive,minlength=int(inverse.max())+1)
    neg=np.bincount(inverse,weights=~positive,minlength=len(pos))
    return float(np.sum(pos*(np.cumsum(neg)-neg+.5*neg))/(p*n))


def evidence(energy,q,source,truth,indices,shape):
    e=np.asarray(energy,dtype=np.float64);q=np.asarray(q,dtype=np.float64)
    s=np.asarray(source).reshape(-1);g=np.asarray(truth).reshape(-1);f=np.asarray(indices)
    if (s.size!=np.prod(shape) or g.shape!=s.shape or f.dtype.kind not in 'iu'
            or f.ndim!=1 or not np.array_equal(f,np.flatnonzero(s>0))
            or e.shape!=(len(f),len(f)) or q.shape!=e.shape
            or not np.isfinite(e).all() or not np.isfinite(q).all()):
        raise ValueError('actual finite induced matrices and exact source support required')
    left,right=np.triu_indices(len(f),1)
    eligible=(g[f[left]]>0)&(g[f[right]]>0)
    left=left[eligible];right=right[eligible]
    same=g[f[left]]==g[f[right]];together=s[f[left]]==s[f[right]]
    ep=e[left,right];qp=.5*(q[left,right]+q[right,left])
    delta=np.abs(f[left]-f[right]);width=shape[1]
    local=(delta==width)|((delta==1)&(f[left]//width==f[right]//width))
    def subset(mask):
        count=int(mask.sum())
        return dict(pairs=count,mean_energy=float(ep[mask].mean()) if count else None,
                    energy_positive_fraction=float((ep[mask]>0).mean()) if count else None,
                    mean_Q=float(qp[mask].mean()) if count else None)
    out={name:subset(mask) for name,mask in dict(
        same_object=same,different_object=~same,
        false_merge=together&~same,false_split=~together&same).items()}
    out['all_pair_energy_AUC']=auc(ep,same)
    out['all_pair_Q_AUC']=auc(qp,same)
    out['spatial_neighbor_energy_AUC']=auc(ep[local],same[local])
    out['spatial_neighbor_Q_AUC']=auc(qp[local],same[local])
    # Split each original group into GT-homogeneous four-connected pieces.
    # This concrete GT-assisted action is NOT an optimum or learnable ceiling.
    from .refine import connected
    induced_lookup=np.full(s.size,-1,np.int64);induced_lookup[f]=np.arange(len(f))
    before=after=0.;source_groups=reference_groups=0
    for label in np.unique(s[f]):
        group=f[s[f]==label];ix=induced_lookup[group];source_groups+=1
        before+=float(e[np.ix_(ix,ix)].sum())*.5
        for target in np.unique(g[group]):
            for piece in connected(group[g[group]==target],shape):
                j=induced_lookup[piece];after+=float(e[np.ix_(j,j)].sum())*.5
                reference_groups+=1
    out['GT_connected_purification_energy_change']=after-before
    out['source_groups']=source_groups;out['GT_connected_purification_groups']=reference_groups
    out['source_foreground_patches']=len(f)
    out['source_foreground_GT_background_patches']=int((g[f]==0).sum())
    return out


def diagnose():
    from . import run
    # Hard GT barrier: all seed endpoints and hashes first, score must exist.
    predictions,_,_=run.validated_predictions()
    summary_path=run.ARCHIVE/'score_v1/summary.json'
    summary=json.loads(summary_path.read_bytes())
    _,binding=run.source.full.validated_all()
    truth=run.source.full.read_gt(binding)[list(run.OFFSETS)]
    rows={};means={}
    for seed in run.SEEDS:
        admission=json.loads((run.archive_for(seed)/'admission.json').read_bytes())['binding']
        records=[]
        for index in range(len(run.OFFSETS)):
            data,_=run.load_row(run.archive_for(seed)/'rows'/f'{index:04d}',run.row_binding(admission,index))
            assert np.array_equal(data['baseline'],predictions[seed]['baseline'][index])
            energy=run.signed.signed_pair_energy(data['q_foreground'],data['a_foreground'],float(data['gain'])) if len(data['foreground']) else np.zeros((0,0))
            row=evidence(energy,data['q_foreground'],data['baseline'],truth[index],data['foreground'],(64,64))
            records.append(dict(image_id=1320+run.OFFSETS[index],**row))
        rows[str(seed)]=records
        keys=('all_pair_energy_AUC','all_pair_Q_AUC','spatial_neighbor_energy_AUC',
              'spatial_neighbor_Q_AUC','GT_connected_purification_energy_change')
        means[str(seed)]={key:dict(images=sum(r[key] is not None for r in records),
            image_mean=float(np.mean([r[key] for r in records if r[key] is not None]))
                if any(r[key] is not None for r in records) else None) for key in keys}
        for kind in ('false_merge','false_split'):
            valid=[r[kind]['energy_positive_fraction'] for r in records if r[kind]['pairs']]
            means[str(seed)][kind+'_positive_energy_fraction']=dict(images=len(valid),image_mean=float(np.mean(valid)) if valid else None)
    result=dict(status='posthoc_completed_energy_GT_alignment_diagnosis',
        prediction_summary_sha256=run.sha(summary_path.read_bytes()),
        seed_scope=list(run.SEEDS),images=16,grid=[64,64],rows=rows,per_seed=means,
        core_forwards=0,optimizer_updates=0,threshold_search=False,
        reference_is_GT_assisted_not_predictor_or_optimum=True,
        notes='Pair AUC and signs diagnose evidence, not global partition optimality, causal gate value, or generalization. Image means exclude explicitly counted undefined AUCs.')
    run.write_json(run.ARCHIVE/'energy_diagnosis_v1/summary.json',result)
    return result


if __name__=='__main__':print(json.dumps(diagnose()),flush=True)
