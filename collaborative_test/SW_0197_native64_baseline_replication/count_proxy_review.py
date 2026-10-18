"""CPU review of raw versus normalized participation proxies; no model changes."""
import json
from pathlib import Path
import numpy as np
import torch
from snn_kuramoto_bidirectional.loss_function import plv_group_count_loss


def normalized_proxy(q):
    """Effective-rank proxy; ideal unequal constant blocks map to block count.

    Not an exact connected-component count or a learned count classifier.
    Unit diagonal is imposed only within this diagnostic proxy.
    """
    if q.ndim!=2 or q.shape[0]!=q.shape[1] or q.shape[0]==0 or not q.is_floating_point():
        raise ValueError('nonempty square floating matrix required')
    if not torch.isfinite(q).all() or (q<0).any() or (q>1).any():raise ValueError('finite [0,1] affinities required')
    w=(q+q.T)*.5;w.fill_diagonal_(1.)
    inverse=w.sum(1).rsqrt()
    normalized=w*inverse[:,None]*inverse[None,:]
    return normalized.diagonal().sum().square()/normalized.square().sum().clamp_min(1e-12)


def ideal_blocks(sizes):
    y=torch.repeat_interleave(torch.arange(len(sizes)),torch.tensor(sizes))
    return (y[:,None]==y[None,:]).to(torch.float64)


def run():
    from . import run as api
    from collaborative_test.SW_0130_phase_state_integration import run as data
    from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
    import h5py
    examples={}
    for name,sizes in [('equal',[30,30,30]),('unequal',[90,5,5])]:
        q=ideal_blocks(sizes);n=q.shape[0];k=len(sizes)
        examples[name]=dict(sizes=sizes,true_blocks=k,raw_participation=float(n*n/q.square().sum()),
            actual_local_plv_group_count_loss_at_true_count=float(plv_group_count_loss(q[None],target_groups=k)),
            normalized_proxy=float(normalized_proxy(q)))
    source_path=api.train.ARCHIVE/'probe_seed0/report.json';source=json.loads(source_path.read_bytes())
    if source['bindings']['fingerprint']!=api.train.fingerprint():raise ValueError('source TRAIN selection changed')
    ids=source['bindings']['ordered_train_ids']
    if len(ids)!=64 or len(set(ids))!=64 or any(1000<=i<1640 for i in ids):raise ValueError('registered TRAIN64 required')
    dataset=Path(data.source97.DATASET)
    with h5py.File(dataset,'r') as stream:pixels=np.stack([stream['mask'][i,...,0] for i in ids])
    modal=clevr_mask_patch(torch.from_numpy(pixels.copy()),2)['patch_labels'].numpy().astype(np.int64)
    if api.train.sha(modal.tobytes())!=source['bindings']['TRAIN64_modal_sha256']:raise ValueError('source TRAIN modal digest changed')
    if not np.array_equal(modal,api.metrics.independent.modal_labels(pixels,patch=2)):raise ValueError('independent TRAIN modal mismatch')
    values=[]
    for labels in modal:
        _classes,counts=np.unique(labels,return_counts=True);k=len(counts);raw=labels.size**2/float(np.square(counts).sum())
        values.append((k,raw,float(np.mean(labels==0)),((raw-k)/k)**2))
    values=np.asarray(values,dtype=np.float64)
    keys=('true_classes_including_background','ideal_raw_PR','background_patch_fraction','ideal_raw_loss_at_true_class_count')
    aggregates={key:dict(mean=float(values[:,i].mean()),minimum=float(values[:,i].min()),maximum=float(values[:,i].max())) for i,key in enumerate(keys)}
    import snn_kuramoto_bidirectional.loss_function as local_loss
    report=dict(status='completed_CPU_count_proxy_review_on_ideal_blocks_and_registered_TRAIN64',
        peer_commit='1fc3906474039f5016efb4f492fccdbb8c223e60',peer_formula_scope='README formula; peer core implementation absent from branch',
        actual_local_loss_function_sha256=api.train.sha(Path(local_loss.__file__).read_bytes()),
        analysis_source_sha256=api.train.sha(Path(__file__).read_bytes()),source_TRAIN_report_sha256=api.train.sha(source_path.read_bytes()),
        ideal_examples=examples,TRAIN64_ideal_affinity_aggregates=aggregates,
        ideal_affinity_from_GT_not_actual_learned_affinity=True,validation_images_used=0,
        normalized_proxy_is_not_exact_connected_count=True,new_GPU_forwards=0,optimizer_updates=0,
        fixed_target_7_5_not_imported=True,production_loss_unchanged=True,performance_improvement_claim=False,
        raw_imagewise_values_included=False)
    out=api.ROOT_ARCHIVE/'count_proxy_review_v1/summary.json'
    if out.exists():
        if json.loads(out.read_bytes())!=report:raise ValueError('preserve different count review')
    else:api.train.write_json(out,report)
    return report


if __name__=='__main__':print(json.dumps(run()),flush=True)
