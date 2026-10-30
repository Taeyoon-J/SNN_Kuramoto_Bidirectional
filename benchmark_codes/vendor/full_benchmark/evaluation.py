"""Inference without truth, followed by immutable benchmark scoring."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from mod_benchmark import common as c
from readout_benchmark.head import activity_features
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
from full_benchmark.fast_partition import foreground_induced_merge
from full_benchmark.loss import regions_from_output

KEYS=('fg_ari','foreground_iou','matched_object_iou')


@torch.no_grad()
def predict(model,rgb,seed):
    image=torch.from_numpy(np.asarray(rgb[None]).copy()).to(next(model.parameters()).device)
    output=model(image,inference_seed=seed,live_tail_steps=0)
    if model.head is not None:
        return model.head.predict(activity_features(output['trace'])[0],regions_from_output(model,output))
    trace=output['trace'];grid=model.grid
    groups=spike_synchrony_components(trace['spikes'].detach().cpu(),synchrony_threshold=.5,
        min_group_size=8 if grid==32 else 32,settle=512,components=trace['component_spikes'].detach().cpu(),
        background='largest_component',affinity_mode='spike',spatial_grid_size=grid)
    support=spatial_components_to_patch_labels(groups,grid,device='cpu').numpy()[0].astype(np.int64)
    gain=float(model.wrapped.core.graph_generator.log_coupling_gain.exp().detach().cpu())
    result=foreground_induced_merge(output['q'][0].cpu().numpy().astype(np.float64),
        output['adjacency'][0].cpu().numpy().astype(np.float64),gain,support,grid_shape=(grid,grid))['labels']
    assert np.array_equal(result>0,support>0)
    return result


def metrics(predictions,dataset,ids):
    cells=c.benchmark_module('make_targets_grid');metric=c.benchmark_module('evaluation')
    grid=predictions.shape[-1];targets=torch.from_numpy(dataset.pixels(ids))
    labels=torch.from_numpy(predictions.astype(np.int64))
    if dataset.dataset=='tetrominoes' and grid==64:labels=cells.mask_cells(labels,32)[0];grid=32
    row,_=cells.cell_index(targets.shape[1],grid,'cpu');col,_=cells.cell_index(targets.shape[2],grid,'cpu')
    expanded=labels[:,row][:,:,col]
    scores=metric.evaluate_patch_masks(expanded,targets)
    result=dict(pixel={k:float(scores['mean'][k]) for k in KEYS},
        valid_count={k:int(v) for k,v in scores['valid_count'].items()},
        single_object_images=int(sum(len(torch.unique(x[x>0]))==1 for x in targets)),
        undefined_fg_ari_excluded=True,prediction_grid=grid,original_pixel_shape=list(targets.shape[1:]))
    if dataset.dataset=='clevr':
        # Preserve the historical fixed64 patch contract alongside primary pixel scores.
        target64=torch.from_numpy(dataset.labels(ids,'fixed320'))
        lifted=labels if grid==64 else labels.repeat_interleave(2,1).repeat_interleave(2,2)
        patch=metric.evaluate_patch_masks(lifted,target64)
        result['historical_fixed64_patch']={k:float(patch['mean'][k]) for k in KEYS}
    return result,scores['per_image']


def evaluate(model,dataset,out,*,count=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);c.immutable()
    if dataset.dataset in ('multi_dsprites','tetrominoes'):c.preflight(dataset.dataset,out/'protected_preflight')
    elif dataset.dataset in ('objects_room','clevrtex_original'):
        from full_benchmark.extra_preflight import preflight
        preflight(dataset,out/'extra_input_preflight')
    ids=dataset.test if count is None else dataset.test[:count]
    rows=[]
    for index,image_id in enumerate(ids):
        path=out/f'{int(image_id):06d}.npy'
        if path.exists():prediction=np.load(path,allow_pickle=False)
        else:
            prediction=predict(model,dataset.rgb(np.array([image_id]),'fixed320' if dataset.dataset=='clevr' else 'test')[0],193000+index)
            temp=path.with_suffix('.tmp')
            with temp.open('wb') as stream:np.save(stream,prediction.astype(np.int64))
            os.replace(temp,path)
        assert prediction.shape==(model.grid,model.grid) and np.min(prediction)>=0
        rows.append(prediction)
        c.write(out/'progress.json',dict(completed=index+1,total=len(ids),stage='truth_free_predictions'))
    stacked=np.stack(rows)
    c.write(out/'frozen_predictions.json',dict(ids=ids.tolist(),GT_used_for_prediction=False,
        hashes={f'{int(i):06d}.npy':c.sha(out/f'{int(i):06d}.npy') for i in ids}))
    result,per_image=metrics(stacked,dataset,ids)
    result.update(count=len(ids),dataset=dataset.dataset,seed=model.seed,variant=model.variant,
        native_model_grid=model.grid,benchmark_unchanged=c.immutable(),GT_used_for_prediction=False)
    if hasattr(dataset,'source_ids'):
        result.update(evaluation_source_ids=dataset.source_ids[ids].tolist(),evaluation_contract=dataset.binding['evaluation_contract'],
            preprocessing=dataset.binding['preprocessing'])
    with (out/'scores.npz').open('wb') as stream:np.savez_compressed(stream,**{k:v.numpy() for k,v in per_image.items()})
    c.write(out/'result.json',result);return result
