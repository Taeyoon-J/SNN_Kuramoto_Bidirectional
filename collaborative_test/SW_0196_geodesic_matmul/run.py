"""Read-only actual native64 operator/gradient/mask and resource comparison."""
import argparse
import json
import os
from pathlib import Path
import time
import types
import numpy as np
import torch
from torch.nn import functional as F
from .operator import geodesic_matmul, install
from collaborative_test.SW_0143_native32_selective_geodesic.geodesic_operator import inverse_softplus
from collaborative_test.SW_0194_native64_train64 import train, evaluate

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results_archive'
PREDECESSOR_ARCHIVE=HERE.parent/'SW_0195_native64_generalization_gap/results_archive'
INDICES=(0,4,8,12)
ARMS=('comparison',)


def fingerprint():
    return dict(files={name:train.sha((HERE/name).read_bytes()) for name in
        ('DESIGN_KO.md','operator.py','test_operator.py','run.py','test_run.py','queue.py')},
        parent=train.fingerprint(),indices=list(INDICES),inference_seed_base=196000,
        optimizer_updates=0,geodesic_product_dtype='float64',training_admission=False)


def prepare_cpu():
    prepared=train.prepare_inputs_cpu()
    found=prepared['found'];model=train.probe.native._state_sha(found.wrapped,found.encoder)
    if model!=prepared['parent_model_sha256']:raise ValueError('wrong completed TRAIN8 source')
    rgb=np.ascontiguousarray(prepared['rgb'][list(INDICES)])
    binding=dict(fingerprint=fingerprint(),source_model_sha256=model,parent_bindings=prepared['parent_bindings'],
        image_ids=[prepared['parent_bindings']['ordered_train_ids'][i] for i in INDICES],
        RGB_rows_sha256=[train.sha(row.tobytes()) for row in rgb],known_TRAIN_GT_available_in_factory=True,
        GT_used_for_prediction=False,optimizer_updates=0)
    return found,None,rgb,binding


def compare_gradient(found,rgb,reference,device):
    graph=found.wrapped.core.graph_generator
    with torch.no_grad():
        gamma=found.encode(torch.from_numpy(rgb[:1]).to(device))
        z0=F.normalize(graph.projection(gamma.transpose(1,2)),dim=-1)
    torch.manual_seed(196900);torch.cuda.manual_seed(196900)
    weight=torch.randn((1,4096,4096),device=device)/4096**2
    gradients={};values={};timing={};peaks={}
    for arm in ('reference','matmul'):
        z=z0.detach().clone().requires_grad_()
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);started=time.perf_counter()
        distance=(reference(z,graph.grid_distance.unsqueeze(0)) if arm=='reference' else
            geodesic_matmul(z,graph.grid_distance.unsqueeze(0),z.new_tensor(inverse_softplus(8.)),
                steps=graph.geodesic_steps,radius=graph.geodesic_radius,
                temperature=graph.geodesic_temperature,cap=graph.geodesic_cap))
        gradient=torch.autograd.grad((distance*weight).sum(),z)[0]
        torch.cuda.synchronize(device)
        timing[arm]=time.perf_counter()-started;peaks[arm]=torch.cuda.max_memory_allocated(device)
        gradients[arm]=gradient.detach().cpu().double();values[arm]=distance.detach().cpu()
        del z,distance,gradient
    a,b=gradients['reference'],gradients['matmul'];norm=float(a.norm())
    relative=float((a-b).norm())/max(norm,1e-30)
    cosine=float(F.cosine_similarity(a.flatten(),b.flatten(),dim=0,eps=1e-30))
    record=dict(reference_z_gradient_l2=norm,relative_gradient_l2_error=relative,
        maximum_gradient_abs_error=float((a-b).abs().max()),gradient_cosine=cosine,
        distance_allclose=bool(torch.allclose(values['reference'],values['matmul'],rtol=2e-5,atol=2e-5)),
        maximum_distance_abs_error=float((values['reference']-values['matmul']).abs().max()),
        seconds=timing,peak_allocated_bytes=peaks,
        gate_passed=norm>0 and relative<=2e-4 and float((a-b).abs().max())<=2e-4 and cosine>=.9999)
    if not np.isfinite([norm,relative,cosine]).all():raise FloatingPointError('nonfinite native64 gradient comparison')
    return record


def row_binding(binding,offset):
    return dict(image_id=binding['image_ids'][offset],RGB_sha256=binding['RGB_rows_sha256'][offset],
        inference_seed=196000+offset,source_model_sha256=binding['source_model_sha256'],
        binding_sha256=train.sha(json.dumps(binding,sort_keys=True).encode()))


def read_row(folder,expected):
    import io
    meta=json.loads((folder/'row.json').read_bytes());raw=(folder/'masks.npz').read_bytes()
    if meta['binding']!=expected or meta['prediction_sha256']!=train.sha(raw):raise ValueError('comparison row binding/hash changed')
    with np.load(io.BytesIO(raw),allow_pickle=False) as stream:
        masks={arm:stream[arm].copy() for arm in ('reference','matmul')}
    if any(v.dtype!=np.int64 or v.shape!=(64,64) or (v<0).any() for v in masks.values()):raise ValueError('invalid production masks')
    return masks,meta


def predict(device='cuda:0',expected_binding=None):
    found,_unused,rgb,binding=prepare_cpu()
    if expected_binding is not None and binding!=expected_binding:raise ValueError('CPU admission changed')
    device=torch.device(device);torch.cuda.set_device(device);train.probe.model64.move_to_device(found,device)
    found.wrapped.requires_grad_(False);found.encoder.requires_grad_(False);found.wrapped.eval();found.encoder.eval()
    graph=found.wrapped.core.graph_generator;reference=graph._geodesic_distance
    gradient_path=ARCHIVE/'gradient.json'
    if gradient_path.exists():
        gradient=json.loads(gradient_path.read_bytes())
        if gradient['binding']!=binding:raise ValueError('saved gradient comparison model changed')
    else:
        gradient=dict(binding=binding,result=compare_gradient(found,rgb,reference,device))
        train.write_json(gradient_path,gradient)
    try:
        for offset in range(len(INDICES)):
            folder=ARCHIVE/'comparison/rows'/f'{offset:04d}';expected=row_binding(binding,offset)
            if folder.exists():read_row(folder,expected);continue
            records={};stored={};geo_values={}
            for arm in ('reference','matmul'):
                graph._geodesic_distance=reference
                if arm=='matmul':install(graph)
                method=graph._geodesic_distance;geo_times=[]
                def measured(self,z,euclidean):
                    torch.cuda.synchronize(device);started=time.perf_counter()
                    result=method(z,euclidean)
                    torch.cuda.synchronize(device);geo_times.append(time.perf_counter()-started)
                    geo_values[arm]=result.detach().cpu()
                    return result
                graph._geodesic_distance=types.MethodType(measured,graph)
                torch.manual_seed(196000+offset);torch.cuda.manual_seed(196000+offset)
                torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);started=time.perf_counter()
                with torch.no_grad():
                    image=torch.from_numpy(rgb[offset:offset+1]).to(device)
                    gamma,trace,adj,z,q,rho=train.probe.native._native_forward(found,image,live_tail_steps=0)
                    labels=evaluate.screen.prediction_from_trace(found,trace,adj,q)
                torch.cuda.synchronize(device)
                if len(geo_times)!=1:raise ValueError('expected one native graph construction')
                records[arm]=dict(seconds=time.perf_counter()-started,geodesic_seconds=geo_times[0],
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
                stored[arm]=dict(labels=labels,adjacency=adj.detach().cpu(),q=q.detach().cpu())
                del image,gamma,trace,adj,z,q,rho
            a,b=stored['reference'],stored['matmul']
            comparisons=dict(mask_exact=bool(np.array_equal(a['labels'],b['labels'])),
                reference_foreground_patches=int((a['labels']>0).sum()),
                geodesic_allclose=bool(torch.allclose(geo_values['reference'],geo_values['matmul'],atol=2e-5,rtol=2e-5)),
                geodesic_max_abs=float((geo_values['reference']-geo_values['matmul']).abs().max()),
                adjacency_allclose=bool(torch.allclose(a['adjacency'],b['adjacency'],atol=2e-5,rtol=2e-5)),
                adjacency_max_abs=float((a['adjacency']-b['adjacency']).abs().max()),
                adjacency_support_differences=int(torch.count_nonzero((a['adjacency']>0)!=(b['adjacency']>0))),
                q_allclose=bool(torch.allclose(a['q'],b['q'],atol=2e-5,rtol=2e-5)),
                q_max_abs=float((a['q']-b['q']).abs().max()))
            stage=folder.with_name(folder.name+f'.{time.time_ns()}.tmp');stage.mkdir(parents=True,exist_ok=False)
            with (stage/'masks.npz').open('xb') as stream:
                np.savez_compressed(stream,**{name:v['labels'] for name,v in stored.items()});stream.flush();os.fsync(stream.fileno())
            train.write_json(stage/'row.json',dict(binding=expected,comparisons=comparisons,timing=records,
                prediction_sha256=train.sha((stage/'masks.npz').read_bytes())))
            stage.rename(folder)
        if train.probe.native._state_sha(found.wrapped,found.encoder)!=binding['source_model_sha256']:
            raise RuntimeError('read-only comparison changed model')
        protocol=dict(binding=binding,gradient_sha256=train.sha(gradient_path.read_bytes()),
            rows_sha256=[train.sha((ARCHIVE/'comparison/rows'/f'{i:04d}'/'row.json').read_bytes()) for i in range(4)])
        path=ARCHIVE/'comparison/protocol.json'
        if path.exists():
            if json.loads(path.read_bytes())!=protocol:raise ValueError('comparison protocol changed')
        else:train.write_json(path,protocol)
    finally:graph._geodesic_distance=reference


def validated_pair():
    protocol=json.loads((ARCHIVE/'comparison/protocol.json').read_bytes());binding=protocol['binding']
    if binding['fingerprint']!=fingerprint() or binding['GT_used_for_prediction'] is not False or binding['optimizer_updates']!=0:
        raise ValueError('invalid read-only comparison protocol')
    if len(binding['image_ids'])!=4 or len(set(binding['image_ids']))!=4 or len(binding['RGB_rows_sha256'])!=4:
        raise ValueError('fixed four-image cohort required')
    gradient_path=ARCHIVE/'gradient.json'
    if train.sha(gradient_path.read_bytes())!=protocol['gradient_sha256']:raise ValueError('gradient proof changed')
    gradient=json.loads(gradient_path.read_bytes())
    if gradient['binding']!=binding:raise ValueError('gradient and forward models differ')
    rows=[]
    for i in range(4):
        folder=ARCHIVE/'comparison/rows'/f'{i:04d}'
        if train.sha((folder/'row.json').read_bytes())!=protocol['rows_sha256'][i]:raise ValueError('row changed')
        masks,row=read_row(folder,row_binding(binding,i))
        if row['comparisons']['mask_exact']!=bool(np.array_equal(masks['reference'],masks['matmul'])):
            raise ValueError('mask equivalence claim differs from actual masks')
        if row['comparisons']['reference_foreground_patches']!=int((masks['reference']>0).sum()):
            raise ValueError('nontrivial mask claim differs from actual masks')
        rows.append(row)
    return rows,gradient


def score():
    rows,gradient=validated_pair()
    graph_parity=all(all(row['comparisons'][key] for key in ('geodesic_allclose','adjacency_allclose','q_allclose','mask_exact')) for row in rows)
    nontrivial=all(row['comparisons']['reference_foreground_patches']>0 for row in rows)
    times={arm:sum(row['timing'][arm]['seconds'] for row in rows) for arm in ('reference','matmul')}
    geo={arm:sum(row['timing'][arm]['geodesic_seconds'] for row in rows) for arm in ('reference','matmul')}
    summary=dict(status='completed_read_only_native64_matmul_comparison',images=4,grid=[64,64],
        whole_prediction_seconds=times,geodesic_seconds=geo,whole_prediction_speedup=times['reference']/times['matmul'],
        geodesic_speedup=geo['reference']/geo['matmul'],mask_and_forward_value_parity=graph_parity,
        gradient=gradient['result'],binding=gradient['binding'],nontrivial_reference_masks=nontrivial,
        forward_and_z_gradient_candidate=graph_parity and nontrivial and gradient['result']['gate_passed']
            and gradient['result']['distance_allclose'] and times['reference']/times['matmul']>=1.5,
        optimizer_proposal_equivalence_verified=False,production_training_admitted=False,optimizer_updates=0,
        FG_ARI_improvement_claim=False,ground_truth_used_for_scoring=False)
    train.write_json(ARCHIVE/'score_v1/summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--score',action='store_true')
    parser.add_argument('--cpu-output',type=Path);args=parser.parse_args()
    if sum((args.worker,args.score,args.cpu_output is not None))!=1:parser.error('choose exactly one action')
    if args.worker:predict()
    elif args.score:print(json.dumps(score()),flush=True)
    else:
        _found,_unused,_rgb,binding=prepare_cpu()
        train.write_json(args.cpu_output,dict(binding=binding,cuda_initialized=torch.cuda.is_initialized()))


if __name__=='__main__':main()
