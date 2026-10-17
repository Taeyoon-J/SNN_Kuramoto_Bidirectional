"""GT-free direct native64 prediction pair and independently checked mini score."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from collaborative_test.SW_0192_native64_incumbent_mini import run as probe
from collaborative_test.SW_0175_critical_component_binding import train_v2 as native

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results_archive'
IDS=np.arange(1320,1352,dtype=np.int64)
ARMS=('baseline','candidate')
SEED=0
sha=probe.sha
write_json=probe.write_json


def fingerprint():
    return dict(experiment='SW0193_native64_probe_screen',
        files={name:sha((HERE/name).read_bytes()) for name in ('run.py','score.py','queue.py','test_run.py','DESIGN_KO.md')},
        parent_probe=probe.fingerprint(),image_ids=IDS.tolist(),arms=list(ARMS),seed=SEED,
        optimizer_updates=0,grid=[64,64],inference_seed_base=193000)


def validate_parent_report(report,bindings):
    if report.get('status')!='complete_native64_training_admission' or report.get('bindings')!=bindings:raise ValueError('successful exact SW0192 admission required')
    history=report.get('history',[])
    if len(history)!=2 or report.get('accepted_updates',0)<1 or report.get('source_CNN_changed') is not True:raise ValueError('parent probe has no committed learning')
    if report.get('validation_data_read') is not False or any(row['credit_failures'] for row in history):raise ValueError('parent probe scope/credit invalid')
    if report.get('actual_optimizer_steps')!=[report['accepted_updates']]:raise ValueError('parent Adam step mismatch')


def prepare_inputs_cpu():
    import h5py
    from collaborative_test.SW_0130_phase_state_integration import run as source
    if torch.cuda.is_initialized():raise RuntimeError('prepare before CUDA')
    prepared=probe.prepare_inputs_cpu()
    report_path=probe.ARCHIVE/'probe_seed0/report.json'
    report=json.loads(report_path.read_bytes());validate_parent_report(report,prepared['bindings'])
    path=probe.ARCHIVE/'probe_work_seed0/boundary.ptx';raw=path.read_bytes()
    envelope=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    data=envelope['payload_bytes']
    if sha(data)!=envelope['payload_sha256']:raise ValueError('probe checkpoint corrupt')
    payload=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
    metadata=probe.checkpoint.inspect_committed_metadata(path.parent)
    if metadata['bindings']!=prepared['bindings'] or payload['completed_updates']!=2 or payload['history']!=report['history']:raise ValueError('parent boundary history mismatch')
    steps=[int(value['step'].item()) for value in payload['optimizer_state']['state'].values()]
    if not steps or any(step!=report['accepted_updates'] for step in steps):raise ValueError('parent checkpoint Adam steps differ from admitted updates')
    state={'wrapped':payload['wrapped_state'],'encoder':payload['encoder_state']}
    from collaborative_test.SW_0179_qce_seed_replication import train as state_api
    if state_api._state_sha_from_payload(state)!=report['final_model_state_sha256']:raise ValueError('parent trained model digest mismatch')
    rgb=np.load(source.VAL_RGB,mmap_mode='r',allow_pickle=False)[:len(IDS)].copy()
    dataset=Path(source.source97.DATASET);stat=dataset.stat()
    with h5py.File(dataset,'r') as stream:
        pixels=np.asarray(stream['image'][IDS[0]:IDS[-1]+1])
    if pixels.dtype!=np.uint8:pixels=np.rint(np.clip(pixels.astype(np.float32),0,1)*255).astype(np.uint8)
    if rgb.shape!=(len(IDS),128,128,3) or not np.array_equal(rgb,pixels):raise ValueError('development RGB cache/canonical mismatch')
    binding=dict(fingerprint=fingerprint(),parent_report_sha256=sha(report_path.read_bytes()),
        parent_boundary_sha256=sha(raw),candidate_model_sha256=report['final_model_state_sha256'],
        baseline_model_sha256=prepared['bindings']['factory_proof']['native64_initial_model_state_sha256'],
        development_RGB_sha256=sha(rgb.tobytes()),development_RGB_row_sha256=[sha(row.tobytes()) for row in rgb],
        dataset=dict(path=str(dataset.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns),
        training_unique_images=8,training_proposals=2,mask_supervised=True)
    return dict(found=prepared['found'],candidate_state=state,rgb=rgb,bindings=binding,dataset=dataset)


def prediction_from_trace(found,trace,adj,q):
    from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components
    from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
    from collaborative_test.SW_0164_foreground_preserving_partition.foreground_signed import foreground_induced_merge
    groups=spike_synchrony_components(trace['spikes'].detach().cpu(),synchrony_threshold=.5,min_group_size=32,
        settle=512,components=trace['component_spikes'].detach().cpu(),background='largest_component',affinity_mode='spike',spatial_grid_size=64)
    support=spatial_components_to_patch_labels(groups,64,device='cpu').numpy()[0].astype(np.int64)
    gain=float(found.wrapped.core.graph_generator.log_coupling_gain.exp().detach().cpu())
    labels=np.asarray(foreground_induced_merge(q[0].detach().cpu().numpy().astype(np.float64),
        adj[0].detach().cpu().numpy().astype(np.float64),gain,support,grid_shape=(64,64))['labels'],dtype=np.int64)
    if labels.shape!=(64,64) or np.any(labels<0) or not np.array_equal(labels>0,support>0):raise ValueError('native64 production support mismatch')
    return labels


def validate_saved_row(folder,expected):
    record=json.loads((folder/'row.json').read_bytes());raw=(folder/'labels.npz').read_bytes()
    if record.get('binding')!=expected or record.get('prediction_sha256')!=sha(raw):raise ValueError('saved inference row binding/hash mismatch')
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:labels=archive['labels'].copy()
    if labels.dtype!=np.int64 or labels.shape!=(64,64) or np.any(labels<0):raise ValueError('invalid native64 saved labels')
    return labels


def predict_pair(prepared,device='cuda:0'):
    device=torch.device(device);torch.cuda.set_device(device)
    found=probe.model64.move_to_device(prepared['found'],device)
    for arm in ARMS:
        if arm=='candidate':
            found.wrapped.load_state_dict(prepared['candidate_state']['wrapped'],strict=True)
            found.encoder.load_state_dict(prepared['candidate_state']['encoder'],strict=True)
        model_sha=native._state_sha(found.wrapped,found.encoder)
        if model_sha!=prepared['bindings'][arm+'_model_sha256']:raise ValueError('wrong inference model')
        found.wrapped.requires_grad_(False);found.encoder.requires_grad_(False)
        found.wrapped.eval();found.encoder.eval()
        rows=[]
        for offset,image_id in enumerate(IDS):
            expected=dict(arm=arm,image_id=int(image_id),inference_seed=193000+offset,
                model_sha256=model_sha,RGB_sha256=sha(prepared['rgb'][offset].tobytes()),
                source_bindings_sha256=sha(json.dumps(prepared['bindings'],sort_keys=True).encode()))
            folder=ARCHIVE/arm/'rows'/f'{offset:04d}'
            if folder.exists():labels=validate_saved_row(folder,expected)
            else:
                torch.manual_seed(193000+offset);torch.cuda.manual_seed(193000+offset)
                started=time.perf_counter()
                with torch.no_grad():
                    rgb=torch.from_numpy(prepared['rgb'][offset:offset+1]).to(device)
                    _gamma,trace,adj,_z,q,_rho=native._native_forward(found,rgb,live_tail_steps=0)
                    labels=prediction_from_trace(found,trace,adj,q)
                stage=folder.with_name(folder.name+f'.{time.time_ns()}.tmp');stage.mkdir(parents=True,exist_ok=False)
                with (stage/'labels.npz').open('xb') as stream:
                    np.savez_compressed(stream,labels=labels);stream.flush();os.fsync(stream.fileno())
                write_json(stage/'row.json',dict(binding=expected,prediction_sha256=sha((stage/'labels.npz').read_bytes()),seconds=time.perf_counter()-started))
                stage.rename(folder)
                del rgb,trace,adj,q,_gamma,_z,_rho
            rows.append(labels)
        if native._state_sha(found.wrapped,found.encoder)!=model_sha:raise RuntimeError('inference mutated trained weights')
        output=ARCHIVE/arm/'protocol.json'
        protocol=dict(arm=arm,bindings=prepared['bindings'],model_sha256=model_sha,image_ids=IDS.tolist(),
            labels_sha256=sha(np.stack(rows).tobytes()),validation_GT_read=False,optimizer_updates=0)
        if output.exists():
            if json.loads(output.read_bytes())!=protocol:raise ValueError('prediction protocol changed')
        else:write_json(output,protocol)
    return dict(status='native64_pair_frozen_before_scoring',bindings=prepared['bindings'],optimizer_updates=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true')
    parser.add_argument('--cpu-output',type=Path);args=parser.parse_args()
    prepared=prepare_inputs_cpu()
    if args.worker:predict_pair(prepared)
    elif args.cpu_output:write_json(args.cpu_output,dict(status='SW0193_CPU_parent_and_baseline_verified',bindings=prepared['bindings'],cuda_initialized=torch.cuda.is_initialized()))
    else:parser.error('choose worker or CPU admission')

if __name__=='__main__':main()
