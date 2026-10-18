"""Immutable emitted-history cache; DEV masks are replayed without DEV GT."""
import io,json,os
from pathlib import Path
import time
import numpy as np
import torch
from collaborative_test.SW_0210_native64_full_seed_reference import run as source
from .features import descriptors
from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ARCHIVE=HERE/'results_archive'
OFFSETS=tuple(range(0,32,2))
SNAPSHOTS=(32,64,128)
sha=source.sha
write_json=source.write_json

def fingerprint():
    names=('DESIGN_KO.md','features.py','center_head.py','cache.py','training.py','run.py','queue.py','test_features.py','test_center_head.py','test_pipeline.py')
    return dict(files={n:sha((HERE/n).read_bytes()) for n in names},source=source.fingerprint(),
        statistics_helper_sha256=sha((ROOT/'collaborative_test/SW_0207_spike_foreground_classifier/classifier.py').read_bytes()),
        checkpoint_helper_sha256=sha(Path(checkpoint.__file__).read_bytes()),
        source_seed=2,source_model_sha256=source.MODELS['2'],TRAIN=32,DEV=16,channels=268,
        primary_steps=128,snapshots=list(SNAPSHOTS),core_updates=0)

def prepare_cpu():
    import h5py
    from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
    if torch.cuda.is_initialized():raise RuntimeError('admit before CUDA')
    found,dev,prior,mini=source.prepare_cpu(2)
    ids64=prior['training_bindings']['ordered_train_ids'];ids=list(map(int,ids64[:32]))
    if len(set(ids))!=32 or any(1000<=i<1640 for i in ids):raise ValueError('TRAIN32 overlaps forbidden evaluation IDs')
    with h5py.File(prior['dataset']['path'],'r') as stream:
        pixels=np.stack([stream['mask'][i,...,0] for i in ids64]);rgb=np.stack([stream['image'][i] for i in ids])
    if rgb.dtype!=np.uint8:rgb=np.rint(np.clip(rgb.astype(np.float32),0,1)*255).astype(np.uint8)
    labels=clevr_mask_patch(torch.from_numpy(pixels.copy()),2)['patch_labels'].cpu().numpy().astype(np.int64)
    if not np.array_equal(labels,source.full.metrics.independent.modal_labels(pixels,patch=2)):raise ValueError('TRAIN modal conversion mismatch')
    if sha(labels.tobytes())!=prior['training_bindings']['TRAIN64_modal_sha256']:raise ValueError('registered TRAIN64 masks changed')
    labels=labels[:32].copy();dev=dev[list(OFFSETS)].copy()
    if rgb.shape!=(32,128,128,3) or dev.shape!=(16,128,128,3):raise ValueError('native128 RGB geometry changed')
    binding=dict(fingerprint=fingerprint(),source_binding=prior,TRAIN_ids=ids,DEV_ids=source.IDS[list(OFFSETS)].tolist(),
        TRAIN_RGB_sha256=sha(rgb.tobytes()),DEV_RGB_sha256=sha(dev.tobytes()),
        TRAIN_RGB_rows_sha256=[sha(v.tobytes()) for v in rgb],DEV_RGB_rows_sha256=[sha(v.tobytes()) for v in dev],
        TRAIN_labels_sha256=sha(labels.tobytes()),DEV_baseline_sha256=sha(mini.tobytes()),DEV_GT_read=False)
    return dict(found=found,TRAIN_rgb=rgb,TRAIN_labels=labels,DEV_rgb=dev,DEV_baseline=mini,binding=binding)

def row_binding(binding,split,index):
    count={'TRAIN':32,'DEV':16}.get(split)
    if count is None or type(index) is not int or index not in range(count):raise ValueError('invalid cache row identity')
    return dict(split=split,index=index,image_id=binding[split+'_ids'][index],
        inference_seed=211000+index if split=='TRAIN' else 193000+OFFSETS[index],
        RGB_sha256=binding[split+'_RGB_rows_sha256'][index],model_sha256=source.MODELS['2'],
        binding_sha256=sha(json.dumps(binding,sort_keys=True).encode()))

def load_row(folder,expected):
    record=json.loads((folder/'row.json').read_bytes());raw=(folder/'data.npz').read_bytes()
    if record['binding']!=expected or record['data_sha256']!=sha(raw):raise ValueError('cache provenance/hash changed')
    with np.load(io.BytesIO(raw),allow_pickle=False) as saved:x=saved['features'].copy();y=saved['baseline'].copy()
    if x.dtype!=np.float32 or x.shape!=(268,64,64) or not np.isfinite(x).all() or y.dtype!=np.int64 or y.shape!=(64,64) or (y<0).any():raise ValueError('invalid actual-history cache')
    return x,y

def extract(prepared):
    binding=prepared['binding'];admission=json.loads((ARCHIVE/'registration_v1/admission.json').read_bytes())
    if binding!=admission['binding'] or admission['cuda_initialized']:raise ValueError('cache CPU admission changed')
    found=prepared['found'];device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.8,0)
    source.source.model64.move_to_device(found,device);found.wrapped.requires_grad_(False);found.encoder.requires_grad_(False);found.wrapped.eval();found.encoder.eval()
    if source.source.native._state_sha(found.wrapped,found.encoder)!=source.MODELS['2']:raise ValueError('wrong actual warm205 model')
    targets=ARCHIVE/'cache/TRAIN/targets.npz'
    if targets.exists():
        with np.load(targets,allow_pickle=False) as saved:
            if not np.array_equal(saved['labels'],prepared['TRAIN_labels']):raise ValueError('TRAIN labels changed')
    else:
        targets.parent.mkdir(parents=True,exist_ok=True);raw=io.BytesIO();np.savez_compressed(raw,labels=prepared['TRAIN_labels'])
        checkpoint._atomic_bytes(targets,raw.getvalue())
    records=[]
    for split,count in (('TRAIN',32),('DEV',16)):
        for index in range(count):
            expected=row_binding(binding,split,index);folder=ARCHIVE/f'cache/{split}/rows/{index:04d}'
            if folder.exists():features,baseline=load_row(folder,expected)
            else:
                torch.manual_seed(expected['inference_seed']);torch.cuda.manual_seed(expected['inference_seed'])
                started=time.perf_counter()
                with torch.no_grad():
                    image=torch.from_numpy(prepared[split+'_rgb'][index:index+1]).to(device)
                    gamma,trace,adj,z,q,rho=source.source.native._native_forward(found,image,live_tail_steps=0)
                    features=descriptors(trace,rho).cpu().numpy().copy()
                    baseline=source.full.screen.prediction_from_trace(found,trace,adj,q)
                if split=='DEV' and not np.array_equal(baseline,prepared['DEV_baseline'][index]):raise ValueError('205 DEV baseline did not exactly replay')
                stage=folder.with_name(folder.name+f'.{os.getpid()}.{time.time_ns()}.tmp');stage.mkdir(parents=True,exist_ok=False)
                with (stage/'data.npz').open('xb') as stream:
                    np.savez_compressed(stream,features=features,baseline=baseline);stream.flush();os.fsync(stream.fileno())
                write_json(stage/'row.json',dict(binding=expected,data_sha256=sha((stage/'data.npz').read_bytes()),seconds=time.perf_counter()-started))
                stage.rename(folder);del image,gamma,trace,adj,z,q,rho
            if split=='DEV' and not np.array_equal(baseline,prepared['DEV_baseline'][index]):raise ValueError('cached DEV baseline changed')
            records.append(sha((folder/'row.json').read_bytes()))
    if source.source.native._state_sha(found.wrapped,found.encoder)!=source.MODELS['2']:raise ValueError('cache extraction mutated core or encoder')
    protocol=dict(binding=binding,row_metadata_sha256=records,TRAIN_targets_file_sha256=sha(targets.read_bytes()),
        DEV_GT_read=False,core_updates=0,source_model_unchanged=True)
    path=ARCHIVE/'cache/protocol.json'
    if path.exists():
        if json.loads(path.read_bytes())!=protocol:raise ValueError('frozen cache protocol changed')
    else:write_json(path,protocol)
    return protocol

def load_arrays():
    protocol=json.loads((ARCHIVE/'cache/protocol.json').read_bytes());binding=protocol['binding']
    if binding['fingerprint']!=fingerprint() or protocol['DEV_GT_read'] is not False or protocol['core_updates']!=0 or protocol['source_model_unchanged'] is not True:raise ValueError('invalid frozen cache protocol')
    arrays={};records=[]
    for split,count in (('TRAIN',32),('DEV',16)):
        rows=[];labels=[]
        for index in range(count):
            folder=ARCHIVE/f'cache/{split}/rows/{index:04d}';x,y=load_row(folder,row_binding(binding,split,index));rows.append(x);labels.append(y);records.append(sha((folder/'row.json').read_bytes()))
        arrays[split+'_features']=np.stack(rows);arrays[split+'_baseline']=np.stack(labels)
    if records!=protocol['row_metadata_sha256'] or sha(arrays['DEV_baseline'].tobytes())!=binding['DEV_baseline_sha256']:raise ValueError('cache rows or baseline changed')
    path=ARCHIVE/'cache/TRAIN/targets.npz'
    if sha(path.read_bytes())!=protocol['TRAIN_targets_file_sha256']:raise ValueError('TRAIN target file changed')
    with np.load(path,allow_pickle=False) as saved:arrays['TRAIN_labels']=saved['labels'].copy()
    if arrays['TRAIN_labels'].dtype!=np.int64 or arrays['TRAIN_labels'].shape!=(32,64,64) or sha(arrays['TRAIN_labels'].tobytes())!=binding['TRAIN_labels_sha256']:raise ValueError('TRAIN target content changed')
    return arrays,protocol
