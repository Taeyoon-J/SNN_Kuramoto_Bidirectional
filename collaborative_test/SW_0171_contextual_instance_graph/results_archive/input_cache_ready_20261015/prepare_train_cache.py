"""Materialize exact original-source TRAIN512 inputs once; never trains."""
import hashlib,json,os,sys,time,uuid
from pathlib import Path
import numpy as np
import torch
root=Path('/Data0/kevinswk/patch_v2_sw');sys.path.insert(0,str(root))
from collaborative_test.SW_0169_normalized_hard_reset import train as producer
from collaborative_test.SW_0155_supervised_spike_relations import train as alignment,objective
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
base=root/'collaborative_test/SW_0171_contextual_instance_graph/results_archive'
output=base/'train_inputs_20261015'
assert not output.exists(),'preserve existing cache; validate and reuse it'
started=time.time();prepared=producer.prepare_training_inputs_cpu()
data=prepared['data'];ids=list(map(int,prepared['source_ids']))
assert len(ids)==len(set(ids))==512
assert not set(ids).intersection(range(1320,1640))
rgb,masks=alignment.load_aligned_train_rows(alignment.HDF5_PATH,ids,data['rgb'],0,512)
assert np.array_equal(rgb.cpu().numpy(),data['rgb'])
modal=objective.modal_train_targets32(masks).cpu().numpy().reshape(512,32,32)
arrays={'rgb':np.asarray(data['rgb'],dtype=np.uint8),
        'source_support':np.asarray(data['supports'],dtype=np.int32),
        'feasible_targets':np.asarray(data['targets'],dtype=np.int32),
        'modal_gt':modal.astype(np.int32,copy=False),
        'image_ids':np.asarray(ids,dtype=np.int64)}
assert arrays['rgb'].shape==(512,128,128,3)
assert all(arrays[key].shape==(512,32,32) for key in ('source_support','feasible_targets','modal_gt'))
assert np.array_equal(arrays['feasible_targets']>0,arrays['source_support']>0)
assert not torch.cuda.is_initialized()
stage=output.with_name(output.name+'.stage-'+uuid.uuid4().hex)
assert stage.resolve().is_relative_to(base.resolve())
stage.mkdir(parents=True,exist_ok=False)
with (stage/'arrays.npz').open('xb') as stream:
    np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
manifest=dict(experiment='SW0171_original_source_TRAIN_input_cache',status='complete',
    count=512,image_ids=ids,arrays_sha256=digest(stage/'arrays.npz'),
    schema={key:dict(shape=list(value.shape),dtype=str(value.dtype)) for key,value in arrays.items()},
    producer='frozen SW0169.train.prepare_training_inputs_cpu DATA ONLY; original SourceT caches',
    producer_implementation=producer.implementation_fingerprint(),producer_bindings=prepared['bindings'],
    worker_sha256=digest(Path(__file__)),canonical_hdf5_path=str(alignment.HDF5_PATH),
    canonical_hdf5_size_bytes=data['hdf5_size_bytes'],canonical_hdf5_mtime_ns=data['hdf5_mtime_ns'],
    modal_GT_verified_against_canonical_RGB_rows=True,source_QCC_mode='original contiguous CPU B4 source spikes, production threshold .5/min8/largest BG',
    candidate_initialization_not_loaded=True,optimizer_updates=0,cuda_initialized=False,
    validation_data_read=False,TRAIN_masks_read=True,elapsed_seconds=time.time()-started)
with (stage/'manifest.json').open('x') as stream:json.dump(manifest,stream,sort_keys=True,indent=2)
with np.load(stage/'arrays.npz',allow_pickle=False) as check:
    assert set(check.files)==set(arrays)
    assert all(np.array_equal(check[key],value) for key,value in arrays.items())
os.rename(stage,output)
print(json.dumps(dict(status='complete_TRAIN512_cache_no_GPU_or_optimizer',output=str(output),manifest_sha256=digest(output/'manifest.json'),elapsed_seconds=manifest['elapsed_seconds'])),flush=True)
