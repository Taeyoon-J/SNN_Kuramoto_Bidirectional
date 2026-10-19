"""Identical239 split, shuffle and tensors, served from the verified canonical cache."""
import hashlib
import json
from pathlib import Path
import numpy as np
from collaborative_test.SW_0239_native64_best_70000 import data as reference
from . import cache

BATCH=reference.BATCH;EPOCHS=reference.EPOCHS;TRAIN_COUNT=reference.TRAIN_COUNT
STEPS_PER_EPOCH=reference.STEPS_PER_EPOCH;TOTAL_STEPS=reference.TOTAL_STEPS
train_ids=reference.train_ids;validate_ids=reference.validate_ids;epoch_ids=reference.epoch_ids
batch_ids=reference.batch_ids;split_binding=reference.split_binding;canonical_rgb=reference.canonical_rgb
modal_labels=reference.modal_labels
ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'collaborative_testing/generated/collaborative_test/SW_0240_native64_70000_cached_io/input_cache'


def cache_binding(verify_files=False):
    raw=(CACHE/'manifest.json').read_bytes();manifest=json.loads(raw)
    if (manifest['status']!='complete_exact_registered_TRAIN70000_input_cache'
            or manifest['binding']['registered_split']!=reference.split_binding()
            or manifest['binding']['registered_input_code_sha256']!=cache.digest_file(Path(reference.__file__))
            or manifest['binding']['builder_code_sha256']!=cache.digest_file(Path(cache.__file__))
            or manifest['source_evaluation_masks_read'] is not False
            or manifest['logical_dataset_changed'] is not False):
        raise ValueError('complete identical canonical TRAIN cache required')
    for name,shape in (('rgb.npy',(70000,128,128,3)),('labels64.npy',(70000,64,64))):
        value=np.load(CACHE/name,mmap_mode='r',allow_pickle=False)
        if value.shape!=shape or value.dtype!=np.uint8:raise ValueError('canonical input cache geometry changed')
        if verify_files and cache.digest_file(CACHE/name)!=manifest['file_sha256'][name]:
            raise ValueError('canonical input cache file changed')
    return dict(manifest_sha256=hashlib.sha256(raw).hexdigest(),file_sha256=manifest['file_sha256'],
        dataset=manifest['binding']['dataset'],logical_dataset_changed=False)


def read_batch(stream,ids):
    rgb,labels=cache.read_batch(CACHE,ids)
    if rgb.shape!=(len(ids),128,128,3) or rgb.dtype!=np.uint8 or labels.shape!=(len(ids),64,64) or labels.dtype!=np.int64:
        raise ValueError('identical original239 batch geometry and dtype required')
    return rgb,labels
