"""Chunk-aligned materialization must never touch reserved evaluation IDs."""
import numpy as np
from collaborative_test.SW_0240_native64_70000_cached_io import cache
from collaborative_test.SW_0239_native64_best_70000 import data


def test_complete_chunk_aligned_TRAIN_coverage_without_reserved_masks():
    blocks=cache.spans(3125)
    ids=np.concatenate([np.arange(begin,end) for begin,end in blocks])
    assert np.array_equal(ids,data.train_ids())
    assert np.array_equal(cache.positions(ids),np.arange(70000))
    assert blocks[:2]==[(0,1000),(1640,3125)]
    for begin,end in blocks:
        assert end<=((begin//3125)+1)*3125


def test_cached_unsorted_batch_preserves_exact_tensor_identity(tmp_path):
    # Reuse SW0239's workspace-owned temporary directory fixture via local conftest.
    import json
    import numpy.lib.format
    (tmp_path/'manifest.json').write_bytes(json.dumps({'fixture':True}).encode())
    rgb=np.lib.format.open_memmap(tmp_path/'rgb.npy',mode='w+',dtype='u1',shape=(70000,1,1,3))
    labels=np.lib.format.open_memmap(tmp_path/'labels64.npy',mode='w+',dtype='u1',shape=(70000,1,1))
    ids=np.array([70639,0,1640,999])
    for i,pos in enumerate(cache.positions(ids)):
        rgb[pos]=i+1;labels[pos]=i+2
    rgb.flush();labels.flush()
    actual_rgb,actual_labels=cache.read_batch(tmp_path,ids)
    assert np.array_equal(actual_rgb[:,0,0,0],np.arange(1,5))
    assert np.array_equal(actual_labels[:,0,0],np.arange(2,6)) and actual_labels.dtype==np.int64
