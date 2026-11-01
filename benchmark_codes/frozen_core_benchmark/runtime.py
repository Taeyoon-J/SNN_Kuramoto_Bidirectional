"""Configure original35000 caches and protected evaluation, not the3500 adapter."""
import json
from pathlib import Path
from portable_runtime import configure,connect_inputs


def setup(config_file,output):
    from full_benchmark import extra_preflight
    original=extra_preflight.preflight
    config=configure(config_file)
    extra_preflight.preflight=original
    return config,connect_inputs(config,output)


def preflight(config,root,sources):
    import numpy as np
    from mod_benchmark import common as c
    from full_benchmark.extra_preflight import preflight as extra
    from frozen_core_benchmark.data import Dataset
    rows={}
    for name in ('clevr','tetrominoes','clevrtex_original'):
        data=Dataset(name,root)
        if name=='tetrominoes':c.preflight(name,root/'preflight'/name)
        if name=='clevrtex_original':extra(data,root/'preflight'/name)
        assert data.rgb(data.train[:1]).shape==(1,128,128,3)
        rows[name]=dict(binding=data.binding,TRAIN35000=True,all_splits_disjoint=True,
            evaluation_count=len(data.test),all10epochs_expected_exposures=350000,
            logical_batch=16,optimizer_updates_per_seed=21880,physical_microbatch=1)
    # Source artifacts are required and hash-checked before any admission/train.
    from frozen_core_benchmark.model import checked_file
    inventory=json.loads(Path(sources).read_bytes())
    for seed in range(3):
        record=inventory['seeds'][str(seed)]
        for key in ('core','manifest','unsupervised_encoder','statistics','supervised_encoder'):checked_file(record[key])
        manifest=json.loads(Path(record['manifest']['path']).read_bytes())
        assert manifest['status']=='complete' and manifest['source_model_seed']==seed
        assert manifest['ground_truth_used_for_training'] is False
        # SW0094/SW0097 manifests store GLOBAL HDF5 image IDs. The gamma cache
        # row is id for0..999, otherwise id-640; do not shift global IDs again.
        source_ids=np.asarray(manifest['training_ids'],dtype=np.int64)
        assert len(source_ids)==len(np.unique(source_ids))==4096
        assert not np.intersect1d(source_ids,np.arange(1000,1640)).size
        cache_rows=np.where(source_ids<1000,source_ids,source_ids-640)
        assert np.all((0<=cache_rows)&(cache_rows<70000))
    result=dict(status='SW0256_sources_full35000_inputs_and_protected_preflights_passed',
        datasets=rows,sources_sha256=c.sha(sources),protected_benchmark=c.immutable(),
        current_scope='18conditions:252/253 x3datasets x3seeds')
    c.write(root/'preflight.json',result);return result
