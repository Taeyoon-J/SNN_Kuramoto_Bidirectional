"""Actual input/evaluation integrity checks for datasets absent from legacy preflight."""
import json
from pathlib import Path
import numpy as np
import torch
from mod_benchmark import common as c


def preflight(dataset,out):
    if dataset.dataset not in ('objects_room','clevrtex_original'):raise ValueError('phase2 only')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);c.immutable()
    manifest=json.loads((dataset.folder/'manifest.json').read_bytes())
    assert manifest['status']=='complete_verified_TRAIN35000' and manifest['unique_train']==35000
    all_ids=np.r_[dataset.train,dataset.validation,dataset.test]
    assert len(all_ids)==len(np.unique(all_ids))==37000
    assert len(dataset.source_ids)==37000 and len(np.unique(dataset.source_ids))==37000
    if dataset.dataset=='clevrtex_original':
        assert np.all((dataset.source_ids[dataset.train]>=10000)&(dataset.source_ids[dataset.train]<50000))
        assert np.array_equal(dataset.source_ids[dataset.validation],np.arange(5000,6000))
        assert np.array_equal(dataset.source_ids[dataset.test],np.arange(1000))
        assert manifest['actual_official_getitem_oracle_checks']==7
        pixel_size=128
    else:
        assert np.array_equal(dataset.source_ids,np.arange(37000))
        assert manifest['ground_truth_background_entities']==[0,1,2,3]
        assert manifest['visible_masks_onehot_verified'];pixel_size=64
    expected_shapes={'rgb128.npy':(37000,128,128,3),'pixels.npy':(37000,pixel_size,pixel_size),'labels64.npy':(37000,64,64)}
    for name,shape in expected_shapes.items():
        path=dataset.folder/name;value=np.load(path,mmap_mode='r')
        assert value.shape==shape and value.dtype==np.uint8
        assert path.stat().st_size==manifest['files'][name]['size'] and c.sha(path)==manifest['files'][name]['sha256']
    selected=np.r_[dataset.train[:8],dataset.validation[:8],dataset.test[:8]]
    pixels=np.load(dataset.folder/'pixels.npy',mmap_mode='r')[selected].astype(np.int64)
    labels=np.load(dataset.folder/'labels64.npy',mmap_mode='r')[selected]
    cells=c.benchmark_module('make_targets_grid');metric=c.benchmark_module('evaluation')
    oracle=torch.from_numpy(pixels) if pixel_size==64 else cells.mask_cells(torch.from_numpy(pixels),64)[0]
    assert np.array_equal(oracle.numpy(),labels)
    # Check pixel lifting against explicit unmodified protected cell boundaries.
    references={}
    for grid in (32,64):
        target=torch.from_numpy(pixels)
        modal=cells.mask_cells(target,grid)[0]
        row,_=cells.cell_index(pixel_size,grid,'cpu');col,_=cells.cell_index(pixel_size,grid,'cpu')
        scored=metric.evaluate_patch_masks(modal[:,row][:,:,col],target)
        references[str(grid)]=dict(mean={k:float(v) for k,v in scored['mean'].items()},
            valid_count={k:int(v) for k,v in scored['valid_count'].items()})
    result=dict(status='actual_phase2_input_split_modal_and_pixel_scoring_preflight_passed',dataset=dataset.dataset,
        input_manifest_sha256=c.sha(dataset.folder/'manifest.json'),all_array_SHA256_verified=True,
        train35000_unique=True,all_splits_disjoint=True,source_id_ranges_verified=True,
        native64_targets_equal_protected_modal=True,checked_rows=selected.tolist(),
        modal_oracle_references_not_claimed_global_upper_bounds=references,
        protected_evaluation_and_cell_builder_unchanged=c.immutable(),
        legacy_gamma_preflight_note='legacy benchmark/preflight.py needs fixed encoder gamma artifacts; these end-to-end extra datasets use actual input/split/modal/scoring checks instead',
        no_dummy_gamma_or_target_artifacts_created=True)
    c.write(out/'preflight.json',result);return result


if __name__=='__main__':
    import argparse,time
    from full_benchmark.data import Dataset
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True,choices=('objects_room','clevrtex_original'))
    parser.add_argument('--root',required=True);parser.add_argument('--output',required=True);a=parser.parse_args()
    directory=Path(a.root)/'inputs'/a.dataset
    while not (directory/'manifest.json').exists():
        launch=json.loads((directory/'launch.json').read_bytes());proc=Path('/proc')/str(launch['pid'])
        if not proc.exists() or not proc.joinpath('cmdline').read_bytes() or proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z':
            raise RuntimeError('input builder terminated before manifest; inspect original stderr, do not auto-restart')
        time.sleep(10)
    torch.set_num_threads(1);print(json.dumps(preflight(Dataset(a.dataset,a.root),a.output)))
