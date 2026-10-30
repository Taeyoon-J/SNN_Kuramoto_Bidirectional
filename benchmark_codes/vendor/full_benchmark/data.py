"""Dataset-specific immutable input contracts and disjoint splits."""
import json
from pathlib import Path
import numpy as np
from mod_benchmark import common as c
from readout_benchmark import runtime as r
from snn_kuramoto_bidirectional.sw240.data import train_ids


class Dataset:
    def __init__(self,dataset,root):
        self.dataset=dataset;self.root=Path(root);self.labels_opened=False
        if dataset=='clevr':
            source,sha=r.canonical_record();self.train=train_ids(35000)
            self.validation=np.arange(1000,1320,dtype=np.int64);self.test=np.arange(1320,1640,dtype=np.int64)
            self.binding=dict(dataset=dataset,canonical_source=source,canonical_manifest_sha256=sha,
                train_ids=self.train.tolist(),validation_ids=self.validation.tolist(),evaluation_ids=self.test.tolist(),
                scoring='native model grid lifted to native128 pixels; plus original fixed64 patch scores',
                split_name='fixed320 reserved evaluation, not an independent original-paper TEST split')
        elif dataset in ('multi_dsprites','tetrominoes','objects_room','clevrtex_original'):
            self.folder=self.root/'inputs'/dataset
            manifest=json.loads((self.folder/'manifest.json').read_bytes())
            assert manifest['status']=='complete_verified_TRAIN35000'
            with np.load(self.folder/'ids.npz',allow_pickle=False) as ids:
                self.train=ids['train'].copy();self.validation=ids['validation'].copy();self.test=ids['test'].copy()
            assert c.sha(self.folder/'ids.npz')==manifest['ids_sha256']
            self.binding=dict(dataset=dataset,manifest_sha256=c.sha(self.folder/'manifest.json'),source=manifest['source'],
                train_ids=self.train.tolist(),validation_ids=self.validation.tolist(),evaluation_ids=self.test.tolist(),
                scoring='original pixels; Tet model64 modal32 then protected lift to35; model253 native32 lift')
            if dataset in ('objects_room','clevrtex_original'):
                with np.load(self.folder/'ids.npz',allow_pickle=False) as ids:self.source_ids=ids['source_ids'].copy()
                self.binding.update(preprocessing=manifest['preprocessing'],evaluation_source_ids=self.source_ids[self.test].tolist(),
                    scoring='standard official cropped/resized128 pixels' if dataset=='clevrtex_original' else 'original64 pixels on explicitly held-out in-distribution scenes',
                    evaluation_contract='fixed1000 official TEST subset' if dataset=='clevrtex_original' else manifest['primary_evaluation'])
        else:raise ValueError('dataset contract not yet implemented; do not silently substitute another dataset')
        assert len(self.train)==len(np.unique(self.train))==35000
        parent=self.train.copy()
        self.train=np.sort(np.random.RandomState(2553500).choice(parent,3500,replace=False))
        self.binding.update(parent_TRAIN_count=len(parent),subset_seed=2553500,
                            train_ids=self.train.tolist(),unique_TRAIN=3500)
        assert len(self.train)==len(np.unique(self.train))==3500
        assert not np.intersect1d(self.train,np.r_[self.validation,self.test]).size
        assert not np.intersect1d(self.validation,self.test).size

    def rgb(self,ids,split='train'):
        ids=np.asarray(ids,dtype=np.int64)
        if self.dataset=='clevr':return r.read_rgb(self.root,self.dataset,split,ids)
        return np.asarray(np.load(self.folder/'rgb128.npy',mmap_mode='r')[ids]).copy()

    def labels(self,ids,split='train'):
        self.labels_opened=True;ids=np.asarray(ids,dtype=np.int64)
        if self.dataset=='clevr':return r.truth(self.dataset,split,ids)
        return np.asarray(np.load(self.folder/'labels64.npy',mmap_mode='r')[ids],dtype=np.int64).copy()

    def pixels(self,ids):
        self.labels_opened=True;ids=np.asarray(ids,dtype=np.int64)
        if self.dataset=='clevr':
            import h5py
            source,_=r.canonical_record()
            with h5py.File(source['path'],'r') as stream:values=np.asarray(stream['mask'][ids,...,0])
            return values.astype(np.int64)
        return np.asarray(np.load(self.folder/'pixels.npy',mmap_mode='r')[ids],dtype=np.int64).copy()
