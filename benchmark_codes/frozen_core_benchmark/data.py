"""Use the full registered35000 TRAIN pool; preserve all held-out IDs and pixels."""
import numpy as np
from full_benchmark.data import Dataset as SubsetDataset
from snn_kuramoto_bidirectional.sw240.data import train_ids


class Dataset(SubsetDataset):
    def __init__(self,name,root):
        if name not in ('clevr','tetrominoes','clevrtex_original'):raise ValueError('outside SW0256 scope')
        super().__init__(name,root)
        if name=='clevr':self.train=train_ids(35000)
        else:
            with np.load(self.folder/'ids.npz',allow_pickle=False) as saved:self.train=saved['train'].copy()
        self.binding.pop('subset_seed',None)
        self.binding.update(train_ids=self.train.tolist(),unique_TRAIN=35000,parent_TRAIN_count=35000)
        assert len(self.train)==len(np.unique(self.train))==35000
        assert not np.intersect1d(self.train,np.r_[self.validation,self.test]).size

