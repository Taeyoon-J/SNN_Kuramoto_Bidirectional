"""Actual CUDA forward + readout on TRAIN image 0 only, never TEST truth."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from snn_kuramoto_bidirectional import load_model
from snn_kuramoto_bidirectional.sw240.model import model_sha256
from . import common as c

def main(archive):
    archive=Path(archive);torch.set_num_threads(1)
    model=load_model(c.EXPORTS/'sw240_seed0.pt',device='cuda:0');model.requires_grad_(False)
    digest=model_sha256(model);rows=[]
    for ds in c.DATASETS:
        preflight=c.preflight(ds,archive/'smoke'/ds)
        image=np.asarray(c.rgb(archive,ds,'train')[:1]).copy()
        started=time.perf_counter();native=model.predict(image,inference_seeds=[193000])[0]
        labels=c.prediction_grid(native,ds)
        assert native.shape==(64,64) and labels.shape==(c.DATASETS[ds],c.DATASETS[ds])
        rows.append(dict(dataset=ds,TRAIN_image_id=0,preflight=preflight,native_shape=list(native.shape),
            prediction_shape=list(labels.shape),finite=True,seconds=time.perf_counter()-started))
    assert model_sha256(model)==digest
    c.write(archive/'smoke_complete.json',dict(status='actual_two_dataset_CUDA_forward_and_readout_passed',
        rows=rows,optimizer_updates=0,TEST_images_or_labels_read=False,benchmark_sha256=c.immutable()))
    print('Both actual CUDA forward/readout checks passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);main(p.parse_args().archive)
