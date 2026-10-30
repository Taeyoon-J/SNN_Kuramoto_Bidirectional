"""Fit initial-encoder normalization on the exact TRAIN35,000, without any masks."""
import hashlib
import numpy as np
import torch


@torch.no_grad()
def fit_clevr(model,*,progress=None):
    from readout_benchmark import runtime as r
    from snn_kuramoto_bidirectional.sw240.data import train_ids
    ids=train_ids(35000)
    source,manifest_sha=r.canonical_record()
    images=np.load(r.CANONICAL/'rgb.npy',mmap_mode='r')
    device=next(model.parameters()).device
    sums=torch.zeros(8,device=device,dtype=torch.float64);squares=torch.zeros_like(sums);count=0
    fast=hasattr(model.encoder,'base_encoder')
    if fast:
        assert bool((model.encoder.context[-1].weight==0).all()) and bool((model.encoder.context[-1].bias==0).all())
    encoder=model.encoder.base_encoder if fast else model.encoder
    identity=hashlib.sha256()
    for name,tensor in sorted(encoder.state_dict().items()):identity.update(name.encode());identity.update(tensor.cpu().numpy().tobytes())
    for start in range(0,len(ids),256):
        rows=ids[start:start+256];positions=np.where(rows<1000,rows,rows-640)
        rgb=torch.from_numpy(np.asarray(images[positions]).copy()).to(device).permute(0,3,1,2).float()/255.
        features=encoder(rgb)
        if start==0 and fast:assert torch.equal(features,model.encoder(rgb))
        double=features.double();sums+=double.sum((0,2,3));squares+=double.square().sum((0,2,3))
        count+=features.shape[0]*features.shape[2]*features.shape[3]
        if progress and (start//256)%10==0:progress(min(start+256,len(ids)))
    if model.variant==253:
        mean=(sums.sum()/(count*8)).repeat(8)
        variance=(squares.sum()/(count*8)-mean[0].square()).repeat(8)
        mode='global_scalar_as_original_unsupervised_encoder'
    else:
        mean=sums/count;variance=squares/count-mean.square();mode='per_channel_as_supervised_family'
    std=variance.clamp_min(1e-8).sqrt()
    model.feature_mean.copy_(mean.float().view(1,8,1,1));model.feature_std.copy_(std.float().view(1,8,1,1))
    model.normalization_provenance='own_random_encoder_exact_selected_TRAIN_no_masks'
    return dict(mean=model.feature_mean.cpu().clone(),std=model.feature_std.cpu().clone(),
        provenance=model.normalization_provenance,mode=mode,images=35000,feature_pixels_per_channel=count,
        train_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),initial_CNN_sha256=identity.hexdigest(),
        canonical_manifest_sha256=manifest_sha,masks_read=False)


@torch.no_grad()
def fit_dataset(model,dataset,*,progress=None):
    ids=dataset.train;device=next(model.parameters()).device
    sums=torch.zeros(8,device=device,dtype=torch.float64);squares=torch.zeros_like(sums);count=0
    fast=hasattr(model.encoder,'base_encoder')
    encoder=model.encoder.base_encoder if fast else model.encoder
    identity=hashlib.sha256()
    for name,tensor in sorted(encoder.state_dict().items()):identity.update(name.encode());identity.update(tensor.cpu().numpy().tobytes())
    for start in range(0,len(ids),256):
        rgb=torch.from_numpy(dataset.rgb(ids[start:start+256])).to(device).permute(0,3,1,2).float()/255.
        features=encoder(rgb)
        if start==0 and fast:assert torch.equal(features,model.encoder(rgb))
        double=features.double();sums+=double.sum((0,2,3));squares+=double.square().sum((0,2,3))
        count+=features.shape[0]*features.shape[2]*features.shape[3]
        if progress and start%2560==0:progress(min(start+256,len(ids)))
    mean=sums/count;variance=squares/count-mean.square();mode='per_channel_as_supervised_family'
    if model.variant==253:
        mean=(sums.sum()/(count*8)).repeat(8);variance=(squares.sum()/(count*8)-mean[0].square()).repeat(8)
        mode='global_scalar_as_original_unsupervised_encoder'
    model.feature_mean.copy_(mean.float().view(1,8,1,1));model.feature_std.copy_(variance.clamp_min(1e-8).sqrt().float().view(1,8,1,1))
    model.normalization_provenance='own_random_encoder_exact_selected_TRAIN_no_masks'
    assert not dataset.labels_opened
    return dict(mean=model.feature_mean.cpu().clone(),std=model.feature_std.cpu().clone(),provenance=model.normalization_provenance,
        mode=mode,images=len(ids),feature_pixels_per_channel=count,train_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        initial_CNN_sha256=identity.hexdigest(),dataset_binding=dataset.binding,masks_read=False)
