"""Prepare genuine TRAIN35k MOD inputs, retaining the original held-out1000 IDs."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from mod_benchmark import common as c
from full_benchmark.tfrecord import records,bytes_features

CONFIG={'multi_dsprites':(64,5,'HWE','multi_dsprites_colored_on_colored.tfrecords',64),
        'tetrominoes':(35,4,'EHW','tetrominoes_train.tfrecords',32)}


def prepare(dataset,output):
    torch.set_num_threads(1);c.immutable();out=Path(output)/dataset;out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'builder.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (out/'manifest.json').exists():
            existing=json.loads((out/'manifest.json').read_bytes())
            assert existing['status']=='complete_verified_TRAIN35000'
            return existing
        size,entities,order,name,grid=CONFIG[dataset]
        raw=Path('/work/USERS/kevinswk/datasets')/dataset/'raw'/name
        acquired=json.loads(Path('/work/USERS/kevinswk/datasets/_acquisition_20261029/manifest.json').read_bytes())
        record=next(x for x in acquired['files'] if x['path']==str(raw))
        raw_sha=c.sha(raw);assert raw_sha==record['integrity']['sha256']
        total=37000;train=np.r_[np.arange(6000),np.arange(8000,total)].astype(np.int64)
        validation=np.arange(6000,7000,dtype=np.int64);test=np.arange(7000,8000,dtype=np.int64)
        assert len(train)==len(np.unique(train))==35000 and not np.intersect1d(train,np.r_[validation,test]).size
        rgb=np.lib.format.open_memmap(out/'rgb128.partial.npy',mode='w+',dtype=np.uint8,shape=(total,128,128,3))
        truth=np.lib.format.open_memmap(out/'pixels.partial.npy',mode='w+',dtype=np.uint8,shape=(total,size,size))
        native=np.lib.format.open_memmap(out/'labels64.partial.npy',mode='w+',dtype=np.uint8,shape=(total,64,64))
        reference=torch.load(c.DATA/f'pixel_labels_{dataset}.pt',map_location='cpu',weights_only=True,mmap=True)
        images=torch.load(c.DATA/f'images_{dataset}.pt',map_location='cpu',weights_only=True,mmap=True)
        previous_labels=torch.load(c.DATA/f'targets_{dataset}_grid{grid}.pt',map_location='cpu',weights_only=True,mmap=True)['patch_labels']
        cells=c.benchmark_module('make_targets_grid');started=time.time();chunk=[];row_ids=[]
        coherence_sum=0.;onehot=True;reference_matches=0
        def flush():
            nonlocal chunk,row_ids
            if not chunk:return
            pix=torch.from_numpy(np.stack(chunk).astype(np.int64))
            target=pix if grid==64 else cells.mask_cells(pix,32)[0]
            for j,index in enumerate(row_ids):
                if index<8000:assert torch.equal(target[j],previous_labels[index])
            values=target.numpy().astype(np.uint8)
            if grid==32:values=np.repeat(np.repeat(values,2,axis=1),2,axis=2)
            native[row_ids]=values;chunk=[];row_ids=[]
        # Match protected make128.py nearest exactly: floor(output*input/128).
        nearest=(np.arange(128)*size)//128
        for index,frame in enumerate(records(raw,total)):
            f=bytes_features(frame)
            x=np.frombuffer(b''.join(f['image']),dtype=np.uint8).reshape(size,size,3)
            masks=np.frombuffer(b''.join(f['mask']),dtype=np.uint8)
            masks=masks.reshape(entities,size,size) if order=='EHW' else masks.reshape(size,size,entities).transpose(2,0,1)
            if not np.isin(masks,(0,255)).all():raise ValueError('nonbinary source masks')
            active=masks>127
            if not (active.sum(0)==1).all():raise ValueError('visible masks must partition every pixel')
            y=active.argmax(0).astype(np.uint8)
            if index<8000:
                assert np.array_equal(y,reference[index].numpy())
                assert np.array_equal(x,images[index].numpy());reference_matches+=1
            rgb[index]=x[nearest[:,None],nearest[None,:]];truth[index]=y
            if index<200:
                coherence_sum+=float(((y[:,1:]==y[:,:-1]).mean()+(y[1:]==y[:-1]).mean())/2)
            chunk.append(y);row_ids.append(index)
            if len(chunk)==256 or index==total-1:flush()
            if index%1000==0:c.write(out/'progress.json',dict(stage='building_exact_source_inputs',completed=index+1,total=total))
        assert reference_matches==8000 and coherence_sum/200>.85
        rgb.flush();truth.flush();native.flush();del rgb,truth,native
        for stem in ('rgb128','pixels','labels64'):os.replace(out/f'{stem}.partial.npy',out/f'{stem}.npy')
        np.savez(out/'ids.npz',train=train,validation=validation,test=test)
        result=dict(status='complete_verified_TRAIN35000',dataset=dataset,source=dict(path=str(raw),sha256=raw_sha),
            count=total,unique_train=35000,train_ranges=[[0,6000],[8000,37000]],validation_range=[6000,7000],test_range=[7000,8000],
            native_pixel_shape=[size,size],mask_axis_order=order,training_target_grid=grid,
            target_lift='exact 2x2 repeat to model64' if grid==32 else 'pixel grid64 identity',
            preprocessing='protected make128 nearest rule; original pixel truth unchanged',
            original8000_RGB_and_pixel_and_patch_labels_exact=True,onehot_visible_masks_verified=True,
            first200_spatial_coherence=coherence_sum/200,seconds=time.time()-started,
            files={p.name:dict(size=p.stat().st_size,sha256=c.sha(p)) for p in sorted(out.glob('*.npy'))},
            ids_sha256=c.sha(out/'ids.npz'),benchmark_unchanged=c.immutable())
        c.write(out/'manifest.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=CONFIG,required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();print(json.dumps(prepare(a.dataset,a.output)))
