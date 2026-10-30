"""CPU-only phase2 inputs. No phase2 training may overtake completed phase1."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import numpy as np
from PIL import Image
import torch
from mod_benchmark import common as c
from full_benchmark.tfrecord import records,bytes_features

BASE=Path('/work/USERS/kevinswk/datasets')
LOADER_URL='https://raw.githubusercontent.com/karazijal/clevrtex/master/ool/clevrtex_eval.py'
LOADER_SHA='d915af96e8c78d0e07ff95bfe5e1a7e785043a4e517c2621d33dc577b91f1335'


def arrays(out,total,pixel_size):
    return [np.lib.format.open_memmap(out/f'{name}.partial.npy',mode='w+',dtype=np.uint8,shape=shape)
        for name,shape in [('rgb128',(total,128,128,3)),('pixels',(total,pixel_size,pixel_size)),('labels64',(total,64,64))]]


def finish(out,metadata,values,ids,source_ids):
    for value in values:value.flush()
    for name in ('rgb128','pixels','labels64'):os.replace(out/f'{name}.partial.npy',out/f'{name}.npy')
    np.savez(out/'ids.npz',**ids,source_ids=source_ids)
    metadata.update(status='complete_verified_TRAIN35000',count=len(source_ids),unique_train=35000,
        ids_sha256=c.sha(out/'ids.npz'),files={p.name:dict(size=p.stat().st_size,sha256=c.sha(p)) for p in out.glob('*.npy')},
        benchmark_unchanged=c.immutable(),phase2_training_started=False)
    c.write(out/'manifest.json',metadata);return metadata


def prepare_room(out):
    """Primary: disjoint in-distribution holdout. Official OOD files stay separate."""
    raw=BASE/'objects_room/raw/objects_room_train.tfrecords'
    acquisition=json.loads((BASE/'_acquisition_20261029/manifest.json').read_bytes())
    expected=next(x for x in acquisition['files'] if x['path']==str(raw))['integrity']['sha256']
    assert c.sha(raw)==expected
    total=37000;rgb,pixels,labels=arrays(out,total,64);coherence=0.
    for index,frame in enumerate(records(raw,total)):
        features=bytes_features(frame)
        image=np.frombuffer(b''.join(features['image']),dtype=np.uint8).reshape(64,64,3)
        masks=np.frombuffer(b''.join(features['mask']),dtype=np.uint8).reshape(7,64,64)
        assert np.isin(masks,(0,255)).all() and ((masks>127).sum(0)==1).all()
        entity=(masks>127).argmax(0)
        # Official sky/floor/two walls are ALL background in FG metrics.
        truth=np.where(entity<4,0,entity-3).astype(np.uint8)
        assert not ((entity<4)&(truth>0)).any() and not ((entity>=4)&(truth==0)).any()
        rgb[index]=image.repeat(2,0).repeat(2,1);pixels[index]=truth;labels[index]=truth
        if index<200:coherence+=float(((entity[:,1:]==entity[:,:-1]).mean()+(entity[1:]==entity[:-1]).mean())/2)
        if index%1000==0:c.write(out/'progress.json',dict(stage='exact_ObjectsRoom_train_source',completed=index+1,total=total))
    assert coherence/200>.85
    ids=dict(train=np.arange(35000),validation=np.arange(35000,36000),test=np.arange(36000,37000))
    metadata=dict(dataset='objects_room',source=dict(path=str(raw),sha256=expected),
        native_pixel_shape=[64,64],mask_axis_order='EHW',native_entities=7,
        ground_truth_background_entities=[0,1,2,3],ground_truth_object_mapping={'4':1,'5':2,'6':3},
        primary_evaluation='1000 held-out in-distribution TRAIN-file scenes; NOT the official OOD variants',
        train_source_ids=[0,35000],validation_source_ids=[35000,36000],test_source_ids=[36000,37000],
        official_OOD_variants_available_separately=['empty_room','six_objects','identical_color'],
        preprocessing='64 to128 nearest replication; original64-pixel truth unchanged',
        source_schema='https://github.com/google-deepmind/multi_object_datasets/blob/master/objects_room.py',
        background_semantics='https://github.com/google-deepmind/multi_object_datasets#objects-room',
        training_target_grid=64,visible_masks_onehot_verified=True,first200_spatial_coherence=coherence/200)
    return finish(out,metadata,[rgb,pixels,labels],ids,np.arange(total))


def official_getter(raw):
    """Execute only pinned official __getitem__ for an independent input oracle."""
    tree=ast.parse(raw)
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='CLEVRTEX')
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='__getitem__')
    class ToTensor:
        @staticmethod
        def to_tensor(array):return torch.from_numpy(array.transpose(2,0,1).copy()).float()/255.
    scope=dict(Image=Image,np=np,torch=torch,Ft=ToTensor)
    exec(compile(ast.Module(body=[method],type_ignores=[]),'pinned_official_CLEVRTEX_getitem','exec'),scope)
    return scope['__getitem__']


def prepare_texture(out):
    raw=urllib.request.urlopen(LOADER_URL,timeout=25).read()
    assert hashlib.sha256(raw).hexdigest()==LOADER_SHA
    (out/'official_loader.py').write_bytes(raw);oracle=official_getter(raw)
    source=BASE/'clevrtex_original/data/clevrtex_full'
    train=np.sort(np.random.RandomState(25535000).choice(np.arange(10000,50000),35000,replace=False))
    # These fixed1000 are an explicit subset of each official5000 holdout split.
    validation=np.arange(5000,6000);test=np.arange(1000)
    selected=np.r_[train,validation,test]
    assert len(np.unique(selected))==37000
    rgb,pixels,labels=arrays(out,len(selected),128);source_digest=hashlib.sha256();chunk=[];rows=[]
    cells=c.benchmark_module('make_targets_grid');oracle_checks=0
    def flush():
        nonlocal chunk,rows
        if chunk:
            labels[rows]=cells.mask_cells(torch.from_numpy(np.stack(chunk).astype(np.int64)),64)[0].numpy().astype(np.uint8)
            chunk=[];rows=[]
    for row,image_id in enumerate(selected):
        folder=source/str(int(image_id)//1000);stem=f'CLEVRTEX_full_{int(image_id):06d}'
        image_path=folder/(stem+'.png');mask_path=folder/(stem+'_flat.png')
        for p in (image_path,mask_path):
            data=p.read_bytes();source_digest.update(p.relative_to(source).as_posix().encode());source_digest.update(hashlib.sha256(data).digest())
        with Image.open(image_path) as original,Image.open(mask_path) as mask:
            assert original.size==mask.size==(320,240)
            # Official default: centered192x192 crop of BOTH, then RGB bilinear,
            # integer mask nearest to128. Preserve alpha during PIL interpolation.
            box=(64,24,256,216)
            image=np.asarray(original.crop(box).resize((128,128),resample=Image.Resampling.BILINEAR))[...,:3].copy()
            truth=np.asarray(mask.crop(box).resize((128,128),resample=Image.Resampling.NEAREST)).copy()
        assert image.dtype==truth.dtype==np.uint8 and truth.shape==(128,128) and truth.min()>=0 and truth.max()<=10
        if row in (0,1,34999,35000,35999,36000,36999):
            from types import SimpleNamespace
            context=SimpleNamespace(_index_with_bias_and_limit=lambda i:int(image_id),index={int(image_id):image_path},
                mask_index={int(image_id):mask_path},crop=True,resize=(128,128),return_metadata=False)
            official_id,official_rgb,official_mask=oracle(context,0)
            assert official_id==int(image_id) and torch.equal(official_rgb,torch.from_numpy(image.transpose(2,0,1).copy()).float()/255.)
            assert np.array_equal(official_mask.numpy()[0],truth);oracle_checks+=1
        rgb[row]=image;pixels[row]=truth;chunk.append(truth);rows.append(row)
        if len(chunk)==256 or row==len(selected)-1:flush()
        if row%1000==0:c.write(out/'progress.json',dict(stage='official_CLEVRTex_cropped_resized_inputs',completed=row+1,total=len(selected)))
    assert oracle_checks==7
    ids=dict(train=np.arange(35000),validation=np.arange(35000,36000),test=np.arange(36000,37000))
    metadata=dict(dataset='clevrtex_original',source=dict(path=str(source),selected_RGB_masks_ordered_sha256=source_digest.hexdigest()),
        official_loader=dict(url=LOADER_URL,sha256=LOADER_SHA),official_split_ranges=dict(train=[10000,50000],validation=[5000,10000],test=[0,5000]),
        train_selection='sorted RandomState25535000 choose35000 without replacement from official40000 TRAIN',
        validation_source_range=[5000,6000],test_source_range=[0,1000],holdout_subset_count=1000,
        raw_pixel_shape=[240,320],pixel_evaluation_shape=[128,128],training_target_grid=64,
        preprocessing='pinned official default center192 crop; RGBA/RGB bilinear128; mask nearest128; modal64 TRAIN targets',
        truth_note='standardized official128 mask, explicitly not full uncropped240x320 pixel evaluation',
        actual_official_getitem_oracle_checks=oracle_checks,variant='Original full, not CAMO/OOD/PBG/VBG')
    return finish(out,metadata,[rgb,pixels,labels],ids,selected)


def prepare(dataset,root):
    torch.set_num_threads(1);c.immutable();out=Path(root)/'inputs'/dataset;out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'builder.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (out/'manifest.json').exists():return json.loads((out/'manifest.json').read_bytes())
        started=time.time();result=(prepare_room if dataset=='objects_room' else prepare_texture)(out)
        result['seconds']=time.time()-started;c.write(out/'manifest.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=('objects_room','clevrtex_original'));p.add_argument('--root',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.dataset,a.root)))
