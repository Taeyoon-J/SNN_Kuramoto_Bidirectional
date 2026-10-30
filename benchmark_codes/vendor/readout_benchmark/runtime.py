"""Resumable SW0251 spike cache, final-mask learning and unchanged metric evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from mod_benchmark import common as c
from snn_kuramoto_bidirectional import load_model
from snn_kuramoto_bidirectional.sw240.model import model_sha256
from snn_kuramoto_bidirectional.sw240.partition import foreground_induced_merge
from snn_kuramoto_bidirectional.sw240.readout import prediction_from_trace
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
from .head import ActivityMaskHead, activity_features, mask_objective

ROOT = Path('/Data0/kevinswk/patch_v2_sw')
OLD = ROOT/'collaborative_testing/generated/SW0250_mod_benchmark'
CANONICAL = ROOT/'collaborative_testing/generated/collaborative_test/SW_0240_native64_70000_cached_io/input_cache'
KEYS = ('fg_ari', 'foreground_iou', 'matched_object_iou')


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    torch.save(value, temp); os.replace(temp, path)


def fingerprint():
    return {p.name:c.sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def source(seed, dataset='clevr', mode='zero_shot'):
    record = c.source_record(seed)
    if mode == 'finetune' and dataset != 'clevr':
        folder = OLD/'jobs'/dataset/'finetune'/f'seed{seed}'
        report = json.loads((folder/'training_report.json').read_bytes())
        path = folder/'trained.pt'
        assert c.sha(path) == report['checkpoint_sha256']
        assert report['updates'] == 3750 and report['unique_train_images'] == 6000
        assert report['binding']['source'] == record
        return path, dict(file=str(path), sha256=c.sha(path), reused_verified_SW0250_encoder_adaptation=True,
                          encoder_updates=3750, encoder_unique_TRAIN=6000, encoder_epochs=10)
    path = c.EXPORTS/record['file']
    assert c.sha(path) == record['sha256']
    return path, dict(record, file=str(path), reused_verified_SW0250_encoder_adaptation=False)


def train_ids(count):
    pool = np.r_[np.arange(1000), np.arange(1640,70640)].astype(np.int64)
    return np.random.RandomState(251).permutation(pool)[:count]


def ids_for(dataset, split, count):
    if dataset == 'clevr':
        if split == 'train': return train_ids(count)
        if split == 'validation': return np.arange(1000,1000+count,dtype=np.int64)
        if split == 'fixed320' and count == 320: return np.arange(1320,1640,dtype=np.int64)
    else:
        if split == 'train': return np.random.RandomState(251).permutation(6000)[:count]
        if split == 'validation': return np.arange(6000,6000+count,dtype=np.int64)
        if split == 'test': return np.arange(7000,7000+count,dtype=np.int64)
    raise ValueError('unregistered split/count')


def canonical_record():
    manifest = json.loads((CANONICAL/'manifest.json').read_bytes())
    assert manifest['status'] == 'complete_exact_registered_TRAIN70000_input_cache'
    record = manifest['binding']['dataset']; path = Path(record['path']); stat = path.stat()
    assert stat.st_size == record['size'] and stat.st_mtime_ns == record['mtime_ns']
    return record, c.sha(CANONICAL/'manifest.json')


def read_rgb(archive, dataset, split, ids):
    if dataset == 'clevr':
        record, _ = canonical_record()
        if split == 'train':
            positions = np.where(ids < 1000,ids,ids-640)
            return np.asarray(np.load(CANONICAL/'rgb.npy',mmap_mode='r')[positions]).copy()
        import h5py
        with h5py.File(record['path'],'r') as stream:
            x = np.asarray(stream['image'][ids])
        if x.dtype != np.uint8:
            assert x.min() >= 0 and x.max() <= 1
            x = np.rint(x*255).astype(np.uint8)
        return x
    if split in ('train','test'):
        x = c.rgb(OLD,dataset,split)
        return np.asarray(x[ids if split=='train' else ids-7000]).copy()
    # Original, protected preprocessing for validation too.
    import subprocess, sys
    folder = Path(archive)/'inputs'/dataset; folder.mkdir(parents=True,exist_ok=True)
    path = folder/'validation128.npy'
    import fcntl
    lock = (folder/'validation128.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX)
    if not path.exists():
        temp = folder/'validation128.partial.npy'
        result = subprocess.run([sys.executable,'-B',str(c.CODE/'benchmark/make128.py'),'--dataset',dataset,
                                 '--src',str(c.DATA),'--split','validation','--out',str(temp)],
                                capture_output=True,text=True,timeout=600)
        (folder/'validation_make128.stdout').write_text(result.stdout,encoding='utf-8')
        if result.returncode: raise RuntimeError(result.stderr)
        os.replace(temp,path)
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    return np.asarray(np.load(path,mmap_mode='r')[ids-6000]).copy()


def truth(dataset, split, ids):
    if dataset == 'clevr':
        if split == 'train':
            assert not np.any((ids>=1000)&(ids<1640))
            pos = np.where(ids<1000,ids,ids-640)
            return np.asarray(np.load(CANONICAL/'labels64.npy',mmap_mode='r')[pos],dtype=np.int64).copy()
        import h5py
        from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
        record,_ = canonical_record()
        with h5py.File(record['path'],'r') as stream: masks=np.asarray(stream['mask'][ids,...,0])
        return clevr_mask_patch(torch.from_numpy(masks.copy()),2)['patch_labels'].numpy().astype(np.int64)
    grid=c.DATASETS[dataset]
    blob=torch.load(c.DATA/f'targets_{dataset}_grid{grid}.pt',map_location='cpu',weights_only=True,mmap=True)
    value=blob['patch_labels'][torch.as_tensor(ids)].numpy().astype(np.int64)
    return np.repeat(np.repeat(value,2,axis=1),2,axis=2) if grid==32 else value


def row_folder(archive,dataset,mode,seed):
    return Path(archive)/'cache'/dataset/mode/f'seed{seed}'


def extraction(archive,dataset,mode,seed,split,count,device):
    c.immutable(); ids=ids_for(dataset,split,count)
    folder=row_folder(archive,dataset,mode,seed); folder.mkdir(parents=True,exist_ok=True)
    checkpoint,record=source(seed,dataset,mode)
    binding=dict(dataset=dataset,mode=mode,seed=seed,source=record,code=fingerprint(),benchmark=c.immutable(),
                 canonical_manifest=canonical_record()[1] if dataset=='clevr' else c.dataset_record(OLD,dataset))
    import fcntl
    lock=(folder/'binding.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    if (folder/'binding.json').exists(): assert json.loads((folder/'binding.json').read_bytes())==binding
    else: c.write(folder/'binding.json',binding)
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    if dataset!='clevr': c.preflight(dataset,folder/'preflight'/f'{split}_{count}')
    images=read_rgb(archive,dataset,split,ids)
    model=None
    for index,(image_id,image) in enumerate(zip(ids,images)):
        offset=int(image_id)-1320 if split=='fixed320' else int(image_id)-7000 if split=='test' else int(image_id)-1000 if dataset=='clevr' and split=='validation' else int(image_id)
        inference_seed=193000+offset if split in ('fixed320','test') else 251000+offset
        path=folder/f'{int(image_id):06d}.npz'; meta=path.with_suffix('.json')
        expected=dict(image_id=int(image_id),RGB_sha256=hashlib.sha256(image.tobytes()).hexdigest(),inference_seed=inference_seed,
                      source_sha256=record['sha256'],code=fingerprint(),GT_used_for_extraction=False)
        if meta.exists():
            saved=json.loads(meta.read_bytes());assert all(saved[k]==v for k,v in expected.items())
            assert c.sha(path)==saved['row_sha256']
            continue
        if model is None:
            model=load_model(checkpoint,device=device);model.requires_grad_(False)
        torch.manual_seed(inference_seed)
        if torch.device(device).type=='cuda':torch.cuda.manual_seed(inference_seed)
        with torch.no_grad():
            output=model(torch.from_numpy(image[None]).to(device))
            features=activity_features(output['trace'])[0].cpu().numpy().astype(np.float16)
            baseline=prediction_from_trace(model,output['trace'],output['adjacency'],output['q'])
            groups=spike_synchrony_components(output['trace']['spikes'].detach().cpu(),synchrony_threshold=.5,
                    min_group_size=32,settle=512,components=output['trace']['component_spikes'].detach().cpu(),
                    background='activity',foreground_threshold=0.,affinity_mode='spike',spatial_grid_size=64)
            support=spatial_components_to_patch_labels(groups,64,device='cpu').numpy()[0].astype(np.int64)
            q=output['q'][0].cpu().numpy().astype(np.float64)
            adjacency=output['adjacency'][0].cpu().numpy().astype(np.float64)
            gain=float(model.wrapped.core.graph_generator.log_coupling_gain.exp().cpu())
            kept=foreground_induced_merge(q,adjacency,gain,support,grid_shape=(64,64))['labels'].astype(np.int64)
            all_support=foreground_induced_merge(q,adjacency,gain,np.ones((64,64),dtype=np.int64),grid_shape=(64,64))['labels'].astype(np.int64)
        temp=path.with_name(path.name+'.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream,features=features,baseline=baseline,keep_groups=kept,all_support=all_support)
        os.replace(temp,path);c.write(meta,dict(expected,row_sha256=c.sha(path)))
        c.write(folder/f'progress_{split}.json',dict(stage='feature_extraction',completed=index+1,total=count,last_id=int(image_id),updated_unix=time.time()))
        del output,q,adjacency,features,baseline,kept,all_support
    if model is not None:
        assert model_sha256(model)==record.get('model_sha256',model.config['model_sha256'])
        del model;torch.cuda.empty_cache()
    c.write(folder/f'complete_{split}_{count}.json',dict(binding=binding,ids=ids.tolist(),GT_used_for_extraction=False))


def read_row(folder,image_id,device):
    path=Path(folder)/f'{int(image_id):06d}.npz';meta=json.loads(path.with_suffix('.json').read_bytes())
    assert c.sha(path)==meta['row_sha256']
    with np.load(path,allow_pickle=False) as value:
        return dict(features=torch.tensor(value['features'].astype(np.float32),device=device),
                    baseline=value['baseline'].copy(),keep_groups=value['keep_groups'].copy(),all_support=value['all_support'].copy())


def values(predictions,targets):
    metric=c.benchmark_module('evaluation')
    result=metric.evaluate_patch_masks(torch.from_numpy(np.stack(predictions)),torch.from_numpy(np.asarray(targets,dtype=np.int64)))
    return {k:float(result['mean'][k]) for k in KEYS}


def evaluate_rows(folder,ids,labels,head,device,arm='residual_mask'):
    predictions=[]
    for image_id in ids:
        row=read_row(folder,image_id,device)
        if head is None:pred=row[arm]
        else:pred=head.predict(row['features'],row['baseline'],row['all_support'])
        predictions.append(pred)
    return values(predictions,labels),predictions


def train_head(archive,dataset,mode,seed,count,validation_count,arm,device,initial=None):
    if arm not in ('learned_background','residual_mask'):raise ValueError('unknown learned arm')
    ids=ids_for(dataset,'train',count);val_ids=ids_for(dataset,'validation',validation_count)
    folder=row_folder(archive,dataset,mode,seed)
    output=Path(archive)/'heads'/dataset/mode/f'seed{seed}'/f'{arm}_TRAIN{count}'
    output.mkdir(parents=True,exist_ok=True)
    binding=dict(dataset=dataset,mode=mode,seed=seed,count=count,validation_count=validation_count,arm=arm,
                 ids=ids.tolist(),validation_ids=val_ids.tolist(),code=fingerprint(),
                 source=json.loads((folder/'binding.json').read_bytes()),initial_sha256=c.sha(initial) if initial else None,
                 core_encoder_frozen_during_final_mask_loss=True,epochs=20,learning_rate=.003)
    if (output/'report.json').exists():
        report=json.loads((output/'report.json').read_bytes());assert report['binding']==binding
        assert c.sha(output/'best.pt')==report['checkpoint_sha256']
        return report
    if (output/'binding.json').exists():assert json.loads((output/'binding.json').read_bytes())==binding
    else:c.write(output/'binding.json',binding)
    head=ActivityMaskHead(background_only=arm=='learned_background',seed=251000+seed).to(device)
    if initial:
        state=torch.load(initial,map_location=device,weights_only=True);assert state['arm']==arm
        head.load_state_dict(state['head'],strict=True)
    if arm=='learned_background':
        head.embedding.requires_grad_(False);head.residual_scale.requires_grad_(False)
    optimizer=torch.optim.Adam([p for p in head.parameters() if p.requires_grad],lr=.003)
    labels=truth(dataset,'train',ids);val_labels=truth(dataset,'validation',val_ids)
    history=[];best=-float('inf');first=0
    if (output/'resume.pt').exists():
        state=torch.load(output/'resume.pt',map_location=device,weights_only=True);assert state['binding']==binding
        head.load_state_dict(state['head'],strict=True);optimizer.load_state_dict(state['optimizer'])
        history=state['history'];first=state['epoch'];best=state['best']
    for epoch in range(first,20):
        losses=[];head.train()
        for index in np.random.RandomState(251000+seed+10000*epoch).permutation(len(ids)):
            row=read_row(folder,ids[index],device);optimizer.zero_grad(set_to_none=True)
            logits=head(row['features'],row['baseline'],row['all_support'])
            if arm=='learned_background':
                y=torch.tensor(labels[index].reshape(-1)==0,device=device,dtype=torch.float32)
                p=logits.softmax(0)[0].clamp(1e-6,1-1e-6)
                parts=[torch.nn.functional.binary_cross_entropy(p[y==v],y[y==v]) for v in (0,1) if bool((y==v).any())]
                loss=torch.stack(parts).mean()
            else:
                loss,_=mask_objective(logits,labels[index],row['baseline'])
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            if not bool(torch.isfinite(loss) and torch.isfinite(norm)) or float(norm)<=0:
                raise FloatingPointError('nonfinite or disconnected readout gradient')
            optimizer.step();losses.append(float(loss.detach()))
        head.eval();score,_=evaluate_rows(folder,val_ids,val_labels,head,device)
        history.append(dict(epoch=epoch+1,mean_loss=float(np.mean(losses)),validation=score,updates=(epoch+1)*count))
        if score['fg_ari']>best:
            best=score['fg_ari'];save(output/'best.pt',dict(head=head.state_dict(),arm=arm,seed=seed,
                                                        binding=binding,best_epoch=epoch+1,validation=score))
        save(output/'resume.pt',dict(head=head.state_dict(),optimizer=optimizer.state_dict(),binding=binding,
                                    history=history,epoch=epoch+1,best=best))
        c.write(output/'progress.json',dict(epoch=epoch+1,total_epochs=20,validation=score))
    report=dict(status='trained_final_readout',binding=binding,history=history,best_validation_FGARI=best,
                checkpoint_sha256=c.sha(output/'best.pt'),trained_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),
                TEST_used_for_training_or_selection=False,validation_used_for_checkpoint_selection=True,
                final_mask_loss_backpropagates_to='readout only; encoder/core frozen',checkpoint=str(output/'best.pt'))
    c.write(output/'report.json',report)
    return report


def load_head(path,device):
    state=torch.load(path,map_location=device,weights_only=True)
    head=ActivityMaskHead(background_only=state['arm']=='learned_background').to(device)
    head.load_state_dict(state['head'],strict=True);head.eval()
    return head


def worker(args):
    torch.set_num_threads(1);device='cuda:0' if torch.cuda.is_available() else 'cpu'
    if device!='cuda:0':raise RuntimeError('server worker requires CUDA; CPU tests are separate')
    torch.cuda.set_per_process_memory_fraction(.8,0)
    archive=Path(args.archive);ds=args.dataset;mode=args.mode;s=args.seed
    if args.task=='cache':extraction(archive,ds,mode,s,args.split,args.count,device)
    elif args.task=='train':train_head(archive,ds,mode,s,args.count,args.validation_count,args.arm,device,args.initial)
    else:raise ValueError('unknown task')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',required=True)
    parser.add_argument('--dataset',choices=('clevr','multi_dsprites','tetrominoes'),default='clevr')
    parser.add_argument('--mode',choices=('zero_shot','finetune'),default='zero_shot')
    parser.add_argument('--seed',type=int,choices=(0,1,2),required=True)
    parser.add_argument('--task',choices=('cache','train'),required=True)
    parser.add_argument('--split',choices=('train','validation','fixed320','test'),default='validation')
    parser.add_argument('--count',type=int,default=16);parser.add_argument('--validation-count',type=int,default=16)
    parser.add_argument('--arm',choices=('learned_background','residual_mask'),default='residual_mask')
    parser.add_argument('--initial');worker(parser.parse_args())
