"""Validate the supplied assets and call unchanged benchmark preprocessing."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from . import common as c

def prepare(archive):
    archive=Path(archive);c.immutable();torch.set_num_threads(1)
    for ds,grid in c.DATASETS.items():
        folder=archive/'inputs'/ds;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'binding.json').exists():
            record=json.loads((folder/'binding.json').read_bytes())
            for name,v in record['source_files'].items():assert c.sha(c.DATA/name)==v['sha256']
            for split,v in record['upscaled'].items():assert c.sha(folder/f'{split}128.npy')==v['sha256']
            continue
        preflight=c.preflight(ds,folder)
        man=json.loads((c.DATA/f'split_manifest_{ds}.json').read_bytes())
        assert man['ranges']=={'train':[0,6000],'validation':[6000,7000],'test':[7000,8000]}
        assert man['mask_order']==('HWE' if ds=='multi_dsprites' else 'EHW')
        targets=torch.load(c.DATA/f'targets_{ds}_grid{grid}.pt',map_location='cpu',weights_only=True,mmap=True)
        pixels=torch.load(c.DATA/f'pixel_labels_{ds}.pt',map_location='cpu',weights_only=True,mmap=True)
        assert targets['names']==[name for split in ('train','validation','test') for name in man['splits'][split]]
        assert pixels.shape==(8000,64,64) if ds=='multi_dsprites' else pixels.shape==(8000,35,35)
        sample=pixels[:200]
        coherence=float(((sample[:,:,1:]==sample[:,:,:-1]).float().mean()+(sample[:,1:,:]==sample[:,:-1,:]).float().mean())/2)
        assert coherence>.85
        source_names=[f'split_manifest_{ds}.json',f'targets_{ds}_grid{grid}.pt',f'images_{ds}.pt',f'pixel_labels_{ds}.pt']
        source_files={name:dict(bytes=(c.DATA/name).stat().st_size,sha256=c.sha(c.DATA/name)) for name in source_names}
        upscaled={}
        for split,n in [('train',6000),('test',1000)]:
            target=folder/f'{split}128.npy';partial=folder/f'{split}128.partial.npy'
            with (folder/f'make128_{split}.log').open('w') as log:
                result=subprocess.run([sys.executable,'-B',str(c.CODE/'benchmark/make128.py'),'--dataset',ds,
                    '--src',str(c.DATA),'--split',split,'--out',str(partial)],stdout=log,stderr=subprocess.STDOUT,
                    timeout=600,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''))
            if result.returncode:raise RuntimeError('protected make128 failed; see '+str(folder/f'make128_{split}.log'))
            x=np.load(partial,mmap_mode='r',allow_pickle=False)
            assert x.shape==(n,128,128,3) and x.dtype==np.uint8
            del x;os.replace(partial,target)
            upscaled[split]=dict(shape=[n,128,128,3],sha256=c.sha(target),interpolation='nearest',row_order=man['splits'][split])
        c.write(folder/'binding.json',dict(dataset=ds,grid=grid,manifest_ranges=man['ranges'],
            source_files=source_files,upscaled=upscaled,mask_order=man['mask_order'],coherence=coherence,
            preflight=preflight,benchmark_files=c.immutable()))
    from snn_kuramoto_bidirectional import load_model
    from snn_kuramoto_bidirectional.sw240.model import model_sha256
    proofs=[]
    for seed in range(3):
        source=c.source_record(seed);path=c.EXPORTS/source['file'];assert c.sha(path)==source['sha256']
        model=load_model(path);assert model_sha256(model)==source['model_sha256']
        if seed==0:
            teacher=load_model(path,teacher=True);teacher.requires_grad_(False)
            optimizer=model.optimizer_for_training();params=[p for g in optimizer.param_groups for p in g['params']]
            for ds in c.DATASETS:
                x=np.asarray(c.rgb(archive,ds,'train')[:16]).copy()
                labels=c.training_labels(ds,np.arange(16))
                loss,_=model.supervised_loss(x,labels,teacher)
                gradients=torch.autograd.grad(loss,params)
                assert all(torch.isfinite(v).all() for v in gradients) and gradients[-1].norm()>0
                proofs.append(dict(dataset=ds,seed=seed,loss=float(loss.detach()),projection_gradient_norm=float(gradients[-1].norm()),optimizer_updates=0))
            del teacher,optimizer
        del model
    c.write(archive/'preparation_complete.json',dict(status='inputs_and_actual_CPU_gradients_passed',proofs=proofs,benchmark_files=c.immutable()))
    print('Two datasets and all three checkpoints admitted; actual two-dataset CPU gradients passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);prepare(p.parse_args().archive)
