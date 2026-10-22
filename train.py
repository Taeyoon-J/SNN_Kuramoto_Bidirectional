"""Additional SW0240 training using its original frozen teacher and supervised objective."""
import argparse
import copy
import json
from pathlib import Path
import os

import numpy as np
import torch
import h5py
from snn_kuramoto_bidirectional import load_model
from snn_kuramoto_bidirectional.sw240.model import model_sha256
from snn_kuramoto_bidirectional.sw240.data import read_batch, train_ids


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True,help='canonical CLEVR HDF5 with image and mask datasets')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--train-count',type=int,choices=(35000,70000),default=35000)
    args=parser.parse_args()
    if args.epochs<1 or args.output.exists():
        raise ValueError('positive epochs and a new output directory are required')
    original=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    seed=int(original['config']['seed'])
    if seed not in (0,1,2):
        raise ValueError('a seed0/1/2 export is required')
    model=load_model(args.checkpoint,device=args.device)
    teacher=load_model(args.checkpoint,teacher=True,device='cpu')
    teacher.requires_grad_(False)
    optimizer=model.optimizer_for_training()
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    ids=train_ids(args.train_count)
    frozen={k:v.detach().cpu().clone() for k,v in model.wrapped.state_dict().items() if k!='core.gamma_channel_proj.weight'}
    args.output.mkdir(parents=True)
    history=[]
    with h5py.File(args.dataset,'r') as stream:
        for epoch in range(args.epochs):
            order=ids[np.random.RandomState(249000+seed+10000*epoch).permutation(len(ids))]
            for start in range(0,len(order),16):
                selected=order[start:start+16]
                rgb,labels=read_batch(stream,selected)
                optimizer.zero_grad(set_to_none=True)
                loss,parts=model.supervised_loss(rgb,labels,teacher)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
                if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):
                    raise FloatingPointError('nonfinite loss/gradient; previous completed epoch is retained')
                optimizer.step()
                history.append(dict(epoch=epoch+1,ids=selected.tolist(),loss=float(loss.detach()),gradient_norm=float(norm)))
            for name,value in frozen.items():
                if not torch.equal(value,model.wrapped.state_dict()[name].detach().cpu()):
                    raise RuntimeError('frozen core changed: '+name)
            bundle=copy.copy(original)
            bundle['config']=dict(original['config'],endpoint='patch_final_additional_training',model_sha256=model_sha256(model))
            bundle['wrapped_state']={k:v.detach().cpu().clone() for k,v in model.wrapped.state_dict().items()}
            bundle['encoder_state']={k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()}
            bundle['training']=dict(completed_epochs=epoch+1,unique_images=len(ids),batch=16,history=history,
                                    optimizer_state=optimizer.state_dict(),initial_model_sha256=original['config']['model_sha256'])
            target=args.output/f'epoch_{epoch+1:02d}.pt'
            temp=target.with_suffix('.tmp')
            torch.save(bundle,temp);os.replace(temp,target)
            print(json.dumps(dict(epoch=epoch+1,updates=len(history),checkpoint=str(target))),flush=True)


if __name__=='__main__':
    main()
