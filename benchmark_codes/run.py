#!/usr/bin/env python3
"""Standalone entrypoint. Imports only this directory plus installed Python packages."""
import argparse,json,os,sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from portable_runtime import HERE,configure,connect_inputs,check_splits
DATASETS=('clevr','multi_dsprites','tetrominoes','objects_room','clevrtex_original')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(HERE/'configs/frontier.json'))
    parser.add_argument('--output',required=True,help='NEW output directory, never the live experiment')
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('preflight');p.add_argument('--dataset',choices=(*DATASETS,'all'),default='all')
    p=sub.add_parser('train');p.add_argument('--dataset',choices=DATASETS,required=True)
    p.add_argument('--model',type=int,choices=(252,253,254,240),required=True)
    p.add_argument('--seed',type=int,choices=(0,1,2),required=True);p.add_argument('--admission',action='store_true')
    p=sub.add_parser('evaluate');p.add_argument('--checkpoint',required=True);p.add_argument('--dataset',choices=DATASETS,required=True)
    p.add_argument('--model',type=int,choices=(252,253,254,240),required=True);p.add_argument('--seed',type=int,choices=(0,1,2),required=True)
    p=sub.add_parser('queue');p.add_argument('--gpus',default='0,1,2,3')
    sub.add_parser('audit')
    args=parser.parse_args();config=configure(args.config);output=connect_inputs(config,args.output)
    if args.action=='preflight':
        result=check_splits(config,output,DATASETS if args.dataset=='all' else (args.dataset,))
    elif args.action=='train':
        from full_benchmark.engine import train
        result=train(args.dataset,args.model,args.seed,output,admission=args.admission)
    elif args.action=='evaluate':
        import torch
        from full_benchmark.data import Dataset
        from full_benchmark.model import FreshModel
        from full_benchmark.evaluation import evaluate
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        model=FreshModel(args.model,args.seed).to('cuda:0');model.load_state_dict(checkpoint['model'])
        result=evaluate(model,Dataset(args.dataset,output),output/'evaluation'/f'model{args.model}'/args.dataset/f'seed{args.seed}')
    elif args.action=='queue':
        from scheduler import run_queue
        return run_queue(args.config,output,[int(x) for x in args.gpus.split(',')])
    else:
        from scheduler import audit_all
        result=audit_all(output)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
