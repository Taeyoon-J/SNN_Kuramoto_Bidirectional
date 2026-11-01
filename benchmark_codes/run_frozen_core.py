#!/usr/bin/env python3
"""SW0256: original pretrained fixed core, full35000x10, three seeds."""
import argparse,json,os
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from portable_runtime import HERE
from frozen_core_benchmark.runtime import setup,preflight


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(HERE/'configs/frontier.json'))
    p.add_argument('--sources',default=str(HERE/'frozen_core_benchmark/sources.frontier.json'))
    p.add_argument('--output',required=True)
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('preflight');sub.add_parser('validate-cpu')
    t=sub.add_parser('train');t.add_argument('--model',type=int,choices=(252,253),required=True)
    t.add_argument('--dataset',choices=('clevr','tetrominoes','clevrtex_original'),required=True)
    t.add_argument('--seed',type=int,choices=(0,1,2),required=True);t.add_argument('--admission',action='store_true')
    t=sub.add_parser('queue');t.add_argument('--gpus',default='0,1,2,3')
    t.add_argument('--legacy-root',default='/work/USERS/kevinswk/experiments/SW0255_full3500')
    sub.add_parser('audit')
    a=p.parse_args();config,root=setup(a.config,a.output)
    if a.action=='preflight':result=preflight(config,root,a.sources)
    elif a.action=='validate-cpu':
        from frozen_core_benchmark.validation import validate_cpu
        result=validate_cpu(a.sources,root)
    elif a.action=='train':
        from frozen_core_benchmark.engine import train
        result=train(a.dataset,a.model,a.seed,root,sources=a.sources,admission=a.admission)
    elif a.action=='queue':
        from frozen_core_benchmark.queue import run_queue
        return run_queue(a.config,a.sources,root,[int(x) for x in a.gpus.split(',')],a.legacy_root)
    else:
        from frozen_core_benchmark.queue import audit_all
        result=audit_all(root)
    print(json.dumps(result))


if __name__=='__main__':main()
