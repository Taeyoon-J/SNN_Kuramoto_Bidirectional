#!/usr/bin/env python3
"""SW0257 separated encoder35000x10; supervised detached head3500x10."""
import argparse,json,os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from portable_runtime import HERE
from frozen_core_benchmark.runtime import setup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(HERE/'configs/frontier.json'))
    p.add_argument('--sources',default=str(HERE/'frozen_core_benchmark/sources.frontier.json'))
    p.add_argument('--heads',default=str(HERE/'separate_encoder_benchmark/heads.frontier.json'))
    p.add_argument('--output',required=True)
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('preflight')
    t=sub.add_parser('train');t.add_argument('--model',type=int,choices=(252,253),required=True)
    t.add_argument('--dataset',choices=('clevr','tetrominoes','clevrtex_original'),required=True)
    t.add_argument('--seed',type=int,choices=(0,1,2),required=True);t.add_argument('--admission',action='store_true')
    q=sub.add_parser('queue');q.add_argument('--gpus',default='0,1,2,3')
    a=p.parse_args();config,root=setup(a.config,a.output)
    if a.action=='preflight':
        from frozen_core_benchmark.runtime import preflight
        from separate_encoder_benchmark.engine import sources
        result=preflight(config,root,a.sources)
        from mod_benchmark import common as c
        from frozen_core_benchmark.model import checked_file
        inventory=json.loads(open(a.heads).read())
        for rows in inventory['datasets'].values():
            for row in rows.values():checked_file(row)
        result.update(experiment='SW0257',encoder_core_backprop=False,encoder_physical_batch=16,
            head_budget='3500x10',heads_sha256=c.sha(a.heads),code=sources())
        c.write(root/'separate_preflight.json',result)
    elif a.action=='train':
        from separate_encoder_benchmark.engine import run
        result=run(a.dataset,a.model,a.seed,root,a.sources,a.heads,admission=a.admission)
    else:
        from separate_encoder_benchmark.queue import queue
        return queue(root,a.config,a.sources,a.heads,[int(x) for x in a.gpus.split(',')])
    print(json.dumps(result))


if __name__=='__main__':main()
