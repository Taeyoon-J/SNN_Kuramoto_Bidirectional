#!/usr/bin/env python3
"""SW0258 fixed core; eligible component reuse or scratch3500x10."""
import argparse,json,os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
from portable_runtime import HERE
from frozen_core_benchmark.runtime import setup
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--config',default=str(HERE/'configs/frontier.json'))
    p.add_argument('--sources',default=str(HERE/'frozen_core_benchmark/sources.frontier.json'))
    p.add_argument('--selection',required=True)
    sub=p.add_subparsers(dest='action',required=True);sub.add_parser('preflight');sub.add_parser('queue')
    t=sub.add_parser('train');t.add_argument('--model',type=int,choices=(252,253),required=True)
    t.add_argument('--dataset',choices=('clevr','tetrominoes','clevrtex_original'),required=True)
    t.add_argument('--seed',type=int,choices=(0,1,2),required=True);t.add_argument('--admission',action='store_true')
    a=p.parse_args();config,root=setup(a.config,a.output)
    if a.action=='preflight':
        from frozen_core_benchmark.runtime import preflight
        from frozen_core_benchmark.model import checked_file
        from selective3500_benchmark.engine import sources
        from mod_benchmark import common as c
        result=preflight(config,root,a.sources);selection=json.loads(open(a.selection).read())
        assert len(selection['cases'])==18
        for case in selection['cases'].values():
            if case['encoder'] is not None:checked_file(case['encoder'])
            if case['head'] is not None:checked_file(case['head'])
        result.update(experiment='SW0258',selection_sha256=c.sha(a.selection),code=sources())
        c.write(root/'selective_preflight.json',result)
    elif a.action=='train':
        from selective3500_benchmark.engine import run
        result=run(a.dataset,a.model,a.seed,root,a.sources,a.selection,admission=a.admission)
    else:
        from selective3500_benchmark.queue import queue
        return queue(root,a.config,a.sources,a.selection)
    print(json.dumps(result))
if __name__=='__main__':main()
