"""Run from an isolated Linux copy: imports, all12 initial states, splits, metrics."""
import argparse,gc,hashlib,json,sys
from pathlib import Path
from portable_runtime import HERE,configure,connect_inputs,check_splits

def initial_states():
    import torch
    from full_benchmark.model import FreshModel
    from full_benchmark.engine import state_digest
    torch.set_num_threads(1);result={}
    for variant in (252,253,254,240):
        for seed in (0,1,2):
            model=FreshModel(variant,seed)
            groups=model.parameter_groups()
            result[f'{variant}/seed{seed}']=dict(state_sha256=state_digest(model),grid=model.grid,
                groups={group['name']:sum(p.numel() for p in group['params']) for group in groups},
                trainable_tensors=sum(p.requires_grad for p in model.parameters()))
            del model;gc.collect()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(HERE/'configs/frontier.json'))
    p.add_argument('--output',required=True);p.add_argument('--reference',required=True)
    args=p.parse_args();configure(args.config);root=connect_inputs(json.loads(Path(args.config).read_text()),args.output)
    from verify_bundle import verify
    integrity=verify()
    import torch
    from mod_benchmark import common as c
    from full_benchmark.data import Dataset
    from full_benchmark.evaluation import metrics
    import full_benchmark,snn_kuramoto_bidirectional,neutral_readout,readout_benchmark,mod_benchmark
    imported={module.__name__:str(Path(module.__file__).resolve()) for module in
              (full_benchmark,snn_kuramoto_bidirectional,neutral_readout,readout_benchmark,mod_benchmark)}
    assert all(Path(path).is_relative_to(HERE/'vendor') for path in imported.values())
    reference=json.loads(Path(args.reference).read_text());states=initial_states()
    assert states==reference['initial_states'],'portable constructors differ from actual live server constructors'
    splits=check_splits(json.loads(Path(args.config).read_text()),root,('clevr','multi_dsprites','tetrominoes','objects_room','clevrtex_original'))
    # Scoring identity: use exact TRAIN-independent fixed evaluation labels.
    import numpy as np
    oracles={}
    for name in splits:
        dataset=Dataset(name,root);ids=dataset.test[:2]
        labels=dataset.labels(ids,'fixed320' if name=='clevr' else 'test')
        actual,_=metrics(labels,dataset,ids)
        assert actual==reference['metric_oracles'][name],name
        oracles[name]=actual
    from scheduler import conditions
    rows=conditions();assert len(rows)==20 and sum(x['phase']==1 for x in rows)==12
    result=dict(status='isolated_bundle_all12_initial_states_all5_splits_protected_preflights_and_metric_oracles_passed',
                integrity=integrity,imports=imported,initial_states=states,splits=splits,metric_oracles=oracles,
                no_other_project_source_imported=True,CUDA_training_claimed=False)
    c.write(root/'validation.json',result);print(json.dumps(result))
if __name__=='__main__':main()
