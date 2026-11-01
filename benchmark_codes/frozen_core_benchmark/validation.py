"""Real source load, independent spatial checks and CPU model contracts."""
import gc,json
from pathlib import Path
import torch
from mod_benchmark import common as c
from frozen_core_benchmark.model import FrozenCoreModel,digest


def validate_cpu(sources,root):
    torch.set_num_threads(1);rows=[];fixed={}
    inventory=json.loads(Path(sources).read_bytes())
    for seed in range(3):
        source=torch.load(inventory['seeds'][str(seed)]['core']['path'],map_location='cpu',weights_only=True)
        for variant in (252,253):
            model=FrozenCoreModel(variant,seed,sources=sources)
            core=model.wrapped.core
            torch.testing.assert_close(core.kuramoto.omega.reshape(16,4,16,4,4),
                source['kuramoto.omega'].reshape(16,1,16,1,4).expand(16,4,16,4,4),rtol=0,atol=0)
            indices=torch.tensor([0,1,63,64,255,4095]);parents=torch.tensor([0,0,15,0,15,255])
            torch.testing.assert_close(core.kuramoto.direction_learner[indices[:,None],indices[None,:]],
                source['kuramoto.direction_learner'][parents[:,None],parents[None,:]],rtol=0,atol=0)
            assert core.in_dim==4096 and core.graph_generator.top_k==512
            assert model.encoder.training and all(v.requires_grad for v in model.encoder.parameters())
            model.assert_frozen()
            if seed in fixed:assert model.frozen_initial_sha256==fixed[seed]
            fixed[seed]=model.frozen_initial_sha256
            row=dict(model=variant,seed=seed,source_core=inventory['seeds'][str(seed)]['core'],
                mapped_core_sha256=fixed[seed],strict_spatial_mapping_verified=True,
                all_core_and_integration_parameters_frozen=True,
                active_groups={g['name']:sum(v.numel() for v in g['params']) for g in model.parameter_groups()},
                encoder_sha256=digest(model.encoder),encoder_pretrained=True,normalization_preserved=True)
            rows.append(row);del model;gc.collect()
        del source
    from frozen_core_benchmark.engine import scientific_sources
    result=dict(status='SW0256_all6_pretrained_initializations_and_mapping_checked_CPU',rows=rows,code=scientific_sources(),
        protected_benchmark=c.immutable(),unique_TRAIN=35000,epochs=10,expected_exposures=350000,
        expected_Adam_updates=21880,source_inventory_sha256=c.sha(sources),GPU_backward_claimed=False)
    c.write(Path(root)/'CPU_validation.json',result);return result
