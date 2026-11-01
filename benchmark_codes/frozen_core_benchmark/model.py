"""Strict pretrained16-to64 mapping; core/integration frozen, encoder gradient live."""
import json,hashlib
from pathlib import Path
import torch
from full_benchmark.model import FreshModel


def digest(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def map_source(source,template):
    if source.keys()!=template.keys():raise ValueError('pretrained core/template state keys differ')
    grid=64;axis=torch.arange(grid)//4;parent=(axis[:,None]*16+axis[None,:]).flatten()
    nodes={'kuramoto.omega':2,'kuramoto.kappa':2,'dendric_layer.tau_n':2,'membrane_layer.tau_m':1}
    result={}
    for name,target in template.items():
        value=source[name]
        if name=='sc':value=torch.eye(grid*grid,dtype=target.dtype)
        elif name=='graph_generator.grid_distance':value=target.clone()
        elif name in nodes:
            assert value.ndim==nodes[name] and value.shape[0]==256,name
            value=value.index_select(0,parent)
        elif name=='kuramoto.direction_learner':
            assert value.shape==(256,256)
            value=value.index_select(0,parent).index_select(1,parent)
        if value.shape!=target.shape:raise ValueError('unregistered shape change: '+name)
        result[name]=value.to(dtype=target.dtype,device=target.device).clone()
    return result


def checked_file(record):
    path=Path(record['path'])
    if path.stat().st_size!=record['bytes']:raise ValueError('source size changed: '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    if h.hexdigest()!=record['sha256']:raise ValueError('source SHA changed: '+str(path))
    return path


class FrozenCoreModel(FreshModel):
    def __init__(self,variant,seed,*,sources):
        if variant not in (252,253):raise ValueError('SW0256 supports252/253 only')
        super().__init__(variant,seed)
        record=json.loads(Path(sources).read_bytes())['seeds'][str(seed)]
        source=torch.load(checked_file(record['core']),map_location='cpu',weights_only=True)
        converted=map_source(source,self.wrapped.core.state_dict())
        self.wrapped.core.load_state_dict(converted,strict=True)
        # Strictly mapped trained source bias replaces FreshModel's random-run
        # initialization policy. Do not reset any pretrained core parameter.
        assert torch.equal(self.wrapped.core.dendric_layer.oscillator_dense.bias,
                           source['dendric_layer.oscillator_dense.bias'])
        del source,converted
        if variant==252:
            saved=torch.load(checked_file(record['supervised_encoder']),map_location='cpu',weights_only=True)
            self.encoder.load_state_dict(saved['encoder_state'],strict=True)
            mean,std=saved['feature_mean'],saved['feature_std']
        else:
            saved=torch.load(checked_file(record['unsupervised_encoder']),map_location='cpu',weights_only=True)
            self.encoder.load_state_dict(saved,strict=True)
            stats=torch.load(checked_file(record['statistics']),map_location='cpu',weights_only=True)
            assert stats['mode']=='standardize' and stats.get('clip',3.)==3.
            mean,std=stats['mean'],stats['std']
        assert torch.isfinite(mean).all() and torch.isfinite(std).all() and (std>0).all()
        self.feature_mean.copy_(mean);self.feature_std.copy_(std)
        self.wrapped.requires_grad_(False);self.wrapped.eval()
        self.encoder.requires_grad_(True)
        self.normalization_provenance='preserved_pretrained_encoder_statistics; no refit or TEST access'
        self.source_provenance=record
        self.frozen_initial_sha256=digest(self.wrapped)
        self.assert_frozen()

    def assert_frozen(self):
        assert all(not p.requires_grad and p.grad is None for p in self.wrapped.parameters())
        assert digest(self.wrapped)==self.frozen_initial_sha256,'fixed core/integration state changed'

    def parameter_groups(self):
        groups=[dict(name='encoder',params=list(self.encoder.parameters()),lr=3e-6)]
        if self.head is not None:groups.append(dict(name='mask_head',params=list(self.head.parameters()),lr=.003))
        assert {id(p) for g in groups for p in g['params']}=={id(p) for p in self.parameters() if p.requires_grad}
        return groups
