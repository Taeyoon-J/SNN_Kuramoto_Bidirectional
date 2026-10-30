"""Fresh constructors and declared active parameter groups for four model variants."""
import math
import types
import numpy as np
import torch
from torch import nn
from neutral_readout.head import NeutralRegionHead
from snn_kuramoto_bidirectional.input_layer_generator import CNNFeatureEncoder
from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.sw240.model import default_config
from snn_kuramoto_bidirectional.sw240.context import ContextualFeatureEncoder
from snn_kuramoto_bidirectional.sw240.integration import PhaseStateIntegration
from snn_kuramoto_bidirectional.sw240.rollout import late_rollout
from snn_kuramoto_bidirectional.sw240.capture import capture_source_coupling
from snn_kuramoto_bidirectional.sw240.correlation import native_component_rho_q
from full_benchmark.geodesic import distance_power2


class FreshModel(nn.Module):
    def __init__(self,variant,seed,*,normalization=None):
        super().__init__()
        if variant not in (240,252,253,254):raise ValueError('unknown model')
        self.variant=int(variant);self.seed=int(seed);self.grid=32 if variant==253 else 64
        # No torch.load, source encoder, learned source state or checkpoint reset.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(255000+seed)
            values=default_config()['hparams'].copy()
            values.update(num_regions=self.grid**2,k=float(self.grid**2),
                          graph_top_k=128 if self.grid==32 else 512,
                          spike_spatial_grid_size=self.grid,gamma_patch_grid_size=self.grid)
            known={k for k in S2NetHyperparameters.__dataclass_fields__}
            hp=S2NetHyperparameters(**{k:v for k,v in values.items() if k in known}).validate()
            core=S2NetCore(hp,device='cpu')
            self.wrapped=PhaseStateIntegration(core,'phase')
            # Standard data-independent zero-bias initialization prevents a
            # shared negative bias from silencing every LIF unit at startup.
            nn.init.zeros_(core.dendric_layer.oscillator_dense.bias)
            base=CNNFeatureEncoder(8,3,3,bias=True)
            self.encoder=base if variant==253 else ContextualFeatureEncoder(base)
            self.patcher=FeaturePatchGammaInitializer(grid_size=self.grid)
        graph=core.graph_generator
        coords=torch.arange(self.grid,dtype=torch.float32)*(16./self.grid)
        dy=coords.repeat_interleave(self.grid);dx=coords.repeat(self.grid)
        graph.grid_distance=torch.sqrt((dy[:,None]-dy[None]).square()+(dx[:,None]-dx[None]).square())
        # Fixed8 squared-geodesic T is an architectural constant in these models.
        # The unused legacy contrast Parameter is explicitly NOT advertised as learned.
        graph.geodesic_contrast.requires_grad_(False)
        def geodesic(module,z,distance):
            raw=torch.as_tensor(math.log(math.expm1(8.)),device=z.device,dtype=z.dtype)
            return distance_power2(z,distance,raw,steps=module.geodesic_steps,
                radius=module.geodesic_radius,temperature=module.geodesic_temperature,cap=module.geodesic_cap)
        graph._geodesic_distance=types.MethodType(geodesic,graph)
        self.encoder.requires_grad_(True);self.wrapped.requires_grad_(True)
        graph.geodesic_contrast.requires_grad_(False)
        self.head=NeutralRegionHead(seed=251000+seed) if variant in (252,254) else None
        self.register_buffer('feature_mean',torch.zeros(1,8,1,1))
        self.register_buffer('feature_std',torch.ones(1,8,1,1))
        self.normalization_provenance='admission_identity_only_not_final_TRAIN_fit'
        if normalization is not None:
            self.feature_mean.copy_(normalization['mean']);self.feature_std.copy_(normalization['std'])
            self.normalization_provenance=normalization['provenance']

    def to(self,*args,**kwargs):
        result=super().to(*args,**kwargs);device=next(self.parameters()).device
        for module in self.modules():
            if hasattr(module,'device'):module.device=device
        return result

    def encode(self,rgb):
        if rgb.dtype!=torch.uint8 or tuple(rgb.shape[1:])!=(128,128,3):raise ValueError('native128 uint8 RGB required')
        features=self.encoder(rgb.to(next(self.parameters()).device).permute(0,3,1,2).float()/255.)
        return self.patcher(((features-self.feature_mean)/self.feature_std).clamp(-3.,3.))

    def forward(self,rgb,*,inference_seed,live_tail_steps=64):
        torch.manual_seed(int(inference_seed))
        if next(self.parameters()).is_cuda:torch.cuda.manual_seed(int(inference_seed))
        gamma=self.encode(rgb)
        trace,adjacency=capture_source_coupling(self.wrapped.core.kuramoto,
            lambda:late_rollout(self.wrapped,gamma,total_steps=1024,live_tail_steps=live_tail_steps))
        rho,q=native_component_rho_q(trace['spikes'],trace['component_spikes'],settle=512)
        return dict(gamma=gamma,trace=trace,adjacency=adjacency,rho=rho,q=q)

    def parameter_groups(self):
        core=self.wrapped.core
        modules=[('encoder',self.encoder),('graph',core.graph_generator),('kuramoto',core.kuramoto),
                 ('dendrite',core.dendric_layer),('membrane',core.membrane_layer),
                 ('sensory_projection',core.gamma_channel_proj)]
        assigned=set();result=[]
        for name,module in modules:
            params=[p for p in module.parameters() if p.requires_grad and id(p) not in assigned]
            assigned.update(map(id,params));result.append(dict(name=name,params=params,lr=3e-5 if name!='encoder' else 3e-6))
        other=[p for p in self.wrapped.parameters() if p.requires_grad and id(p) not in assigned]
        assigned.update(map(id,other));result.append(dict(name='integration',params=other,lr=3e-5))
        if self.head is not None:
            params=list(self.head.parameters());assigned.update(map(id,params))
            result.append(dict(name='mask_head',params=params,lr=.003))
        assert assigned=={id(p) for p in self.parameters() if p.requires_grad}
        assert all(g['params'] for g in result)
        return result
