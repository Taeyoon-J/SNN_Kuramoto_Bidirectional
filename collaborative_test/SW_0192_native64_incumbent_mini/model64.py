"""Strict learned native32 -> native64 transfer; no common module mutations."""
import types
import numpy as np
import torch

GRID, NODES, OLD_GRID = 64, 4096, 32
NODE_RANKS = {'kuramoto.omega':2,'kuramoto.kappa':2,
              'dendric_layer.tau_n':2,'membrane_layer.tau_m':1}
PAIR_KEYS = {'kuramoto.direction_learner'}


def parent_index(old_grid=OLD_GRID,grid=GRID):
    if type(old_grid) is not int or type(grid) is not int or old_grid<1 or grid%old_grid:
        raise ValueError('integer nested grids required')
    if grid<old_grid:raise ValueError('target grid must not shrink')
    cells=torch.arange(grid)//(grid//old_grid)
    return (cells[:,None]*old_grid+cells[None,:]).reshape(-1)


def map_state(source,template,*,old_grid=OLD_GRID,grid=GRID):
    from collaborative_test.SW_0135_native32_spike_binding.resolution import grid_distance
    if set(source)!=set(template):raise ValueError('core state keys differ')
    index=parent_index(old_grid,grid);result={};n=grid*grid
    for key,target in template.items():
        value=source[key]
        if key=='sc':
            if tuple(value.shape)!=(old_grid**2,old_grid**2):raise ValueError('source SC geometry')
            mapped=torch.eye(n,device=target.device,dtype=target.dtype)
        elif key=='graph_generator.grid_distance':
            if tuple(value.shape)!=(old_grid**2,old_grid**2):raise ValueError('source distance geometry')
            mapped=grid_distance(grid,16./grid,device=target.device,dtype=target.dtype)
        elif key in NODE_RANKS:
            if value.ndim!=NODE_RANKS[key] or value.shape[0]!=old_grid**2:raise ValueError('node tensor geometry')
            mapped=value.index_select(0,index.to(value.device))
        elif key in PAIR_KEYS:
            if tuple(value.shape)!=(old_grid**2,old_grid**2):raise ValueError('pair tensor geometry')
            p=index.to(value.device);mapped=value.index_select(0,p).index_select(1,p)
        else:
            if value.shape!=target.shape:raise ValueError('unregistered state shape change: '+key)
            mapped=value
        mapped=mapped.to(device=target.device,dtype=target.dtype).clone()
        if mapped.shape!=target.shape or not torch.isfinite(mapped).all():raise ValueError('mapped state invalid: '+key)
        result[key]=mapped
    return result


def encode64(encoder,patcher,mean,std,clip,rgb):
    if rgb.dtype!=torch.uint8 or rgb.ndim!=4 or tuple(rgb.shape[1:])!=(128,128,3):raise ValueError('native128 uint8 RGB required')
    if not np.isfinite(float(clip)) or clip<=0:raise ValueError('invalid feature clip')
    device=next(encoder.parameters()).device
    feature=encoder(rgb.to(device).permute(0,3,1,2).float()/255.)
    if feature.ndim!=4 or feature.shape[:2]!=(len(rgb),8):raise ValueError('eight feature channels required')
    mean=torch.as_tensor(mean,device=device,dtype=feature.dtype)
    std=torch.as_tensor(std,device=device,dtype=feature.dtype)
    if mean.shape!=std.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<=0).any():raise ValueError('invalid source statistics')
    gamma=patcher(((feature-mean)/std).clamp(-clip,clip))
    if gamma.shape!=(len(rgb),8,NODES) or not torch.isfinite(gamma).all():raise ValueError('native64 gamma geometry/finiteness')
    return gamma


class Native64Foundation:
    def __init__(self,source):self.__dict__.update(vars(source))
    def encode(self,rgb):
        return encode64(self.encoder,self.patcher,self.feature_mean,self.feature_std,self.feature_clip,rgb)


def move_to_device(found,device):
    device=torch.device(device)
    found.wrapped.to(device);found.encoder.to(device);found.patcher.to(device)
    # These recurrent layers allocate transient state using a stored device.
    for module in found.wrapped.modules():
        if hasattr(module,'device'):module.device=device
    return found


def build_from_incumbent(seed,inputs,device='cpu'):
    from collaborative_test.SW_0191_partition_error_feedback import source as warm
    from collaborative_test.SW_0175_critical_component_binding import train_v2 as util
    from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_operator as geo
    from collaborative_test.SW_0094_aligned_joint_pilot.run import hparams
    from collaborative_test.SW_0154_native64_resolution_compare.native64_model import configure_grid_hparams
    from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
    from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
    found,_discarded,_families,binding,_setup=warm._make_fresh('product_q_ce',seed,inputs,'cpu')
    # Keep the entire trained wrapper/encoder; replace only dimensioned core state.
    old_core=found.wrapped.core
    hp=configure_grid_hparams(hparams('raw'),grid_size=GRID,graph_top_k=512)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(192000+seed)
        core=S2NetCore(hp,device='cpu')
    core.load_state_dict(map_state(old_core.state_dict(),core.state_dict()),strict=True)
    def distance(self,z,euclidean):
        fixed=torch.as_tensor(geo.inverse_softplus(8.),device=z.device,dtype=z.dtype)
        return geo.geodesic_distance_power2(z,euclidean,fixed,steps=self.geodesic_steps,
            radius=self.geodesic_radius,temperature=self.geodesic_temperature,
            cap=self.geodesic_cap,chunk=1,log4=True)
    core.graph_generator._geodesic_distance=types.MethodType(distance,core.graph_generator)
    core.graph_generator._sw0143_geodesic_arm='T'
    core.graph_generator._sw0143_effective_contrast=8.
    found.wrapped.core=core
    found.patcher=FeaturePatchGammaInitializer(grid_size=GRID)
    found=Native64Foundation(found)
    del _discarded,_families,old_core
    move_to_device(found,device)
    found.encoder.eval();found.wrapped.eval()
    optimizer,families=util._make_optimizer(found)
    families=warm.enable_source_encoder(found,optimizer,families)
    if optimizer.state or set(families)!=warm.FAMILIES:raise ValueError('fresh five-family optimizer required')
    if warm._base_encoder_sha(found)!=binding['base_encoder_sha256']:raise ValueError('learned source CNN not preserved')
    proof=dict(seed=seed,incumbent_lineage=binding['incumbent_lineage'],
        native32_model_state_sha256=binding['initial_model_state_sha256'],
        native64_initial_model_state_sha256=util._state_sha(found.wrapped,found.encoder),
        source_CNN_sha256=warm._base_encoder_sha(found),
        optimizer_groups=warm.parent._optimizer_contract(optimizer),
        grid=[GRID,GRID],graph_top_k=512,geodesic_chunk=1,
        validation_data_read=False,mask_supervised=True,adam_state_entries=0)
    return found,optimizer,families,proof
