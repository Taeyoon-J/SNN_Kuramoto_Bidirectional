"""Actual 4096-node dynamics, initialized from the published native16 source.

This is a resolution conversion, not a newly trained native64 checkpoint.
The SW0259/SW0260 source snapshots remain immutable.
"""
import torch
from torch.nn import functional as F

GRID = 64
NODES = GRID * GRID
TOP_K = 512
NODE_RANKS = {'kuramoto.omega': 2, 'kuramoto.kappa': 2,
              'dendric_layer.tau_n': 2, 'membrane_layer.tau_m': 1}


def parent_index(device=None):
    axis = torch.arange(GRID, device=device) // 4
    return (axis[:, None] * 16 + axis[None, :]).flatten()


def map_state(source, template):
    if set(source) != set(template):
        raise ValueError('source and native64 state keys differ')
    result = {}
    for key, target in template.items():
        value = source[key]
        p = parent_index(value.device)
        if key == 'sc':
            mapped = torch.eye(NODES, device=target.device, dtype=target.dtype)
        elif key == 'graph_generator.grid_distance':
            axis = torch.arange(GRID, device=target.device, dtype=target.dtype) / 4
            xy = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1).reshape(NODES, 2)
            mapped = (xy[:, None] - xy[None]).square().sum(-1).sqrt()
        elif key in NODE_RANKS:
            if value.ndim != NODE_RANKS[key] or value.shape[0] != 256:
                raise ValueError('unregistered source node tensor: ' + key)
            mapped = value.index_select(0, p)
        elif key == 'kuramoto.direction_learner':
            if tuple(value.shape) != (256, 256):
                raise ValueError('unregistered source pair tensor')
            mapped = value.index_select(0, p).index_select(1, p)
        else:
            mapped = value
        if mapped.shape != target.shape:
            raise ValueError('unregistered state shape change: ' + key)
        result[key] = mapped.to(device=target.device, dtype=target.dtype).clone()
    return result


def build_wrapped(blob, device='cpu'):
    # Imports resolve to the immutable SW0259 snapshot installed by portable.load.
    from SW_0094_aligned_joint_pilot.run import hparams
    from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
    from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
    from collaborative_test.SW_0143_native32_selective_geodesic.geodesic_operator import install_operator
    hp = hparams('raw')
    hp.num_regions = NODES
    hp.k = float(NODES)
    hp.graph_top_k = TOP_K
    hp.graph_spatial_grid_size = GRID
    hp.spike_spatial_grid_size = GRID
    hp.gamma_patch_grid_size = GRID
    hp.validate()
    core = S2NetCore(hp, device=device).to(device)
    core.load_state_dict(map_state(blob['core16_state'], core.state_dict()), strict=True)
    install_operator(core.graph_generator, 'T')
    wrapped = PhaseStateIntegration(core, 'phase')
    with torch.no_grad():
        for key in ('a_d', 'a_m', 'b'):
            getattr(wrapped, key).copy_(blob['integration'][key].to(device))
    return wrapped


def native_routing(self, row):
    """All features and proposals come from native64 activity, without lifting."""
    features, masks, prior = row['features'], row['masks'].bool(), row['prior']
    if features.shape != (NODES, 144) or row['gamma'].shape != (NODES, 8):
        raise ValueError('actual native64 activity and gamma required')
    if masks.shape != prior.shape or masks.shape[1] != NODES:
        raise ValueError('native64 proposals required')
    z = F.normalize(self.head.embedding(features) + self.fine(row['gamma']), dim=-1)
    prototypes = F.normalize(masks.float() @ z / masks.sum(1, keepdim=True).clamp_min(1), dim=-1)
    similarity = prototypes @ z.T
    radius = self.config['radius']
    allowed = F.max_pool2d(masks.float().reshape(-1, 1, GRID, GRID),
                          2*radius+1, stride=1, padding=radius).flatten(1).bool()
    weights = allowed.float()
    mean = (similarity*weights).sum(0) / weights.sum(0).clamp_min(1)
    std = (((similarity-mean).square()*weights).sum(0) /
           weights.sum(0).clamp_min(1)).clamp_min(.0025).sqrt()
    logits = (prior + 2*((similarity-mean)/std).tanh()).masked_fill(~allowed, -1e4)
    return logits, features, prior
