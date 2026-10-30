import numpy as np

def prediction_from_trace(found,trace,adj,q):
    from ..spike_classifier import spike_synchrony_components
    from ..evaluation import spatial_components_to_patch_labels
    from .partition import foreground_induced_merge
    groups=spike_synchrony_components(trace['spikes'].detach().cpu(),synchrony_threshold=.5,min_group_size=32,
        settle=512,components=trace['component_spikes'].detach().cpu(),background='largest_component',affinity_mode='spike',spatial_grid_size=64)
    support=spatial_components_to_patch_labels(groups,64,device='cpu').numpy()[0].astype(np.int64)
    gain=float(found.wrapped.core.graph_generator.log_coupling_gain.exp().detach().cpu())
    labels=np.asarray(foreground_induced_merge(q[0].detach().cpu().numpy().astype(np.float64),
        adj[0].detach().cpu().numpy().astype(np.float64),gain,support,grid_shape=(64,64))['labels'],dtype=np.int64)
    if labels.shape!=(64,64) or np.any(labels<0) or not np.array_equal(labels>0,support>0):raise ValueError('native64 production support mismatch')
    return labels
