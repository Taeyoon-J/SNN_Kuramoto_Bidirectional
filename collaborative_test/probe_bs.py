"""One batch size per process, so a failed allocation cannot poison the next.

PV2_0120's first probe ran batch 8, 4, 2 and 1 in one process. After the batch-8 OOM the
exception object held references to the tensors, so empty_cache() could not return the
memory and every later size ran on top of it -- which is how "batch 1 OOMs at 23.16 GB"
was recorded when batch 1 actually needs 4.59 GB.
"""
import sys, torch
sys.path.insert(0, "/export_home/tkim1")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")
from hyperparameter import S2NetHyperparameters
from s2net_cls import S2NetCore
from loss_function import signal_synchrony

B = int(sys.argv[1]); GRID, N, T = 64, 4096, 64
hp = S2NetHyperparameters(
    num_feature_maps=8, num_regions=N, sc=None, gamma_drive_mode="static",
    num_time_steps=T, theta_init="gamma", gamma_phase_mode="standardize_tanh",
    osc_dim=4, freq_gain=2.0, graph_mode="learned", graph_top_k=32, k=float(N),
    graph_spatial_decay=0.35, low_n=-4., high_n=0., membrane_vth=0.06,
    membrane_low_m=-4., membrane_high_m=0., gate_mode="raw",
    spike_per_component=True, membrane_threshold_mode="population",
    membrane_threshold_k=0.5, geodesic_steps=3, geodesic_radius=1.5,
    geodesic_contrast=2.0, geodesic_chunk=8,
    spike_classify_method="spatial_components",
    spike_spatial_grid_size=(GRID, GRID)).validate()
core = S2NetCore(hp, device="cuda").to("cuda")
core.gamma_channels = 8; core.gamma_graph_extra = 0
g = torch.randn(B, 8, N, device="cuda")
_, spikes, _, _ = core(g, return_core_out=True, return_theta=True)
comp = core.last_component_spikes
loss = spikes.mean() + torch.stack([signal_synchrony(comp[:, d], settle=32).clamp_min(0)
                                    for d in range(comp.shape[1])]).prod(0).mean()
loss.backward()
print("  batch %-2d  OK   peak %6.2f GB" % (B, torch.cuda.max_memory_allocated()/2**30))
