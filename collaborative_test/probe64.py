"""How large a batch fits at grid 64 WITH the loss?

PV2_0120 died on torch.OutOfMemoryError inside signal_synchrony, at
torch.bmm(trace, trace.transpose(1, 2)) -- the N x N spike synchrony matrix that
--spike-plv-weight 5.0 asks for. At N = 4096 that is 16.7M entries per component per
batch element, and with spike_per_component there are osc_dim of them.

The earlier probe measured a forward pass only and cleared at 2.08 GB, which is why
this failure was not caught: the forward pass is not what allocates the big tensor.
This probes the training step -- forward, loss, backward -- at several batch sizes.
"""
import sys, torch
sys.path.insert(0, "/export_home/tkim1")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")
from hyperparameter import S2NetHyperparameters
from s2net_cls import S2NetCore
from loss_function import signal_synchrony

DEV, GRID, N, T = "cuda", 64, 4096, 64
print("card %s" % torch.cuda.get_device_name(0))
for B in (8, 4, 2, 1):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
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
        core = S2NetCore(hp, device=DEV).to(DEV)
        core.gamma_channels = 8; core.gamma_graph_extra = 0
        g = torch.randn(B, 8, N, device=DEV)
        _, spikes, _, theta = core(g, return_core_out=True, return_theta=True)
        # the allocation that killed PV2_0120: the per-component spike synchrony
        comp = getattr(core, "last_component_spikes", None)
        if comp is not None:
            s = torch.stack([signal_synchrony(comp[:, d], settle=32).clamp_min(0)
                             for d in range(comp.shape[1])]).prod(0)
            loss = s.mean() + spikes.mean()
        else:
            loss = spikes.mean()
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 2**30
        print("  batch %-2d  OK      peak %.2f GB of 23.47" % (B, peak), flush=True)
        del core, g, spikes, loss
    except torch.OutOfMemoryError:
        print("  batch %-2d  OOM     peak %.2f GB" % (B, torch.cuda.max_memory_allocated()/2**30), flush=True)
    except Exception as exc:
        print("  batch %-2d  ERROR   %s" % (B, str(exc).split("\n")[0][:70]), flush=True)
    torch.cuda.empty_cache()
