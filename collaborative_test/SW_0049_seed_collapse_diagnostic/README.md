# SW0049 - Seed2 upstream-collapse diagnostic

This is a follow-up diagnostic for SW0042 seed2. The three HDF5-aligned runs
have increasingly lower final reconstruction losses but sharply worsening
short-window spike-readout FG-ARI: final losses are 18.8295 / 17.3599 /
14.2687 for seeds 0/1/2, while T256/settle64 threshold .35 FG-ARI is
.5820/.4285/.1866. The mean phase product-PLV is .4454 on seed2, and its
per-image phase-PLV mean has standard deviation .00627, versus .1433/.1707
for seeds0/1. Together these figures motivate checking whether seed2's
representation becomes unusually input-insensitive upstream of binary spike
thresholding; they do not by themselves identify the cause. Exact supplied
figures are preserved in `results/seed_collapse_evidence.json`.

The watcher reuses the existing SW0034 `diagnose.py` on SW0042 seed2 and keeps
its full aligned stage analysis: phase, gating, h-wave, membrane, gated spike,
and binary threshold distance-controlled pair AUCs; activation/constant-node
statistics; readout controls; and gradient connectivity. It uses HDF5-aligned
IDs 1320-1639, matching gamma/manifest and targets, vth .06, shared dendritic
projection, graph decay .35, geodesic steps 3/radius 1.5/contrast 2/temperature
.5/cap 16, factorized Kuramoto, and T256/settle64.

```bash
bash collaborative_test/SW_0049_seed_collapse_diagnostic/launch_after_seed2_spatial.sh GPU_ID
```

The watcher polls every 20 seconds until seed2's `core.pt`, base short and
long validation JSONs, SW0044 `SPATIAL_EVALUATED` marker, aligned gamma,
manifest, HDF5, and an idle selected GPU are present. It rechecks readiness
and output collisions before starting, writes
`trained_models/SW_0042_HDF5_aligned_BIM6_s2/stage_signal_validation320.json`,
and creates `STAGE_SIGNAL_DIAGNOSED` only on completion. Existing output or
marker makes it refuse overwrite. It never terminates other processes. The
watcher is prepared but has not been launched.
