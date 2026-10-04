# SW0045 - HDF5-aligned stage signal diagnostic

This reruns the SW0034 frozen-core stage diagnostic on SW0042 seed0, using the
aligned HDF5 validation gamma IDs 1320-1639 and the matching HDF5 masks. It
retains the distance-controlled foreground-pair AUCs for phase, gating,
h-wave, membrane, gated spike, and binary-threshold stages; activation and
constant-node statistics; spatial-only, membrane-times-spatial, and
gated-spike-times-spatial readout controls; and phase/membrane/spike loss
gradient-connectivity checks. Ground truth is used for pair labels and scoring
only, after model outputs are formed.

The checkpoint is loaded with SW0042's configuration: vth .06, shared
dendritic projection, graph spatial decay .35, geodesic steps 3, radius 1.5,
contrast 2, temperature .5, cap 16, and factorized Kuramoto backend. Readout is
T256/settle64; gradient connectivity remains a separate T64/settle32 diagnostic
as in SW0034. `diagnose.py` gained backward-compatible CLI parameters; its old
defaults still use full-tensor global IDs and the original SW0034 core setup.
The original SW0034 command and stored results are unchanged.

`launch_after_sw0042.sh GPU_ID` waits every 20 seconds for the seed0 checkpoint,
the SW0042 short and long result JSONs, the SW0044 `SPATIAL_EVALUATED` marker,
aligned gamma plus manifest, HDF5, and a free selected GPU. Requiring the
spatial marker serializes this watcher behind the SW0044 watcher when both are
assigned to GPU3. It rechecks before launch, runs only the full320 stage
diagnostic, and writes `STAGE_SIGNAL_DIAGNOSED` after the JSON is complete.
Output is `stage_signal_validation320.json` under the SW0042 seed0 model folder.
Existing output/marker causes an overwrite refusal; the watcher never stops
other processes.

The full seed0 diagnostic completed. Distance-controlled macro pair AUC is
`.856734` at phase PLV, `.848851` at gating, `.840436` at h-wave, `.841045`
at membrane, `.835516` at gated spike, and `.815982` at the ungated binary
threshold. Thus the good seed already contains strong object structure at the
phase stage and loses it gradually downstream rather than at one catastrophic
layer boundary. The sigma1.5/k10 spectral controls score
`.723731/.211605/.370475` for membrane-times-spatial and
`.705382/.210102/.364788` for spike-times-spatial, compared with spatial-only
`.491272/.190837/.178026`. Unlike the earlier weak-core SW0033 result, learned
activity now adds substantial signal beyond geometry. Phase loss still does
not connect to dendrite/membrane parameters; membrane and spike losses do.

Synthetic local checks:

```bash
python -m pytest collaborative_test/SW_0045_aligned_stage_signal/test_aligned_inputs.py -q
```
