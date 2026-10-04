# SW0044 - Spatially weighted spike affinity

This experiment adds an opt-in spatial Gaussian to the existing product of
per-component spike synchrony affinities. For patch locations `i,j`, it forms
`S_ij * G_ij`, where `G_ij = exp(-d_ij^2 / (2 sigma^2))` and distance is in
16x16 patch-grid units. The unchanged `spike` mode is the default. The new
evaluator compares `spike` (S-only), `spike_spatial` (S times G), and
`spatial_only` (G-only) over sigma 1.0, 1.5, 2.0, infinity and synchrony
thresholds .05, .10, .15, .20, .35, .50. S-only uses the same threshold grid.
Choose `short` for T256/settle64 or `long` for T1024/settle512; each invocation
uses a single window so those rollout protocols are not mixed.

The largest-connected-component background convention also applies to the
spatial-only control. A Gaussian grid can connect most or all patches at low
thresholds, after which removing that largest component may yield an empty
foreground; the empty-image count is explicitly reported. Compare predicted
groups/image, predicted foreground fraction, empty-image count, and all three
mask metrics when selecting a common sigma/threshold. Ground-truth masks are
used only after predictions for scoring/count diagnostics; `ground_truth_used_for_prediction`
is recorded as false in the JSON.

The evaluator is paired with the SW0042 HDF5-aligned checkpoint contract:
validation gamma IDs 1320-1639 and its manifest, membrane threshold .06,
shared dendritic projection, graph spatial decay .35, geodesic steps 3 / radius
1.5 / contrast 2 / temperature .5 / cap 16, and factorized Kuramoto backend.
The only readout change is the affinity mode/sigma sweep. The collaborator's
reported spatial-sigma result is provisional and based on validation 300; it
should not be treated as a final result or pooled with this fixed split. This
folder only prepares an evaluation; no server evaluation or training has been
started.

```bash
bash collaborative_test/SW_0044_spatial_spike_affinity/evaluate.sh GPU_ID CHECKPOINT OUTPUT.json short
bash collaborative_test/SW_0044_spatial_spike_affinity/evaluate.sh GPU_ID CHECKPOINT OUTPUT.json long
```

For the SW0042 seed0 checkpoint, `launch_after_sw0042.sh GPU_ID` waits in
20-second intervals until `core.pt`, `validation_short_T256_settle64.json`,
and `validation_long_T1024_settle512.json` exist and the selected GPU has no
compute process. It rechecks readiness before launch, runs short and long
evaluations sequentially, and writes `SPATIAL_EVALUATED` only after both
complete. Outputs are `spatial_affinity_short.json` and
`spatial_affinity_long.json` under
`trained_models/SW_0042_HDF5_aligned_BIM6_s0`. Existing outputs or marker cause
an immediate refusal to overwrite. It never stops other processes. This
watcher is prepared but has not been run.

For any SW0042 seed, `launch_seed_after_sw0042.sh GPU_ID SEED` supports seeds
0, 1, and 2. It waits for that seed's checkpoint, short/long baseline
validation JSONs, and an idle GPU, then runs the corresponding spatial sweeps
sequentially into that seed's directory and writes its `SPATIAL_EVALUATED`
marker. Existing result files or marker cause refusal; it does not terminate
compute processes. The existing seed0-only watcher remains separate and is not
modified by this launcher.

After SW0042 base and SW0044 spatial short/long validation JSONs exist for all
three seeds, create a validation summary with:

```bash
python collaborative_test/SW_0044_spatial_spike_affinity/summarize_three_seed.py \
  --model-root /Data0/kevinswk/patch_v2_sw/trained_models \
  --output /Data0/kevinswk/patch_v2_sw/trained_models/SW_0044_spatial_affinity_three_seed_summary.json
```

The summarizer requires both SW0042 baseline and SW0044 spatial short/long
results from every seed. It reports mean and sample standard deviation for
the common `(affinity_mode, sigma, threshold)` configurations, separately by
window, and includes the SW0042 spike-only threshold control. It does not rank
configurations or reselect from ground truth; it reads only already-produced
validation metrics and prediction diagnostics.

Focused local tests:

```bash
python -m pytest collaborative_test/SW_0044_spatial_spike_affinity/test_spatial_affinity.py -q
```

All three seed sweeps completed and the common-configuration summary is stored
in `results/three_seed_summary.json`. The best common long FG-ARI row is
spike-times-spatial, sigma 1.5, threshold .20:
`.418171 / .414471 / .301746`. The best common long matched-object IoU row is
sigma 1.0, threshold .10: `.416896 / .415005 / .301948`. The best foreground
IoU remains spike-only threshold .20 at
`.415117 / .420557 / .293158`. Thus the spatial readout adds only a small
three-seed ARI/object-IoU gain because the collapsed seed2 dominates the mean;
it does not solve core stability.
