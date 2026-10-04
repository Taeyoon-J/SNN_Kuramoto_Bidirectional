# SW0047 - Deterministic permuted-spatial control

This is an opt-in control for SW0044's spike-times-Gaussian affinity. It forms
`S_ij * G_{p(i),p(j)}`, where `S` is the existing per-component spike
synchrony product, `G` is the same 16x16 Gaussian spatial kernel, and `p` is a
deterministic node permutation generated from each recorded seed. Applying one
permutation to both kernel axes preserves the kernel's values and degree
distribution while breaking its correspondence with the actual patch
positions. It does not alter the model, spikes, default affinity, or SW0044
outputs.

The aligned seed0 evaluation uses SW0042's validation IDs 1320-1639 and
checkpoint with the same inference contract as SW0044. It evaluates short
T256/settle64 and long T1024/settle512 separately, sigmas 1.0, 1.5, 2.0,
thresholds .05, .10, .15, .20, and permutation seeds 0, 1, 2. Each row and the
top-level provenance records the permutation seed. Predictions depend on the
model spikes, kernel, threshold, and fixed permutation only; target masks enter
after prediction for scoring, as recorded by `ground_truth_used_for_prediction`
being false.

The `spatial_only` control from SW0044 can produce an empty foreground when a
Gaussian kernel becomes one large connected component and largest-component
background removal removes it. That does not test whether spatial correspondence
helps or harms a learned spike affinity. The permuted control keeps the same
kernel's values and degree distribution and changes only its alignment with
patch positions, providing that comparison.

Run an individual evaluation with:

```bash
bash collaborative_test/SW_0047_permuted_spatial_control/evaluate.sh \
  GPU_ID CHECKPOINT OUTPUT.json short
bash collaborative_test/SW_0047_permuted_spatial_control/evaluate.sh \
  GPU_ID CHECKPOINT OUTPUT.json long
```

For the seed0 SW0042 checkpoint, `launch_after_sw0044.sh GPU_ID` waits in
20-second intervals for the checkpoint, SW0042 short/long baseline JSONs, the
SW0044 `SPATIAL_EVALUATED` marker, and an idle selected GPU. It rechecks output
collisions and GPU availability before launch, runs both windows sequentially,
and writes `PERMUTED_SPATIAL_EVALUATED` only after success. Existing output or
marker files cause refusal; it never terminates another process.

The seed0 control completed. Its best permuted FG-ARI row used permutation 0,
sigma 2.0 and threshold .05: short scored
`.319195 / .494502 / .254317`, and long scored
`.324880 / .480193 / .262987` (FG-ARI / foreground IoU / matched-object IoU).
The correctly aligned SW0044 spike-times-spatial rows reach about
`.599264 / .608529 / .463727` short and
`.601268 / .603618 / .466832` long at their best-FG-ARI settings. Breaking
position correspondence therefore causes a large loss, so the spatial gain
depends on the actual patch layout rather than only the Gaussian kernel's
value or degree distribution. This validates spatial correspondence as a
useful readout prior; it is not evidence that the learned core improved.
