# SW0077 - GT-free RGB veto for spike components

This is a fixed 32-image seed1 pilot on the frozen SW0072 core, IDs 1320-1351,
with the same aligned gamma, T256/settle64, vth .06, graph/geodesic settings,
and .35 synchrony cutoff as SW0072's pilot. It targets SW0072 seed1's low
foreground IoU while preserving the spike connected-component grouping.

The evaluator emits four rows: the current per-component spike affinity,
binary-crossing-only affinity, and the same two affinities after one RGB
foreground veto. Binary-only affinity binarizes each per-component trace as
`components != 0` before applying the same per-component correlation product;
it removes any continuous amplitude weighting while preserving crossing
identity. The peer reports
gated and binary-only affinities correlate at `r=.982`; this pilot checks their
readout consequences directly.

The RGB cue estimates each image's background from the 60 perimeter patches.
It takes the per-channel median patch RGB and robust scale `max(1.4826*MAD, 4
intensity levels)`. Each spike component gets the median diagonal standardized
RGB distance of its member patches from that border model. A component is
retained only when the distance exceeds `3.37`, the nominal 99th-percentile
radius for a 3-D chi-square color model. That single threshold is fixed before
scoring; no labels, masks, object counts, or validation scores set it.

This is a whole-component veto. Retained tuples remain byte-for-byte identical;
the classifier neither splits nor merges groups. The largest-component
background rule and minimum-size filter remain the same. The base gated and
binary-only rows provide controls for the veto applied to each affinity.

Earlier foreground rules motivate the conservative test. SW0030's border
fraction cutoff `.2` moved foreground coverage from `.8648` to `.2283`, but
FG-ARI fell `.5301 -> .2610`; `.4` coverage `.7776` retained FG-ARI `.5148`.
SW0031's independent global foreground intersection gained foreground IoU
while reducing FG-ARI/object IoU. SW0032's border-limited veto improved the
two IoUs slightly but reduced FG-ARI (`.4946 -> .4677` for width2). SW0067's
per-image fixed-area search likewise reduced FG-ARI/object IoU and worsened
count MAE. These results show that reducing foreground area alone can discard
object patches or entire groups.

Expected failure modes are border patches contaminated by objects, object
colors close to the scene background, spatially varying backgrounds, and
components that mix background-colored and object-colored patches. The median
component score may remove a legitimate background-colored object wholesale.
The pilot is a classifier diagnostic only, not evidence of generalization.

Run on GPU0 or GPU1 only after confirming it is idle:

```bash
bash collaborative_test/SW_0077_rgb_border_component_veto/evaluate.sh 0
```

The script refuses existing result/log files. No remote evaluation has been
launched.

## Result

| seed1 32-image pilot | FG-ARI | foreground IoU | matched-object IoU | count MAE |
|---|---:|---:|---:|---:|
| gated spike baseline | 0.708757 | 0.468105 | 0.509881 | 1.37500 |
| gated spike + RGB veto | 0.476408 | 0.515963 | 0.311105 | 3.15625 |
| binary crossing | 0 | 0 | 0 | 5.53125 |
| binary crossing + RGB veto | 0 | 0 | 0 | 5.53125 |

The RGB veto raises foreground IoU by `0.047858` but removes too many true
object groups: predicted foreground falls from `0.37769` to `0.10608`, and both
grouping metrics and count accuracy collapse. It is a rejected foreground-IoU
tradeoff; do not tune its threshold.

The binary result differs from the peer because this checkpoint is saturated in
the settled window. Across `[32,4,256,192]` settled component traces, crossing
rate is `0.999938`, `98.82%` of units are always on, and mean per-unit temporal
standard deviation is `0.000855`. Binary-only and gated affinity correlate only
`0.00342`; just `4.79e-6` of binary off-diagonal affinities are positive. This
confirms that SW0072 grouping is carried by the continuous gate amplitude under
this configuration. Stop both SW0077 directions.
