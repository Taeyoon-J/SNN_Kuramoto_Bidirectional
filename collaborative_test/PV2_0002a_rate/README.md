# PV2_0002a — constrain the firing rate

**failed.** Parent: `PV2_0001`. Contract v1, split v1.

## Hypothesis

`PV2_0001` came out at std 0.07 to 0.10, wider than most differences this project
measures. Its spike rate varied sevenfold across seeds -- 0.043, 0.224, 0.324 --
and the lowest-rate seed was last on all three metrics, while nothing in the
objective constrains the rate. If the rate was driving the spread, pinning it
should narrow the spread. Success was a visible drop in std; the mean was allowed
to stay where it was.

## Change

Loss only; no architecture change.

```
-  --loss-signal sigmoid_membrane  --spike-rate-weight 0
+  --loss-signal spikes            --spike-rate-weight 1.0  --spike-target-rate 0.2
```

`--loss-signal spikes` is needed: the rate term reads `loss_signal`, and at its
default of `sigmoid_membrane` it pushes the membrane rather than the firing rate.

## Result

Test split, 300 images, three seeds, thresholds unchanged from `PV2_0001`.

| metric | seed 0 | seed 1 | seed 2 | mean | std | PV2_0001 | delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `patch_fg_ari` | 0.2781 | 0.3042 | 0.1126 | 0.2316 | 0.1038 | 0.4457 +/- 0.0706 | -0.2141 |
| `patch_foreground_iou` | 0.3302 | 0.4643 | 0.2219 | 0.3388 | 0.1214 | 0.4115 +/- 0.0980 | -0.0727 |
| `patch_matched_object_iou` | 0.2202 | 0.2940 | 0.1778 | 0.2307 | 0.0589 | 0.3438 +/- 0.0924 | -0.1131 |

Spike rate 0.193, 0.178, 0.193, against 0.043, 0.224, 0.324 before.

## Insight

**The intervention worked and the hypothesis was wrong.** The rate is pinned --
sevenfold spread down to nothing -- and the metric spread did not follow: fg_ari
std went 0.071 to 0.104 and foreground IoU 0.098 to 0.121. Only
matched-object IoU narrowed, 0.092 to 0.059, and all three means fell.

So the seed variance has a source other than the firing rate. The rate was a
visible correlate of it -- the slowest-firing seed really was the worst -- but
constraining the correlate does not constrain what drives it.

Worth noting what else that loss term does: it is the first term in this project
to send gradient into the dendrite, 1.45e-01 against exactly 0 from the PLV
objective. It did not help, which is a second small result: simply giving the
spiking layers a gradient is not enough, it has to be a gradient about something
useful, and firing rate is not.

## Next

Dropped. `PV2_0002b`, the geodesic graph, is the live line.
