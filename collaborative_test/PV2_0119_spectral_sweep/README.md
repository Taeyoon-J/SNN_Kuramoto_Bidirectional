# PV2_0119 — the sweep `PV2_0091` owed, and the dial peaks at 0.7862

Status: **completed**. Settles an argued-but-unmeasured claim `ROUTE_MAP_0.9_v3` leans on.
One checkpoint (`BIM6_s1`), `--limit 150 --num-time-steps 512`, exactly as `PV2_0091`'s
Limitations specified. Contract v1, grid 16, validation.

## Why it was owed

`PV2_0091` concluded *"the gap is in what the spikes encode, not in how they are
thresholded"* — and said in its own Limitations that the values behind it came from
**different runs and configurations gathered from the records**, not one checkpoint swept
in one session. The clean sweep was launched once, produced the components row, and was
stopped at 25 minutes per row under contention with the 32×32 training.

## Result

| readout | fg_ari | fg_iou | obj_iou | groups | **pred_fg** |
| --- | ---: | ---: | ---: | ---: | ---: |
| components (no k) | 0.7259 | **0.7158** | 0.4835 | 5.25 | **0.1266** |
| spectral k=6 | 0.5744 | 0.5432 | 0.4027 | 4.94 | 0.1782 |
| k=8 | 0.6620 | 0.5079 | 0.5099 | 7.00 | 0.2693 |
| k=10 | 0.7198 | 0.4662 | 0.5765 | 9.00 | 0.3732 |
| k=12 | 0.7598 | 0.4032 | 0.6254 | 11.00 | 0.4780 |
| k=14 | 0.7704 | 0.2970 | 0.6532 | 13.00 | 0.5990 |
| k=16 | 0.7767 | 0.2148 | 0.6692 | 15.00 | 0.7099 |
| **k=20** | **0.7862** | 0.1516 | 0.6948 | 19.00 | 0.8391 |
| k=24 | 0.7544 | 0.1320 | **0.7023** | 23.00 | 0.8954 |
| k=28 | 0.7132 | 0.1271 | 0.6997 | 27.00 | 0.9189 |
| k=32 | 0.6765 | 0.1262 | 0.7001 | 31.00 | **0.9349** |

The recorded values reproduce: components 0.7300 → **0.7259** here, k=14 0.7603 →
**0.7704**, k=20 0.7718 → **0.7862**. The small differences are the cheaper setting
(150 images, 512 steps against 300 and 1024), not a disagreement.

## What the sweep adds to `PV2_0091`'s argument

**1. The dial has a peak, and it is below the bar.** fg_ari rises to **0.7862 at k=20** and
then *falls* — 0.7544, 0.7132, 0.6765. `PV2_0091` argued from collected values that the
maximum was far below the bar; this measures that it is a maximum, on one checkpoint, and
that pushing past it loses even the free points. Aligned Slot Attention is **0.8102** and
the goal is **0.9**.

**2. `pred_fg` is the whole story.** Every point of fg_ari above the components readout is
bought by calling more of the image foreground: 0.1266 → **0.9349** across the sweep,
against a true foreground of **0.1152**. At k=20, where fg_ari peaks, the model calls
**84% of the image foreground** and foreground IoU has collapsed 0.7158 → 0.1516, a factor
of **4.7**.

**3. It is the same trade `PV2_0090` caught Slot Attention making.** Slot calls 85%
foreground, scores 0.8102 fg_ari, and has IoUs of 0.1151 / 0.1025. Our k=20 calls 84%,
scores 0.7862, and has 0.1516 / 0.6948. The dial moves our readout along Slot's axis —
and even at the far end there is no 0.9.

**4. matched-object IoU saturates at 0.70.** It climbs to 0.7023 at k=24 and stops, while
foreground IoU keeps falling. So there is no setting where both IoUs are good: the honest
column tops out at 0.70 and the figure/ground column is already ruined by then.

## Conclusion

`PV2_0091`'s claim is now measured rather than argued: **the readout dial trades FG-ARI
against foreground IoU and cannot raise both.** Its best FG-ARI is 0.7862 at a predicted
foreground of 0.8391, which is not a result a calibration gate would ever accept.

Combined with `PV2_0118`, every route in `ROUTE_MAP_0.9_v3` is now closed except **grid 64**
(route C, ~96 GPU-hours, untested).
