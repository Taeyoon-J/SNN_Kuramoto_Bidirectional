# PV2_0011 — fg_ari does not prefer a lower spike synchrony weight (hypothesis wrong)

**completed, hypothesis refuted.** Parent `PV2_0008`. Validation 300, three seeds.

The weight 5 had been chosen to maximise foreground IoU, and a seed-0 sweep on
100 images suggested fg_ari wanted 0 or 1 instead (0.5993 and 0.5958 against
0.4560 at weight 5). At three seeds that reverses:

| `--spike-plv-weight` | fg_ari, 3-seed mean |
| --- | --- |
| 0 | 0.558 |
| 1 | 0.546 |
| **5** | **0.611** |

Weight 5 is best for fg_ari as well. The claim that the configuration was "close
to the worst choice for the metric now being asked for" was wrong, and rested on
a single seed at 100 images -- the fourth single-seed reading in this project to
need correcting.

Instability shows up across weights too, not only at weight 5: `ARI_w0_s1` and
`ARI_w1_s2` both reach predicted foreground near 0.5 with foreground IoU 0.16.
