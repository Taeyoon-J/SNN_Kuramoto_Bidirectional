# PV2_0016-0018 — giving the transduction capacity (failed, and instructive)

**failed.** Parent `PV2_0013`. Seed 0, validation 300.

`PV2_0015` found the phase-to-spike map is 8 weights shared across all 256
regions, so `--dendrite-per-region` gave each region its own, 12 transduction
parameters becoming 3,072. Verified before running: the update keeps its form,
and tying the per-region weights reproduces the shared layer to 2.4e-7.

| arm | fg_ari | foreground_iou | phase fg_ari |
| --- | --- | --- | --- |
| `BIM6_s0` baseline | **0.6211** | **0.6680** | **0.6769** |
| per-region, random init | 0.3066 | 0.4386 | 0.5368 |
| + alignment 10 | 0.3286 | 0.4123 | 0.4877 |
| + alignment 40 | 0.0655 | 0.1639 | 0.2659 |

fg_ari halves. Independently initialised per-region weights make two units at the
same phase respond differently, so synchrony stops reaching the spikes at all.
**The 8 shared weights are not a bottleneck to remove; they are the prior that
makes phase legible downstream.**

That experiment moved two things at once, though -- capacity and the prior -- so
`PV2_0018` separated them with tied initialisation. Step 0 then reproduced the
bar to four decimals (0.6211 / 0.6679 / 0.4255), and the alignment term finally
optimised for real, 0.477 -> 0.391, where under the shared map it had moved only
in its sixth decimal.

And fg_ari still fell, 0.6211 -> 0.5826.

So the premise behind `PV2_0007`, `PV2_0014`, `PV2_0015` and this one is refuted:
the spike readout does not lag because its synchrony matrix has the wrong shape.
It was made possible to match the phase matrix, it was matched, and the metric
got worse.
