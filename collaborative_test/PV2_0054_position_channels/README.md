# PV2_0054 — widening gamma killed the spike path

**failed, for an implementation reason, not a scientific one. Superseded by
`PV2_0055`, which tests the same hypothesis correctly.** Parent `PV2_0053`.

## What was run

`PV2_0053` found that 38.8% of validation images hold an object pair the graph's
feature cosine cannot separate, and that appending position coordinates takes that
to 18.1% at w=1. This widened gamma from 8 to 10 channels and retrained at
w in {0.5, 1, 2}, seed 0, 40 epochs.

## Result: identical collapse at every setting

| weight | fg_ari | fg_iou | obj_iou | groups | pred_fg |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 0.0033 | 0.0000 | 0.0000 | 0.00 | 0.00 |
| 1 | 0.0033 | 0.0000 | 0.0000 | 0.00 | 0.00 |
| 2 | 0.0033 | 0.0000 | 0.0000 | 0.00 | 0.00 |

Nine rows, three weights by three sync thresholds, **all exactly 0.0033 with zero
groups and zero predicted foreground**. Identical values across settings is a
degenerate failure, not a dose-response, so the number is not a measurement of
position.

## Cause: standardisation is joint over channels

`s2net_cls.py:413`, in `_to_phase`:

```python
dims = tuple(range(1, gamma.dim()))
mean = gamma.mean(dim=dims, keepdim=True)
std = gamma.std(dim=dims, keepdim=True, unbiased=False).clamp_min(1e-6)
```

`dims` spans channels **and** regions, so the eight appearance channels are divided
by a mean and standard deviation that two extra channels move. The appearance
signal is rescaled, everything downstream shifts with it, and the absolute membrane
threshold of 0.06 stops being crossed — the dead-threshold regime of `PV2_0023`.

Three independent signs agree:

- training loss sat at **86.8 -> 85.9** against the 8-channel run's 32.5 -> 17.9;
- `spike_plv_collapse = 9.210340`, exactly log(10^4), the clamp floor, from the
  first epoch — no spikes at all;
- re-thresholding the finished checkpoint at the population mean, **no weight
  changed**, gives 0.0100 -> **0.5577**.

## The design error

The hypothesis concerned one line, `graph_generator.py:123`, where edges are built
from the cosine of appearance channels. But gamma has **two** consumers — the graph
and the phase path — so widening it changed the graph's code *and* the scale of the
phase initialisation. Two variables, one of them unintended. The code should have
been read for consumers before the tensor was widened.

`PV2_0055` separates them with `graph_extra_channels`.
