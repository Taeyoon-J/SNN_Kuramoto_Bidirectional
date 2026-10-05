# SW0075 - preserve the foreground activation profile

SW0074 shows that small gamma RMS drift does not preserve foreground IoU.
SW0073's learned features alter the fixed SW0072 core's settled patchwise mean
sigmoid-membrane profile by only MSE `1.4e-4`, yet foreground IoU falls about
`.037`. The relevant invariant is therefore the downstream activation profile,
not raw feature distance.

SW0075 freezes the complete SW0072 seed1 core and distills its per-patch settled
membrane activity from native gamma while the encoder also receives the
reconstruction and binding objectives. Only the time-mean activity is matched,
so temporal rhythm and phase grouping can still improve. Coarse weights `1000`
and `10000` make the measured drift contribute about `.14` and `1.4` loss units.
An arm advances only if all three seed1 metrics exceed SW0072. No masks, counts,
or validation labels enter training.

## Result

| seed1 pilot | FG-ARI | foreground IoU | matched-object IoU | count MAE |
|---|---:|---:|---:|---:|
| SW0072 baseline | 0.708757 | 0.468105 | 0.509881 | - |
| activity anchor 1,000 | 0.679576 | 0.441083 | 0.499716 | 1.28125 |
| activity anchor 10,000 | 0.706576 | 0.477694 | 0.500626 | 1.34375 |

Neither arm passes the registered all-three gate. Weight 10,000 nearly preserves
FG-ARI and improves foreground IoU by `0.009590`, but matched-object IoU falls by
`0.009254`. Weight 1,000 lowers all three metrics. The frozen core and graph have
exactly zero parameter change in both arms.

The stronger anchor reduces gamma RMS drift to `0.005403`, so the negative result
is not explained by a loose constraint. Matching time-averaged membrane activity
preserves foreground extent better but does not preserve or improve object-wise
binding. Stop this scalar activation-profile route and test preservation of the
image-conditioned graph topology directly.

