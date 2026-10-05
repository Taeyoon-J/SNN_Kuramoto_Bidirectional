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

