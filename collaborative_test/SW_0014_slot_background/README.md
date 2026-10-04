# SW_0014: Is adaptive-slot background selection the bottleneck?

Status: completed, negative for all-metric gain. No training or production classifier change.

SW_0013's adaptive slots raised FG-ARI but lowered the two IoUs and still
predicted too many groups. This experiment held the same SW_0011 40-epoch
checkpoint, validation IDs 1320–1639, spike histories, slot update rule, and
evaluation fixed. Only the *prediction-only background rule* changed:

- `largest`: the biggest slot is background (SW_0013's current rule).
- `border`: any slot with at least two image-edge patches and border fraction
  above 0.1, 0.2, or 0.3 is background.
- `largest_plus_border`: both rules together.
- `none`: only constant/no-spike histories are background.

These rules never consult the ground-truth mask. We tested slot thresholds and
initial counts (0.3,1), (0.5,3), and (0.7,6). On the 320-image validation:

| Rule at threshold .7, six initial slots | FG-ARI ↑ | FG IoU ↑ | object IoU ↑ | predicted groups | FG precision | FG recall |
|---|---:|---:|---:|---:|---:|---:|
| Largest slot (SW_0013) | .247376 | .248332 | .204032 | 17.37 | .2619 | .8259 |
| Border only, fraction .1 | .071969 | .337518 | .177311 | 13.39 | .5702 | .4797 |
| Largest + border, fraction .1 | .071979 | .338240 | .177311 | 13.39 | .5726 | .4796 |
| No slot background | .247376 | .216821 | .207310 | 18.37 | .2168 | 1.0000 |

The border rule improves foreground precision and IoU by suppressing many
patches, but also removes almost half the true foreground and damages ARI and
object IoU. No tested background policy improves all three metrics over the
existing readout. Therefore background selection alone does not solve the
overfragmentation, and these rules are not adopted. This is a diagnosis, not
proof that future learned background inference cannot help.

`evaluate.py` is new and imports the SW_0013 slot routine with its new
`assign_background=False` diagnostic option. With the default `True`, SW_0013
behavior and recorded results are unchanged. Full results: `validation320.json`.
