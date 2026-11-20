# PV2_0118 — the γ-contrast loss fails its promotion, and the 8-epoch gain is explained

Status: **completed — FAILS the registered gate at three seeds. The registered falsifier
fired.** Parent: `PV2_0115` (8-epoch screen, which it passed). Contract v1, grid 16,
40 epochs, seeds 0/1/2, validation 300. Regression row passed first: `0.7239 0.7251 0.4951`.

## Result

| three-seed mean | fg_ari | fg_iou | obj_iou | pred_fg |
| --- | ---: | ---: | ---: | ---: |
| control | 0.5963 ± 0.2251 | 0.5892 | 0.4095 | 0.1296 |
| γ-contrast 5.0 | 0.5690 ± 0.1276 | 0.5188 | 0.3457 | **0.0986** |
| change | **−0.0273** | **−0.0704** | **−0.0639** | −0.0310 |

Per seed, the mean is not describing a treatment effect:

| seed | control | γ-contrast 5.0 | Δ fg_ari | control `pred_fg` → treated |
| --- | ---: | ---: | ---: | --- |
| 0 | 0.3374 | 0.5176 | **+0.1802** | 0.1272 → 0.0964 |
| 1 | 0.7064 | 0.7143 | +0.0079 | 0.1249 → 0.1160 |
| 2 | 0.7451 | 0.4752 | **−0.2699** | 0.1366 → 0.0735 |

**The control itself ranges 0.3374 to 0.7451**, a sample std of 0.2251. Seed variance
dominates the mean, so the −0.0273 is not evidence of harm any more than the 8-epoch
+0.0070 was evidence of help.

## The one consistent effect, and why it is the failure

`pred_fg` falls at **every** seed: 0.1272 → 0.0964, 0.1249 → 0.1160, 0.1366 → 0.0735. The
term reliably reduces predicted foreground. That is exactly what it was built to do, and
what `PV2_0114`'s arithmetic asked for — but only while there is excess to remove.

| regime | control `pred_fg` | treated | against a true 0.1152 |
| --- | ---: | ---: | --- |
| `PV2_0115`, 8 epochs | 0.3699 | 0.3122 | both far **over**; the term helps |
| `PV2_0118`, 40 epochs | 0.1296 | **0.0986** | control is near target; the term **overshoots into under-prediction** |

**The term is a one-way pressure with no knowledge of the target level.** At eight epochs
the control was over-predicting by 3×, so any reduction helped. At forty the control is
already at 0.1296 against a true 0.1152, and the same pressure pushes it to 0.0986 — seed 2
reaches 0.0735, and that is where its fg_ari collapses by 0.27.

**So `PV2_0115`'s gain was an artefact of the 8-epoch regime's over-prediction, not a
general property of the loss.** The screen's own recorded limitation said its control
breached the calibration gate at 0.3699 and only the comparison to control was readable;
what it could not show is that the comparison itself depended on that breach.

## A reproducibility check that came free

`GCP_ctrl_s0` scores **0.3374 / 0.3246 / 0.2383**, identical to four decimals to
`PV2_0116`'s `DS_1000`. Same configuration, same seed, same data — it reproduces exactly.

That also says this training line does **not** reproduce the published 0.7239 at seed 0.
The control is internally valid for the comparison but is not the published model, and
`PV2_0115`'s and this test's absolute numbers must not be read against the published line.

## Verdict

The falsifier was registered as: *if the three-seed means do not clear the control, the
loss joins the closed list*. They do not. **It joins the closed list as written.**

What survives is the mechanism, which is worth keeping: the term does what it claims —
it moves spike activity toward the γ deviation (`gamma_contrast` −0.55 to −0.70 at the end
of training) and it reduces predicted foreground at every seed. It fails because reducing
foreground is only useful when there is excess, and nothing in it measures that.

A version that scaled its weight by the gap between predicted and target foreground would
be a different experiment, not a rescue of this one, and it would need the target — which
is a hyperparameter selectable on validation, not an oracle. That is recorded as a
possibility, not claimed as a result.

## Where this leaves the route to 0.9

Every route in `ROUTE_MAP_0.9_v3` is now closed except two that were never run:

| route | state |
| --- | --- |
| readout: add unclaimed patches | closed (`PV2_0112`, base rate 0.73%) |
| readout: widen groups | closed (peer `SW_0207`) |
| readout: merge groups | closed (`PV2_0107b`, `SW_0234` — loses obj_iou even with a GT oracle) |
| readout: delete spurious groups | closed (no signal, AUC 0.500) |
| readout: `--target-foreground` | closed (`PV2_0117`, shatters the partition) |
| data scale | closed (`PV2_0116`) |
| "fire on small objects" | refuted (`PV2_0113`) |
| **γ-contrast loss** | **closed here** |
| C grid 64 | **untested**, ~96 GPU-hours |
| D `PV2_0091`'s owed sweep | **untested**, cheap |

The honest position, stated plainly: **no route to 0.9 has been demonstrated on this
contract, and eight have been closed by measurement.** The two that remain are a grid
change and a sweep that closes an argued-but-unmeasured claim.
