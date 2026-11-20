# PV2_0085 — motion as a PLV target does not help, and the pilot is too weak to be the platform

**completed. Closes the route `PROPOSAL_0.9.md` named, with two caveats that matter more
than the headline.** Parent `PV2_0084`. **MOVi-A contract (`movi-1`) — nothing here is
comparable with contract v1, and the test result 0.7603 is untouched.**

## What was built and run

Motion entered as the **external target of a loss term**, which is what the proposal
specified: the cosine affinity of per-patch RGB change between frame t and t+6, masked
where a patch barely moves. The mechanism mirrors `slot_reconstruction_loss`, whose target
is pooled RGB — external to the model, so it cannot be gamed by changing the features.

| | |
| --- | --- |
| patches above the magnitude gate | 23.3% |
| patch pairs surviving the mask | 6.0% |
| affinity on kept pairs | mean 0.505, **std 0.376** |

The spread of 0.376 says the target is discriminative rather than uniform. Three arms,
identical but for the term's weight, 1,000 MOVi-A scenes, 40 epochs, seed 0.

## Result

| weight | final loss | motion term | fg_ari | fg_iou | obj_iou | pred_fg |
| --- | --- | --- | --- | --- | --- | --- |
| **0 (control)** | 28.25 -> **23.48** | — | 0.4181 | 0.1350 | 0.2682 | **0.5847** |
| 10 | 30.22 -> 25.62 | 0.2033 | 0.4150 | 0.1349 | 0.2566 | 0.5916 |
| 100 | 46.97 -> 43.14 | 0.1810 | 0.5090 | 0.1358 | 0.3455 | 0.6145 |

The term worked — it falls from 0.203 to 0.181 as its weight rises, so the model really
did pull its PLV toward the motion affinity. But **predicted foreground is 0.58-0.61
against a target of 0.1061**, five times too high in every arm including the control, so
by the criterion used since `PV2_0064` none of these rows is interpretable.

Recovering calibration with the population membrane threshold (`PV2_0023`'s rescue,
inference only):

| threshold | control | motion w=100 |
| --- | --- | --- |
| population k=1.0 | **0.3042** (pred_fg 0.179) | **0.0823** (pred_fg 0.142) |
| population k=2.0 | 0.0734 (pred_fg 0.137) | 0.0196 |
| population k=3.0 | 0.0000 | 0.0000 |

**In the calibrated regime motion is far worse than the control** — 0.0823 against
0.3042. So the answer is negative either way.

## Two caveats, and the second is the important one

**The term does not convert the signal.** Motion's between-object margin is 1.4x
standardised appearance's (`PV2_0084`), so the information is in the data, but pulling the
PLV toward it lowers the score. That is the **sixth** time a better descriptor or target
failed to move the metric — after `PV2_0058`, `PV2_0062`, `PV2_0064`, `PV2_0078` and
`PV2_0084`.

**But the pilot is too weak to be the platform.** The control itself reaches only 0.3042
calibrated, against contract v1's 0.7250, because its gamma is **3-channel RGB patch
means** rather than the trained 8-channel encoder behind contract v1, and the operating
point — `membrane_vth` 0.06 and the rest — was tuned for that encoder's distribution. So
this compares two poorly-calibrated models, and a negative result on it is weaker evidence
against motion than the numbers suggest.

**What a fair test needs:** a gamma initialiser trained on MOVi-A, the way contract v1's
8-channel gamma was produced, bringing the control to a comparable level before the term
is judged. That is a day of work, not an afternoon, and it is the honest next step for
anyone continuing.

## Two protections the pilot exposed, both now fixed

`evaluate_model.py` refused the MOVi targets because their `contract_version` is not 1 —
**the guard working as designed.** Rather than weaken it, `--expect-contract` now makes the
expected version explicit, defaulting to 1 so every existing command keeps the same
protection and a different contract has to be named.

More serious: the report dict wrote `"contract_version": 1` **literally**, so a MOVi-A run
would have been recorded as a contract-v1 result — exactly the confusion the guard exists
to prevent. It now carries the contract actually evaluated. The contract-v1 regression
reproduces 0.7239 / 0.7251 / 0.4951 after both changes.

## Where this leaves part 3

`PROPOSAL_0.9.md`'s route is run and negative, with the caveat that the platform was too
weak for the result to be strong. Thirty routes are closed. The architecture's gap to 0.9
on contract v1 remains model error (`PV2_0082`, `PV2_0083`), the failure remains coherent
object merges, and motion — the one untested source of the missing information — did not
convert through this term on this pilot.
