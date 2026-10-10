# Evidence required after the SW0134 pilot

This document records prospective follow-up requirements. It does not change
the running seed1 pilot, its 4096-update endpoint, its three arms, or its
registered FG-ARI expansion test. These follow-ups are not running or queued.
The pilot can establish downstream spike utility and joint-learning utility;
it cannot establish every individual gate's contribution or data scaling.

## Decision after the fixed pilot endpoint

Freeze all four complete 320-image predictions before opening masks. Validate
the checkpoint, optimizer steps, actual training orders and prediction hashes.
Apply the original paired FG-ARI expansion test against source97, gate_joint
and actual_frozen. Report all three metrics even when expansion fails.

If expansion passes, repeat the unchanged recipe on seeds0/2 before promoting
it. A seed1 gain alone cannot replace the three-seed incumbent. If it fails,
use the already registered TRAIN-only assignment/RGB diagnostic to identify
collapse or representation failure; do not rescue the result with the
secondary QCC readout or a new evaluation contract.

## Individual contributions need controlled retraining

Inference-only removal is a fragility diagnostic, not sufficient proof that
each mechanism improves a model trained with an alternative. Train matched
controls from identical seed-specific starting states, with the same image
orders, updates, binder, decoder and evaluation endpoint. Register each exact
intervention and its initialization/gradient checks before launching it.

| Mechanism | Controlled question | Interpretation limit |
|---|---|---|
| Kuramoto coupling | Compare the full model with coupling removed while preserving the uncoupled phase evolution and downstream modules. | Tests coupling, not the existence of any rhythmic input. |
| Sinusoidal modulation | Replace only the selected delayed modulation by a TRAIN-derived constant; retain the phase carrier and the remaining dynamics. | Name every altered carrier/hold/emission path; a blanket constant replacement confounds them. |
| Dendritic temporal integration | Remove the previous dendritic state contribution while retaining the same learned projection and branch reduction. | Tests temporal state, not the entire dendritic computation. |
| Membrane retention | Remove previous membrane retention while keeping the threshold, event function and emission modulation. | Preserve and report any native hold predicate; do not silently remove two mechanisms. |
| Thresholded event | Compare native threshold events with a separately specified continuous-output control under the same downstream budget. | This changes signal amplitude/distribution; report activity and a TRAIN-only amplitude-matched diagnostic. |

Each control needs a code-level equation, not just a name such as "gate off".
Log activation activity, gradient connectivity and assignment occupancy on
fixed TRAIN images. A nonzero gradient alone is not usefulness evidence.
Evaluate complete models on the fixed patch contract, reporting per-seed
scores and paired image differences. Three-seed replication is required for
the paper's main contribution claims. A control that matches or improves the
full model leaves that mechanism's claimed benefit unproved and directs the
next architecture/loss change.

## Data scaling must start before exposure to the large pool

Do not reuse source97 as a clean small-data initialization: it has already
seen70k. Reconstruct one reproducible clean initialization per seed and clone
its exact weights, buffers and optimizer state across nested2500/10k/70k
conditions. Disclose any encoder pretraining. Jointly train encoder, graph,
core, integration and binder/decoder; keep the selected recipe unchanged.
Calibrate any TRAIN-derived constants on the common small prefix, never on
the large pool before the small-data arm is trained.

Use two distinct comparisons:

- Equal compute:4375 batch16 updates,70000 image presentations per condition.
  This measures diversity at fixed compute, with28/7/1 passes respectively.
- Equal repetition:ten complete passes, retaining the final partial batch.
  This gives1570/6250/43750 updates and25000/100000/700000 presentations.
  The per-image loss must be normalized for the partial batch. Ten passes
  are a registered budget, not a claim of convergence.

Use seeds0/1/2, the same fixed320 evaluation IDs and the same mask/readout
contract. Persist actual image IDs, initialization hashes and optimizer
counts. Report differences for2500->10k and10k->70k with their uncertainty;
do not infer a scaling trend from a single favorable seed. If the large-data
TRAIN loss is still falling, a longer-training comparison needs its own
prospective budget and no validation-based endpoint selection.

The final70k model must strictly exceed the comparable own70k Slot Attention
three-seed mean on FG-ARI, foreground IoU and matched-object IoU. Keep the
reserved independent set unread until the final recipe is frozen. Scores
from a different renderer, oracle allowed-label evaluation, or a different
patch grid cannot satisfy this objective.
