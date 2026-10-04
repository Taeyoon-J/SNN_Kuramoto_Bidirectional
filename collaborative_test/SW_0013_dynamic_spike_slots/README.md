# SW_0013: spike pattern adaptive slots

Status: completed, mixed result. No model retraining and no production classifier change.

## What this tested

The existing connected-component classifier produced too many masks: the
40-epoch SW_0011 checkpoint produced 77.03 object groups/image at its
component-product threshold 0.50, while the ground truth had 6.20.
The user proposed making random slots from each patch's spike-history vector,
assigning similar patches, updating the slot vector, and spawning new slots
until every pattern is assigned. This test changes **only the readout**.

For each of 256 patches, the actual 256-step spike history is trimmed to
steps 64–255, centered and L2-normalized. Initial slots are randomly sampled
from observed nonconstant spike vectors rather than arbitrary random
directions; that guarantees that each slot starts with at least one valid
member and the algorithm terminates. Unassigned patches join the nearest slot
when cosine similarity exceeds the threshold. The slot centroid updates from
assigned members; if no more patches join, a new slot is seeded from a
remaining patch. Constant/no-spike traces are background. The largest slot is
also assigned background. No GT object count is used in prediction.

Same 40-epoch SW_0011 checkpoint, same validation IDs 1320–1639, same patch
metric implementation. Assignment seed 0 in the 320-image validation;
64-image pilot also screened seeds 0, 1, 2. The full validation screened
thresholds 0.3, 0.5, 0.7 and initial slots 1, 3, 6.

| 320-image validation, seed 0 | FG-ARI ↑ | FG IoU ↑ | matched object IoU ↑ | predicted groups/image | exact count |
|---|---:|---:|---:|---:|---:|
| Existing SW_0011 component-product, threshold .50 | .179760 | .271468 | .295250 | 77.03 | 0% |
| Adaptive slots, threshold .70, 6 initial | .247376 | .248332 | .204032 | 17.37 | 0% |
| Adaptive slots, threshold .30, 1 initial | .087132 | .319554 | .158197 | 10.09 | see `validation320.json` |
| Ground truth | — | — | — | 6.20 | — |

The proposal **helps ARI and sharply reduces fragmentation**, but worsens
the two IoUs at the ARI-best setting. Lower thresholds reduce count further
but reduce ARI and object IoU. The exact object count still does not match for
any of the 320 images at the ARI-best setting. Therefore this is not an
all-metric improvement and has not replaced the production classifier.

The likely remaining issue is that an arbitrary largest-slot background rule
and threshold-based births do not identify foreground or stop near the true
object count reliably. This is a hypothesis, not a proven cause. A next
diagnostic should separate foreground/background errors from overfragmented
foreground groups without using GT at inference.

## Files and changes

- `evaluate.py`: new standalone classifier-only validation script; no
  `S2NetCore`, training loss, or evaluation metric changes.
- `pilot64.json`: first 64 validation images, 3 assignment seeds.
- `validation320.json`: all 320 validation images, assignment seed 0.
- Checkpoint: `/Data0/kevinswk/patch_v2_sw/trained_models/SW_0011_graph_teacher_40ep_seed0/core.pt`.
- Server outputs/logs: `/Data0/kevinswk/patch_v2_sw/trained_models/SW_0013_dynamic_slots/`.

`patch_sw`'s `edge_membrane_separation_loss` is a separate future training
ablation. It needs original RGB images aligned with gamma/patch indices,
whereas current core training batches load only cached gamma. It was not
silently added to this classifier test.
