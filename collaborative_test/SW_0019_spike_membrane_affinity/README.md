# SW_0019: spike + membrane affinity classifier

Status: 64-image pilot complete, mixed result. No classifier replacement.

## Question

SW_0016 increased membrane boundary contrast but weakened spike grouping. Can
using both actual spike and membrane temporal histories in the readout improve
all three patch metrics, without changing or retraining the core?

## Controlled settings

- Use the completed SW_0011 seed-0, 40-epoch graph-teacher checkpoint, not
  SW_0016. This isolates a classifier change from the RGB-edge training change.
- Validation-only CLEVR HDF5 IDs 1320-1383 for the pilot; no reference-test
  images and no GT masks in prediction. All evaluation remains at 16x16 patches.
- Roll out 256 steps, discard first 64; use per-component actual spike and
  membrane traces. Center each trace in time, cosine-correlate oscillator
  pairs, clamp negative similarities to zero, and multiply across four
  components. Form `(1-membrane_share) * spike_affinity +
  membrane_share * membrane_affinity`; threshold and take connected components.
  Largest component is assigned background. Compare membrane shares
  0/.25/.5/.75/1 and thresholds .10/.25/.50/.70/.80/.90/.95.
- `membrane_share=0` is the existing pure-spike readout and acts as a direct
  control. If the pilot yields a plausible all-metric gain, recheck on all
  fixed 320 validation images before any reference-test claim.

## Code changes

Only this new evaluation script and report. No core, loss, training, or
production-classifier code is changed.

## Pilot result and decision

On the same first 64 fixed validation images, the pure-spike control
(`membrane_share=0`, threshold .50) scored FG-ARI .157762, foreground IoU
.257242, and matched-object IoU .296283, with 76.73 predicted groups/image.

| Readout | FG-ARI | Foreground IoU | Matched-object IoU | Groups/image |
|---|---:|---:|---:|---:|
| Pure spike, threshold .50 | .157762 | .257242 | .296283 | 76.73 |
| 25% membrane, threshold .50 | .188753 | .268401 | .259725 | 42.64 |
| Pure membrane, threshold .90 | .201389 | .236243 | .259767 | 59.42 |

The 25% membrane mixture improved FG-ARI and foreground IoU but reduced
matched-object IoU. Pure membrane at threshold .90 gave the highest FG-ARI
in the sweep but lowered both IoUs. No all-metric improvement was observed,
so the production classifier is unchanged and a 320-image confirmation of
this exact sweep is not warranted yet. The 64-image pilot was used for
selection only; no reference-test claim or three-seed result is made.
All 35 combinations are in `pilot64.json`.
