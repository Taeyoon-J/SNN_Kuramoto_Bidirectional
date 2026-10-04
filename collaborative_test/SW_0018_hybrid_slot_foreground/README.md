# SW_0018: Low-spike-synchrony foreground + adaptive slots

Status: completed, negative for all-metric gain. No model retraining or production classifier change.

The peer `patch_v2` branch observed that patches with low average spike
synchrony can indicate foreground on its different CLEVR render. We tested
that insight on **our** SW_0011 checkpoint and fixed validation IDs 1320–1639.
For each image, actual centered spike histories after step 64 form a cosine
matrix. The lowest-mean-synchrony 20%, 30%, 40%, or 50% of patches are
foreground candidates. SW_0013 adaptive slots group patches; patches outside
the candidate set become background. We tested both keeping all candidate
slots and also removing the largest candidate slot as background. The
percentage is a global validation-selected setting, not a per-image GT count.
Threshold/initial-slot pairs were (0.3,3), (0.5,3), (0.5,6), (0.7,6).

| 320-image validation | FG-ARI ↑ | FG IoU ↑ | object IoU ↑ | groups/image |
|---|---:|---:|---:|---:|
| Existing SW_0011 component-product readout | .179760 | .271468 | .295250 | 77.03 |
| SW_0013 slots only, .7/6 | .247376 | .248332 | .204032 | 17.37 |
| Hybrid, bottom 50%, .7/6, keep slots | .212862 | .276301 | .205796 | 17.17 |
| Hybrid, bottom 50%, .5/3, remove largest | .130456 | .291989 | .177966 | 13.02 |

The peer-derived foreground signal can raise foreground IoU over slots alone,
but ARI drops and matched-object IoU remains below the existing readout.
The best foreground-IoU setting has aggregate precision .3983 and recall
.5518, indicating substantial missed foreground. No tested setting is an
all-metric improvement. Do not merge the peer heuristic based only on its
different-dataset result. See `validation320.json` for the full sweep.

Only `evaluate.py` was added. The core, losses, checkpoint, and patch metrics
were unchanged. Ground truth was used only after prediction to score it.
