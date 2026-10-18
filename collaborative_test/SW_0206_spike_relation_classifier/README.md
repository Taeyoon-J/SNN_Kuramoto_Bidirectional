Native64 learned spike-relation classifier; seed2 screening before replication. Pending preflight, no score.

Ten local and ten Linux tests, all source AST checks, queue CLI validation and
actual CPU model/TRAIN relation-plan admission passed before registration.
Supervisor2101577 scans all physical GPUs0..3 using shared exclusive leases.
Actual GPU1 worker2102905 subsequently started. NVIDIA PID and all8 frozen TRAIN
correlation caches verified; source model is original independent197 seed2,
b2c1a56f..., not the negative205 candidate. Classifier is97 parameters, only
signed emitted-spike correlations enter it. No core gradient/optimizer updates.
Actual four-component gather ordering was separately verified on CPU.
No validation score or full320/.90 result is claimed at start.

Completed and independently audited on fixed16 development images, seed2:

| Arm | FG-ARI | Foreground IoU | Object IoU |
| --- | ---: | ---: | ---: |
| original product-Q grouping | .8276711245 | .4448827161 | .6892385263 |
| learned relation classifier | .8292020333 | .4448827161 | .6812687164 |

FG difference+.0015309087, CI[-.0003676247,.0043684974]; object IoU declines.
Gate failed, recipe closed; no width/LR/threshold micro-rescue. The97-parameter
classifier completed200 Adam steps on65536 pairs from8 TRAIN images. TRAIN pair
accuracy89.38% with the symmetrized component product and89.75% with the MLP is
not an image segmentation result. CPU refit exactly reproduced the saved head
parameters (maximum difference0), and actual Adam/moments plus32 frozen old/new
masks independently verified. Core optimizer steps0; foreground support stays
identical. No full320 or three-seed .90/gate-causality claim follows. This does
not rule out other learned readouts or joint core/classifier training.
