# SW0208: full320 confirmation

**COMPLETE.** Native64 fixed320 seed0: warm192
**.877709/.791224/.696284**; component204 **.876242/.790093/.693262**.
Metric order: FG-ARI/foregroundIoU/matched-objectIoU. Neither reaches .90 on
all320. The three paired204-minus-warm95%CIs include zero. Keep warm192 as the
slightly higher-mean seed0 reference; no statistically established recipe gain.
See [RESULT_KO.md](RESULT_KO.md) and
`results_archive/completed_v1/score_v1/summary.json`.

Comparable Slot three-seed mean: .786902/.198201/.195203; Slot best FG seed2
.878278. Our two arms are seed0, supervised and inherited70k lineages, with
only8/16 additional native64 TRAIN images. No new70k training, independent
three-seed native64 win, unsupervised improvement or .90 goal is established.
All640 masks and all48 overlapping mini masks were checked; production and
independent metric implementations agree within2.22e-16. All queues ended.

## Historical registration

Registered on Frontier after11 local and11 Linux tests, AST/CLI checks and
actual checkpoint/full320 canonical RGB CPU admission. See DESIGN_KO.md.
This experiment compares actual204 and stronger warm192 on all320 images,
without further training. Seed0 only; it cannot establish a three-seed .90 claim.

Four exclusive-GPU shard supervisors2115329/2115330/2115331/2115332
were verified alive at registration. Each owns80 fixed images and evaluates
both models. All GPUs0..3 are eligible; foreign processes retain priority.
Completed rows are immutable and resume after an ownership interruption.
The last completed shard triggers GT-last scoring and matched full320 Slot
comparison. Registration artifacts: results_archive/registration_receipt.json
and results_archive/registration_v1/admission.json. No new70k training has begun.
