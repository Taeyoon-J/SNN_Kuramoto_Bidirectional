# SW0208: full320 confirmation

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
