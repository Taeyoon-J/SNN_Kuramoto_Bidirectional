# SW0202 finite Adam, original QCE

Tests whether retaining finite Adam steps, including uphill minibatch steps,
unblocks useful learning. QCE, model, data, live64 credit, clipping and production
readout stay fixed. Same real TRAIN8 model/Adam/RNG and two new B4 proposals as
the frozen199 guarded control. No count loss or full512 credit is introduced.

Nonfinite proposals restore the full transaction. History records finite-step
application and the previous descent test separately; actual Adam steps must
match applied steps. Acceptance of an update alone is not segmentation success.
Registered extension criteria still require actual fixed16 mask improvements.
See DESIGN_KO.md. No performance claim before actual GPU training/evaluation.

2026-10-18:14 local and14 Linux checks passed, including real nonempty Adam
uphill-step retention, nonfinite loss/rollout/parameter rollback and checkpoint/
RNG preservation. All Python ASTs and2 local/Linux CLI entrypoints passed.
Actual CPU model/Adam/RNG/frozen-control admission completed, source Adam steps2.
Supervisor2032092 registered for all four GPUs; no GPU training start is claimed
at registration. Actual raw minibatch descent and finite-step flags will remain
distinct in reports and final audit.
