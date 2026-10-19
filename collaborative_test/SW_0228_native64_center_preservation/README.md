# SW0228

Source-center loss weight16 vs frozen223 weight1; same TRAIN64/128updates/joint encoder+projection.
All3 actual CPU admissions passed,14 Linux tests and3 CLI checks passed.
Three queues wait until each SW0227 seed has received its first GPU allocation.
No GPU training/performance result at this snapshot. See START_KO.md and DESIGN_KO.md.

Post-score independent verification is prepared in `completion_audit.py`.
It checks the three committed encoder/projection/Adam checkpoints, immutable
source teachers and their center16 loss decomposition, matched center1 control,
and all 144 candidate/control/baseline masks before independently recomputing
metrics, paired confidence intervals and candidate gates. It performs no new
model forwards or training. Four audit tests passed locally and on Frontier,
including rejection of altered intervals/nonfinite scores and correct handling
of an IoU regression or failure to improve over center1. This preparation is
not a completed performance result; the full320 incumbent remains unchanged.
