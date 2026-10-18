# SW0207 learned foreground before object grouping

Modern197 seed2 predicts too much BG as FG;205 short adaptation does not fix it,
and206 pair-relation calibration is ineffective. Freeze original197 seed2 core,
CNN and product-Q. Learn a225-parameter12->16->1 Tanh node classifier from actual
four component spike rates, membrane means and membrane standard deviations over
steps512..1023. No RGB/gamma/position/count feature or validation GT is supplied.

TRAIN8:4096 positive nodes sampled object-class-uniformly and4096 BG nodes per
image. Train-sample mean/std normalization (floor1e-4, clip5), CPU full-batch
Adam1e-3 for200 steps, seed207002. Fixed sigmoid threshold.5, no coverage target.
Predict all4096 nodes; induce the original Q/A partition on this new support.
This is BEFORE grouping, unlike0031 posthoc membrane-mask AND or0023 oracle
post-filtering, which cannot undo already merged object IDs.0067's failed fixed
.22 coverage heuristic is not reused. Old0023 single aggregate cues were weak on
an early native16 model; modern multi-component supervised separability is open.

Compare exact original197 masks on fixed16 development offsets0,2,...30 and193
RNG seeds. Store new probabilities and verify labels have precisely thresholded
support. Freeze classifier/masks before GT scoring. Require FG gain>=.01 with
positive paired CI lower bound and nonnegative FG/object IoU changes. No full320,
three-seed .90 or gate-causality claim from a seed2 mini. No core optimizer step.
No post-failure threshold/width/LR refinement. Shared exclusive GPUs0..3.
