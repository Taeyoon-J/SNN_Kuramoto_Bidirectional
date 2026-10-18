# SW0204 native64 per-component learning credit

Hypothesis: positive-product Q=product_d ReLU(rho_d) gates off all pair learning
when one component correlation is negative. QCE on signed component correlations
can provide credit before that hard product without bypassing emitted spikes.
Objective = original class-balanced QCE(Q) + mean of the four class-balanced
QCE(rho_d), using the same temperature .1. No new loss weight or target margin.
Unlike SW0175 this uses all class-balanced query/prototype comparisons, not
minimax edges of a GT route forest. Unlike SW0199 it acts before the product,
not on the calibrated final Q. This is supervised mask training.

Only objective changes versus SW0202 finite-Adam control. Same actual SW0192
TRAIN8 model/Adam/RNG, two B4 proposals on indices8..15, live64, original graph,
encoder, gating structure and production inference. Finite uphill steps retained,
nonfinite proposals rolled back. Same sixteen development IDs, GT after masks.
Require paired FG improvement >=.01 with bootstrap lower bound >0 against202,
nonnegative FG versus warm, nonnegative object IoU against both, applied update
and changed CNN. Mini success is not full320/three-seed .90 or causal proof of
gating usefulness. No LR/weight micro-rescue after a failed screen.
