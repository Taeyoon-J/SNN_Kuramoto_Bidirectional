# SW_0004 — graph-to-membrane synchrony pilot

Status: code smoke-tested; training pending.

## Motivation

The phase-only baseline leaves all dendritic and membrane parameters without
gradients. A diagnostic on validation IDs 1320–1351 found the learned graph's
mean edge weight to be 0.254288 for pairs in the same true object, versus
0.021720 for pairs in different foreground objects. Ground truth was used only
to inspect the graph, not to train it. Direct spectral clustering of the graph
gave FG-ARI 0.305600 / foreground IoU 0.227293 / object IoU 0.160476 on
these 32 images, but this **graph-only readout is not a valid goal result**.

## Change

Retain the baseline phase PLV loss, core computation, and 40-epoch baseline
settings. Add an optional loss that aligns *centered membrane-trace synchrony*
with the per-image learned graph. The graph is detached as a teacher; its
off-diagonal row-normalized edge weights form a distribution and each
oscillator's membrane synchrony forms a softmax distribution. Their KL
divergence is weighted by 0.1. This reaches the dendritic and membrane path
while the original phase PLV objective still trains theta. No CLEVR masks are
used during training. The pilot is 10 epochs and therefore not a matched-epoch
causal comparison to the 40-epoch baseline.

Source change: standalone `graph_teacher_synchrony_loss` in
`snn_kuramoto_bidirectional/loss_function.py`; optional
`--graph-teacher-weight` and `--graph-teacher-temperature` in
`training/train_s2net_core.py`, both leaving original behavior unchanged when
the weight is zero. `test_graph_teacher_loss.py` confirms finite nonzero
membrane gradients and no gradient into the detached teacher.

Exact server command is in `run.sh`. Evaluate both spike and original
spatial-components readouts on validation IDs 1320–1639 using the same
`evaluate_fixed_split.py` contract. Do not select on the reference test split.
