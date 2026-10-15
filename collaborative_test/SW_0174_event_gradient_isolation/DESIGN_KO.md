# SW0174 event-gradient isolation audit

SW0173 found that a full-horizon virtual Adam direction did not improve the
registered Q-only route loss on its fixed TRAIN cohort and transfer images.
SW0174 measures one narrow causal intervention: detach the output of the
existing `SW0134.rollout.act_fun_adp` call while leaving its input, forward
value, membrane update, recurrent reset, graph, encoder, objective, and readout
unchanged. The raw event output is detached at every step; the final spike
trace is not substituted or rewritten.

The run is admitted only when the frozen SW0173 report has its registered
SHA256, complete status, exact SW0173 implementation fingerprint, and the same
source/input proof as current CPU admission. A fresh same-RNG B1 source forward
compares native and detached-event traces, gamma, Q, A, projection, and both
registered source-F/native-QCC partitions before the frozen SW0173 audit is
reused. SW0173's audit runs inside the scoped intervention and keeps its exact
TRAIN cohort (0–3), transfer set (256–259), horizons, fresh Adam direction,
scales, routes, metrics, and no-update behavior. Its original report is never
overwritten.

The primary check is the full1024 direction at scale 1: native route loss must
decrease on cohort and transfer, while transfer source-F and native-QCC
foreground ARI must each be at least the native source baseline and the frozen
SW0173 full1024 primary result. Scale 1/16 is diagnostic only. This audit does
not train, tune, add a loss, or inspect validation/reserve data. A positive
result only motivates a separately registered short pilot.
