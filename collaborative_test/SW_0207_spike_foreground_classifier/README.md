Learned native64 foreground support before induced grouping; core and Q remain frozen. COMPLETE; not adopted.

Eleven local and eleven Linux tests, AST checks, queue CLI and real CPU model/
TRAIN-node-plan admission passed before scheduling. Tests verify settled trace
axes, train-only normalization, actual225-parameter Adam learning, immutable
cache/head bindings, probabilities matching predicted support, and the crucial
BG-bridge distinction: removing a BG node before grouping can split objects;
post-filtering the same already-merged IDs cannot. This is a distinct experiment
from old0031/0067 heuristics. No core/CNN/Q/graph update or mask-area prior.

Supervisor2107852 targets all physical GPUs0..3. Actual GPU1 worker2107988 began;
NVIDIA PID and spike/membrane TRAIN-node caches verified. The source is the
original independent197 seed2 baseline, not the negative205 candidate or206 head.
Prediction uses no GT or object count. No validation outcome claimed at start.

Completed fixed16 seed2: original .827671/.444883/.689239; learned foreground
.773808/.616158/.659627 (FG-ARI/foregroundIoU/objectIoU). FG-ARI difference
-.053863, paired95%CI[-.107010,-.015820]. ForegroundIoU improves+.171275,
but instances fragment more. This recipe is closed without threshold/LR rescue.

The independent completion audit refit all200 actual Adam steps from frozen
TRAIN nodes with zero parameter discrepancy and recomputed every old/new mask
metric (maximum error1.11e-16). No core/CNN optimization occurred.
Summary SHA bd517e1b5a8f45cf481c9bd38a210e6e584c7a5a3df37f5ecf8218d7d79c892a.

Posthoc, zero-forward diagnosis: TRAIN sampled balanced foreground accuracy
.927826; validation exact object-macro/BG balanced accuracy .880745, versus
original .788263. Sampling differs, so the .047081 gap does not isolate pure
overfitting. Background false positives fall .321240->.105527, but foreground
misses rise .033341->.126684 and same-object pair splits .122598->.198906.
Splits also rise on common retained GT foreground, .100711->.137764. Better
binary foreground classification therefore did not improve instance grouping.
These are posthoc diagnoses, not new validation-selected predictor scores.

Authoritative artifacts: results_archive/seed2/score_v1/{summary,completion_audit}.json
and results_archive/seed2/diagnosis_v1/summary.json. No full320 or three-seed
native64 .90 result is established.
