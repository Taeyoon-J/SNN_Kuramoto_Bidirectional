# SW_0207_spike_foreground_classifier

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `d14f506ee63ed5752ff90b97bdda131dc8aa5613e6b058f5d830c6d0a212f3f5`

````markdown
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

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `d3bf7ab13d86dc98647d5899f96c3cc196369bf7e4d3d50b8091f20b381f70ee`

````markdown
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

````

</details>
