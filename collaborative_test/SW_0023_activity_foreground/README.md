# SW_0023: inspect activity-derived foreground cues

Status: activity-feature diagnostic and oracle diagnostic bound complete.
No deployable score improvement claimed.

## 쉽게 설명하면

SW_0022의 결과는 물체끼리 묶는 것보다 배경을 물체로 잘못 보는 문제가
크다는 것을 보여줍니다. 그래서 spike가 자주 튀는 patch나 membrane이
많이 변하는 patch가 실제 물체 patch와 구분되는지 먼저 확인합니다.

## Exact diagnostic

- Frozen SW_0003 seed-0 40-epoch checkpoint; fixed validation IDs 1320-1639;
  256 time steps with first 64 discarded; 16x16 patch labels.
- Record spike rate, membrane mean/absolute mean/std/max/range/temporal
  change, and the SW_0022 membrane-border-template similarity for each patch.
- Use GT foreground/background *only after inference* to report means and
  per-image ranking AUC. An AUC of .5 is uninformative; values above .5 mean
  higher feature values tend to indicate foreground, below .5 the reverse.
- This experiment does not create object masks or select a test threshold.
  Any promising cue must be tested separately with the fixed patch metrics.

## Code changes

Only this standalone diagnostic and report. The model, loss, and classifier
are unchanged.

## Feature-separation result

Across all 320 validation images, the GT foreground fraction is .21682.
Per-image AUCs for single actual SNN features were weak: spike rate .41463
(inverse direction .58537), membrane maximum .54145, membrane standard
deviation .51849, membrane mean .49615, and border-template similarity
.38893 (inverse direction .61107). In each case .5 is uninformative.
These are GT-labeled *post-hoc diagnostics*, not prediction scores or
deployable features chosen on the test split. Full eight-feature results are
in `diagnostic320.json`.

No single feature looks strong enough to safely reject the roughly .46 excess
predicted foreground fraction seen in SW_0022. `oracle_bound.py` therefore
measured what grouping quality would remain if the background were known
perfectly. It deliberately applies GT foreground **only after frozen slot
prediction**. Its result must never be reported as actual model performance.

| Diagnostic only | FG-ARI | FG IoU | Object IoU |
|---|---:|---:|---:|
| Existing largest-slot background | .357053 | .209870 | .113992 |
| Existing grouping + oracle foreground | .357053 | .716590 | .312692 |
| All slots + oracle foreground | .357053 | 1.000000 | .384556 |

The oracle background changes neither FG-ARI result: the metric already
evaluates only GT-foreground patches, and relabeling predicted background
outside those patches does not repair which foreground patches were grouped
together. Therefore the **FG-ARI bottleneck is grouping**, not foreground
selection. Better foreground detection can still greatly improve both IoUs.
Future ARI-first work must change the grouping representation or algorithm,
while foreground/background can be optimized as a separate track. This
diagnosis corrects the earlier hypothesis that background selection alone
could close the FG-ARI gap. Full result: `oracle_bound320.json`.
