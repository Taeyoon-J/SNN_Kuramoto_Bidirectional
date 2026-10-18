# SW0195 native64 generalization diagnostic

Read-only comparison of the completed TRAIN8 and TRAIN64 models on16 fixed TRAIN
images, reusing and independently rechecking the32 development predictions.
No optimizer updates, architecture changes, or threshold tuning.

Completed and independently rescored actual TRAIN16 and development32 masks.
The source and final model hashes, cohort bindings, paired means and confidence
intervals were checked; maximum production/independent metric error2.22e-16.
No optimizer updates occurred during this diagnostic.

| Cohort | TRAIN8 FG-ARI / FIoU / object IoU | TRAIN64 FG-ARI / FIoU / object IoU |
| --- | --- | --- |
| TRAIN16 | .915137 / .838085 / .752900 | .913519 / .827964 / .738010 |
| Development32 | .918586 / .832867 / .733686 | .900166 / .830834 / .731658 |

Both mean FG scores decreased. TRAIN delta95%CI[-.005617,.001131]; development
delta95%CI[-.047440,.001273]. The registered positive-TRAIN/negative-development
signal was false. This does not prove overfitting absent; neither FG change is
resolved from zero, and only seed0/small cohorts were measured. It also does not
establish that a fixed computation structure caused the changes. This unchanged
QCE continuation showed no gain and is not selected as an improved recipe.

Public audit and aggregate summary: results_archive/score_v1/completion_audit.json
and summary.json. Actual imagewise vectors remain local and ignored.

Known TRAIN masks are already present in the model factory. Prediction does not
use them; it only freezes the same production spike/Q masks before scoring.
Only aggregate results may be shared; imagewise GT-derived scores remain private.

See DESIGN_KO.md for fixed cohorts, uncertainty estimates, and interpretation.
