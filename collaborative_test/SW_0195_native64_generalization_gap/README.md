# SW0195 native64 generalization diagnostic

Read-only comparison of the completed TRAIN8 and TRAIN64 models on16 fixed TRAIN
images, reusing and independently rechecking the32 development predictions.
No optimizer updates, architecture changes, or threshold tuning.

Prepared; local tests passed. Server admission and queue registration are tracked
in results_archive. This document does not claim that GPU inference completed.
The queue waits for SW0194 completion before admitting its final checkpoint,
then requests an exclusively idle GPU. Saved prediction rows resume after yielding
to another owner. Full model/Adam retraining is never part of this diagnostic.

Known TRAIN masks are already present in the model factory. Prediction does not
use them; it only freezes the same production spike/Q masks before scoring.
Only aggregate results may be shared; imagewise GT-derived scores remain private.

See DESIGN_KO.md for fixed cohorts, uncertainty estimates, and interpretation.
