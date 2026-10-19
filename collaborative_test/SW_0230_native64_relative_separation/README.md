# SW0230: source-relative class-center separation

Prepared matched TRAIN64/128-update encoder+projection comparison against SW0223.
Original center weight1 retained; add a squared one-sided source-distance deficit,
normalized by source distance. No new absolute margin or center16 anchoring.
No model performance result yet. Frozen16/three-seed admission and candidate gates
are specified in DESIGN_KO.md. Source targets use TRAIN masks; not unsupervised.

Independent post-score audit is prepared in `completion_audit.py`: committed
encoder/projection/Adam state, immutable teacher/control, original-plus-relative
loss decomposition and all144 masks are checked before independently scoring
the fixed16 cohort and reproducing paired intervals/candidate gates. No extra
model forward or training is required. Four audit tests passed locally and on
Frontier, including invalid CI/NaN rejection and separate FG/balanced gates.
This preparation does not establish improved model performance.
