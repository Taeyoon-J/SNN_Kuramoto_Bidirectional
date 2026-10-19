# SW0230 COMPLETE: relative separation rejected

Fixed16/three-seed .852402/.624603/.711310 vs matched SW0223 .859634/.645611/.706340.
FG paired95% CI is strictly negative; candidate gates failed. All144 masks independently audited.
No full320 or70k expansion; full320 incumbent unchanged. See RESULT_KO.md.

Earlier preparation notes follow.

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
