# SW0235: fixed232 convergence models, secondary64 validation

No additional training. Actual accepted512 checkpoints; retain the original16
gate failure, reuse16 masks and infer48 additional images per seed. Require exact
mask and frequency replay. Compare against the same64 baseline and actual128
control. All three candidate/control protocols precede GT. This is reused
validation, not untouched test or a fresh70000-image training result.

Completed and independently audited: [results](RESULT_KO.md). Fixed64 three-seed
mean .893972/.625892/.680334; registered FG and balanced gates passed. Expand the
same checkpoint in SW0237, preserving the original16 failure and full320 incumbent.
