# SW0187 ? joint encoder, unique1024 versus repeated512

Registered: three training supervisors, three prediction supervisors, GT-last
scoring and an independent final auditor. All eight parents were verified live.
At that observation all training runs were waiting for exclusive idle GPUs;
no actual GPU training or new performance score is claimed yet.

The improved SW0186 recipe keeps its original RGB CNN trainable. Both arms share
one contemporaneous32-proposal prefix, then fork exact model/Adam/RNG state.
Control repeats512 images; candidate uses1024 distinct images at the same
logical64-proposal/1024-exposure budget. Shared computation costs96 actual
proposals per seed. This tests unique data in this additional training stage,
not whether images were unseen by the pretrained Source097 foundation.

Local41 tests and six actual CLI checks passed before deployment. Server45 tests,
all six actual CLIs, all three five-family factories and initial gamma parity,
the canonical TRAIN1024 cache and all three actual Slot archives passed before
registration. Initial tensors also matched the archived SW0186 initialization.
Sources/queues are frozen while live. No datasets, checkpoint weights or keys
were transferred. The original preregistration README is archived in cpu_v1.

The primary scaling gate compares candidate versus its contemporaneous control,
with a positive paired-image FG confidence bound. Slot70k, SW0186, SW0181,
conditional masks and both IoUs are reported separately. Fresh70k, FG.87,
unsupervised improvement and gate causality remain outstanding.

See [the registered design](DESIGN_KO.md), `results_archive/cpu_v1` and
`results_archive/start_v1`. Best supervised FG remains SW0186; unsupervised SW0165.
