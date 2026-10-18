Native64 original QCE plus signed-component QCE. Ready for preflight; no performance claim. See DESIGN_KO.md.

2026-10-18:18 local and18 Linux tests, AST checks, two Linux CLI checks
and actual CPU model/Adam/RNG admission passed. All four GPUs are eligible.
Supervisor2049352 registered; no actual GPU start or improved score claimed yet.
Control is the completed finite-Adam202 candidate, not the guarded199 control.
The legacy report completion status mentions finite_Adam because the step policy
is retained; experiment bindings and fingerprint specify the new component loss.

Actual GPU1 worker2052928 started after203 finished. NVIDIA PID, initial
model, Adam moments and CPU/CUDA RNG restoration independently verified.
Start evidence is under results_archive/started_v1. No evaluation result yet.

Completed, independently audited: fixed16 development images, seed0.

| Arm | FG-ARI | Foreground IoU | Object IoU |
| --- | ---: | ---: | ---: |
| component credit | .9005389599 | .8605773316 | .7276216788 |
| matched finite-Adam202 control | .9001477304 | .8595864826 | .7281376470 |
| frozen TRAIN8 warm | .9244014538 | .8598964167 | .7276384095 |

Two new finite steps applied, actual Adam step4, CNN/model changed. Actual48
masks, checkpoint, Adam/history and inherited model/RNG were independently
verified. FG difference versus202 +.0003912294, CI[-.0009170428,.0018393708];
object IoU slightly lower. Warm FG difference -.0238624940. Extension gate
failed; recipe closed. No loss-weight/LR micro-rescue. This does not establish
that component learning is generally useless or prove gate contribution.
No full320/three-seed .90 result. Best full-evaluation incumbent is unchanged.
