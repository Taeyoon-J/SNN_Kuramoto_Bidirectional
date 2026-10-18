# SW0213 single center-head convergence check

211/212 failed with one-center fallback even on TRAIN. Offset learning was
present (TRAIN error1.5185 versus zero-offset4.2434); center detection was not.
Continue the exact parent128 head/Adam/RNG to one fixed512 endpoint, using cached
TRAIN32/native64 features. Same loss, LR, width and readout thresholds. No core
forwards or core/encoder updates. See DESIGN_KO.md.

Primary512 preserves original spike foreground. Learned foreground and two
frozen-input controls are secondary endpoints. DEV GT opens after all masks
validate. Source211 and its failed128 verdict are preserved. This is a separate
posthoc-motivated convergence experiment, not a relabeling of211 success.

Local continuation tests verify exact parity with uninterrupted Adam, original
parent bytes/history preservation, resumed endpoint identity and changed-data
rejection. The initial local runtime-guard test caught a missing deterministic
cuDNN context; it was corrected before deployment and all five local tests pass.
Linux tests, actual parent/native64 CPU gradient admission and queue receipts
will be recorded under results_archive before model execution.

Five local/five Linux tests, actual parent128/native64 CPU gradient and saved
software/math-setting admission passed before launch. Supervisor2347302,
GPU1 worker2347432 completed the fixed512 continuation. Actual Adam512 and
original128 history/source bytes plus96 mask endpoints independently audited.
Primary .589666/.445474/.325129 versus original .788414/.445474/.674335:
no gain. TRAIN center recall98.77%, fallback0; extending budget fixed center
detection but not grouping. Count exact4/16 is retained as a limited pilot.
No further budget/threshold sweep. See RESULT_KO.md. Full320 incumbent unchanged.
