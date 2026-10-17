# SW0193 native64 before/after mini screen

Compare the exact mapped SW0186 native64 model with the successfully admitted
SW0192 model (eight TRAIN images/two proposals) on predetermined32 development
images. Same inference RNG and production spike-derived readout. Score only
after both masks are frozen; independently verify actual metrics and publish
aggregates only. This is a seed0 mini screen, not a three-seed FG.90 result.

The screen is running on GPU2 after SW0192 successfully completed. Local/server
seven tests include real native64 mask conversion and foreground merge, paired
prediction resume/hash checks, actual production/independent metric parity and
GT-access ordering. Exact parent checkpoint/model/Adam and baseline CPU admission
passed before GPU leasing. Current parent PID1836851, worker PID1837070.

The original inference attempt failed because the mask conversion function was
imported from spike_classifier instead of evaluation. Original state/log/source
and CPU admission were preserved under failure_v1 on the server; the failed queue
state/log are also archived locally. The import and a real conversion regression
test were corrected before seven server tests passed and inference resumed.
No training was repeated and no prediction rows or validation GT were consumed
by that failed attempt. The initial recovery launcher observed the same parent
before its CPU admission wrote state; the same PID was subsequently verified,
with no duplicate relaunch. See start_v2 and cpu_v2 for evidence. The canonical
cpu_v1/admission.json contains the current exact model binding; its fingerprint
identifies the fixed source version. The earlier six-test receipt is retained
as the pre-fix record, not evidence for the corrected readout path.
