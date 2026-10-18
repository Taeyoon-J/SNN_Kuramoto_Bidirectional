# SW0209 foreground pair-ARI loss

**COMPLETE, expansion gate failed.** Fixed16/seed0: candidate
.901220/.862279/.736022; matched QCE202 .900148/.859586/.728138;
warm192 .924401/.859896/.727638. FG gain versus matched control+.001072,
CI[-.000057,+.003147]. The tiny mean improvements are not established gains.
No full320/seed expansion or coefficient rescue of this recipe. Warm192 remains
the full320 seed0 reference. Candidate checkpoint and all evidence are preserved.

Actual two new Adam updates applied, both descent; all five families and CNN
learned. The standalone new term reached emitted spikes with norms.005706/.006321.
Its TRAIN surrogate losses fell .249533->.237221 and .207382->.197430, but that
did not establish better hard-mask grouping. Independent boundary/Adam/RNG,
48 masks and metrics audit passed (maximum error2.22e-16).
See [RESULT_KO.md](RESULT_KO.md), results_archive/score_v1/summary.json and
results_archive/score_v1/completion_audit.json. No new70k training or .90 claim.

## Historical registration

RUNNING after23 local and23 Linux tests and actual model/Adam/RNG/native4096
CPU admission. Supervisor2249349 and actual GPU1 worker2249387 verified.
New QCE+foreground-pair-ARI surrogate,
unchanged actual native64 spike readout, warm192 source and matched202 control.
See DESIGN_KO.md. No new70k training or improvement is claimed.

The first Linux preflight found an optional sklearn import in the test only.
Before admission/training it was replaced with the existing independent
project ARI implementation. No package installation, model/loss/data change,
GPU training retry or gate relaxation occurred. The failed test log and the
unstarted failure summary remain under results_archive/preflight_v1_failure/.

Registration and actual process receipts are archived. A read-only waiter
PID2250255 waits for this exact supervisor, then independently checks the
completed model/Adam/RNG transaction, standalone new-loss spike credit,
48 actual masks and all metrics. Actual new-loss spike credit and performance
are not claimed until those completed records exist. All GPUs0..3 are eligible
through the shared exclusive leases; foreign owners are not interrupted.
