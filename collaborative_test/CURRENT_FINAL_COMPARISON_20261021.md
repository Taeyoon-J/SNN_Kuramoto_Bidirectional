# Current user-authorized goal: final four-model comparison only

Open-ended architecture/loss improvement is stopped. Do not revive old
exploratory queues. Active work is SW0249, documented in
`SW_0249_four_model_35000/DESIGN_KO.md`.

- Models240,237,244-within16,245-within16; seeds0/1/2; twelve final jobs.
- Same35,000 distinct TRAIN images, ten passes,21,880 updates,350,000 exposures.
- Same fresh optimizer/LRs, whole encoder and actual sensory projection trainable;
  other core/readout remain fixed. Preserve each model's existing loss.
- Original244 seed1 exists. User explicitly approved reproducing original244
  seed0/2 with the unchanged128-update recipe before common35k training.
- Same fixed320/native64 evaluation, three metrics, all final protocols before GT.
- Shared GPUs allowed when sufficient memory remains; never automatically yield
  on another process's arrival; never signal other users' processes.
- Quiet periodic SSH observation, beginning/end reports, preserve checkpoints.
- Final scoring/audit/report/publication completes this goal; no followup search.

At initial preparation, nine local CPU tests passed, including exact35k coverage,
last8-image batch, source routing, CUDA visibility, and actual Adam checkpoint
roundtrip. Frontier SSH subsequently returned Broken pipe/banner timeout from
both WSL and Windows. Initial server deployment was not confirmed and the new
twelve-job queue must not be claimed as registered until server receipts exist.
The user's fresh connection request is pending. A local quiet45-second observer
is running; server preflight/queue registration will resume on restored access.

Connection was subsequently restored. The nine frozen research files were
deployed; ten local tests and ten Linux tests passed. Actual source-checkpoint
and CPU-gradient admissions are running before all twelve GPU queues are
released. Do not equate the deployment/preflight process with GPU training.

Old goal's benchmark endpoint was achieved by supervised240, independently
audited and recorded/pushed. Its history does not establish gate causality or
unsupervised superiority. The active task is now this bounded comparison.
