# SW0095 — complete70k continuation with classifier-aligned spike loss

Registered following the SW0094 pilot's positive_frozen gain on all3 metrics
against the old-loss continuation. SW0094 full320 confirmation remains pending
at registration; the queue waits for its completion before launching this test.

Three source-model seeds0/1/2: continue each SW0090 epoch1 **whole core**, not
just the graph. Replace absolute component spike affinity in the loss with the
actual prediction classifier's positive-correlation product. Other loss weights,
graph/encoder freezing, spatial priors and classifier remain unchanged.

- One complete additional pass through the same70,000 unique training images.
- 4,375 optimizer updates x16 images, without replacement. Source cores already
  saw one complete70k pass; total downstream exposures are140,000 per seed.
- Native encoder/graph pretraining provenance remains that of SW0090; this is
  continuation rather than a from-scratch or budget-matched ten-pass comparison.
- Shuffle seeds17/18/19; each source-core seed remains separately identified.
- Adam lr3e-5, clip1, training64 steps/settle32.
- Training IDs0–999 and1640–70639 exactly; all IDs1000–1639 excluded.
- Full320 validation IDs1320–1639, same1024/512 spike readout and threshold.50.
- Full gamma pool, actual training IDs, source hashes, per-step finite loss and
  gradient norms, checkpoints, per-image metrics and count diagnostics recorded.
- No GT in training or prediction; validation selection only.

Runner is SW0094 `run.py` with explicit `--source-seed`, `--steps 4375`,
`--batch 16`, `--validation-count 320`. Default SW0094 arguments remain unchanged.
Real-data CPU preflights for new source seeds1/2 passed before queue creation;
source0 already passed the same recipe in SW0094. Existing running SW0094 Python
workers retain their loaded code and are not restarted by this parameterization.

Queue checks actual GPU compute processes and uses GPUs0/1/2. It first
waits for positive_frozen full320 evaluation to finish; the independent slow
encoder arm is not a prerequisite. With explicit `--allow-own-sharing`, it may
share a GPU only if every compute PID belongs to our UID and free VRAM is at
least8GiB; at most one SW0095 training job runs per GPU. Observed frozen-pilot
training uses about4.5GiB. It never shares other users' GPUs. User authorized GPU2; GPU3 is
reserved for the colleague. Duplicate queue copies are prevented by flock;
failed/incomplete outputs require diagnosis and are never silently retrained.

Interpret against the same held-out Slot70k epoch10 three-seed mean:
FG-ARI .774933 / foreground IoU .203589 / matched-object IoU .206937. Report the
training-budget difference explicitly. This candidate must improve three-seed
performance, and independent test evidence is still required to finish the goal.
The released Slot checkpoint .894641 is a separate transfer reference. Neither
an80-image pilot nor a single-seed gain proves full70k success.

### Seed0 completed (2026-10-07)

Full320 validation, original evaluator batch8, unchanged .50 spike classifier:

| Source seed0 checkpoint | FG-ARI | Foreground IoU | Matched-object IoU |
|---|---:|---:|---:|
| Before continuation: SW0090 epoch1 | .818470 | .714993 | .660334 |
| SW0095 additional complete70k pass | .824063 | .718180 | .653774 |

Deltas are +.005592 / +.003187 / -.006560. Aligned-loss continuation modestly
improves FG-ARI and foreground IoU but loses matched-object quality. It is not
an all-metric improvement over its own source. Whether the three-seed mean
exceeds the matched Slot benchmark remains unproven; seed1 is now training
(PID3578370, GPU0), seed2 remains queued. No running job was restarted.

Checked actual manifest: exactly70,000 distinct IDs, equal to0–999 plus
1640–70639; no held-out IDs. 4,375 updates of batch16, all recorded losses and
gradient norms finite. Source-model seed0 and whole-core source SHA match the
registered source. Evaluation320 IDs1320–1639,1024/512 spike readout and
threshold.50; GT absent from training/prediction. Source already had one70k
pass, so total downstream exposures are140k versus Slot's700k; encoder/graph
pretraining provenance differs and this is not a budget-matched scratch claim.

Loss endpoints18.654997→16.718990 are on different minibatches, not a
same-image loss comparison. Mean predicted object count is6.64375 versus
GT6.196875; aggregate mean count alone does not prove accurate per-image counts.
Raw manifest/history/evaluation and completion marker are archived in
`results/seed0/`. The reused runner's legacy `pilot:true` manifest field does
not describe data exposure; explicit training IDs/steps establish the full pass.
