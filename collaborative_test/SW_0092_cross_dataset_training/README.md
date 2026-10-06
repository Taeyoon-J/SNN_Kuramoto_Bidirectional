# SW0092 — reciprocal cross-dataset training

This experiment runs both directions requested after obtaining the released
Slot Attention TFDS data.

1. **Our model on released Slot data.** TFDS CLEVR 3.1.0 train is processed in
   released-code order: skip the first 512 rows, keep scenes with at most six
   objects, center-crop, and bilinear-resize to 128x128. This yields 34,766
   unique training images. The strongest fixed-feature/frozen-graph SNN recipe
   trains for 1/3/10 epochs and is evaluated on our fixed mask-bearing HDF5
   validation IDs1320-1639.
2. **Slot Attention on our data.** The official architecture is trained from
   scratch on our 70,000 unique training IDs for ten exact passes. It uses 11
   slots for the up-to-ten-object HDF5 scenes. A real batch64 backward preflight
   exceeded 24GB; batch32 passed, so all registered runs use batch32. Epochs
   1/3/10 are evaluated on the same validation IDs and three metrics.

The first direction must use our validation domain for mask scores because TFDS
CLEVR 3.1.0 does not provide instance masks. Consequently it is explicitly a
cross-domain transfer test, while the second direction is a matched-data test.

## Active scheduling (2026-10-06)

Experiment 1 has completed all three seeds, epoch1/3/10 evaluations and mask
visualizations. Experiment 2 seed1 remains live on GPU1; experiment 3 seeds1/2
remain live on GPUs2/3. GPU0 currently belongs to another user.

`resume_our_official.py` replaces the original experiment 2 waiting queue.
It preserves existing live/completed seeds and uses whichever of GPUs0/1 is
free for remaining seeds and evaluations. Only the old queue and its waiting
seed0 wrapper were stopped; the live seed1 training process was preserved.
Its log is `trained_models/SW0092/our_resume_queue.log`, and its state is
`trained_models/SW0092_OUR_OFFICIAL_QUEUE.json`.

`queue_slot_finish.sh` independently finishes experiment 3 on GPUs2/3:
after seed1 completes, train seed0 on GPU2 while evaluating the completed
seeds1/2 on GPU3. Then evaluate seed0 at epochs1/3/10 and write the three-seed
summary. No model modification tests are queued.
After both experiments finish, retrieve results, record comparison, and pause
goal mode as requested in `../ACTIVE_REQUEST_TEST123.md`.

Experiment 2 reuses our existing encoder and frozen graph pretrained on our
data; its downstream training uses released Slot data. It is not a complete
model trained from scratch exclusively on released Slot data.

### Pending coordinator update after SSH reconnect

The new local coordinator launches pending seeds on both available GPUs0/1,
reserving a GPU immediately so two jobs cannot select it before CUDA registers.
CPU-only `preflight_queue.py` passes three checks: preserve an existing seed,
reserve distinct GPUs, and refuse partial output without launching a job.
Windows sandbox temporary-file access failed; the identical checks passed when
run with filesystem access outside that restriction.

SSH multiplex broke during deployment. The pending local scp/SSH mutation
commands were cancelled; this update is **not verified as deployed**. Last
verified live training handles were experiment2 seed1 PID3415859 and experiment3
seeds1/2 PID3336397/PID3336399. Last verified queue handles were experiment2
PID3417653 and experiment3 PID3419065. After reconnect, inspect those handles,
the remote coordinator file and its log before replacing any queue. Preserve
all live training; never restart based only on this observation failure.

### Reconnected and deployed

The user reconnected successfully after forcing curve25519 key exchange for
both SSH hops. The default WSL hybrid exchange stalled at the raptor key
exchange reply; forcing curve25519 reached authentication. The underlying
network cause remains unconfirmed.

All four requested training jobs survived the observation outage. Experiment2
seed0 started automatically when GPU0 became free (PID3428389, wrapper3428384),
while seed1 continued on GPU1 (PID3415859). Experiment3 seeds1/2 continued on
GPUs2/3. The parallel experiment2 coordinator is now deployed, remotely compiled
and confirmed live as PID3442230. Only its previous parent PID3417653 was
stopped; both training wrappers and trainers remained live. Its state explicitly
reports running seeds0/1 and waiting seed2, which will start on the first free
GPU0/1. Experiment3 finish queue PID3419065 remains live. No training restart
or architecture change was performed.

### First experiment3 evaluations and GPU3 reuse

Experiment3 seeds1/2 completed ten epochs (21880 updates each); their six
epoch1/3/10 evaluations are now saved under `results/slot_our70000/`. All six
were checked for exact held-out IDs1320-1639, 320 valid scores per metric,
16x16 patches, seed/checkpoint identity, the prescribed 70000 training IDs and
no GT use during training or prediction. `partial_summary_seed12.json` reports
**two seeds only**, not a final three-seed result:

| Epoch | FG-ARI | Foreground IoU | Matched-object IoU |
| --- | --- | --- | --- |
| 1 | 0.617100 | 0.216267 | 0.176199 |
| 3 | 0.745254 | 0.218597 | 0.202319 |
| 10 | 0.853923 | 0.200172 | 0.212717 |

Experiment3 seed0 is training on GPU2 (PID3444264). Once all six earlier
evaluations finished, GPU3 became free and experiment2 seed2 started there
(PID3447194). The coordinator permits GPU3 only after those six scoring
completion markers exist and nvidia-smi confirms the GPU is free. Its new
live handle is PID3447179; experiment2 seeds0/1 were preserved on GPUs0/1.
This follows the user's request to use all four available GPUs. Experiment3's
remaining seed0 evaluations use GPU2, so GPU3 is no longer required by its
finish queue. Three-seed conclusions remain pending.

### Experiment2 seed1 completed; missing marker recovered

The trainer completed epoch10 and saved `core.pt` plus epochs01/03/10, but its
wrapper did not write `TRAINING_COMPLETED`. The queue stopped conservatively;
the other three training jobs remained alive. Recovery loaded all four saved
state dictionaries on CPU, checked matching keys and finite values, and proved
the final model tensors exactly equal the epoch10 model tensors. It also
required the trainer's epoch10 and final-save log lines. Only then was the
completion marker restored. Hash evidence and the completed training log are
saved under `results/our_on_official/`; training was not repeated.

The marker omission's root cause is unconfirmed. A launcher was updated while
older wrappers were running, so future shell deployments should use a new
file/inode rather than overwrite an executing script in place.

The current coordinator PID3467736 reports completed seed1 and live seeds0/2.
`evaluate_seed1_early.sh` PID3467737 evaluates seed1 epochs1/3/10 on the now-free
GPU1 while the remaining jobs train on GPUs0/2/3. The final coordinator waits
for this early-evaluation wrapper before reading or generating evaluation
outputs, avoiding duplicate writes. Actual first evaluation PID3467744 was
confirmed on GPU1. Model architecture and training parameters are unchanged.

### Experiment2 seed1 early evaluation completed

The three seed1 evaluations completed and are saved in
`results/our_on_official/seed1_epoch{1,3,10}.json`. Validation checked 320
held-out images, correct checkpoint/epoch, 1024 steps with settle512,
membrane threshold .06, spike threshold .50, 16x16 patch scoring, finite
scores with 320 per-image values and no GT used for prediction.
`partial_summary_seed1.json` is explicitly a single-seed cross-domain result:

| Epoch | FG-ARI | Foreground IoU | Matched-object IoU |
| --- | --- | --- | --- |
| 1 | 0.758584 | 0.345658 | 0.580763 |
| 3 | 0.748688 | 0.371336 | 0.581524 |
| 10 | 0.732275 | 0.409210 | 0.593392 |

The early evaluator exited successfully and its completion marker exists.
Experiment2 seeds0/2 and experiment3 seed0 are still training; both final
three-seed summaries remain pending.

