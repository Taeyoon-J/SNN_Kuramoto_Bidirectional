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

