# SW0087: fixed-contract matched Slot goal comparison

SW0087 waits for the completed SW0056 three-seed matched-data Slot summary and
compares it with the completed SW0072 three-seed mean. It accepts only the
aligned HDF5 validation slice IDs 1320-1639, seeds 0/1/2, 320 images, and
GT-free Slot prediction provenance.

The goal flag is true only when SW0072's mean is strictly greater than the
SW0056 mean for FG-ARI, foreground IoU, and matched-object IoU independently.
No metrics are averaged into a composite score, and equality fails the gate.
The JSON and Markdown outputs retain every mean, delta, and Boolean decision.

The durable CPU-only watcher performs no model inference and consumes no GPU:

```bash
nohup bash collaborative_test/SW_0087_matched_slot_comparison/wait_and_compare.sh \
  </dev/null >trained_models/SW0087_GOAL_COMPARISON_QUEUE.log 2>&1 &
```
