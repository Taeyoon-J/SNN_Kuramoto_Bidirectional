# SW0241: event information at the production readout

SW0238 found positive-gate event occupancy of 99.3265–100% on six actual TRAIN traces. Check whether the small remaining binary-event differences influence the final masks, before designing a learned membrane intervention.

Use exactly the same accepted232 seed0/1/2 checkpoints, two TRAIN images per seed and inference seeds as SW0238. One unchanged native64 T1024 forward supplies each actual trajectory. Keep its gates and graph fixed. The counterfactual replaces only downstream component spikes with the observed gates, rebuilds mean spikes, recomputes component correlations/Q and runs the identical production support/partition readout. Compare masks including foreground support and object-ID-invariant partitions.

This is an event-information readout intervention, not removal of membrane recurrence or dendritic feedback during dynamics. It neither trains a model nor measures held-out performance or causal necessity of the full membrane layer. No ground-truth labels are read. Retain full actual component spikes, gates, adjacency and paired masks for later exact recomputation; no raw arrays are published. Validate source/checkpoint/row/tensor hashes and unchanged model state.

Run CPU unit tests and real CPU restore admissions before the existing exclusive-GPU queue can execute. Other jobs, including SW0240 single-GPU TRAIN70000 and SW0237 full320 evaluation, remain independent. Six cases alone must not be generalized to the full data distribution.
