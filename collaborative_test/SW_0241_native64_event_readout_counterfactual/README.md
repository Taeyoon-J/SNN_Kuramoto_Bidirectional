# SW0241 event information at readout

Compare actual emitted spikes with observed-gate-only readout on the same six frozen TRAIN cases as SW0238. Both mask paths use the production readout; the intervention rebuilds mean spikes and component correlations/Q. Full actual spikes, gates, adjacency and paired masks are retained privately.

This tests downstream event information. The native forward and its membrane/dendritic recurrence are unchanged. No ground-truth labels, optimizer updates or new performance claims are involved. An unchanged mask would not establish that removing membrane dynamics preserves performance.

CPU tests cover input preservation, production component-gate geometry, mean/Q recomputation, real native Q sensitivity, label-permutation-safe comparisons and correct exclusive-GPU worker routing. Real CPU checkpoint restoration must pass for all three seeds before GPU queues start. Source and generated results stay separate.
