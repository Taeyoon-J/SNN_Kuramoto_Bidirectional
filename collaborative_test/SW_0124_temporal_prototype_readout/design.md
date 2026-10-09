# SW0124: temporal-prototype readout, frozen-source stage 1

SW0123's assignment head produced mostly one foreground group on the first
training batches and many all-background fixed-validation images. This stage
tests whether a deterministic, affinity-seeded prototype readout of the same
frozen SW0097 actual spike histories provides a better fixed readout. It does
not train a model or alter the source core.

For each seed, validate the SW0097 core, matched SW0123 legacy calibration RMS,
and SW0042 validation gamma cache. Run the same source core at 1024 steps on
the registered 320 validation images. On settled actual component traces,
subtract the temporal mean and divide by the frozen four-channel TRAIN RMS.
Select detached anchors from the registered positive-product affinity: first
the lowest-index maximum row-sum patch, then the lowest-index minimum
maximum-affinity patch among distinct traces, stopping at affinity 0.50 or
11 anchors. Exact duplicate traces are excluded; no numeric novelty cutoff is
used. If all traces are identical, the readout has one prototype.

Gather live traces at those anchors. Apply three negative temporal-MSE soft
assignment / weighted-prototype updates at temperature 0.1 and compute final
assignments. Pad unused prototype columns with zero. Map labels with the
existing SW0123 largest-slot-background and minimum-two-patch rule. Report
the unchanged production QCC output beside it.

Save all predictions and hashes before loading masks. Score source QCC first
and require all three per-image metrics to reproduce the immutable SW0097
reference within 1e-10. Any mismatch ends the run before prototype scores can
be considered. Compare the prototype readout to the registered source metrics
with the shared-image, three-seed bootstrap and fixed Slot IoU references in
`protocol.json`. A failed gate closes this readout; it does not authorize
threshold, seed, or prototype tuning.
