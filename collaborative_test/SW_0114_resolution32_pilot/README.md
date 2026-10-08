# SW0114: matched 32x32 resolution pilot

The candidate uses1024 patch oscillators rather than256. This is a matched4096-image continuation from each whole SW0097 seed, with256 logical B16 updates per arm. It keeps the native frozen encoder, graph parameters, actual raw-gated spike assignment and original training images. It does not require a full70k run to obtain an initial result.

`protocol.json` fixes the source conversion, geometry, memory checks and scoring. Root review is required before implementation: the soft-geodesic quadrature correction and doubled coherence weight are explicit proposed physical-scale adaptations, not implicit defaults.

## Source conversion and geometry

Omega, kappa, dendritic tau_n and membrane tau_m require nearest2x2 spatial replication. The phase-lag direction matrix requires replication on BOTH node axes. Shared dendritic projection and every other parameter remain exact. Regenerate identity SC and graph-distance buffers; never flatten-repeat a spatial vector or silently ignore checkpoint keys.

At32, use K1024, top-k128 and minimum component size8, against256/32/2 at16. Measure graph distances in old8-pixel patch units. Keep geodesic radius, cap, temperature and3 iterations fixed, with a fixed log4 intermediate-node normalization to remove duplicated-node softmin entropy. Coherence weight1 versus.5 preserves derivative scale. These choices define a physically calibrated resolution comparison; they do not isolate changing N alone.

## Memory and matched optimization

The existing geodesic implementation creates an N-cubed temporary:4GiB even at B1 for1024 nodes. Freeze graph execution and chunk the outer node axis32 while keeping the full reduction axis. Use B1 gradient accumulation16 in BOTH arms; divide each image loss by16 and clip/step once per16. Enabled objectives are per-image reductions, so this preserves the logical loss, subject to the registered numerical preflight. Measure actual memory and runtime before admitting GPU training. Do not downscale T, N or features to hide failure.

Only selected pilot RGB images and320 validation images need32-grid gamma. Apply the registered encoder, standardization and clipping before adaptive pooling32. Upsampling cached16-grid gamma would not test finer image information. The126x126 valid-convolution feature map has adaptive pooling boundaries; this retains the native convention and is disclosed.

## Comparable results

The primary comparison pools predicted32-grid labels by a GT-free2x2 majority rule and scores the original16-grid320-image contract. Native32 results use modal4-pixel GT and are a separate endpoint. Upsample the control predictions to32 for a same-target secondary comparison. Historical Slot16 means cannot substantiate Slot32 superiority; frozen Slot predictions must be rescored at32 before that claim.

All masks are formed from actual component spikes, frozen before GT scoring. The unread reserve stays untouched. The all-three-seed primary promotion gate, paired-image bootstrap and IoU floors are fixed in the protocol. A matched70k extension is conditional on improvement and a further reviewed execution plan.

This folder contains design only. No server payload, job or transfer is authorized by writing these files while the parent's automatic-review approval question is pending.
