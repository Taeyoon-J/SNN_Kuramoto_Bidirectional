# SW0241: six-case gate-only substitution, with a layout limitation

Four of six frozen TRAIN cases retained identical masks. Seed0/case0 changed three foreground-support patches while retaining five objects. Seed2/case1 retained foreground support but changed the partition from23 objects to22. There is no GT-based score here, so neither change establishes an improvement.

The CPU archive audit re-read the actual full spike/gate tensors, exactly reproduced the prior238 tail-event statistics in all six cases, and recomputed comparisons from the saved paired masks. All source/checkpoint/row bindings and hashes passed. This audit does not re-execute the readout.

A post-completion code check found that the gate-only replacement preserved values but not native component-spike strides. Native rollout stacks time-first and returns a noncontiguous spike tensor; cloning the contiguous gate yields a different layout. Correlations are mathematically equivalent under layout changes, but floating-point reductions near a hard partition threshold can differ. The two observed mask differences therefore cannot yet be attributed exclusively to binary-event information. The all-one-event seed1 controls retained identical masks, which does not rule out numerical sensitivity in the other cases.

The required next control is a readout replay using retained arrays, with actual native spike strides restored and the gate-only signal copied into the same layout. First reproduce the original actual and original gate-only masks, then compare the matched-layout substitution. This needs no new training or native trajectory rollout. Preserve this original experiment and its limitation rather than changing its registered source after completion.

This is not a full membrane recurrence ablation, held-out performance result, proof of causal necessity or new best model. SW0237 remains the audited full320 reference. The ongoing70k training is unchanged.
