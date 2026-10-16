# SW0178: Temporal class competition

## Question

Does class-balanced leave-one-out competition over full temporal labels improve
the registered native32 task, and does using signed component mean affinity
help relative to the matched production positive-product affinity?

## Frozen comparison

Two fresh model instances start from identical pretrained SW0097 SourceT
weights and zero-context parameters. Neither arm reuses trained weights,
heads, or rollout hooks from SW0175, SW0176, or SW0177. Both use the same SW0175
four optimizer families, rates, shuffle order, and native rollout:

- `mean_rho_ce`: for each image, use the mean of the four signed component
  correlations as pair affinity.
- `product_q_ce`: use the frozen production affinity, the product of the four
  component correlations after each component is clamped below at zero.

Both arms train on all 512 registered TRAIN images for two passes (64 logical
updates, B16 with microbatches of four), using seed 171001 plus the zero-based
logical update index for each update. The two NumPy shuffle permutations are
drawn consecutively from `RandomState(171001)`. The
rollout is the original SourceT recurrence with 1024 steps, settle point 512,
and live tail 64. This experiment does not reuse learned weights from earlier
namespaces and does not add a SourceF restriction to the training objective.

## Objective

For each image, each patch is labeled with its full modal target, including
background label 0. For a query pixel, each class prototype is the mean
affinity to pixels of that class. The query is omitted from its own class
prototype. A singleton query is omitted because it has no leave-one-out own
prototype; singleton classes remain eligible as competitors for other queries.
For each query, cross-entropy uses temperature 0.1 over the available classes.
Losses are averaged within each query class, then over classes with valid
queries, then over images. An image with no valid class contributes zero to
the batch mean. GT classes/counts are used only by this TRAIN loss; inference
does not use GT count or a GT-derived K.

The `mean_rho_ce` arm uses the signed mean of four live component correlations;
the `product_q_ce` control uses the exact native positive-product Q. The latter
is computed with the frozen production operation order. No additional loss,
lambda, threshold sweep, or source-target foreground restriction is added.

## Readout and decision

Inference uses actual emitted spikes S, native production Q, and the original
SW0165 solver. Conditional readout uses the registered SourceT foreground;
standalone readout uses its own native QCC foreground. All predictions for
both arms are frozen and validated before opening validation ground truth.
Each arm is assessed against SW0164 seed 1 (registered means: FG-ARI
0.7709320986336607, ForegroundIoU 0.6149934302630647, MatchedObjectIoU
0.6305234820304301) with paired 10,000-resample bootstrap, seed 178. At both
endpoints, mean FG-ARI must be strictly above the SW0164 mean and the paired
95% confidence lower bound must be above zero; mean ForegroundIoU and
MatchedObjectIoU must each meet or exceed their SW0164 means. The mean-rho
mechanism is promoted over product-Q only if it also passes these same gates
against that matched control. Validation is fixed to registered IDs
1320..1639; reserve data is never opened. If both arms fail, close the line
without sweeps.

## Cache boundary and scope

The SW0161 cache stores float32 component spikes with shape `[4,1024,512]`;
these are already settled traces. Cache-boundary tests consume them as settled
inputs with `settle=0` and must not slice a second 512-step warmup. Live
rollouts use 1024 steps with `settle=512`. CPU-only preparation is limited to
the first eight ordered TRAIN rows and their full 1024-node modal labels.
Cached-trace gradients are not evidence of model credit. Ground-truth classes
and counts are used only in the TRAIN loss; inference does not consume GT
count or a GT-derived K. No validation/reserve rows, GPU rollout, training, or
performance claim is part of this initial implementation slice.
