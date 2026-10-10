# SW0135 prospective throughput amendment

Sol6.1 review, 2026-10-10. Resource implementation change only; no model/objective/endpoint/budget change. Register before the new probe; preserve the existing B1 observation and artifact hashes.

## Observed foundation

The actual native32 disposable lambda1 probe completed rc0: logicalB16 via16 microB1 contributions took56.87983477s, peak allocated933536768 bytes/reserved1124073472 bytes. All20 declared joint tensors and23 head tensors changed; reconstruction credit was nonzero in the reported families. These are resource/path proofs, not segmentation or gate-usefulness evidence. Simple extrapolation gives about64.7h per4096-update arm, excluding warmup, calibration, evaluation and other overhead. It is not an adequate reason to launch that full budget blindly.

## Recommended next measurement

Measure **microB4 with four accumulation calls per logicalB16** first, on the same exclusive-GPU policy, fresh source0, identical16 TRAIN IDs/order and disposable lambda1. Keep genuine1024 nodes, native gamma32, T1024/settle512/live-tail64, full intermediate-node geodesic reduction/checkpointing and the unchanged binder/decoder. No scientific training admission follows this probe.

If memory permits and throughput warrants it, prospective resource measurements at microB8 or16 may follow. Choose the execution batch from measured capacity/throughput only, never endpoint masks or scores. Record CUDA synchronization boundaries, peak allocated/reserved memory, joint/head gradient and optimizer-state allocation, wall time and projected warm32/calibration/4096-update/evaluation cost. Include initialization separately; one-step timing remains a forecast. The frozen arm has a different cost and requires its own estimate.

## Same logical objective and optimizer semantics

For each microbatch of size m, multiply its imagewise mean old loss and R contribution by **m/16**. Equivalently accumulate the sum of each image loss divided by16. The joint receives old+lambda*R; head/decoder receive unweighted R exactly once through their separate gradient computation. Zero gradients once, accumulate all16 images, clip each unique optimizer union once, perform one Adam step per union, and project integration coefficients once. Restore the same warm Adam32 moments when scientific training begins. Do not clip or step per microbatch.

The source encoder/binder use no inter-image normalization, and old/R losses reduce independently over images. Thus B4 accumulation represents the same logicalB16 mathematical objective. Calibration must use norms of the **accumulated joint gradient vector**, preserving first4 logical batches, rather than averaging microbatch gradient norms. Warm32 remains32 logical updates/512 images, not128 B4 updates.

Before GPU execution, verify batching/accumulation on a small deterministic CPU fixture, including old/R means, separate joint/head gradients, one clipping/Adam step and parameter update. Compare with explicit per-image accumulation using prespecified numerical tolerances, not bitwise equality. Actual GPU resource checks must additionally record batch-layout trace/Q/loss and gradient differences against the retained B1 reference. Float32 kernels and recurrent threshold crossings can amplify small batching differences: mathematical independence is not a claim of identical CUDA trajectories.

## Scientific admission remains separate

Always mark the disposable probe `resource_only=true`, `training_admission=false`, `ground_truth_used=false`, with source, IDs, implementation, microbatch/accumulation and optimizer configuration hashes. Do not reuse its changed source, head or optimizer as warm/training artifacts. A zero R-family gradient in the random-head resource probe is recorded, not used to reject the model hypothesis.

After a practical execution batch is selected, run native32 source/adaptor parity at that batch configuration, frozen-backbone warm32, fresh source0 lambda calibration, and the original real family-credit/update preflight. Preserve every existing scientific guard; do not redefine parity to accept a failed input/source path. Record the throughput amendment in warm/preflight/training provenance. Only then admit the registered seed1 paired pilot. Its final4096-update endpoint and native32 evaluation contract remain unchanged.

If no measured batch makes the registered budget practical, report the measured limitation and require a separately registered budget/resource decision before launching. Do not silently shorten the sequence, freeze the graph, reduce1024 patches or replace the final endpoint with an early checkpoint. No new16 experiment or server action is authorized by this review document.
