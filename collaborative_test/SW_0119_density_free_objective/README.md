# SW0119 density-free objective proposal

The proposed single change removes the fixed synchrony-density penalty from SW0117's joint analytic-RGB objective: `L119 = L117 - 10 B(PLV) - 50 B(Q)`, where `B(S)` is the image mean squared error between off-diagonal synchrony density and `0.867`. This is an affinity-density prior, not a ground-truth foreground-area target.

Keep the whole SW0097 seed-0 source, jointly trainable encoder/graph/core, fresh Adam, learning rates `3e-5 / 3e-6`, clipping at 1, exact 4096 training-ID order, batch 16, 256 updates, 64/32 training steps, analytic RGB loss and immutable SW0117 coefficient `7.865416617457706`. Keep all other loss weights and the original 320-image, 1024/512-step production classifier and patch metrics unchanged. SW0118's prediction-conditioned diagnostic metric is excluded from promotion.

Before any new training, require identity with SW0117 for the retained loss, trace and hard partition; prove the loss difference equals exactly the removed balance contribution, including gradient agreement. Existing source/cache/encoder and real-update guards remain required. Reuse completed SW0117 endpoints only when all source, input order, optimizer and coefficient bindings match. Promote only with FG-ARI gains of at least 0.01 over both original SW0097 and retained-density SW0117 RGB control, positive paired-image CI lower bounds against both, and both IoU margins in SW0117's registered gate. Failure ends this recipe without density/coefficient/window tuning.

The read-only first-four-training-batch diagnostic completed with no parameter updates or ground-truth access. Actual Q densities were 0.6573–0.7131; balance-vs-R joint gradient cosines had both signs. Centered component trace norms had no near-zero values. These results support testing the density prior but do not establish the cause of SW0117's extreme later gradient outliers.

Training is not yet started. The next prerequisite is an exact 45-update SW0117 candidate prefix replay, followed by forward and diagnostic backward at the recorded update-46 outlier. Compare against the original history before attributing the outlier. Preserve all original training artifacts and perform no update 46 in this replay.

## Outlier attribution completed at an earlier verified step

The update-46 replay stopped before update 13 because gradient parity exceeded the fixed `1e-5` relative tolerance. Loss disagreement was only about `1.9e-6`, but the failed gradient guard was preserved; no attribution is made for update 46. The first three steps had already reproduced the original history within `9.14e-8` relative difference. Original update 3 itself had a preclip norm of `22,349,246`, so a separate bounded replay applied only the first two optimizer updates and diagnosed step 3 without applying that update.

This earlier replay passed. At the outlier, one of 16,384 centered component traces was exactly zero. The weighted spike-prior and RGB derivatives with respect to Q had norms `0.0564` and `0.0505`; their derivatives with respect to component traces had norms `85,127` and `362,005`. The primary phase path was far smaller. This supports a Pearson-normalization derivative amplifier at this verified outlier, not a claim about every outlier. Evidence and provenance are in `early_outlier_attribution_20261009.json`.

Before testing density removal, separately verify a forward-preserving gradient guard for flat component traces. Its effect must be isolated from the density intervention; no guard benefit is yet claimed and no density-free training has started.
