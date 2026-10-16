# SW0179 — supervised QCE seed replication

## Status and purpose

The audited SW0178 seed-1 product-QCE result is supervised exploratory evidence. Its preregistered task gate failed, so it does not replace the prior incumbent or support an unsupervised claim. SW0179 repeats the same fixed product-QCE procedure from fresh SW0097 source seeds 0 and 2. It does not reuse learned weights or optimizer state from SW0178, SW0177, or earlier candidate experiments. The existing SW0178 seed-1 checkpoint, prediction, and score are the candidate's seed-1 result only.

## Fixed training protocol

- For each new seed, load the matching source with `foundation.load_native32_foundation(seed, device, verify_rgb_assets=False)` and apply `install_operator(graph, "T")`. Do not use the seed-1-only SW0169 source loader.
- All three seeds use the validated shared SW0171 TRAIN512 RGB and full modal instance labels. SW0171 producer lineage refers to source seed 1 and proves only common ordered data/assets; it is recorded as dataset provenance. Its seed-1 feasible targets, support masks, and spike cache are not model inputs for seeds 0 or 2.
- Construct a fresh zero-context wrapper with context seed `171000 + source_seed`. Draw the two consecutive 512-image permutations from `RandomState(171001)`. Use 2 passes, 64 logical updates, B16/micro4, T1024, settle512, live tail64, and per-update seed `171000 + source_seed + zero_based_update`.
- Keep the frozen SW0178 product-Q leave-one-out class competition loss and temperature 0.1, including full TRAIN modal labels and background class 0. Do not change architecture, optimizer, learning rates, clipping, integration projection, or loss. Each source seed starts a fresh Adam. The first two committed updates must show finite positive emitted-S gradient and finite positive gradient and parameter change for each of the four registered optimizer families.
- This is supervised training: `mask_supervised=true`, `unsupervised_leaderboard_eligible=false`. Do not read validation or holdout labels during training or selection.

## Evaluation and promotion gates

Freeze all new-seed predictions, protocols, IDs, and hashes for the fixed validation IDs 1320–1639 before opening validation ground truth. Use the existing SW0178 product-QCE prediction as candidate seed 1. The primary endpoint is standalone prediction from actual S → native component Q → native QCC → SW0165 solver. Report the conditional SourceT endpoint separately.

For each image, average metrics across the three candidate seeds first; bootstrap the 320 paired image rows, not 960 seed-image observations. Compare the three-seed standalone result to the three-seed SW0165 incumbent (FG-ARI 0.8042989149474198, foreground IoU 0.6099754484530633, matched object IoU 0.6511489750258587) and to the matched three-seed Slot Attention baseline (FG-ARI 0.7816519719609135, foreground IoU 0.20436012776655546, matched object IoU 0.2031649861884501). FG-ARI must strictly exceed both FG baselines, with paired 10,000-resample 95% lower bounds above zero for both comparisons (bootstrap seed 178). Foreground IoU and matched object IoU must strictly exceed the Slot Attention means. Record the separate 0.87 FG-ARI goal; it does not replace these gates. No unsupervised leaderboard promotion claim is allowed.

## Provenance

Bind each seed's source checkpoint and manifest, shared SW0171 dataset manifest/arrays and source-seed-1 producer provenance, per-seed source state, initial zero-context state, encoder/preprocessing asset hashes, optimizer groups, exposure order, runtime/device, durable boundaries, final model, and prediction hashes. Do not claim SW0161 seed-1 cached spike evidence for seeds 0 or 2. Do not read validation ground truth or reserve data before every candidate prediction is immutable.
