# SW0224

Independent zero-update feature-cell alignment model test. Existing126x126
features get one replicated border before native64 pooling. Original128 RGB,
2px modal GT, source parameters and spike readout remain fixed. This changes
the inference adapter and is not a numerically equivalent optimization.
Three source seeds, same16 images. See DESIGN_KO.md; no result claimed yet.

Actual start2026-10-18:7 local/7 Linux tests and all3 source/immutable-checkpoint/actual126-feature CPU admissions passed. Three exclusive idle GPU queues registered; START_EVIDENCE.json binds actual NVIDIA workers. Source parameters/statistics/RGB/GT remain fixed; only the declared feature-border/pooling adapter changes. Zero optimizer updates. No new segmentation result; full320 incumbent .848615/.592533/.672047 unchanged.
