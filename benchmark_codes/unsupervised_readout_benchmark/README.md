# SW0259: SW0165 foundation with a mask-free252 readout

This is a new independent experiment, not a relabelling of historical253 results.
The same-seed SW0165 native32 foundation, source encoder/statistics, graph-T,
1024-step rollout and settle512 remain frozen. Every fixed320 baseline prediction
must exactly replay its archived SW0165 mask. No source or benchmark file is edited.

Only a randomly initialized, unchanged252 NeutralRegionHead and shared RGB decoder
learn. Inputs are144 actual spike/membrane features per patch. All-image signed
partition proposals and the head's original12 feature proposals include every
region; no region is designated or removed as background. RGB is a reconstruction
target only, never a head/decoder input. The decoder receives one pooled32D latent
per group and spatial coordinates, and produces group RGB composed by soft masks.

Fast first round: same256 TRAIN images across seeds,20epochs, no TRAIN masks;
validation RGB IDs1000–1031; evaluation IDs1320–1639. TRAIN IDs exclude1000–1639.
Three prospectively declared arms: RGB MSE, RGB+activity consistency(.01), and
RGB+activity consistency(.01)+mask entropy(.001). Same initial weights and orders.
One common arm and per-seed checkpoint are selected by validation RGB MSE before
evaluation truth is opened. Do not select an arm by fixed320 FG-ARI.

Core activities are cached per seed/input, and reused across all arms. Final
scores use the original immutable SW0165 native32 patch contract, with same-seed
baseline replay and saved prediction hashes. ForegroundIoU of new all-positive
labels is GT foreground fraction, so its semantic meaning differs from SW0165.
The primary requested comparison is FG-ARI0.804299, three-seed mean.

Additional loss/decoder refinements are authorized if this screen performs poorly;
preserve every arm and use separate versions and validation for new decisions.
Keep core/encoder and unchanged252 head as the first comparison. Never claim the
new readout isolates only supervision: it also changes background handling and
introduces the reconstruction decoder relative to historicalSW0165.

Run `test_model.py`, then launch `coordinator.py --output NEW_OUTPUT`. The detached
coordinator queues actual-CUDA admission, per-seed caching/training and final scoring.
Physical one image/update, Adam.003, clipping1.0; source weights receive no gradients.
