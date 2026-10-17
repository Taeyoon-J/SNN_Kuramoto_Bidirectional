# SW0190: TRAIN-only oracle-count global partition admission

SW0189 observed both fragmentation and inter-object merging in the incumbent.
Before training a count predictor, test whether supplying the correct TRAIN
object count can make a different global assignment useful at all.

Use the completed matching-seed SW0186 model, all three seeds0/1/2, and the same
16 TRAIN512 positions0,32,...,480. Native32 RGB→all four gates→actual component
spikes→unchanged production Q, T1024/settle512, B4, live-tail0. No optimizer
updates, new encoder/head, validation RGB/masks, or model tensor changes.
The baseline is the actual QCC plus foreground-induced merge production path.

Two fixed alternatives use the same Q: normalized symmetric spectral embedding
with topK eigenvectors, row normalization, deterministic farthest initialization
and50 Lloyd steps, no selected restart or parameter sweep. The first keeps the
baseline foreground support and partitions those nodes into K groups. The second
partitions all patches into K+1 groups and chooses background by maximum overlap
with the model's existing background, then cluster size and ID. Only the scalar
TRAIN modal visible-object count K is oracle information used by the readout;
no target pixel membership, GT background geometry or object correspondence
enters the assignment. Counts are not predicted yet.

Record paired TRAIN metrics for baseline and both alternatives, every source
seed and all16 images, plus exact unchanged model SHA and Q/z digests.
Do not call the oracle metric a deployable model result or update a leaderboard.
The small TRAIN cohort is an admission test, not a validation improvement.

Admission to a learned count/readout experiment requires at least one fixed
alternative's mean TRAIN FG improvement of .01 or more, no mean Foreground IoU
decrease, and positive mean FG difference in at least two of three seeds.
Report all alternatives and ties. Failure closes this spectral/count recipe;
do not perform a small clustering/threshold sweep. A pass only justifies training
a predictor without GT counts at inference and evaluating full fixed320 later.

CPU tests and all three actual frozen-model admissions precede GPU execution.
Reuse the exclusive-owner/yield queue, never share a foreign GPU. The pilot may
use a free GPU while SW0188 continues; it never stops or changes SW0188.
Detailed TRAIN imagewise reports remain private; share code and cohort aggregates.
Unsupervised, native32 joint70k and gate causality are not established here.
