# SW0096 — frozen three-seed holdout comparison

Registered after SW0095 finished all three seeds. Its validation mean
(.775645/.574029/.599779) numerically exceeds Slot's matched70k epoch10 mean
(.774933/.203589/.206937), but the FG-ARI margin is only .000711.
Validation is used for selection, not final independent confirmation.

Freeze all three SW0095 final cores and all three SW0092 Slot epoch10 cores.
No further training, threshold changes, background changes, seed selection or
selection of a holdout based on its scores. Same original preprocessing and
8px modal16x16 patch metrics for both models. Our spike evaluation retains
batch8,1024 steps/512 settle,.50 threshold,largest-component background.
Slot keeps batch1,11 slots,3 iterations,inference seed0,perimeter background.
Model-specific inference batches are the unchanged registered settings.

Primary independent confirmation: HDF5 IDs90000–90319 (320 images), outside
all registered70k downstream training IDs and previous documented selection
slices. Dataset metadata confirms100,000 images. Check actual training
manifests and Slot protocols before scoring. Previous experiments and native
encoder/graph pretraining provenance must also be considered when describing
independence; an absence of recorded use cannot prove absence of unrecorded use.

Secondary reference: IDs1000–1319, disclosed as already inspected during
baseline work. Use both splits without choosing the easier result. Never
retune on either. Raw legacy evaluators hardcode `validation` as a role label;
actual image IDs and this experiment manifest define the holdout roles.

GPU0/1 only; check live compute PIDs before each evaluation and never share
another user's process. No training is launched. A failure is recorded and
not automatically rerun. Prepare-only CPU checks verify all six checkpoint
paths, complete training ID exclusion, and cached-gamma reproduction before
the GPU queue starts. Preserve raw per-image results and model provenance.

Training budgets remain different: our downstream cores saw140k exposures
plus earlier encoder/graph pretraining; Slot saw700k exposures from scratch.
This is a same-data performance comparison, not equal-budget training.

## Completed frozen comparison

All12 evaluations completed without model changes. Same dataset, seed0/1/2,
image IDs and16x16 metric grid for the two models within each split:

| Split | Model mean | FG-ARI | Foreground IoU | Matched-object IoU |
|---|---|---:|---:|---:|
| Independent90000–90319 | SW0095 | .759817 | .562696 | .588620 |
| Independent90000–90319 | Slot epoch10 | **.777602** | .196114 | .201021 |
| Reference1000–1319, previously inspected | SW0095 | .763595 | .561184 | .584245 |
| Reference1000–1319, previously inspected | Slot epoch10 | **.773277** | .194499 | .200663 |

Two-IoU advantage persists but FG-ARI numerical exceedance does not. The
validation margin (.000711) is not sufficient evidence of a general victory.
The final objective is **not complete**. Preserve all splits and seeds rather
than selecting a favorable split or excluding Slot's weaker/stronger seeds.
Raw per-image scores, checkpoint fingerprints, manifests and launch/state
records are in `results/`; `results/summary.json` compares all means.

Both90000–90319 and1000–1319 are now observed. No classifier tuning or further
model selection may use their scores. Future improvements use the original
validation1320–1639. Reserve90640–90959 for a later frozen final confirmation;
do not read their images/masks or produce their scores during model tuning.
The weak source-seed disparity and graph-adaptation hypothesis were already
visible on the original validation; investigate these there.

The first daemon request was rejected by automatic review due to suspected
duplicate execution. Read-only checks established no queue.pid/state.json and
no Python processes owned by our UID. The same authorized request was then
approved and started once (queue PID3620227), without a workaround, duplicate
evaluation or GPU contention. Final queue state is complete.
