# SW0135 native32 spike binding

Preparation started following the user's instruction to use32x32=1024
patches for all new tests. The immediate target is to exceed comparable Slot
Attention under the same native32 patch evaluation contract; the final
three-seed/all-three-metric requirement remains intact.

See [native32 contract](../evaluation_contract32.md). No native32 learned-binder
training or new performance result is claimed yet. SW0134's already-running
16-grid training is kept immutable while this new experiment is prepared.

Server inspection confirmed all three own70k Slot epoch10 prediction files:
`trained_models/SW0092_slot_our70000_eval/seed{seed}_epoch10/predictions.npz`.
Each contains320 native128x128 pixel-label images for IDs1320..1639. These
can be rescored directly at4-pixel patches without rerunning Slot training
or substituting upsampled16 patch labels.

Implementation review must address native1024 feature extraction, strict
source-node parameter conversion, differentiable graph memory, full spike
history input,4-pixel decoder geometry, and a real GPU optimizer update
before main training. Grid-size constants alone are insufficient.

The previous SW0114 candidate's native32 seed0 scores were
`.817187/.760535/.641405`, with frozen encoder and graph. They are historical
single-seed observations, not a native32 Slot victory or this experiment's
result. Its pooled16 result is not the primary contract for the new work.
