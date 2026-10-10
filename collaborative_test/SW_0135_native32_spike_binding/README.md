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

## Comparable native32 Slot baseline

CPU scoring completed on2026-10-10; original16 scores reproduced for all
three seeds from the same frozen native128 predictions. Native32 seed means
are FG-ARI .781651972, foreground IoU .204360128 and matched-object IoU
.203164986. All320 images are valid for each metric and seed.

See results_archive/slot70k_native32_baseline.json and
results_archive/slot32_execution_record.json. This is baseline scoring,
not new candidate training or evidence of a native32 victory.

## Native32 implementation

Implemented genuine1024 raw-spike binder,4px decoder geometry, strict spatial
source conversion and a differentiable checkpointed full-reduction graph.
Eight focused CPU tests passed, including graph values/gradients against a
dense registered reference and a real native32 core construction with live
graph parameters. See implementation_review.json and protocol.json.

Actual1024 GPU rollout/optimizer/resource checks remain pending; CPU tests
do not certify main-training readiness or any candidate performance.

Native32 encoder/source foundation is implemented and four focused CPU
checks passed. It regenerates gamma from registered uint8 RGB, maps the
complete source before the integration wrapper, and uses native32 losses.
See foundation_review.json. Actual server assets,1024 GPU rollout/parity
and logical-batch resource admission remain pending.

## Exclusive GPU resource execution

All21 model/foundation/resource CPU checks passed on Frontier. Registered
seed0 native1024 resource probe is actually running on exclusive GPU0,
supervisor2658960 / child2658986 (observed2026-10-10 05:59 UTC). See
resource_execution_start_20261010.json. This disposable logicalB16 Adam
measurement uses lambda1 before warmup; it is not candidate training,
scientific calibration, temporal-parity certification or performance.
