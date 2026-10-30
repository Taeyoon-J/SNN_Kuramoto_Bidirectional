# Data locations and checkpoint requirements

## What is and is not required

**No existing learned checkpoint is required to start any current SW0255 job.**
The four names240/252/253/254 refer to architecture/loss families, not loading
their historical best weights. The teacher used by the supervised families is
made from their own fresh initialization.

To resume a newly trained job, keep that job's `initial.pt`, `resume.pt`,
`binding.json`, `effective_code.json` and history together. Do not transplant a
checkpoint across different data, code paths or backward recipes.

Legacy240 seed0/1/2 exports are optional historical assets, not training inputs:

`/Data0/kevinswk/patch_v2_sw/collaborative_testing/generated/patch_final_exports/sw240_seed{0,1,2}.pt`

Their file sizes/SHA are in `contracts/legacy_checkpoint_manifest.json`.
Do not load them into this scratch protocol and still call it scratch training.

Current checkpoint pattern:

`/work/USERS/kevinswk/experiments/SW0255_full3500/jobs/model{252,253,254,240}/{dataset}/seed{0,1,2}/{initial,resume}.pt`

Previous35k jobs were retired with their original files retained. Verification
records are in `snapshot/previous35000_retirement.json`. They are not silently
reused for the3500 subset.

## Dataset/cache paths on Frontier

| Asset | Location |
|---|---|
|Our CLEVR HDF5|`/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5`|
|Our CLEVR RGB/labels cache|`/Data0/kevinswk/patch_v2_sw/collaborative_testing/generated/collaborative_test/SW_0240_native64_70000_cached_io/input_cache`|
|Four prepared dataset caches|`/work/USERS/kevinswk/experiments/SW0255_full35000/inputs/{multi_dsprites,tetrominoes,objects_room,clevrtex_original}`|
|Corrected peer MOD artifacts|`/work/USERS/tkim1/mod` (use `multi_dsprites`, never `mdsprites`)|
|Protected preflight reference|`/work/USERS/tkim1/clevr/with_masks/targets_v1.pt`|
|Official raw Multi-dSprites|`/work/USERS/kevinswk/datasets/multi_dsprites/raw/multi_dsprites_colored_on_colored.tfrecords`|
|Official raw Tetrominoes|`/work/USERS/kevinswk/datasets/tetrominoes/raw/tetrominoes_train.tfrecords`|
|Official raw Objects Room|`/work/USERS/kevinswk/datasets/objects_room/raw/objects_room_train.tfrecords`|
|CLEVRTex Original full PNG/masks|`/work/USERS/kevinswk/datasets/clevrtex_original/data/clevrtex_full`|

`contracts/assets.json` is the executable inventory: exact file paths, size, SHA
and portable destination. It totals approximately16.6GB, copied only when moving
to a new data location. The70k/35k sizes of parent caches do not mean those many
images are trained in the current3500-subset experiment.

`materialize_assets.py` fetches those assets and writes a path config. On Frontier
you can use them in place with `configs/frontier.json`, without copying16.6GB.
Public source URLs and integrity evidence for all downloaded raw datasets are
in `contracts/acquisition_manifest.json`. COCO/OOD artifacts in that historical
acquisition log are not part of the current five-dataset matrix.

## Splits and preprocessing

- CLEVR parent train: fixed35000 selection from0–999 plus1640–70639; no reserved
  IDs1000–1639. Validation1000–1319, evaluation1320–1639. All3500 IDs included.
- Multi-dSprites/Tet parent train:0–5999 plus8000–36999. Validation6000–6999;
  TEST7000–7999. RGB nearest128. Multi-dSprites masks are HWE in raw source,
  Tet masks EHW; the corrected cache has been checked against peer pixel labels.
- Objects Room parent train0–34999; validation35000–35999; TEST36000–36999.
  Raw entities0–3 (sky/floor/two walls) merge to GT background0;4–6 map to1–3.
- CLEVRTex parent train is a fixed35000 subset of official10000–49999, selected
  with RandomState25535000. Validation source5000–5999, TEST source0–999.
  Compact cache row IDs map to raw source IDs via `*_parent_ids.npz`.
  RGB and mask both center-crop box(64,24,256,216), then RGB bilinear128 and mask
  nearest128. TRAIN targets use protected modal64. Official loader SHA is pinned.

All actual selected3500 IDs, validation IDs and evaluation IDs are stored in
`contracts/*_split.json`; the runtime checks them against the reproduced Dataset.
No answer-mask resizing is left implicit.
