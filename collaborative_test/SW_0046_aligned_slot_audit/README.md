# SW0046 - Aligned single-checkpoint Slot Attention audit

This audit reruns the available official pretrained Slot Attention checkpoint
on the same HDF5 validation IDs 1320-1639 used by SW0042. It is a single
transfer checkpoint, not a comparable three-seed training mean and not a
threshold-selection experiment. The checkpoint was pretrained on a different
CLEVR distribution; it is an audit/reference for the aligned validation slice,
not a matched training comparison.

The TensorFlow runner is based on the saved official inference protocol:
`model.build_model((128,128), 1, 11, 3, model_type="object_discovery")`, seed
0, batch size 1, full 128x128 HDF5 RGB image scaled to [-1,1], and no extra
crop. It hard-assigns each pixel to the maximum soft mask, selects the
background slot by most hard-assigned one-pixel perimeter pixels (smallest
slot ID breaks ties), then maps background to 0 and foreground slot IDs to
1-based instance labels. Ground truth is not used in prediction formation.

The scorer converts both predicted labels and HDF5 masks through the repository
`clevr_mask_patch(..., 8)` path and scores with `evaluate_patch_masks`. It saves
the patch predictions/targets, per-image CSV, and summary JSON. The `run.sh`
CPU launcher targets existing assets at
`/Data0/kevinswk/patch_v2/trained_models/slot_attention_official_patch_eval_20261002`
and refuses to overwrite results.

## Completed aligned result

On HDF5 IDs 1320-1639 (320 valid images), the official single pretrained
checkpoint scores FG-ARI **0.8946414575**, foreground IoU **0.2235111789**,
and matched object IoU **0.2459682554**. The earlier IDs 1000-1319 result was
0.8901145722 / 0.2122510283 / 0.2354868700, so the split/evaluation path shift
is small and stable. Relative to this single-checkpoint reference, the current
SW0042 seed0 gap is concentrated in ARI; its aligned-validation foreground and
object IoUs already exceed this reference. This is still one transfer
checkpoint, not a comparable three-seed training mean or evidence that the
three-seed goal is achieved.

```bash
bash collaborative_test/SW_0046_aligned_slot_audit/run.sh
```

Pure NumPy protocol checks:

```bash
python -m pytest collaborative_test/SW_0046_aligned_slot_audit/test_protocol.py -q
```
