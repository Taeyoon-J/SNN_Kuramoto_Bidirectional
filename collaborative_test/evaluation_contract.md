# Patch evaluation contract, version 1

- Images: CLEVR HDF5 dataset at `/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5`.
- Train IDs: 0–999. Validation IDs: 1320–1639. Reference test IDs: 1000–1319.
- Ground truth: `mask` array, 128×128, converted with `clevr_mask_patch(..., patch_size=8)` to a 16×16 instance label grid. Background ID is 0. A patch takes its modal pixel ID; ties choose the smallest ID.
- Prediction: one integer instance ID per patch on the same 16×16 grid. Unassigned patches are background (0). Overlapping object masks must be resolved without the ground truth.
- Metrics: `patch_fg_ari`, `patch_foreground_iou`, and `patch_matched_object_iou` from `snn_kuramoto_bidirectional/evaluation.py`. Report each image's score and the mean of valid images; do not expand patches to pixels.
- Selection: classifier choices, thresholds, and loss weights use validation only. The number of true objects and the ground-truth mask must not set the classifier's cluster count or background group.
- Final readout: spike or membrane derived mask. Theta/PLV may be a training loss or diagnostic, but not the final prediction.
- Caveat: the reference test split was already inspected during the starting baseline. Any final success claim must disclose this and confirm the comparison on an untouched split if a comparable Slot Attention prediction becomes available.
