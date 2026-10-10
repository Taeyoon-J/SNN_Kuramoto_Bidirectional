# Native32 patch evaluation contract

Effective for new experiments after the user's32x32 instruction on2026-10-10.
The primary target is now1024 genuine image patches, not256 predictions
upsampled to32 and not native32 predictions pooled back to16.

- Images remain native128x128 CLEVR RGB from the same canonical HDF5.
- Training pool remains70000 unique IDs0..999 and1640..70639. A smaller pilot
  must disclose its actual subset, initialization exposure and update budget.
- Validation remains the same320 IDs1320..1639. Reserved90640..90959 stays unread
  until a final recipe is frozen. Existing inspected splits remain disclosed.
- Each4x4 pixel cell produces one instance label on a32x32 grid. Ground truth
  uses the production `clevr_mask_patch(mask, patch_size=4)` modal rule, with
  ties choosing the smallest instance ID. Background is0.
- Our model must compute1024 patch features and1024 patch dynamics from the
  actual image/encoder. Replicating a16-grid feature cache or label grid is
  not a native32 candidate. Spatial replication of node parameters may be a
  disclosed warm start; it does not replace native32 feature extraction.
- Final predictions must come from spike/membrane signals. No mask, true
  object count or ground-truth label enters training or prediction selection.
- Freeze the complete320 predictions and their provenance/hashes before
  reading validation masks. Score with unchanged `evaluate_patch_masks`:
  `patch_fg_ari`, `patch_foreground_iou`, `patch_matched_object_iou`.
- For the comparable Slot baseline, reuse the own70000-trained seed0/1/2
  epoch10 checkpoints' frozen native128 pixel labels, preserving their existing
  perimeter-background mapping. Apply the same4x4 modal rule to those labels.
  This changes evaluation resolution, not Slot training or inference.
- First reproduce each baseline's original16 scores from those same pixel
  labels to bind the new evaluation to the already audited models. The
  original16 evaluation is a provenance audit, not a new16 training experiment.
- Report every seed, each metric's valid-image count, all per-image scores
  and the unweighted seed0/1/2 mean. Slot and our candidate use the same IDs,
  ground-truth grid and metric implementation. Keep their prediction-only
  background rules explicit; do not tune them against validation masks.
- Success requires the candidate's three-seed mean to strictly exceed the
  comparable native32 Slot mean on all three metrics. FG-ARI is the immediate
  development priority; success on FG-ARI alone does not complete the goal.

Historical16 scores, SW0114's pooled16 endpoint, released checkpoints trained
on another renderer, oracle allowed-label diagnostics and a single best seed
cannot substantiate this native32 success claim. Existing already-running
SW0134 remains immutable historical16 work; no new16 training is scheduled.
