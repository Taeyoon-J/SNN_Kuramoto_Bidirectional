# SW0063 - label-free edge-aware gamma consistency

The peer feature curve shows that reducing within-object feature variation is
the largest remaining lever. SW0062 established that off-the-shelf DINOv2 is a
poor CLEVR-domain match after PCA: its normalized within/between centroid ratio
is `1.018`, versus `0.225` for the native gamma. SW0063 therefore preserves the
native encoder basis and applies a label-free edge-aware local consistency step.

For each 16x16 patch, native gamma is averaged over its 3x3 neighborhood with
weights from RGB patch similarity and spatial distance, then mixed 50/50 with
the original gamma. RGB comes from the same image; masks, counts, and labels are
never used. The seed0 checkpoint, IDs1320-1351, T256/settle64, raw gate,
spike-CC threshold .35, and every downstream setting stay fixed.

This is a 32-image direction pilot. It advances only if all three fixed metrics
improve over the native-gamma SW0054 reference.

The label-free transform reduced the normalized within/between feature ratio
from `.2254` to `.2193`. FG-ARI improved from `.668520` to `.697825`, but
foreground IoU and matched-object IoU changed by `-.002495/-.002087`. This is
evidence that local feature consistency improves grouping while the gain is
lost before correct foreground/object overlap. Because all three endpoints did
not improve, the transform is stopped before training. Reports are in
`results/`; SW0064 traces where the gain disappears.
