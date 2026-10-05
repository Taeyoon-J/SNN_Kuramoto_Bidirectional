# SW0067 - seed-stable foreground classifier

SW0066 fixed the first unstable activation transition. It improved grouping and
matched-object IoU for both weak seeds, but seed1 foreground IoU fell. The
diagnosis is specific: fixed threshold `.35` predicts foreground fractions
`.4075` and `.5544` for the two candidates while the fixed validation slice is
near `.1854`. Seed1 count simultaneously improves from mean `5.00` to `5.375`
against target `5.531`, so the remaining seed1 loss is foreground selection,
not object grouping.

SW0067 changes only the label-free readout. Instead of applying one absolute
synchrony threshold to models with different affinity scales, it searches a
per-image threshold whose non-background components cover a fixed `.22` of
patches. The value is registered from earlier full-validation foreground
diagnostics, before this experiment; no image's mask or count enters prediction.
The largest synchrony component remains background and connected components
still determine object count.

Stage 1 uses SW0055 seed0 and the SW0066 seed1/2 candidates on fixed
IDs1320-1351, T256/settle64. It compares the adaptive classifier against the
same checkpoints' fixed `.35` readout. Promotion requires the three-seed mean
to improve all three mask metrics and mean count MAE not to increase.

## Result

The fixed readout mean is `.6187/.4785/.4369`; adaptive coverage scores
`.5120/.4917/.3464`. Foreground IoU gains only `.0132`, while FG-ARI and
matched-object IoU fall `.1066/.0905`, and count MAE worsens `1.927 -> 2.490`.
Matching foreground area removes object patches and collapses groups. No
coverage or threshold refinement is warranted; the direction is stopped.
