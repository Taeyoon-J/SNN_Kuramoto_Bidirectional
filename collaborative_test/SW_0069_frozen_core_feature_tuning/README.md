# SW0069 - feature tuning against a frozen good core

SW0068 component swaps show that co-adaptation is the main failure. With the
joint LR `3e-5` encoder, learned features passed through the original SW0066
core improve FG-ARI and matched-object IoU (`+.0055/+.0156`) but lose
foreground IoU (`-.0171`). The jointly updated core loses both IoUs even when
fed the original gamma.

SW0069 therefore freezes the complete SW0066 seed1 core and trains only the
native RGB encoder through the existing phase/spike losses. Two coarse arms use
the partially promising encoder LR `3e-5`: no gamma anchor and an anchor weight
`100` that limits drift from the pretrained gamma. Both run five epochs on the
same 2,500 aligned RGB scenes without masks or counts. The core must remain
bitwise unchanged. An arm advances only if all three fixed seed1 pilot metrics
exceed SW0066.

## Result

Neither arm advances. No-anchor scores `.6094/.3821/.4380`; anchor100 scores
`.6141/.3790/.4586`, against `.6317/.3852/.4617`. Anchor100 reduces gamma RMS
drift from `.0487` to `.0090` and nearly preserves object IoU, but all three
metrics remain below baseline. The existing binding loss is not a suitable
feature-generator objective, even with the core frozen. Stop LR, anchor, and
epoch refinement on this loss.
