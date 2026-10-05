# SW0068 - joint image-to-feature generator and core

Frozen-feature substitutions, smoothing, and self-bootstrap failed because they
do not add object-boundary information. The native encoder was trained only for
image reconstruction and the current core trainer consumes cached gamma, so its
binding loss cannot improve feature content or channel ordering.

SW0068 puts the pretrained native RGB feature generator in the differentiable
training path. RGB and cached gamma use the verified aligned training IDs
`0-999,1640-3139`; masks and object counts are never loaded. The first two
epochs reproduce the fixed-feature core recipe, followed by eight joint epochs.
Only the encoder learning rate differs between the two coarse pilot arms
(`3e-6` and `3e-5`). Core LR remains `3e-4`, and graph initialization remains
fixed at seed0 based on SW0064-66.

The trainer must reproduce cached gamma before its first update, show a finite
nonzero encoder parameter change, and export learned validation gamma directly
from HDF5 RGB. Stage 1 uses seed1 and fixed IDs1320-1351. An arm advances only
if it improves all three mask metrics over the SW0066 seed1 checkpoint; feature
drift and object count are diagnostics, not selection substitutes.

The first launch was stopped after epoch1 because a fresh explicit DataLoader
generator changed the baseline shuffle order. The corrected `SW0068b` launch
uses the same global RNG state reached after core construction as SW0066, so
the warmup isolates the later encoder intervention. Corrected epoch1 loss is
`23.7410427216`, matching SW0066's `23.74104272`.

## Result

Neither joint arm passes. Against SW0066 seed1
`.6317/.3852/.4617`, LR `3e-6` scores `.6006/.3532/.4081` and LR
`3e-5` scores `.6153/.3539/.4221`.

A component swap localizes the loss. LR `3e-5` learned features in the frozen
SW0066 core score `.6371/.3681/.4773`, improving FG-ARI and object IoU while
trading off foreground IoU. The jointly trained core with native gamma scores
`.6402/.3325/.4015`: grouping rises but both IoUs collapse. Feature diagnostics
show only a small ratio change (`.22540 -> .22609`), so the main failure is
harmful core co-adaptation rather than wholesale feature collapse. SW0069
therefore freezes the good core and tunes only the encoder, with and without a
strong gamma anchor.
