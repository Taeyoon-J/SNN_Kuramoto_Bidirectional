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
