# SW0066 - fixed graph initialization across run seeds

SW0064 located seed instability at graph-to-Kuramoto conversion. SW0065 then
showed that copying only the good seed0 graph generator into seed2 raises the
32-image result from `.3895/.2003/.1822` to `.7036/.3081/.5376`; swapping the
input projection or Kuramoto parameters does not rescue it.

SW0066 tests the smallest train-time correction: all runs initialize only the
learned graph generator from model seed 0, while the declared run seed still
controls every other module and data order. The graph remains trainable. This
does not copy a validation-selected trained graph and uses no masks, counts, or
labels.

Stage 1 retrains SW0055 seeds 1 and 2 for the same 10 epochs on the same 2,500
unique scenes and evaluates candidates plus their original baselines on fixed
IDs1320-1351, T256/settle64, threshold .35. It advances to full 320-image long
evaluation only if both seeds improve all three metrics. SW0055 seed0 is the
natural graph-init-seed0 arm and need not be retrained for the final mean.
