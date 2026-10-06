# SW0090 — large unique-scene scaling and qualitative masks

This experiment keeps the strongest SW0072 architecture and readout contract
fixed, while scaling the native frozen-encoder training set from 2,500 to
70,000 unique scenes, matching the official Slot Attention CLEVR6 scene count.
IDs 1320–1639 remain untouched validation data.

The same run saves epochs 1, 3, and 10 to measure whether additional passes over
the larger set help. Three seeds are evaluated on all 320 held-out
scenes with the fixed 1024-step spike connected-component readout at threshold
0.50. Qualitative output uses three pre-registered validation IDs (1320–1322)
and exports originals, ground-truth masks, predicted masks, and mask grids.

## Early seed0/1 result

While seed2 continues, the fixed full-320 evaluation is complete for seeds 0
and 1. Their epoch-1 mean is `.782584/.615775/.623642`. Against the same-seed
SW0072 mean `.785606/.553558/.607371`, 70,000 unique scenes change the metrics
by `-.003023/+.062217/+.016271`: foreground and matched-object IoU improve,
while FG-ARI is essentially flat but slightly lower. Epoch 3 trades more FG-ARI
for IoU (`.764199/.651509/.624770`); epoch 10 degrades to
`.708914/.541804/.583084`. This establishes early stopping as necessary on the
larger dataset. Final claims wait for seed2.

