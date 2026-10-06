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

