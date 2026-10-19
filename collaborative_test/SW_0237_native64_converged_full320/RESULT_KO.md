# SW0237: new audited native64 full320 reference

| Model | FG-ARI | Foreground IoU | Matched-object IoU |
|---|---:|---:|---:|
| SW0237 three-seed mean | 0.872680 | 0.626110 | 0.676331 |
| Previous SW0227 mean | 0.858916 | 0.605286 | 0.674313 |
| Comparable Slot Attention mean | 0.786902 | 0.198201 | 0.195203 |

The new checkpoint improves the previous full320 reference means by0.013764 /0.020824 /0.002017. It strictly exceeds the comparable Slot Attention means on all three metrics. The native64 FG-ARI0.90 target remains unmet. This is a comparison of three-seed means; individual seeds do not all exceed Slot Attention on every metric.

The best full320 FG-ARI seed is seed0:0.885240 /0.798583 /0.697130. Seed1 is0.869394 /0.592512 /0.675732 and seed2 is0.863406 /0.487234 /0.656130.

The evaluated models are the exact accepted SW0232 checkpoints:64 unique TRAIN images per seed,512 cumulative Adam updates, whole encoder plus the actual8-to4 sensory projection trained. The remaining core and production spike readout are fixed. SW0237 adds no training; it confirms these checkpoints on the fixed320 native64 validation images. Longer optimization improves the mean relative to the128-update reference. This does not demonstrate a benefit from more unique training images.

All three prediction protocols and comparable Slot artifacts were validated before evaluation GT. Independent validation rescored2,880 frozen masks across the original baseline, candidate and Slot Attention and reproduced means/bootstrap statistics with maximum error5.55e-16. Each seed also passed exact mask/frequency replay on its registered replay image. The full private audit and model references are preserved; `PUBLIC_RESULT.json` omits raw arrays and private dataset/server metadata.

This lineage uses TRAIN mask supervision. Slot Attention uses its reconstruction-training objective, so this is not an unsupervised leaderboard claim. These are repeatedly used validation images, not an untouched final test. Cross-layer gating causality remains unproven. Spike-to-Kuramoto and synchrony-to-graph feedback remain disabled; local previous-spike dendritic input and membrane-reset paths remain present.

SW0237 is now the tracked full320 reference. The previously registered70k training continues as SW0240, preserving its initial SW0227 checkpoint selection and Adam history; its final scores are pending. SW0241 separately tests downstream event information with these same accepted SW0232 models.
