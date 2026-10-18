# SW0215 TRAIN-derived loss/decision alignment

Four local/four Linux tests passed. Frozen214 CPU replay was exact before new
mask scoring. No optimization or core forward. TRAIN-only log-odds shift3.263823
corrects the ideal positive/negative normalized BCE objective's decision scale.

Corrected primary .781651/.445474/.664934 versus original
.788414/.445474/.674335: no improvement established. Corrected geometry merges
nothing and equals the original. No DEV threshold sweep. See RESULT_KO.md and
the completed_v1/ 80-endpoint independent audit. Full320 .848615 mean is unchanged.
