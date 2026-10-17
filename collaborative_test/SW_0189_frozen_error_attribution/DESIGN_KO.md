# SW0189: posthoc frozen-mask error attribution

While SW0188 is training, inspect only the already completed SW0186 and SW0187
fixed320/native32 masks and comparable Slot70k archives. No new GPU inference,
optimizer steps, readout thresholds or metrics are introduced. Hash-pin both
complete source scores, every prediction archive and protocol before opening GT.
Recompute the three registered metrics against every saved seed/image vector.

Separately describe foreground misses, background false positives, raw positive
label count accuracy, same-object split pairs and cross-object merged pairs.
Predicted background label0 remains a cluster for FG pair errors, as in FG-ARI;
also identify cross-object pairs merged specifically through that background
cluster. Report patch-pair-weighted and object-balanced rates distinctly.
Report all three seeds and actual GT object-count strata, without selecting
favorable images. These diagnostics are descriptive, not a causal explanation,
a trained count classifier or a replacement FG-ARI evaluation.

Use the completed experiment comparisons to choose between support prediction,
fragmentation and inter-object separation as the next hypothesis. Do not promote
a new incumbent from this analysis or use validation masks for gradient updates.
No claim of unsupervised improvement, gate causality, unseen-data scaling or
full native32 joint70k training follows from this diagnostic.
