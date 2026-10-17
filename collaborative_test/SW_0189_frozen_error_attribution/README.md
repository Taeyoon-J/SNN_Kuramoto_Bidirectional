# SW0189 ? completed frozen-mask error attribution

CPU-only analysis of 6,720 completed endpoint/seed/image rows. No model forward passes or optimizer steps. Six tests passed; all registered metrics recomputed within 3.34e-16. Fragmentation, cross-object merging, foreground misses and background leakage remain visible. Raw mask-label count is not a trained count classifier.

See [findings and limits](RESULT_KO.md). SW0188 remains the running model experiment; no incumbent changes.

Publication scope: code, cohort aggregate summary and execution receipt only. The image-level report remains private and is ignored by Git. Automatic approval review rejected its public push; no image-level diagnostic payload is included in the replacement public commit.
