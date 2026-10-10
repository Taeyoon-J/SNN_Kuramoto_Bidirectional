# SW0139 seed 1 partition-error analysis

This read-only, post-hoc decomposition covers the same 320 validation images (IDs 1320–1639) and the frozen source97, control, and context_residual predictions. It used the saved predictions; it did not rerun inference or alter training. The source evaluation and prediction hashes are recorded in the [diagnostic result](results_archive/partition_diagnostic_seed1.json) (SHA-256 `27d15a4d2a225cdf37a6bf81f1923e76f46919e3c82672b835957ddebdde4ab0`).

| Arm | False-merge pairs / all true-FG pairs | False-split pairs / all true-FG pairs | Missed true-FG patches | Predicted nonzero labels | True objects | BG-column merge pairs / all true-FG pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source97 | 0.047292 | 0.046832 | 0.130703 | 5.815625 | 6.312500 | 0.017335 |
| control | 0.054390 | 0.051094 | 0.155309 | 5.687500 | 6.312500 | 0.022439 |
| context_residual | 0.086560 | 0.047700 | 0.256129 | 4.503125 | 6.312500 | 0.055274 |

The reported pair fractions are means of per-image ratios over true-foreground patch pairs. They are not fractions of all image patches. False merges count pairs of true-foreground patches from different true objects that share a predicted label; false splits count pairs from the same true object assigned different predicted labels. The background-column merge fraction is the subset of false-merge pairs assigned predicted label 0. Predicted-label counts are image means; the nonzero-label count is measured on the full prediction, while the true-object count comes from the target.

Relative to source97, context_residual increases the false-merge fraction by 0.039268, of which 0.037939 is in the predicted-background column (96.6%). Its false-split fraction changes by only 0.000868, while the missed-foreground fraction rises by 0.125426 and the mean number of nonzero predicted labels falls by 1.312500. This points to foreground loss and background-label merging as the main observed error pattern, rather than a comparable increase in object fragmentation. Control shows smaller increases in both merge and split fractions and a missed-foreground increase of 0.024606 relative to source97.

This is a descriptive decomposition on one seed, not a causal-layer result or a model-promotion test. It does not establish gating or scaling, and it does not change the source97 incumbent. Use it only to inform a separately preregistered next hypothesis after the SW0140 endpoint is complete; do not change the closed SW0139 recipes based on this post-hoc analysis.