# SW0186 ? full source RGB encoder plus unchanged QCE

Completed and independently audited. All three training runs and fixed320
native32 conditional/standalone predictions are complete. The original CNN
actually learned in every seed. Standalone three-seed means are
**0.8346805 FG-ARI / 0.5690396 foreground IoU / 0.6898022 object IoU**.

The preregistered FG gate versus SW0181 passed, including a positive paired-image
95% CI. All three metrics exceed comparable Slot70k. Foreground IoU decreased
versus SW0181, so this is the supervised FG incumbent rather than an improvement
in every metric. The unsupervised incumbent remains SW0165.

This is supervised continuation from Source097 with TRAIN512x2/64 proposals,
not fresh70k training. FG0.87, data scaling and gate causality remain unproven.
See [the full result](RESULT_KO.md) and [registered design](DESIGN_KO.md).

CPU admissions, actual GPU credit, complete training manifests/histories, frozen
prediction receipts, the raw score and independent audit are under
`results_archive`. No final metric was used to select prediction thresholds.
