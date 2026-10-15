# SW0169 completed training; fixed320 evaluation pending

Both registered TRAIN512 arms completed all 64 B16 updates (two passes).
The public CPU validator rechecked actual model, optimizer, complete exposure
order, diagnostics, final durable boundary and source fingerprints. Root then
independently checked the collected manifests, all update counters and current
local code hashes. No validation masks were opened for this collection.

| TRAIN8 diagnostic | Initial, both arms | Final history_live | Final history_detached |
|---|---:|---:|---:|
| Mean raw event-on fraction | .061436 | .058467 | .057854 |
| Mean event-transition fraction | .117920 | .112863 | .112115 |
| Mean all-pair Q | .001785 | .001856 | .001717 |
| Mean standalone foreground patches / 1024 | 2 | 5 | 4 |
| Standalone foreground patches min / max | 0 / 8 | 0 / 16 | 0 / 16 |
| Mean constrained maximin loss | .496256 | .496956 | .498149 |

These are the same fixed eight TRAIN examples, not validation scores. Q means
include the diagonal and are not evidence of same-object separation. Selection
is nonconstant, but this alone has not produced useful standalone foreground:
the readout is almost empty, not identically empty in every row. The fixed-eight
objective increased slightly; shuffled training-update loss is a different
quantity (first8/last8 updates: live .506217/.505480; detached .506529/.503736).

Full fixed320 prediction automatically started from the two final checkpoints.
No favorable endpoint, intermediate checkpoint or threshold is selected. The
registered co-primary gate is still pending; these TRAIN diagnostics do not
establish its final outcome. Existing SW165 remains the performance incumbent.

Final manifest SHA256: live
`678886c1f18602608a08dcdc144d9a2c42b3f11e3710a15967240ae51e2f64bb`;
detached `7856158558ea0a062701c89663d0f95bb3e886afbdcd9ff6a758883060d585e2`.
Evidence: `results_archive/completed_training_20261015/server_records.json`
and `independent_summary.json`. No dataset or checkpoint bytes are published.
