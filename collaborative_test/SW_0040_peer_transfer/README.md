# SW_0040: Peer spike-synchrony readout transfer

This evaluator applies the peer's `spike_synchrony_components` classifier to an
existing checkpoint on our fixed validation IDs 1320-1639. It reads matching
gamma rows and CLEVR HDF5 masks, then sweeps synchrony thresholds
`0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50`. Inference uses the local
SW_0038/SW_0035 protocol: 256 rollout steps, settle 64, membrane threshold 2.0,
and minimum component size 2. The default classifier forms
connected components from product-combined per-component spike-trace
correlations and drops the largest component as background. No target object
count or mask participates in prediction.

For every threshold the JSON reports foreground ARI, foreground IoU, matched
object IoU, and object-count exact accuracy, MAE, bias, and within-one accuracy.
Count targets are computed from unique nonzero HDF5 instance IDs only after
predictions are formed. The report also includes a same-checkpoint phase PLV summary as a diagnostic,
separate from the spike-derived predictions. Passing `--peer-targets /work/USERS/tkim1/clevr/with_masks/targets_v1.pt` and `--peer-manifest` additionally scores the same predictions against peer labels. The verified identity map 1320->1320 through 1639->1639 is allowed for this cross-target diagnostic: our checkpoint trained only on IDs 0-999, while those rows are peer-train rows. The JSON marks this as diagnostic-only and records the pairing evidence (same-index binary IoU .27338 and object-count correlation .94091; offset +1 and reversed alignments were near chance). This does not score a peer checkpoint; formal peer-checkpoint validation must use peer validation IDs 6000-6999.

Usage: `run.sh GPU_ID CHECKPOINT OUTPUT_JSON`. The script only runs when invoked;
this transfer setup has not been launched on the server. Set non-default
per-region or geodesic construction flags directly when using `evaluate.py` so
the strict checkpoint load matches how the core was trained. SW_0039 training
remains on hold and must not be launched as part of this evaluation.

## Peer BIM6 count diagnostic

`count_peer.py` evaluates prediction-derived connected-component counts on peer
validation IDs 6000-6999 for each BIM6 seed. Ground-truth instance counts are
read only after predictions are formed. The three-seed results at 256 steps,
settle 64, synchrony threshold .35 are:

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Exact count accuracy | .192 | .189 | .204 | .195 |
| Count MAE | 1.635 | 1.624 | 1.543 | 1.601 |
| Within-one accuracy | .522 | .547 | .572 | .547 |
| Count bias | -.819 | -.852 | -.227 | -.633 |

Predicted mean counts were 5.416 / 5.383 / 6.008 versus target mean 6.235.
Dynamic count inference works, but accurate count is not solved: exact
agreement remains about 19.5%, with systematic undercounting. Seed JSONs are
under `results/peer_count/`. `count_peer.py` accepts `--steps`, `--settle`, and
`--synchrony-threshold` to reproduce alternate inference protocols, including
the peer's long-rollout settings.

### Long-rollout count diagnostic

The peer BIM6 checkpoints were also evaluated on the same 1000-image peer
validation split at 1024 steps, settle 512, synchrony threshold .10. Relative to
the 256/64/.35 count readout above, results improve modestly, but exact count
remains low:

| Count metric | Seed 0 | Seed 1 | Seed 2 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Exact count accuracy | .215 | .187 | .209 | .203667 |
| Count MAE | 1.499 | 1.595 | 1.488 | 1.527333 |
| Within-one accuracy | .571 | .552 | .581 | .568 |
| Count bias | -.211 | -.867 | -.358 | -.478667 |

Predicted mean counts were 6.024 / 5.368 / 5.877 versus target mean 6.235.
Dynamic count inference improves slightly under long rollout, but accurate
count is not solved (20.4% exact agreement on average). Per-seed JSONs are in
`results/peer_count_long/`.
