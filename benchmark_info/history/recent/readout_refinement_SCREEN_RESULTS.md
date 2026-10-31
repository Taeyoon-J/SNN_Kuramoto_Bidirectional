# readout_refinement/SCREEN_RESULTS.md

<details>
<summary>SCREEN_RESULTS.md</summary>

Source SHA256: `2bdba860145bc6c061cd94bead9653abb836d8fdf504da3c04cf00187b152b2d`

````markdown
# SW0251 controlled readout screen

This is a small validation experiment, not the fixed320 preservation gate or a new-dataset benchmark result.
Same actual SW240 activity, TRAIN64, validation IDs1000–1015, 20 epochs, 1280 Adam steps, identical final-mask loss.
Core and encoder remain frozen. Benchmark code and original checkpoints are preserved.

| Readout | Seed0 FG-ARI | Seed1 FG-ARI | Two-seed mean | Mean foreground IoU | Mean object IoU |
|---|---:|---:|---:|---:|---:|
| Original SW240 | .880281 | .882659 | .881470 | .815321 | .711661 |
| Soft prior (1), metric gain 1 | .860626 | .881055 | .870840 | .708322 | .682559 |
| Original prior (4), metric gain 16 | .882094 | .884013 | .883053 | .814160 | .723516 |

Best epochs were selected by validation FG-ARI independently for each seed and configuration.
Soft prior loses .010630 mean FG-ARI and is rejected at this screen; it is not being extended to seed2.
Strong metric changes mean FG-ARI by +.001583 on these two seeds. Its seed2 confirmation is queued after the exact source cache is complete and GPU memory is admitted.

The selected strong-metric checkpoints have **zero positive-group-to-other-positive-group reassignments** on validation.
Their improvement comes from changed background/foreground assignments. Other epochs can change object assignments, but this experiment has not demonstrated that these changes improve validation results.
Do not describe this as a successful instance reassignment learner or a completed original-data preservation gate.

The original three-seed v1 comparison continues independently. No configuration is promoted to the new datasets on the basis of this screen alone.
Authoritative reports are SHA-verified copies under SW0251_background_mask/collected/rf_seed*_*.json; the baseline and queue receipt are in refinement_seed2_queue_launch.json.

## Completed three-seed screen

Seed2 strong-metric training also completed all1280 steps; best epoch6, validation FG-ARI .826703, foreground IoU .795725, object IoU .656926. Its selected checkpoint likewise has zero positive-group reassignments. The full same-data v1 comparison is now complete:

| Readout | Three-seed mean FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| Original SW240 | .863203 | .794869 | .695189 |
| Learned background only | .854945 | .623144 | .661761 |
| V1 final-mask supervised residual | .864096 | .814651 | .701184 |
| Strong metric gain16 | .864270 | .808015 | .701320 |

The gain16 head exceeds the ordinary residual by only .000174 mean FG-ARI, with lower foreground IoU. This small validation study does not justify adding it to production selection or extending its training. The unchanged planned v1 TRAIN256/VAL64 comparison continues across all three seeds. No new refinement model is promoted, no fixed320 gate is claimed, and no new-dataset result is inferred.

The background-only head loses .008259 mean FG-ARI at this screen. The direct final-mask residual retains FG-ARI and improves the other two metrics here. These observations support continuing that controlled comparison, without establishing generalization or an object-reassignment benefit.

````

</details>
