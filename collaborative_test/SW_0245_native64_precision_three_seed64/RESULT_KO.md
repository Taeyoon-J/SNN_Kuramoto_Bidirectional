# SW0245 result: small improvement, expansion gate not met

Native64, three seeds0/1/2,64 evenly spaced fixed validation images per seed. All models use the same frozen mask-scoring contract. TRAIN64,128 additional updates after the512-update source; whole core/projection frozen. Mask-supervised teacher lineage, not an unsupervised leaderboard result.

| Model | Mean FG-ARI | Foreground IoU | Matched-object IoU |
|---|---:|---:|---:|
| Parent237 on these64 | .879551 | .636343 | .699194 |
| Matched original-loss control | .887473 | .634625 | .693765 |
| Within16 loss | **.892688** | **.672348** | .689237 |
| Comparable Slot | .789959 | .198504 | .198333 |

Within16 versus matched control: +.005215 / +.037723 / -.004529. The grouping gain is smaller than the preregistered +.01 expansion threshold. Do not change that threshold after seeing scores; SW0246 full320 expansion is closed. No full320 best-model update. Incumbent237 remains .872680 / .626110 / .676331 on320 images and three seeds.

Within16 FG by seed:0=.890886,1=.897797,2=.889381. Matched control:0=.900827,1=.880930,2=.880661. Within16 loses FG to control in seed0; the eight-image seed1 .9188 screen overstated the broader effect.

Six actual encoder/Adam checkpoints were audited; every stage has128 actual updates, unchanged core/projection and verified history. Independent rescoring of768 frozen masks reproduced per-image vectors, seed means, model means and deltas within8.89e-16. No training or inference was repeated for the audit recovery.

The original standalone auditor completed all six checkpoint audits, then failed to import the repository during final metric verification because launching an absolute script did not put the repository on sys.path. Preserve that failure. A separate module-based recovery explicitly activates the repository import path and reuses the six verified training audits; ten local/Linux tests including an isolated-process import regression passed. The raw-mask audit then completed. The original246 prerequisite runner stopped on that audit failure; it is not restarted because the independently recovered245 result fails the registered performance gate anyway.

Repeating seed1 training did not produce bitwise-identical final weights to244. Among the same original eight images, masks matched bitwise in4/8 control cases and6/8 within16 cases. This is retained as a numerical/reproducibility limitation; no cause or magnitude of metric variation is inferred from those counts. Before another loss refinement, distinguish training numerical variation from readout randomness using existing checkpoints and arrays. Do not claim the gates became necessary or that70k scaling improved from this screen.

A follow-up CPU comparison canonicalized foreground IDs before comparing the same eight masks. Changed cases are actual partition/support changes, not just label renaming:4/8 control and6/8 within16 partitions were identical. Across all eight, foreground support changed17 patches in control and4 in within16. Repeated-minus-old FG means were -.000103 and -.000427 respectively; within16 on those same eight remains .918373. Thus the broader seed1/64 score .897797 mainly reflects using the broader sample in this specific comparison, while a small repeated-training prediction difference also exists. This is one repeat on eight images, not an estimate of general run-to-run variance or proof of a numerical cause. See REPEAT_DIAGNOSTIC.json; no GPU, training or inference was used.
