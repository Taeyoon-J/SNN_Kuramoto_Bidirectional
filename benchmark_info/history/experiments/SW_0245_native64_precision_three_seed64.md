# SW_0245_native64_precision_three_seed64

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `98251ac53bce5efe12f4ccc2f4ff5dc0a5c65d496b27e0c43e7256e221e0d762`

````markdown
# SW0245: three-seed64 matched frequency precision confirmation

SW0244's fixed seed1/eight-image screen raised FG-ARI from matched control0.86484 to0.91880, but object IoU fell from0.69670 to0.64295. This is a positive grouping screen, not a full320 result or a new best. Confirm whether the grouping improvement generalizes over seeds and a broader predetermined sample before committing to full320 evaluation.

Keep both registered SW0244 objectives unchanged: original teacher loss versus original plus15 times class-balanced within-object frequency variance. Start each seed0,1,2 from its actual accepted232/237 checkpoint after512 updates. Freeze the whole core including sensory projection. Train each arm with identical fresh Adam, TRAIN64, batch16,128 additional updates, identical sampling/RNG and learning rates. No coefficient sweep, new gate or changed data. Source512 and new Adam128 are separate counters.

Evaluate64 predetermined, evenly spaced offsets0,5,...315 from the same canonical320 validation images, native64 T1024/settle512 and the unchanged production spike readout. Both arms and all three seeds must freeze complete masks; validate the original237 and comparable Slot references before GT is read. Compare means and retain individual seed scores. Six independent exclusive queues use available GPUs0..3 without sharing another user's occupied device. The existing single-GPU70k job is untouched.

Require within16 mean FG at least0.01 above both matched control and parent237, and both IoUs strictly above scoped Slot, before a separate full320 confirmation. Report object-IoU regression honestly. This reused validation sample is not an untouched test; supervised teacher lineage is retained. No64-image score proves the full320 FG0.90 goal or large-data scaling. Actual CPU restoration/gradient admissions and local/Linux tests precede queue starts. Restore and audit actual final encoder/Adam/history and frozen core before accepting results.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `4ccb71ff6f93fec1a7b237569886aa6576eaceb57ff6404d194ccf6f67457631`

````markdown
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

````

</details>
