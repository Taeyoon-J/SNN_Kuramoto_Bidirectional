# SW0244 result

On eight predetermined native64 validation images, seed1, strengthening class-balanced within-object frequency precision improved FG-ARI substantially. These are mask-supervised screening results, not the full320 three-seed benchmark.

| Model | FG-ARI | Foreground IoU | Matched-object IoU |
|---|---:|---:|---:|
| Parent237 | .857103 | .462292 | .690716 |
| Matched extra-training control | .864838 | .465304 | .696699 |
| Within16 loss | **.918801** | **.516084** | .642951 |
| Comparable Slot seed1 | .806661 | .208314 | .221033 |

Candidate versus matched control: +.053963 / +.050780 / -.053748. The registered FG-led expansion gate passed, but object IoU regressed. All three candidate metrics exceed this scoped Slot reference. Do not claim every metric improved versus our parent.

Both actual final encoder/Adam checkpoints were restored on CPU. Adam counters/history contain128 new updates; all core/projection parameters remained unchanged. Independent rescoring of32 masks (four models times eight images) reproduced all means/deltas within1.11e-16. Both original objectives started from the same512-update seed1 checkpoint and used identical data/batches. TRAIN within-object variance fell from .00701223 to .00274537 in the candidate, versus .00654130 in the control; source-center fidelity and gamma anchoring worsened, so this is a precision/grouping tradeoff, not proof of a solved mechanism.

Proceed to SW0245 with unchanged objectives and three seeds on64 evenly spaced validation images. The full320 incumbent remains SW0237 (.872680 / .626110 / .676331). The full320 FG0.90 target and70k scaling remain unproven. No gate contribution claim follows from this loss screen.
