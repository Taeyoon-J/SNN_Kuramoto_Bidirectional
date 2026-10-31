# SW_0070_graphinit_full_mean

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `c122347dba2de239ae072fc9fbd94fec2032c65056460f3c759ddaf402785ebb`

````markdown
# SW0070 - full three-seed graph-initialization mean

SW0066's registered per-seed gate failed because seed1 foreground IoU fell on
32 images. Its actual three-seed pilot mean nevertheless improved all three
goal metrics. Since the goal is explicitly the seed0/1/2 mean, SW0070 measures
the full fixed contract rather than discarding this evidence.

Seed0 is the natural graph-init0 SW0055 checkpoint and reuses its validated
320-image long result. SW0066 seeds1/2 are evaluated on IDs1320-1639,
T1024/settle512 and the registered primary spike threshold `.50`. The summary
compares the resulting mean directly with SW0055/SW0057 spike, the current
best three-seed all-metric reference.

## Result

The full seed0/1/2 mean is `.677448/.435439/.479878`. This improves the prior
SW0055 three-seed spike reference by `+.050715/+.012099/+.039759` for FG-ARI,
foreground IoU, and matched-object IoU. The seed results are
`.817116/.730071/.643890`, `.668158/.344169/.496278`, and
`.547068/.232078/.299466`. The graph-init0 intervention therefore becomes the
new stable-core baseline even though seed spread remains the main limitation.

````

</details>
