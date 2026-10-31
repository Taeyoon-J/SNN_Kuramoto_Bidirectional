# SW_0087_matched_slot_comparison

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `6319752241d4f1714e27536bac736631a2086a831ab9017d0ade3cb4bd235f4f`

````markdown
# SW0087: fixed-contract matched Slot goal comparison

SW0087 waits for the completed SW0056 three-seed matched-data Slot summary and
compares it with the completed SW0072 three-seed mean. It accepts only the
aligned HDF5 validation slice IDs 1320-1639, seeds 0/1/2, 320 images, and
GT-free Slot prediction provenance.

The goal flag is true only when SW0072's mean is strictly greater than the
SW0056 mean for FG-ARI, foreground IoU, and matched-object IoU independently.
No metrics are averaged into a composite score, and equality fails the gate.
The JSON and Markdown outputs retain every mean, delta, and Boolean decision.

The completed comparison passes the strict goal on all three metrics. SW0072
scores `.781407/.478038/.616746`, while matched Slot scores
`.571056/.225541/.171659`; the respective margins are
`+.210352/+.252498/+.445087`. The absent watcher was resumed from the two
complete summaries after correcting its SW0072 server path; no evaluation was
repeated.

The durable CPU-only watcher performs no model inference and consumes no GPU:

```bash
nohup bash collaborative_test/SW_0087_matched_slot_comparison/wait_and_compare.sh \
  </dev/null >trained_models/SW0087_GOAL_COMPARISON_QUEUE.log 2>&1 &
```

````

</details>

## results/SW0087_goal_comparison.md

<details>
<summary>SW0087_goal_comparison.md</summary>

Source SHA256: `f7d892576df3108612facacf9e25012d91634a70e8f0d768ad849e8f5a9c7e4d`

````markdown
# SW0087 matched Slot goal comparison

HDF5 validation IDs1320-1639; 16x16 patch masks; three-seed means.

| Metric | SW0072 mean | Matched Slot mean | Delta | Strictly exceeds |
|---|---:|---:|---:|:---:|
| fg_ari | 0.781407 | 0.571056 | +0.210352 | true |
| foreground_iou | 0.478038 | 0.225541 | +0.252498 | true |
| matched_object_iou | 0.616746 | 0.171659 | +0.445087 | true |

Goal achieved on all three metrics: **true**

````

</details>
