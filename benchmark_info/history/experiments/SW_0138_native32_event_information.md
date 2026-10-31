# SW_0138_native32_event_information

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `1765dfd35aea20c380d64b1b9adebcbbd9a6e9b3b07d003733667686ae27124c`

````markdown
# SW0138 native32 event information

Compare actual emitted spike QCC and retained gate QCC on the same frozen trajectories. Native event E comes directly from saved membrane and the actual wrapped threshold; exact S=E*g is required, including zero gate. Recurrence, feedback and reset are unchanged; this is readout information, not individual gate removal.

Three native32 source seeds, fixed320 images, full1024/settle512, B1. Actual labels must exactly reproduce SW0137; six prediction arrays are frozen before GT.

Local and server CPU tests:14/14. Supervisor2699428, seed0 PID2699504 GPU0 and seed2 PID2699498 GPU1 verified live; seed1 queued. No score yet. Publication remains pending scope approval.

Completed all three seeds. Actual spike means .7819186737/.6054271336/.5970138067; gate-readout means .7818825951/.6057708257/.5965825859. FG-ARI and object-IoU paired intervals include zero; foreground-IoU has a small positive interval, without FG improvement. Actual scores exactly reproduce SW0137. No gating-necessity claim. A terminal NameError prevented automatic scoring; completed predictions were preserved and CPU score-only recovery succeeded. Local regression15/15; changed queue was not redeployed while139 fingerprints the original.

````

</details>
