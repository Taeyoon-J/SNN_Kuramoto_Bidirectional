# SW_0248_native64_convergence_full320

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `d68c690f9f3b3771e2207b923eeaefb5dffc14aeac717acde323571f513e83e8`

````markdown
# SW0248: conditional selected-arm convergence full320 confirmation

SW0246 remains closed because245 failed its loss-only expansion criterion. This separate branch expands only a qualified optimizer-budget arm from247. Preserve247's preregistered rule: FG mean gain at least0.01 over that arm's128-update controls and parent237, with all three metrics above scoped Slot; among qualifying arms choose highest mean FG, then arm name for ties. Validate all six actual512-update checkpoint audits and the independent576-mask audit before accepting selection. A negative247 screen closes this branch without GPU acquisition.

For each seed0/1/2, restore the selected247 control or within16 checkpoint exactly. The source model has512 prior updates plus512 fresh stage updates, TRAIN64 and a frozen entire core/projection. No optimizer step or architectural/readout change occurs here. Reuse its32 frozen predictions and generate the remaining288 on the original fixed320 native64 images. Replay offset0 mask and frequency bitwise. Keep T1024/settle512, canonical RGB, inference seeds and per-row recovery unchanged.

Require all three full protocols, parent237 and comparable Slot predictions before evaluation GT. Reproduce the accepted237 and Slot full320 scores, retain all seeds, and independently rescore2880 masks. The FG0.90 claim requires mean FG at least0.90 plus all three means strictly above Slot. Masks remain supervised-lineage and evaluation is reused validation, not an untouched test. The other512 and128 arms remain evaluated only on32; no matched-control full320 contribution claim follows.

Run local/Linux selection, real serialized encoder restoration, isolated-import and scoring-order tests. The CPU-only prerequisite runner waits for an audited positive247 screen, then restores and admits all three actual selected models before exclusive GPU queues0..3. A module-launched auditor performs three actual CPU restorations and independent full320 rescoring automatically. Keep registered sources immutable, other users' GPU ownership intact and the existing70k job unchanged. No full320 best-model update until the completion audit finishes.

````

</details>
