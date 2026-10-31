# SW_0137_native32_source_qcc

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `f948b0b65f6648429ce5a2d20f7394041f73d0b04035810779da1aab906582b1`

````markdown
# SW0137 native32 source QCC evaluation

Evaluate the three registered SW0097 source checkpoints at genuine 32x32 patches against the matched Slot32 baseline. No optimizer updates or new training; GT is read only after all prediction archives are fixed.

Server CPU tests: 11/11 passed; all three actual source states loaded on CPU. Supervisor 2681756; verified seed0 on GPU2 and seed2 on GPU1, seed1 queued with validated SW0136 reuse or fresh inference fallback.

No performance result yet. This does not establish gate necessity or data scaling. Research record is local; GitHub publication remains pending scope approval.

````

</details>
