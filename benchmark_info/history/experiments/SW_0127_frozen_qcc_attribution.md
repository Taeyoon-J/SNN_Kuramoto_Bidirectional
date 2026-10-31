# SW_0127_frozen_qcc_attribution

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `81b8efb162ab03cbdb61b4d3d50b11e9531d708fd39edfbbe935801a4dab1e66`

````markdown
# SW0127 frozen QCC attribution

This read-only diagnostic compares the SW0097 seed-1 native core, the same source core with an untrained SW0126 history adapter, and the two completed SW0126 seed-1 trained cores. Every arm uses the registered QCC classifier on actual component-spike traces with the original 16×16 readout settings.

The CLI first supports a small source-rollout parity check on cached TRAIN and validation gamma. Full evaluation saves and verifies all four prediction grids before it opens the canonical HDF5 masks. It then applies the existing modal 8×8 target and standard three-metric scorer. The source97 per-image scores must reproduce within `1e-10`; a mismatch is retained as a failed result.

This is a frozen-checkpoint attribution diagnostic. It makes no training changes and cannot establish that any one SW0126 mechanism caused an endpoint difference.

Example commands:

```text
python -m collaborative_test.SW_0127_frozen_qcc_attribution.run --preflight-only --output results_archive/preflight.json --device cuda:0
python -m collaborative_test.SW_0127_frozen_qcc_attribution.run --output results_archive/qcc_attribution --device cuda:0
python -m collaborative_test.SW_0127_frozen_qcc_attribution.dispatcher
```

The dispatcher holds one shared GPU lease at a time, waits for an idle GPU, runs the preflight before evaluation, and preserves failed attempts without automatic retry.

````

</details>
