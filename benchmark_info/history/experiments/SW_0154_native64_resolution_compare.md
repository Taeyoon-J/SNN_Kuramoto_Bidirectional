# SW_0154_native64_resolution_compare

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `14e521be1e37ce77b363f6c1ab4c63762d96fa8ce5bd3c63bc4701a4263cbd6c`

````markdown
# SW0154 native64 resolution comparison

`resolution64.py` and `score_slot64.py` are the frozen direct-modal Slot64 baseline/scorer. Their hashes are recorded in the completed Slot score artifacts. Candidate source mapping and the one-image resource probe live separately in `native64_model.py` and `resource_probe.py` so they do not alter that scoring contract.

The candidate probe requires the exact registered SW0135 seed1 native32 parity receipt and current source RGB asset validation before constructing a native4096 core. It maps source16 node tensors by the registered 4x4 child-parent relation, maps pairwise state on both axes, and regenerates the identity structural matrix and 64-grid distance buffer. It uses one first-in-order source TRAIN image, batch1, 1024 steps with settle512, and no optimizer or masks. It reports peak CUDA memory and separate timings for core graph creation, rollout, and actual-spike Q/QCC. The rollout timing includes its own graph construction, so graph and rollout times must not be added. A successful single-image run is only a resource/finite-state observation; it is not a native64 score, training admission, or scaling claim.

Local checks:

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'patch_v2_sw')
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m unittest collaborative_test.SW_0154_native64_resolution_compare.test_resolution64 collaborative_test.SW_0154_native64_resolution_compare.test_score_slot64 collaborative_test.SW_0154_native64_resolution_compare.test_native64_model
```

The resource probe is not launched by these CPU tests. A measured resource report must be reviewed before any fixed320 native64 evaluation is considered.

````

</details>
