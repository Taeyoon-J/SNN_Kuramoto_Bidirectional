# SW_0142_native32_masked_completion

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `a7b8ae906b5b0eb57ddede52107a59f85e086a57c18e0e64f77a2a8598fcfb34`

````markdown
# SW0142 native32 masked completion

SW0142 tests whether a fixed native32 spike-slot model improves through masked-image completion, comparing a frozen-core `encoder_graph` arm with a `core_trainable` arm. The source is the already-trained SW0097 seed1 model, mapped through the frozen SW0135 native32 stack. It has prior 70,000-image training at 16×16 resolution; this pilot is not fresh native32 pretraining and does not establish a scaling result.

The nine-stage queue is: write the 4,096-image float16 masked spike cache; warm the shared binder/decoder for 256 B16 updates; run one disposable B16 preflight per arm; train each arm for 256 B16 updates; predict the same fixed 320 images per arm; then score on CPU. Cache/warm/training/prediction do not read ground truth. The scorer validates and freezes prediction/reference hashes before loading the 320 targets. It reports the primary native32 P-derived hard-slot masks and the unchanged QCC readout. The registered gate uses 10,000 paired-image bootstrap samples with `RandomState(139)` and common resample indices.

The protocol also records first-64-TRAIN reconstruction diagnostics for the original assignments, row-permuted P, and within-coarse-RGB-bin permuted P. These diagnostics are descriptive, with no pass threshold. A reconstruction or nonzero-gradient result alone does not show object segmentation or gate necessity. Failure closes this recipe; it does not authorize sweeps or automatic seed expansion.

From the repository root, run the CPU tests with CUDA hidden:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m unittest collaborative_test.SW_0142_native32_masked_completion.test_run collaborative_test.SW_0142_native32_masked_completion.test_queue -v
```

The registered CLI is `run.py --stage {cache,warm,preflight,train,predict,score}`. `stage_queue.py --dry-run` displays the fixed stage plan without creating outputs. Running the queue creates artifacts and requires the reviewed owner-aware GPU lease environment; preserve existing outputs and do not restart an interrupted stage automatically.

````

</details>

## results_archive/completion_20261012/RESULT.md

<details>
<summary>RESULT.md</summary>

Source SHA256: `16c10eded06f4102917b94afef18323cc014b5531a6b0d4715ab159132eaa640`

````markdown
# SW0142 completion, 2026-10-12

The registered seed1 masked-completion pilot failed its promotion gate and is closed without seed expansion or hyperparameter sweeps. Both 256-update/4096-exposure arms and both 320-image prediction stages completed; the CPU scorer froze references before reading GT. All 18 metric means and six paired bootstrap intervals were independently recomputed from the archived per-image arrays.

| Seed1 model/readout | FG-ARI | Foreground IoU | Matched object IoU |
|---|---:|---:|---:|
| Encoder/graph trained, P | .184229 | .277616 | .083670 |
| Encoder/graph/core trained, P | .291697 | .412178 | .151097 |
| Encoder/graph trained, QCC | .748880 | .607114 | .580991 |
| Encoder/graph/core trained, QCC | .749859 | .604018 | .578109 |
| Source seed1, QCC | .754135 | .606900 | .582201 |
| Comparable Slot seed1 | .841819 | .203417 | .222115 |

The joint P assignment improved over the frozen-core control (FG +.107468, paired 95% CI [.078425, .135617]) but remains substantially below source QCC (FG -.462438, CI [-.490126, -.434414]). It also fails the registered object-IoU margin. The unchanged QCC readout remains near source; this points toward a primary assignment/objective mismatch rather than disappearance of all core separation.

GT-free first64 TRAIN completion diagnostics also show weak instance-specific pressure: permuting P within coarse RGB bins reduces loss from .937748 to .907844 in the encoder/graph arm and from .974090 to .945565 in the joint arm. These descriptive interventions suggest color-compatible assignment can satisfy the objective. They do not establish the sole cause, prove gate necessity, or justify another blind reconstruction sweep.

This is a one-seed native32 continuation pilot, not a three-seed or fresh native32 70,000-image result. SW0137 remains the incumbent; its FG advantage over comparable Slot is only .000267 and is not a robust or substantial victory. SW0143 independently tests selective graph boundaries with unchanged spike QCC and all three seeds; its queue remains active. No checkpoint, dataset, mask array or key is included in this record.

Post-hoc descriptive mask statistics on the same frozen 320 images show under-separation: joint P predicts 3.328 foreground labels per image versus 6.313 in GT and 5.784 in joint QCC. The largest P object contains 61.87% of its predicted foreground, versus 40.17% in GT. P foreground occupies 36.43% of patches versus 21.84% in GT. These observations locate merging/background confusion in the primary readout; they do not justify changing the evaluation contract or selecting a new threshold from validation masks. See mask_diagnostics.json.

````

</details>
