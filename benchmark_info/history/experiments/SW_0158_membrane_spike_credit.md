# SW_0158_membrane_spike_credit

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `70b7729eb2b616cef637f54fa6119c7ee7bdaa48ff3fcddf704cd14fcd6659eb`

````markdown
# SW0158 membrane-to-emitted-spike credit

SW0158 tests one registered amplitude path on the frozen SW0143-T native32 source. The native binary event and gate remain unchanged. Four zero-initialized scalars modulate the actual emitted spike by a bounded membrane-margin term, and that same value becomes the native recurrent spike state used by both dendritic input and membrane reset on the next step. The native reset equation and reset coefficient are unchanged.

The matched detached control has the same forward equation and recurrent values. It detaches membrane only in the added amplitude factor, leaving the native event surrogate path intact. The component scales are fixed from the equal-size source97-T TRAIN batches in the completed SW0157 report; no validation labels are used to calibrate them. SW0157's settled event occupancy was saturated, so SW0158 does not claim to desaturate binary events.

The prospective pilot uses the unchanged SW0155 relation objective and TRAIN-mask loader for 128 updates, logical batch 16, four microbatches, T=1024, settle=512, live tail=64. The source RGB encoder, native32 patcher, and SW0143-T graph weights are frozen. Native core, the existing 12 integration scalars, and eta are trainable. Source-T baseline, live-credit arm, and detached control use the same seed1 source and ordered TRAIN images. A disposable two-update preflight must prove exact eta-zero source-T trace/Q/QCC parity and measure same-state live-versus-detached membrane credit before any full training admission.

The preflight uses only the first four registered TRAIN IDs and their canonical TRAIN masks. It measures live/detached gradients at the shared post-update-1 state before update 2, checks repeated-live numerical residual, and independently verifies that the emitted spike is the next-step input to both dendrite and native reset. It also reports same-state Q change against eta zero and per-node temporal variance of the amplitude factor on active event-gate support; neither is a performance claim. To run the GPU-only preflight as an owner-lease child after source and lease review:

```powershell
python -m collaborative_test.SW_0158_membrane_spike_credit.preflight --device cuda:0 --output collaborative_test/SW_0158_membrane_spike_credit/results_archive/preflight_seed1.json
```

Evaluation keeps the native actual-spike QCC readout and registered fixed320 images. All prediction artifacts must be validated and frozen before the mask dataset is opened. A failed preflight or promotion closes this fixed recipe; it does not authorize a sweep or additional seeds. No server deployment or GPU run is included in this local implementation.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `e2e39a1e74dd1803a0310544da15ed4cfceb38bb65ef2894d92f5401ef097b1f`

````markdown
# SW0158 ?? ??

?? ??? FG-ARI / foreground IoU / matched-object IoU?? fixed320, 32?32, seed1 ???. ? arm ?? sourceT?? ??? TRAIN mask 2,048?, 128 updates? ????.

| ?? | FG-ARI | foreground IoU | matched-object IoU |
|---|---:|---:|---:|
| ?? SW143-T | .755273 | .614993 | .579514 |
| membrane amplitude credit | .753971 | .611127 | .578847 |
| added M gradient detached control | .753829 | .615467 | .579047 |

?? ??? FG ??? ?? ?? -.001302 (paired95% CI [-.005753,.003144]), ??? ?? +.000142 ([-.004264,.004875])?. ??? ? ?? ?? ??? ?? ????. ? recipe? ???? ?? seed? ? sweep? ?? ???. ?? unsupervised ??? ???? ???.

?? emitter? recurrence? membrane-dependent amplitude? ???? ?? gradient? ???? ?? GPU preflight? ????. ??? segmentation ??, binary event desaturation, membrane ???, supervised 70k ??? ???? ???.

?? SW159??? sourceT?? ?? threshold scalar ???? TRAIN scale? ?? ? ? ????. Threshold? event cutoff? reset ? ? ???? event selector?? ablation?? ???? ???. ?? ?? event on/off? ?? credit? ???? ????, ??? ?? ?? ????.

??: results_archive/completed_evaluation_20261014/0_evaluation_seed1.json ? root_review.json. ?? protocol/README? ?? ??? provenance ??? ?? ???? ???.

````

</details>
