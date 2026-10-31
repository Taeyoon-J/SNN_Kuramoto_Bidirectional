# SW_0174_event_gradient_isolation

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `13ce29a3fcf75079500c75d67921f524bd6fd468c6cc9db27b18c4c8d371190e`

````markdown
# SW0174 event-gradient isolation audit

SW0173 found that a full-horizon virtual Adam direction did not improve the
registered Q-only route loss on its fixed TRAIN cohort and transfer images.
SW0174 measures one narrow causal intervention: detach the output of the
existing `SW0134.rollout.act_fun_adp` call while leaving its input, forward
value, membrane update, recurrent reset, graph, encoder, objective, and readout
unchanged. The raw event output is detached at every step; the final spike
trace is not substituted or rewritten.

The run is admitted only when the frozen SW0173 report has its registered
SHA256, complete status, exact SW0173 implementation fingerprint, and the same
source/input proof as current CPU admission. A fresh same-RNG B1 source forward
compares native and detached-event traces, gamma, Q, A, projection, and both
registered source-F/native-QCC partitions before the frozen SW0173 audit is
reused. SW0173's audit runs inside the scoped intervention and keeps its exact
TRAIN cohort (0–3), transfer set (256–259), horizons, fresh Adam direction,
scales, routes, metrics, and no-update behavior. Its original report is never
overwritten.

The primary check is the full1024 direction at scale 1: native route loss must
decrease on cohort and transfer, while transfer source-F and native-QCC
foreground ARI must each be at least the native source baseline and the frozen
SW0173 full1024 primary result. Scale 1/16 is diagnostic only. This audit does
not train, tune, add a loss, or inspect validation/reserve data. A positive
result only motivates a separately registered short pilot.

````

</details>

## RECOVERY_KO.md

<details>
<summary>RECOVERY_KO.md</summary>

Source SHA256: `1c431503df75fc86bb632dd62c7c79aa2a87c5234b96f05fd1cef75a8d39c187`

````markdown
# SW0174 CUDA 파일 읽기 오류와 동일 실험 재개

첫 실행은 source baseline의 `edge_rho.clamp_min(0).prod(dim=0)`에서
`basic_filebuf::underflow error reading the file: iostream error`로 종료됐다.
성능 결과나 학습 업데이트가 만들어지기 전의 실행 오류이며, 과학적 실패로 분류하지 않는다.
기존 parent 3786424 / worker 3786474의 종료를 확인하고 첫 실행 파일을 보존했다.

별도 GPU 1 진단에서 실제 사용한 여덟 edge 개수의 reduction forward/backward를
기본 캐시와 캐시 비활성 환경의 새로운 프로세스로 각각 실행했다. 두 실행 모두 성공했고
출력과 gradient 해시가 전부 일치했다. 기존 캐시 파일의 크기와 SHA도 유지됐다.
따라서 최초 오류는 이 작은 재현에서 반복되지 않았으며, 캐시 손상을 입증하지 못했다.

공식 PyTorch는 `USE_PYTORCH_KERNEL_CACHE=0`으로 CUDA JIT 파일 캐시를
비활성화하는 방법을 제공한다.
[PyTorch CUDA 문서](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/cuda.md)

이 환경 설정으로 동일 SW0174 코드를 새 `event_gradient_audit_v2_20261015`
실행 경로에서 재개했다. source SHA, 모델, loss, 입력, 비교 scale은 바꾸지 않았다.
실제 GPU 1 worker 3787993의 `/proc`에서 캐시 비활성 환경을 확인했다.
이는 원인 확정이 아닌 실행 우회이며, 개선 점수로 해석하지 않는다.
증거는 `results_archive/cache_bypass_recovery_20261015/`에 기록한다.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `dbf66f602a61ecc5375dcc8cc6c0ba4950b43af539f5448c0f9e718af7e5d3d7`

````markdown
# SW0174 결과: event surrogate 역전파 분리만으로는 개선되지 않음

별도 캐시 비활성 실행이 정상 종료됐다. 첫 CUDA 파일 읽기 오류와 구분한 과학적 결과다.
원본 SW0173과 같은 TRAIN cohort 4장 및 transfer 4장을 사용했고,
새 학습 업데이트나 고정 320장 평가를 실행하지 않았다.

동일 RNG 비교에서 gamma, 실제 A, projection, Q, 여섯 trace와 두 mask 출력이
정확히 일치했다. 여덟 source baseline도 기존 보고서와 일치했다.
65,536회 raw event의 역전파만 분리됐으며 scope와 원본 모델 상태가 복원됐다.

사전 지정한 full1024 / scale 1 결과:

| TRAIN 진단 항목 | 변경 전 | event 역전파 분리 후 |
|---|---:|---:|
| cohort Q loss | 0.26514053 | 0.26498338 |
| transfer Q loss | 0.35634674 | 0.35643243 |
| transfer 최종 source_F FG-ARI | 0.60379075 | 0.60171174 |
| transfer native_QCC FG-ARI | 0.60038102 | 0.60197111 |

앞단 전체 gradient norm은 context 755,906 / graph 505,522 / core 89,837로
여전히 매우 크며, full/tail 방향 cosine도 -0.89887이다.
기존 SW0173의 full gradient 폭증과 방향 불일치가 event 역전파 분리 후에도 남았다.
따라서 event surrogate만을 해당 문제의 원인으로 단정할 근거가 없다.

transfer loss 감소와 최종 mask 보존을 동시에 충족하지 못해 등록된 확대 기준은 실패했다.
64-update 학습으로 확대하지 않는다. 작은 scale의 일부 개선을 선택해 성공으로 보고하지 않는다.
이 결과는 모든 학습률에서의 실패나 spike forward의 무용함을 입증하지 않는다.

서버 raw report SHA: `7583c48c6909e2acdcf141a369c3e6b9f65c7c2681c146fb1e1ed226945e72c3`.
원시 결과와 독립 재계산은 `results_archive/completed_audit_20261015/`에 보존한다.
고정 32×32 세 seed 최고 기록은 SW0165의 **0.8043 / 0.6100 / 0.6511**로 유지한다.

````

</details>
