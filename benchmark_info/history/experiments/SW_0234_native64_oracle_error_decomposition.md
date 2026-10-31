# SW_0234_native64_oracle_error_decomposition

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `77895c591453d7ba5d2f493c63690726f74ae88890d25595b5e2b33ab90a3d64`

````markdown
# SW0234: GT-assisted error decomposition, not a model score

Peer review at origin/patch_v2 commit087cd82 found PV2_0104 three-seed native32
contract-v2 validation300 FG .7109; oracle merge .8242, coverage .8111,
both .9329. These use ground truth and are not trained-model improvements.
Their Slot contract and dataset differ from ours. No scores are imported.

Transfer the diagnostic question to our independently audited SW0227 native64
full320 masks: apply prescribed merge-only, coverage-only and combined GT-assisted
operations on CPU. Preserve pure-background groups in merge-only so foreground
support and foreground IoU remain identical. Coverage fills only GT foreground
currently predicted background. It can invent a group for entirely missing objects.

These operations do not prove a mathematical upper bound or that a learned readout
can recover the improvements. Existing straddling groups are not split. No mask,
checkpoint, loss, GPU queue or best-model record changes. No model forwards.
Prior SW0214–216 learned coarse-merger failures remain evidence; a high oracle
result alone does not justify repeating that approach. Use this diagnostic to
prioritize remaining error types after SW0232's convergence comparison.

````

</details>

## RESULT_KO.md

<details>
<summary>RESULT_KO.md</summary>

Source SHA256: `005762e92dedeed226c3d98d994a4871044a0db8290ff106553517cf6af97ea2`

````markdown
# SW0234: 조각 합치기와 전경 복구를 함께 확인해야 함

동료 branch087cd82의 PV2_0105·0106을 확인하고 같은 진단 질문을 우리 SW0227의 고정320장·native64·세 seed에 적용했다. 동료 결과는 native32/contract-v2 validation300이며 우리 점수와 섞지 않는다.

아래 변환은 **정답 mask를 보고 수행한 분석이며 학습된 모델 성능이 아니다.** 기존 checkpoint, 실제 예측 mask, 최고 모델 기록은 바꾸지 않았다.

| 정답 보조 변환 | FG-ARI | Foreground IoU | Object IoU |
|---|---:|---:|---:|
| 실제 SW0227 | .858916 | .605286 | .674313 |
| 조각 합치기만 | .885762 | .605286 | .667059 |
| 누락된 전경 복구만 | .883696 | .641168 | .762060 |
| 두 변환 함께 | .911458 | .641168 | .755105 |

조각 합치기에서는 기존 foreground support를 보존하고 순수 배경 위에 생긴 예측 그룹도 지우지 않았다. 따라서 foreground IoU가 정확히 유지된다. 전경 복구는 정답 물체 중 예측 background였던 patch를 채우며 완전히 누락된 물체에는 새 그룹도 만든다. 서로 다른 물체를 이미 합친 그룹은 분리하지 않는다.

두 변환을 합하면 .90을 넘지만 실제 모델이 이 정보를 스스로 알아낼 수 있다는 증거는 아니다. 목표 .90에 가려면 이 변환의 FG 증가분 .052542 중 약78%를 회수해야 하므로 작은 조정으로 충분하다고 판단하지 않는다. 합치기만 하면 Object IoU는 오히려 내려간다. 따라서 FG만 보고 단순 merge 방향을 밀어붙이지 않는다.

SW0214–216의 학습형 coarse-merger 실패와 SW0207의 foreground 개선/FG 하락 기록은 여전히 유효하다. 이번 결과는 그 방법을 그대로 반복할 근거가 아니다. SW0232의 추가 학습 결과와 함께, 누락 영역 복구와 잘못된 join을 함께 제어하는 학습 신호를 설계하는 데 사용한다. 모든 gating 기여나 데이터 증가 효과는 별도 검증이 필요하다.

로컬·Linux 각각 세 검사를 통과했다. 세 checkpoint/전체 예측 protocol을 검증한 뒤 정답을 읽었고, 실제 SW0227 점수 재계산 오차는 1.12e-16 이하다. 추가 GPU 실행·학습은 없다. 이 처방적 GT 변환을 수학적으로 최적인 상한이라고 부르지 않는다.

````

</details>
