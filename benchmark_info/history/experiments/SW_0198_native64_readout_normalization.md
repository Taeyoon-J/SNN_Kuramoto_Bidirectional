# SW_0198_native64_readout_normalization

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `5444fd472162561d7f1f3390a7b90261010419df766bc0865bdf2fa98d14f614`

````markdown
# SW0198: native64 readout의 node 수 정규화 비교

관찰:194번 추가 학습 후 foreground support보다 동일 물체의 fragmentation이
악화된 경향이다.32장 posthoc 진단의 CI는0을 포함하므로 확증으로 주장하지 않는다.
현재 production merge energy는(1+1024*A_sym/gain)*logit(Q)이다. 이1024는
native32의 patch 수이며 native64에서도 그대로다. 실제 노드 수4096을 사용한
별도 버전을 비교한다. 이것을 미리 원인이나 성능 개선으로 판정하지 않는다.

원래 모델·CNN·graph·gate·spike·loss·foreground support는 그대로다. 두 수치만
비교한다: 등록된1024와 실제 Q.shape[0]인4096. coefficient sweep을 하지 않는다.
기존 solver를 별도 파일에 복사하고 pair-energy 계산만 변경한다. 공유 실행
코드나 이미 진행 중인197/195/196의 fingerprint를 바꾸지 않는다.

모델 두 개: 완료된 TRAIN8 및 TRAIN64. 각 모델에서 같은 원래 native64 forward의
Q/A를 공유해 기존 및 새 readout을 계산한다. 기존 mask가193/194의 frozen mask와
완전히 같은지 확인한다. 입력은 기존 development32장 중 offsets0,4,...,28의
8장, RNG193000+offset. 총16 forward이고 새 학습은0회다. GT를 열기 전에 모든
arm의 prediction을 고정한다. 모델/입력/정규화 개수는 이미지마다 선택하지 않는다.

Primary는 TRAIN8에서 새−기존 FG gain>=.01 및 paired bootstrap95%CI 하한>0,
object IoU 평균 비감소다. Foreground support가 같으므로 foreground IoU는
완전히 같아야 한다. TRAIN64 및 두 모델 간 차이도 secondary로 모두 보고한다.
실패하면1024/4096 사이의 작은 가중치 탐색으로 이 결과를 구제하지 않는다.
성공해도 전체320장/세 seed 확인이 필요하다. 데이터 증가의 causal proof나
각 gate의 causal contribution을 입증했다고 표시하지 않는다.

표준 FG-ARI/IoU 계산, modal GT, 데이터 및 학습은 변경하지 않는다. 이 테스트는
readout 비교이며 새로운 학습 성과가 아니다. 모든 GPU0~3에서 독점 idle lease를
기다리고 다른 owner가 합류하면 우리 worker만 양보하며 frozen row부터 재개한다.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `691f044074b9ce624c2465fe0d0b5ac1b3d0fe6c91768b0f010aec2ccd0d1a09`

````markdown
# SW0198 native64 graph weighting at readout

One independent all-GPU ready job. Two frozen models, TRAIN8 and TRAIN64,
each use eight predetermined development images. One original forward supplies
both the registered1024 and actual4096 graph-weighted signed partitions.
Reference masks must exactly replay frozen193/194 masks and foreground support
must remain identical. No optimizer updates or data changes.

This tests resolution normalization at the decoder. Separately, a CPU regression
case demonstrates the relative-QCE/absolute-0.5 readout ambiguity: shifting all
off-diagonal affinities can preserve QCE but split a correct partition. That
example does not prove it caused the actual194 decline; normalization alone
cannot guarantee calibration of Q around0.5.

Do not present this screen as full320/three-seed superiority or gate causality.
The existing shared classifier and running jobs remain unchanged.

Completed; actual32 masks (8images x2models x2readouts), frozen-reference replay,
foreground equality and all three metrics/bootstrap intervals independently
verified, maximum metric error2.22e-16.

| Frozen model | Reference1024 FG / FIoU / objectIoU | Actual4096 FG / FIoU / objectIoU |
| --- | --- | --- |
| TRAIN8 | .954783 / .883223 / .799757 | .955533 / .883223 / .800246 |
| TRAIN64 | .878008 / .865682 / .766942 | .878008 / .865682 / .766942 |

Primary FG delta+.000749,95%CI[0,.002248], below the registered+.01/positive-lower
gate. ObjectIoU delta+.000489,CI[-.000156,.001623]. TRAIN64 masks/metrics did not
change. The normalizer alone did not rescue the continuation decline; close this
screen without intermediate-coefficient tuning or changing the production code.
These are eight predetermined images, not the fixed32 or full320 scores; the .9555
number does not establish the native64 three-seed .90 objective. Incumbent unchanged.

````

</details>
