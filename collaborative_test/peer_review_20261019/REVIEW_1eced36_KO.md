# 동료 최신 연구 확인: 1eced36

`origin/patch_v2`를087cd82에서1eced36으로 fetch했다. 현재 작업 브랜치나 실행 중인
237·238번 소스는 바꾸지 않았다. 아래 기록은 동료 결과 검토이며 새 모델 성능이 아니다.

## 확인 범위

전체를 읽은 기록은 `peer_updates/REVIEW_0007_20261019.md`,
`PV2_0107b_merge_threshold/README.md`와`metrics.csv`,
`PV2_0098_data_scale/README.md`다. 추가로 `STATUS.md`와
`evaluate_model.py`의087cd82 이후 diff를 확인했다. 그 밖의 동료 실험을 전부
검토했다고 주장하지 않는다.

## 적용할 근거

동료0107b는 native32·contract-v2·validation300·세 seed다. 같은 checkpoint에서
merge를 끄면 약.7110/.7130/.6588, threshold .0001이면 .7592/.7130/.6303이다.
세 seed 모두 FG가 오르고 Object IoU가 낮아졌다. 공개된 반올림 seed 평균을
별도 재계산한 근거는 `peer_merge_recompute.json`이다. 다른 threshold의 seed 수가
다른 행이나 native16 점수를 이 비교에 넣지 않았다.

이는 우리234번의 GT 보조 merge에서도 Object IoU가 낮아진 관측과 같은 방향이다.
따라서 단순 merge/linkage sweep의 우선순위를 낮춘다. 그러나 GT 보조 변환이
수학적 최적 oracle은 아니며, 이 관측만으로 모든 linkage의 가능성을 배제하거나
pair 선택이 중요하지 않다고 증명하지 않는다.

동료0098의 TRAIN1000/6000은 학습이 끝났지만 파일시스템 문제로 두 평가가
완료되지 않았다. 낮은 loss를 patch 성능 향상으로 세지 않는다. 같은40 epoch는
동일 update 예산도 아니므로 우리236번의 matched512-update 비교를 대체하지 않는다.
우리236번은 이미 평가·독립 검증이 끝났고 FG 데이터 증가 효과는 불확실하며,
Object IoU 개선과 Foreground IoU 평균 하락을 구분해 기록했다.

동료0110의 graph feedback 비용 약27배는 STATUS가 보고한 진행 중 wall-time
관측이다. 평가 성능은 아직 없고, 다른 GPU 부하·backend·step 수가 있는 우리
native64 비용으로 그대로 환산하지 않는다. 효과 근거 없이 feedback을 활성화하지
않으며 현재 frozen237번을 변경하지 않는다.

## 그대로 받아들이지 않을 해석

동료는 누락 물체에 새 그룹을 만드는 coverage를 다음 가설로 제안했다. 우리0207과
234번은 학습 방식과 GT 사용도 달라, 새 그룹 생성만의 효과를 분리한 비교가 아니다.
따라서 이는 검증할 수 있는 가설이며 성능 향상이 보장된 구현 지침은 아니다.
누락 그룹과 false-positive 배경을 함께 구분하는 근거 없이 support를 넓히지 않는다.

동료 REVIEW_0007의 우리 Slot 기준을 single published checkpoint라고 쓴 부분은
현재227번 기준과 다르다. 현재64-grid 기준은 `SW0092_slot_our70000_eval/seedN_epoch10`
세 artifact를 검증하며,0092 experiment3은 같은70,000장으로 세 seed를 각각10 passes
학습한 기록이다. 공식 pretrained single-checkpoint transfer .894641은 별도16-grid
비교다. 근거는 `SW_0092_cross_dataset_training/README.md`,
`SW_0154_native64_resolution_compare/{evaluate_native64.py,score_slot64.py}`다.
우리 mask-supervised lineage와 동료의 unsupervised line을 같은 leaderboard로 합치지 않는다.

237번의 전체320장 확인과238번의 실제 event/gate 진단은 그대로 진행한다.
70,000장 신규 작업은 사용자의 학습/추론 범위 답변을 기다린다.
