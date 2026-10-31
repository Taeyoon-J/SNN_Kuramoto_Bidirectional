# readout_refinement/DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `7eef9bc8fb165da0519d600e0bb19f34c9a169961ef05aa224f16e50216ad2e6`

````markdown
# SW0251 readout score calibration

실제 TRAIN16/validation16의 네 loss 비교에서 기존 positive group끼리 재배정된 patch는 0개였다.
보존 prior4와 cosine logit의 bound(2×scale)가 원인이다. 실제 best scale .56이면 최대 점수차1.12로 prior4를 넘지 못한다.
따라서 FG-ARI 유지가 물체 분할 학습 성공을 뜻하지 않는다.

기존 core, cache, 정답/예측 생성 분리, final mask loss, LR, 데이터 순서,20 epochs를 유지한다.
control(prior4,gain1)과 soft_prior(prior1,gain1), strong_metric(prior4,gain16)을 비교한다.
세 설정 모두 초기 hard mask는 기존240과 정확히 같지만 두 새 설정은 작은 scale에서도 재배정할 수 있다.
매 epoch validation FG-ARI와 실제 positive→다른 positive 재배정 patch 수를 함께 기록한다.
초기 작은 비교가 통과해도 원래 fixed320·seed0/1/2 평균 FG-ARI 감소 .005 이내라는 최종 보존 gate는 유지한다.
이 실험의 결과로 기존 240 또는 benchmark 파일을 덮어쓰지 않는다.

````

</details>
