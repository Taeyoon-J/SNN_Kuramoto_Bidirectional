# SW176 대기 중 동료 연구 확인

확인한 동료 브랜치 `origin/patch_v2`의 최신 commit은 `cab6c676e24a149a45ea960586c04e579f78d384`다. 새 변경은 없었다. `PV2_0092_signed_coupling/README.md`와 `ROUTE_MAP_0.9.md`를 직접 읽었다.

동료는 양수 결합만 있는 그래프에서 물체들이 일관되게 합쳐지는 오류를 지적하며, 일부 이웃 간 반발 결합을 도입한 seed0 실험을 진행 중이다. balance 0/0.25/0.5/1의 최종 결과는 문서상 pending이다. 이는 후보 가설이며 성공 결과가 아니다. 우리 모델의 실제 A와 오류 유형을 확인하기 전 그대로 적용하지 않는다.

동료의 중요한 교훈은 surrogate loss 감소나 연결 분리 통계 개선만으로 최종 mask 개선을 주장할 수 없다는 것이다. foreground를 크게 늘려 FG-ARI만 높이는 경우도 있어 세 지표와 실제 foreground를 함께 확인해야 한다. 우리의 고정 native32 계약은 유지한다. 동료의 native16 계약·학습 supervision 제한과 우리의 authorized supervised pilot은 다르므로 절대 점수를 직접 비교하지 않는다.

현재 SW176은 기존 matched loss와 데이터를 유지한 gradient 전달 정책 비교다. 그 결과가 확정되기 전 새 구조 학습을 중복 실행하지 않는다. SW176 실패 시 같은 정책의 LR·threshold 미세 조정 대신 오류 진단으로 다음 구조 가설을 선택한다. 이 기록은 새로운 성능 결과 또는 gating 인과 검증이 아니다.
