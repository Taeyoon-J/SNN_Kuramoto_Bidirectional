# SW0231: 정확한 128회 prefix 검증 실패

seed0·2의 재계산은 128회 경계에서 accepted SW0223와 bitwise 동일하지 않아 중단됐다. Projection 최대 절대 차이는 각각 1.1920929e-7, 2.9802322e-8, encoder는 8.9406967e-8, 4.4703484e-8이다. 최초 history 차이는 두 seed 모두 update2의 gradient norm에서 나타났다. 수치 차이의 근본 원인은 확정하지 않았다.

검증 허용 오차를 완화하지 않았다. seed1도 같은 재계산을 반복하지 않도록 해당 실험의 프로세스만 종료했고, 세 supervisor가 모두 종료됐음을 서버 /proc에서 확인했다. 실패 checkpoint와 로그는 보존했다. 512회 학습이나 mask 평가 결과는 없으며 모델 성능 실패로 해석하지 않는다.

후속 SW0232는 accepted SW0223의 실제 가중치·Adam·history·Torch RNG를 직접 복원한다. 복원한 128회 상태를 동일한 bitwise 검사로 확인한 뒤 384회만 추가 학습한다. 기존 SW0227 전체 평가와 원본 checkpoint는 유지한다.
