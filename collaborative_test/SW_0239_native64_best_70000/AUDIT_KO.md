# 완료 후 독립 검증

completion_audit.py는 학습 source fingerprint에 들어 있지 않은 별도 검사 도구다.
등록된 학습 코드·선정 모델·CPU admission·큐를 바꾸지 않는다. 검사 결과에는
이 도구 자신의 source SHA를 별도로 기록한다.

완료 시 실제 checkpoint의 Adam counter, 43,750개 update 이력과 각 pass의
고유 70,000장 완전 사용 및 seed별 shuffle 순서를 검증한다. 원래 모델에서
최종 checkpoint를 CPU로 복원해 전체 모델 hash와 고정 core·입력 통계를 확인한다.
이것은 전체 학습을 다시 실행해 재현하는 검사는 아니다.

세 seed의 완성된 prediction protocol과 Slot artifacts를 먼저 확인한 후,
실제 mask 1,920개를 독립 metric 구현으로 다시 채점하고 평균·paired CI·목표 판정을
검증한다. 사전 지정한 seed0·이미지 3장의 양쪽 모델 figure 6개도 확인한다.

로컬 검사 2개에서 완전한 합성 이력을 통과시키고 반복·누락·순서 변경·비정상 loss와
비교 배열 오류를 거절했다. 아직 실제 70,000장 학습 완료 자료에 이 검사를 실행한
결과는 없다. 서버 배포·실행과 실제 완료 증거는 이후 별도로 기록한다.
