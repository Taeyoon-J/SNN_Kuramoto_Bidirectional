# SW0175 첫 실행 입력 adapter 오류

두 arm은 GPU 1/2에서 원본 모델을 구성했으나, 초기 TRAIN8 진단의
`forest_routes`에서 `support must be boolean`으로 종료됐다.
실제 캐시의 `source_support`는 object label grid이며 foreground boolean이 아니다.
기존 SW171 caller의 `np.asarray(support) > 0` 변환을 SW175에서 빠뜨린 구현 오류다.

로컬·서버 CPU 13개 검사는 통과했지만, 해당 검사는 실제 캐시의 multiclass support를
새 loss caller에 전달하는 경우를 다루지 못했다. source/cache admission 성공을
실제 입력 adapter 검증으로 확대 해석한 점을 바로잡는다.

기존 로그·초기 checkpoint·source manifest는 보존한다. 첫 두 worker와 parent의
종료를 확인하고, checkpoint 메타데이터로 두 arm 모두 update 0임을 확인한다.
이는 모델 성능의 음성 결과가 아니다. 세 seed 최고 기록은 SW165 그대로다.

수정은 caller의 foreground 변환 복구와 새 실행 namespace 분리에 한정한다.
모델·loss 수식·TRAIN IDs·64-update schedule은 바꾸지 않는다.
재실행 전 multiclass support 회귀 검사와 실제 TRAIN cache의 caller 검사를 추가한다.
첫 실행 증거는 `results_archive/failed_v1_20261015/`에 기록한다.
