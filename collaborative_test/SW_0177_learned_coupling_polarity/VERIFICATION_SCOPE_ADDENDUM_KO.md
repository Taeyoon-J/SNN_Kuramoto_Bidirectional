# SW177 초기 parity 검증 범위 정정

2026-10-16. GPU preflight 합격 결과와 학습 업데이트를 관측하기 전에 기록한다. Frozen DESIGN_KO.md와 실행 파일, 실행 중 큐는 수정하지 않는다.

DESIGN_KO.md의 “Source와 두 zero-head arm” GPU 직접 비교라는 문구는 현재 구현보다 넓다. 실제 preflight.py의 직접 GPU 비교 대상은 **SourceT와 signed zero-head arm**이다. 독립 nonnegative GPU rollout은 실행하지 않았다. 이 문서는 이를 두 arm의 GPU 실측으로 보고하지 않도록 검증 범위를 정정한다.

Nonnegative의 초기 동등성은 다음 근거로만 판정한다.

- 두 실제 CPU factory의 초기 source/head state가 같고 final head weight/bias가 0이다. 서버 CPU 검증에서 combined initial state SHA는 두 arm 모두 `b2b59d00641935b9deb5ecd13200da020d096641acd46867f9108705144e805b`, head SHA는 `35cb30c146b60260e7f8e66b2cde86829effe8e64af62cc82b90429411363bab`였다.
- 같은 유한 입력에서 초기 r=0, m=1−2tanh(r)=1이다. 두 mode의 차이는 m에 대한 abs뿐이므로 abs(1)=1이고 해당 지점의 도함수도 1이다. CPU operator 검사는 두 mode의 native tuple, 초기 forward 및 gradient 동등성을 검증한다.
- **실제 signed GPU source-parity preflight가 합격해야** 이 근거를 GPU 학습 진입에 적용한다. CPU 검사나 대수적 동일성으로 GPU 합격 receipt를 대체하지 않는다. 합격 receipt가 나오면 원본 report 경로와 byte SHA를 별도 결과 기록에 연결한다.

이 논증은 초기 zero-final 상태에 한정한다. 학습 후 signed/nonnegative 궤적, magnitude 또는 mask의 동등성을 의미하지 않는다. 각 arm은 별도 fresh optimizer로 학습하며 각각의 실제 산출물을 평가한다.

## 고정 근거 파일

| 파일 | SHA256 |
|---|---|
| DESIGN_KO.md | 293fa158f9dd165b6114a9acd8c54b5c673c8b9175c91665c2e7176396030f50 |
| polarity_operator.py | f85de0ae29fe5f71bce67cffc1f4865de565a417b5508c013fc2ed8f04ae6099 |
| test_polarity_operator.py | ceea3ca0a5724fbb3d1288ca38cd162d32368f9bf0469854e2361f8c9a3a401f |
| preflight.py | 31160b3accd94f72ccc1486b248c2da199fd71bb456852343b4d9113c19160e8 |
| train.py | d9c1e3d77a6218b6663954362deebd947ce1ef6095beb1b26d50c119091777a4 |
| test_train.py | 3e9554ec83e55254322fad8e2c970dac6722f98eb5cd39d39ea0be586d0b83bf |

학습 데이터·순서·64 updates·loss·두 평가 endpoint·baseline/control 비교·bootstrap 및 IoU 통과 조건은 DESIGN_KO.md 그대로다. 이 정정은 성능 결과나 repulsion/gating 기여의 증거가 아니다.
