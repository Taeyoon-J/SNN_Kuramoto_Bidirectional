# Fixed-support FG-ARI의 인증 상한

이 문서는 SW0164/165의 예측 foreground membership을 고정했을 때 가능한 FG-ARI의 상한을 증명한다. 실제 solver를 변경하거나 GT를 prediction에 사용하는 방법이 아니라, 모든 prediction을 먼저 고정한 뒤 수행하는 GT-last 진단이다.

## 범위와 기호

Production FG-ARI와 같이 **GT foreground patch만** 모집단으로 삼는다. 그 수가 `n>=2`일 때 `T=choose(n,2)`, GT가 같은 pair 수를 `C`, prediction이 같은 pair 수를 `B`, GT와 prediction 모두 같은 pair 수를 `A`, `c=C/T`라 두면

`ARI = (A-cB) / (0.5*C + (0.5-c)*B)`.

SourceT가 background로 정한 patch는 prediction label 0으로 고정하고, foreground로 정한 patch는 0이 아닌 label을 가져야 한다. 상한 계산의 relaxation은 positive label 수를 제한하지 않고 disconnected group도 허용한다. SW0164 실제 solver는 각 positive group에 4-connectivity를 요구하므로, relaxation의 최대값은 실제 solver가 달성 가능한 값의 유효한 상한이다. 실제 solver가 disconnected group을 허용한다고 주장하지 않는다.

## 최적화 증명

임의의 admissible positive partition에서 여러 GT row를 섞은 positive cluster를 GT row별로 나눈다. 이때 `A`는 그대로이고 `B`만 감소한다. 분모가 양수인 비퇴화 경우

`dARI/dB | A = [A(c-0.5) - 0.5*c*C] / D^2`.

`c<=0.5`이면 음수이고, `c>0.5`이면 `A<=C`를 써서 최대 `0.5*C*(c-1)<0`이다. 따라서 mixed positive cluster를 나누면 ARI가 감소하지 않는다.

이제 같은 GT row에 속한 positive cluster들을 합친다. `A`와 `B`가 같은 양만큼 증가할 때 도함수의 부호를 정하는 항은

`0.5*C*(1-c) + (0.5-c)*(B-A)`.

`c<=0.5`이면 음수가 아니다. `c>0.5`이면 `B-A<=T-C=T*(1-c)`이므로 이 항은 최소 `0.5*T*(1-c)^2>0`이다. 따라서 같은 GT row의 positive cluster를 합쳐도 ARI가 감소하지 않는다.

두 변환을 반복하면 fixed label-0 patch는 그대로 두고, covered GT foreground에는 GT object별로 하나의 positive label을 준 partition에 도달한다. 그러므로 이 `fixed-support GT-induced FG-ARI`가 위 relaxation에서의 인증 최대값이며, 4-connected 실제 solver에 대한 인증 상한이다.

## 퇴화 경우

- `c=1`: GT foreground가 한 object뿐이다. missed와 covered patch가 모두 있으면 모든 admissible partition의 최댓값은 0이고 GT-consistent covered partition이 이를 달성한다. 전부 covered이거나 전부 missed이면 GT-consistent partition과 GT가 각각 단일 cluster이므로 분모가 0이며 production 구현 규칙대로 1이다.
- `c=0`: 모든 GT cluster가 singleton이다. `B=0`이면 분모가 0이고 production 규칙대로 1이다. 고정 background에 둘 이상의 patch가 있어 `B>0`이면 ARI는 0이며 GT-consistent covered partition도 최대값을 달성한다.
- GT foreground patch가 두 개 미만이면 production metric은 NaN을 반환한다. 상한 helper도 NaN을 반환하며 mean에서는 production `valid_count` 처리와 정확히 같은 방식으로 제외해야 한다.

`test_ari_bound.py`는 2~6 patch의 모든 GT partition, 모든 fixed-support mask, covered patch의 모든 positive partition을 열거해 후보 점수가 이 상한을 넘지 않음을 확인한다. 별도 소수 예제만 sklearn과 대조한다.
