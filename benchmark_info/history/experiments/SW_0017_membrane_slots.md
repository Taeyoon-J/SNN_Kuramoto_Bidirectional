# SW_0017_membrane_slots

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `7076cfac3f6ca064b73be468567d42564cb67746ed64fa0c7b5046534de02d7a`

````markdown
# SW_0017: Adaptive slots on membrane histories

Status: completed, mixed result. No retraining and no production classifier change.

SW_0013 used each patch's binary spike history as its adaptive-slot vector.
To test whether thresholding had discarded useful grouping information,
SW_0017 changed only that source to the **actual continuous membrane history**
from the same SW_0011 checkpoint. The same centering, cosine slot update,
largest-slot background, fixed validation IDs 1320–1639, assignment seed 0,
and 0.3/0.5/0.7 thresholds × 1/3/6 initial slots were used. Ground-truth
object count was never used in prediction.

| Source, threshold .7, six initial slots | FG-ARI ↑ | FG IoU ↑ | object IoU ↑ | groups/image |
|---|---:|---:|---:|---:|
| Binary spikes (SW_0013) | .247376 | .248332 | .204032 | 17.37 |
| Continuous membrane (SW_0017) | .261507 | .246264 | .207637 | 18.18 |
| Existing component-product spike readout (SW_0011) | .179760 | .271468 | .295250 | 77.03 |

Membrane improves FG-ARI over binary spikes by .0141 and object IoU by .0036,
but slightly lowers foreground IoU and still predicts about 18 groups versus
6.20 true objects. Compared with the existing readout it remains worse on
both IoUs. Thus binary thresholding is not the sole reason for the weak masks;
the slot grouping/count and foreground inference remain unresolved. This is a
diagnostic inference, not a proof that membrane signals are generally worse.

The only code change is a `--source membrane` option in SW_0013's standalone
evaluation script (default `spikes` unchanged). Raw scores:
`validation320.json`. Existing model, loss and evaluation metric code unchanged.

````

</details>
