# SW_0091_official_slot_data

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `3877912eebf22c63f290a3b7bbe838aa98d95657f3af3a36f809333c63a3ab90`

````markdown
# SW0091 — official released Slot Attention data

The released Google object-discovery code loads TFDS `clevr:3.1.0`, whose
official train split contains 70,000 images. It then reserves the first 512
train rows for `train_eval`, filters training scenes to at most six objects,
applies the fixed center crop `(29:221, 64:256)`, resizes to 128x128, and trains
with seven slots.

`inspect_and_prepare.py --download` downloads and prepares the exact TFDS
release under `/Data0/kevinswk/datasets/tfds`. This is separate from our
100,000-image `clevr_10-full.hdf5` render and must remain a separate data-domain
condition.


````

</details>
