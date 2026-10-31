# SW_0015_gamma_image_alignment

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `b4d03cf661ec029f55e35fd24a3f6de3e17a2ce003524499e7ba3da5e6b21a4c`

````markdown
# SW_0015: Does cached gamma follow HDF5 image numbering?

Status: completed diagnostic. No training/evaluation algorithm changed.

This matters before testing the suggested RGB-boundary membrane loss: it
would be invalid to pair gamma from one image with the RGB edges of another.
The split manifest had previously recorded image-index alignment as unverified.

First, `check.py` compared 12 numbered CLEVR_1k PNGs with the first 1000
HDF5 images after resizing to 32×32. None of the same-numbered HDF5 images
was the nearest image. These PNGs are **not** evidence that our `wm...gamma`
is misaligned: the `wm` tensors may come from a different image source.

Therefore `check_gamma.py` directly compared the actual cached gamma to the
actual HDF5 image. For each of 128 train indices and 128 validation indices,
it formed a 16×16 patch-boundary contrast fingerprint from gamma and from the
HDF5 RGB image. It ranked the same-index HDF5 image among 128 candidates:

| Split/gamma | Same index ranked #1 | Top 5 | Median rank | Random median |
|---|---:|---:|---:|---:|
| Train `wm_patch_gamma_seq_k8_grid16.pt` | 27/128 | 65/128 | 5 | 64.5 |
| Validation `wm10k_full_grid16.pt` | 26/128 | 73/128 | 4 | 64.5 |

The matched cross-modal boundary similarity averages .3326 (train) and
.3513 (validation), compared with all-pair means .1616 and .1835. This is
strong empirical evidence for image-index alignment of the **actual** gamma
and HDF5 data, unlike the unrelated PNG set. It is not an exact provenance
or bitwise identity proof. Exact generation scripts/metadata would strengthen
that claim further. No GT mask was used in the matching.

Scripts: `check.py`, `check_gamma.py`. Raw results: `png_result.json`,
`gamma_result.json`. This supports, but does not yet implement, a separate
RGB-edge loss ablation.

````

</details>
