# SW_0203_native64_consensus_units

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## DESIGN_KO.md

<details>
<summary>DESIGN_KO.md</summary>

Source SHA256: `ca671a7c6eddfbc029b5159a72988c08b05f18569c8b0760b48e4bda14ae5285`

````markdown
# Native64 consensus units, fixed eight-image paired screen per seed

Four component correlations are multiplied in the current Q. Equal positive
correlations of .7 therefore give Q=.2401, below the .5 signed-energy boundary.
Compare Q with its geometric mean Q**(1/4), using the actual four components.
Only foreground-induced grouping changes; preserve baseline foreground support,
original 1024 energy normalizer, graph, encoder, gates and frozen weights.
No new optimizer steps; no validation GT for prediction or choosing exponent.
Seeds0/1/2 each use offsets0,4,...28 from the same32 development images.
Every original mask must replay exactly before storing either prediction.
This is inference calibration, not new training or proof of gate contribution.
Extend only if mean FG improvement >=.01 and paired bootstrap lower bound>0,
with nonnegative object IoU. Full320/three-seed .90 goal remains unverified.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `049ed42f55be9b2b00e4357243d5f6cf669205c110f0d355988386e49631fe43`

````markdown
Native64 paired readout calibration; three independent seed queues scanning GPUs0..3. Pending actual execution. See DESIGN_KO.md.

Completed: fixed8 development images per seed, independent seeds0/1/2,
4096 patches, no optimizer steps. All48 old/new masks and CPU model admissions
independently audited; maximum metric discrepancy2.22e-16.

| Three-seed mean, same eight IDs | FG-ARI | Foreground IoU | Object IoU |
| --- | ---: | ---: | ---: |
| original product Q | .8946055391 | .5522615349 | .7391562767 |
| geometric consensus units | .8952023021 | .5522615349 | .7389540016 |
| comparable Slot | .7978265641 | .1805270567 | .2059092105 |

FG change+.0005967631, paired image CI[.0001360351,.0011160475].
Despite a small positive difference, the registered+.01 gain criterion fails,
and object IoU mean declines. No adoption or exponent/threshold micro-rescue.
This is a readout diagnostic, not newly learned weights. Our source models are
mask-supervised; this is not an unsupervised benchmark win. Eight images are not
the full320 evaluation, nor the previous32-image mini. Full320/three-seed .90
remains unverified. See aggregate_v1/summary.json under results_archive.

````

</details>
