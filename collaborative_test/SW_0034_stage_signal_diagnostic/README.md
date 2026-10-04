# SW_0034: stage-wise object signal and gradient diagnostic

Status: completed on 64-image pilot and full 320-image validation. No other user training processes were observed; GPU 3 was idle at launch.

## Pilot finding (real checkpoint, 64 images)

`pilot64.json` confirms the SW_0033 control scores exactly. After settle,
binary threshold activation mean is 1.0, temporal standard deviation is 0,
and constant fraction is 1.0 for all components: actual gated spike histories
in this subset therefore reflect gate modulation, not changing threshold events.
Distance-stratified macro AUC: phase .5437, gating .5484, h-wave .5494,
membrane .5492, gated spike .5524, binary threshold .5000. Signal is weak
already upstream; these different affinity definitions do not establish a
causal loss at one layer. Phase loss bypasses all downstream parameters;
membrane/spike loss connects all four downstream tensors with finite gradients.
Full320 confirmed the binary threshold saturation and weak stage signal.

| Pilot readout | FG-ARI | FG IoU | Object IoU |
|---|---:|---:|---:|
| Spatial only | .525719 | .179896 | .180391 |
| Membrane × spatial | .530110 | .184325 | .176630 |
| Gated spike × spatial | .487071 | .174560 | .160534 |

All predict nine foreground groups; predicted foreground fractions are .883,
.865, .858 respectively. No three-seed or test improvement claim.

## Full validation finding (320 images)

After controlling pair distance, macro AUC is phase `.578814`, gating
`.587110`, h-wave `.584243`, membrane `.587027`, gated spike `.586344`, and
binary threshold `.500000`. The signal is weak already in phase/gating and is
roughly preserved through continuous downstream activations; this diagnostic
does not isolate one abrupt information-loss layer. Binary threshold activation
is exactly 1.0 with constant fraction 1.0 after settle on all 81,920 aggregate
patch histories, so the apparent gated-spike rhythm is gate amplitude rather
than changing threshold events. Branch cancellation ratio is `.947275`, which
does not support severe branch cancellation as the current main bottleneck.

| Full320 readout | FG-ARI | FG IoU | Object IoU | Foreground fraction |
|---|---:|---:|---:|---:|
| Spatial only | .491272 | .190837 | .178026 | .882813 |
| Membrane × spatial | .494639 | .191070 | .170568 | .867578 |
| Gated spike × spatial | .479762 | .185651 | .164179 | .858936 |

Membrane/spike PLV losses connect finite gradients to dendritic and membrane
parameters; phase PLV bypasses them. The next controlled core candidates should
therefore include threshold/range dynamics and an object-specific upstream
signal, while classifier count/grouping baselines continue in parallel.

## 무엇을 확인하는가

SW_0033에서 높은 ARI의 대부분은 공간 거리만으로 설명됐습니다.
이번에는 같은 이미지에서 theta, gating 입력, dendritic h-wave,
membrane, gate가 곱해진 spike, gate를 곱하기 전 binary threshold 신호를
비교합니다. 어디서 물체 구분 정보가 약해지는지 다음 학습과 classifier
실험을 고르는 근거로 사용합니다. 이는 다음 작업을 막는 엄격한 단계별
통과 조건이 아니며, classifier와 core 진단은 서로 피드백을 주며 병행합니다.

기존 nearby-pair AUC는 같은 물체 patch가 더 가깝다는 영향이 섞입니다.
따라서 patch 간 squared distance 1, 2, 4, 5, 8, 9를 각각 고정하고
same/different-object 평균과 AUC를 측정합니다. 두 종류 쌍이 모두 있는
거리만 macro 평균에 포함하고 실제 spatial-only AUC를 빼 추가 신호를
보고합니다. 이는 pooled AUC와 구분하며 통계적 유의성 주장은 하지 않습니다.

## Fixed configuration and provenance

- Source HEAD before this addition: `28d24b0e8f81c759e13cf7082e837eb146c404e7`.
- Frozen checkpoint: `/Data0/kevinswk/patch_v2_sw/trained_models/SW_0003_membrane_plv_40ep_seed0/core.pt`; runtime SHA256 recorded in result.
- Gamma: `/work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt`.
- GT: `/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5`, mask modal patch size 8.
- Pilot IDs 1320–1383; full validation IDs 1320–1639. No reference-test selection.
- Diagnostic rollout 256 steps, discard first 64, matching SW_0033.
- Separately, training-loss connectivity uses 64 steps/32 settle and first two of the same validation IDs, without optimizer updates.
- All PLV loss settings match SW_0003, with phase/membrane/spikes source switched separately; report each parameter's connected status, norm, and finite check.
- Spectral k10, sigma1.5, largest group background: spatial-only, aggregate membrane×spatial, aggregate gated spike×spatial. GT is never used for prediction.
- Aggregate and per-component waveform absolute correlations are exploratory diagnostics, not new final classifiers. Phase uses its existing mean PLV definition, so AUC differences between definitions are not a causal isolation.
- Branch cancellation reports `abs(sum(branches))/sum(abs(branches))`; smaller values mean stronger cancellation.
- Each waveform also reports constant-node fraction, temporal standard deviation and activation mean. Near-constant histories can make correlation uninformative; inspect these before interpreting AUC. Binary threshold mean measures firing fraction, while gated-spike mean also includes gate amplitude.
- Every readout reports three metrics, foreground fraction, and predicted group count. No result values are filled until execution.

## Exact code changes

Only standalone `diagnose.py` and `run.sh` are added. Forward hooks observe
actual dendrite inputs/output/branch states and membrane threshold decisions;
they do not alter core outputs, parameters, loss implementation, or evaluator.
Hooks are removed before gradient measurement. Checkpoint is never saved over.

## Reproduction and checks

After restoring SSH master, inspect live processes, paths and all GPU memory/utilization.
Upload these two scripts to the matching server directory, then run:

```bash
nohup bash collaborative_test/SW_0034_stage_signal_diagnostic/run.sh GPU_ID 64 > trained_models/SW_0034_stage_signal_diagnostic/pilot.log 2>&1 &
# After pilot review, use the same command with count 320 and validation.log.
```

Create the log parent directory before nohup. GPU_ID must be an inspected,
available single GPU; never kill another user's processes.
Server outputs: `trained_models/SW_0034_stage_signal_diagnostic/validation64.json`
and `validation320.json`; copy small results into this experiment directory.

Local Python syntax compilation passed. A synthetic CPU smoke check in the
existing `anaconda3/envs/kuramoto` environment passed: two samples, eight steps;
membrane observation hooks leave outputs exactly equal; binary threshold times
actual gate reconstructs aggregate spikes exactly in B,D,N fold order; a membrane
synchrony loss gives finite connected gradients to all dendritic and membrane
parameters. This is a core-path check, not real-data/checkpoint or full diagnostic
validation. That environment lacks h5py. Base Windows torch import fails with
duplicate OpenMP runtime; no unsafe runtime override was enabled.
Server master socket is absent. Peer fetch succeeded
on 2026-10-04 and remains `b99fac0`, with no new tested results.

## Next decision

Use stage AUC and gradient connectivity as evidence for controlled training,
while continuing classifier baselines in parallel when they can clarify count
or grouping limits. Preserve spatial-only control and matched readout; record
count/foreground tradeoffs. No training sweep is queued yet.
