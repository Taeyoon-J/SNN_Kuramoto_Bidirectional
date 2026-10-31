# SW_0124_temporal_prototype_readout

이 문서의 인용 블록은 기존 문서 원문이다. 서로 다른 시점의 계획/결과를 소급 수정하지 않는다. 원문 안의 상대 경로는 ZIP 안의 원래 디렉터리를 기준으로 해석한다.

## design.md

<details>
<summary>design.md</summary>

Source SHA256: `1b77df2974fd1225f505e61182fc8f7af1a55deee809aa45618a9ca1c4096a3b`

````markdown
# SW0124: temporal-prototype readout, frozen-source stage 1

SW0123's assignment head produced mostly one foreground group on the first
training batches and many all-background fixed-validation images. This stage
tests whether a deterministic, affinity-seeded prototype readout of the same
frozen SW0097 actual spike histories provides a better fixed readout. It does
not train a model or alter the source core.

For each seed, validate the SW0097 core, matched SW0123 legacy calibration RMS,
and SW0042 validation gamma cache. Run the same source core at 1024 steps on
the registered 320 validation images. On settled actual component traces,
subtract the temporal mean and divide by the frozen four-channel TRAIN RMS.
Select detached anchors from the registered positive-product affinity: first
the lowest-index maximum row-sum patch, then the lowest-index minimum
maximum-affinity patch among distinct traces, stopping at affinity 0.50 or
11 anchors. Exact duplicate traces are excluded; no numeric novelty cutoff is
used. If all traces are identical, the readout has one prototype.

Gather live traces at those anchors. Apply three negative temporal-MSE soft
assignment / weighted-prototype updates at temperature 0.1 and compute final
assignments. Pad unused prototype columns with zero. Map labels with the
existing SW0123 largest-slot-background and minimum-two-patch rule. Report
the unchanged production QCC output beside it.

Save all predictions and hashes before loading masks. Score source QCC first
and require all three per-image metrics to reproduce the immutable SW0097
reference within 1e-10. Any mismatch ends the run before prototype scores can
be considered. Compare the prototype readout to the registered source metrics
with the shared-image, three-seed bootstrap and fixed Slot IoU references in
`protocol.json`. A failed gate closes this readout; it does not authorize
threshold, seed, or prototype tuning.

````

</details>

## README.md

<details>
<summary>README.md</summary>

Source SHA256: `c4c91355ec187c723f21e37bd26e09e5e1a76e32466145423b15934ca2ea8bfa`

````markdown
# SW0124 temporal prototype readout

This is a read-only stage-1 evaluation of a deterministic temporal-prototype
readout over frozen SW0097 source spike traces. It does not train or modify a
checkpoint. The protocol and exact operation order are in `protocol.json` and
`design.md`; the implementation is in `prototype_readout.py` and `evaluate.py`.

The readout centers and applies the registered SW0123 legacy four-channel
TRAIN RMS to actual settled component-spike histories. Detached production
positive-product affinity selects up to eleven distinct trace anchors. Three
soft assignment/prototype updates use negative mean squared temporal distance,
followed by a final assignment. Existing SW0123 largest-slot-background and
minimum-two-patch labeling stays fixed. Production QCC is reported as the
comparison readout.

The evaluator writes all three seeds' GT-free predictions and verifies their
hashes before opening validation masks. It then requires exact per-image
SW0097 source-QCC reproduction within `1e-10` before calculating any temporal
prototype scores. It uses the fixed 320-image validation contract and shared
image indices for the registered three-seed bootstrap. Existing outputs are
never overwritten; provide a new `--output-dir` for a new attempt.

The create-once real-data preflight is one B8/T1024 batch per seed and writes
only a small provenance record. Evaluation requires all three matching
preflights. `coordinator.py` defines those three preflight tasks followed by one
three-seed evaluation; `dispatcher.py` holds the shared per-GPU flock across
each child, admits only a GPU with no compute owner and at most 512 MiB used,
monitors owned process descendants, and never retries a failed task.

Focused CPU checks:

```powershell
python -m unittest collaborative_test.SW_0124_temporal_prototype_readout.test_prototype_readout collaborative_test.SW_0124_temporal_prototype_readout.test_evaluate collaborative_test.SW_0124_temporal_prototype_readout.test_dispatcher -v
```

The suite validates variable temporal lengths, deterministic anchors,
identical-trace K=1 behavior, live spike gradients, source-QCC baseline
rejection, and the shared-image bootstrap contract. It does not substitute for
the registered source/GPU evaluation.

## Actual execution (2026-10-09)

Registered commit3d4b462 was deployed as10 SHA-verified files. Local and Frontier14 CPU tests passed; all3 immutable source/evaluation/calibration bindings passed beforelaunch. Supervisor1491982 is live; GPU0 source seed0 actual B8/T1024 preflight passed and seed1 was observed running. All3 preflights precede full320 three-seed comparison. No optimizer updates or checkpoint edits. Result pending; no gating/scaling/promotion claim. Peer branch697eeb6 unchanged thiscycle. See server_launch_evidence_20261009.json.

## Completed outcome (2026-10-09)

All3 actual preflights and full320 evaluations passed execution validation; supervisor1491982 and evaluation1492295 are terminal. Original source QCC reproduced all per-image metrics within1e-10. Prototype3seed means FG/foregroundIoU/objectIoU: .751922/.397001/.552672 versus source97 .776348/.577070/.601537 and matchedSlot .774933/.203589/.206937. Every seed lostFG; paired image-bootstrap FG delta -.024426,95%CI[-.030902,-.017989]. Promotion failed. Incumbent97 unchanged. No threshold/temperature/count/init sweep, no70k expansion or joint training of this failed recipe. Relative temporal structure retained substantial instance information versus123 global mean-pooled assignment, but this deterministic competition worsened existing partitions and foreground selection. This does not prove gating usefulness or data scaling. Full source/prediction hashes and per-image results are in completed_results_20261009.json. Earlier running text is historical.

````

</details>
