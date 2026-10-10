# SW0130: phase-conditioned state integration

The opt-in model adapter is implemented. Six focused CPU tests passed, including exact native initialization parity for both arms at 64 and 1024 steps on a synthetic 64-region learned-graph fixture. The model equations passed a separate Sol6.1 review. The three actual SW0097 source checkpoint hashes were verified on Frontier.

The TRAIN preflight runner is now implemented and its import/CLI, compilation and eight total focused CPU tests passed. A separate review verified both arms' disposable gradient/update path and the shared source0 lambda provenance. The runner remains frozen for the actual screens; later training/evaluation stages will be separate files so they cannot invalidate a passed preflight fingerprint.

These results establish code behavior, not improved segmentation. The server passed all 12 focused CPU tests after hash-verified deployment of 12 research files and verification of 16 dependencies. The exclusive-GPU preflight queue is live: supervisor PID 2245879, seed0 child PID 2245982 on physical GPU0, with seeds1/2 queued after seed0. No paired training has started. All three actual source seeds must pass the registered TRAIN checks before the paired pilot can start. See server_preflight_launch_20261009.json for the launch observation.

`model.py` wraps a strictly loaded native core and leaves production modules untouched. Both arms have the same twelve added parameters. The phase arm gates the dendritic and membrane state-update amount with the native continuous gate; the control uses a constant integration signal while retaining the native rhythmic carrier and emitted-spike multiplier. Zero added parameters preserve native dynamics. The classifier remains actual-spike QCC.

See `scientific_design_20261009.md` and `protocol.json` for the objective, calibration, budget and decision rules; `source_contract_20261009.json` and `local_cpu_review_20261009.json` for verification scope. An improvement in this pilot would not by itself establish every gating contribution or data scaling.
