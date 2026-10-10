# SW0130: phase-conditioned state integration

The opt-in model adapter is implemented. Six focused CPU tests passed, including exact native initialization parity for both arms at 64 and 1024 steps on a synthetic 64-region learned-graph fixture. The model equations passed a separate Sol6.1 review. The three actual SW0097 source checkpoint hashes were verified on Frontier.

These results establish local adapter behavior, not improved segmentation. No SW0130 GPU screen or training has started. The TRAIN preflight runner is being implemented; all three actual source seeds must pass its registered checks before the paired pilot can start.

`model.py` wraps a strictly loaded native core and leaves production modules untouched. Both arms have the same twelve added parameters. The phase arm gates the dendritic and membrane state-update amount with the native continuous gate; the control uses a constant integration signal while retaining the native rhythmic carrier and emitted-spike multiplier. Zero added parameters preserve native dynamics. The classifier remains actual-spike QCC.

See `scientific_design_20261009.md` and `protocol.json` for the objective, calibration, budget and decision rules; `source_contract_20261009.json` and `local_cpu_review_20261009.json` for verification scope. An improvement in this pilot would not by itself establish every gating contribution or data scaling.
