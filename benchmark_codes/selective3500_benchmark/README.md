# SW0258: fixed core, eligible reuse, otherwise scratch3500x10

This user-requested revision replaces the longer SW0257 encoder schedule.
Scope:252/253 x our CLEVR HDF5/Tetrominoes/CLEVRTex Original x seed0/1/2.
Same-seed SW0097 native16 core is mapped64 and remains bitwise fixed, including
graph/projection/integration. No new core training. Protected benchmark unchanged.

Reuse requires the same dataset, seed and model family, compatible native64
interface, **at least3500 unique TRAIN images and35000 image exposures**.
Actual historical data counts and originating core are recorded, never relabelled
as an identical3500x10 run. Eight current252 encoders qualify. One earlier252
Tetrominoes seed0 head qualifies; it was trained with a different core, so this is
explicit component transfer while retaining the requested SW0097 core. Original
253's1000x100 encoder fails the unique-image floor and is not silently counted as
3500 unique images. Source selection and file SHA are in the runtime selection.json.

Missing components start from random initialization. Fresh encoders train on
3500 unique registered TRAIN images x10 epochs, physical16/final12:35000 exposures,
2190 Adam updates.252 uses the existing direct gamma/frequency teacher objective;
253 uses CNN RGB reconstruction MSE with an auxiliary decoder. Recurrent core
simulation/backward is forbidden during encoder fit. Fixed source preprocessing
statistics are retained for the preserved core. No TRAIN mask supervision for253.

Fresh252 heads use the existing neutral144-feature CE/Dice architecture and loss,
3500 TRAIN images x10,35000 single-image updates. Qualifying reused heads receive
no extra fit.253 has no learned head. Previously computed spike/membrane feature
caches are hardlinked only after encoder/core/IDs/source-statistics inventory and
all shared scientific source hashes match. Head weights do not generate features.
No cached activity is reused when a model/input binding differs.

Fixed evaluation: CLEVR320, Tet1000, CLEVRTex1000, protected native-pixel scoring,
plus historical CLEVR64-patch scores. Each condition passes actual-CUDA admission
before full execution; a final independent audit checks fresh TRAIN IDs/updates,
source eligibility, saved predictions and recomputed metrics. Results report
reuse/fresh histories per condition. No architecture search or data substitution.

```sh
python -B run_selective3500.py --output NEW_OUTPUT --selection SELECTION_JSON preflight
python -B run_selective3500.py --output NEW_OUTPUT --selection SELECTION_JSON queue
```

All prior SW0257 encoder/Adam/RNG and feature files remain preserved. The new
queue reserves8GiB per worker, keeps the original13GB allocator, limits two jobs
per GPU and does not interrupt foreign jobs. Terminal failures preserve outputs
and require diagnosis before resume. Full completion requires18 independent audits.
