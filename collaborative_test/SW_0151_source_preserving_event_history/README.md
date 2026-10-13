# SW0151: source-preserving adaptive event history (candidate)

This folder contains an isolated CPU-tested event/spike helper, rollout wrapper, and a read-only actual-source parity/calibration preflight driver for a prospective experiment. It is not admitted for training, and it makes no performance or mechanism claim.

The existing SW0134 rollout computes an integrated membrane value, uses the source threshold `0.06 * exp(b)` for both the event decision and the unchanged membrane reset equation, then emits `event * gate`. The helper here consumes that already-computed membrane, gate, and existing positive threshold. It adds four learned decay values initialized to 0.95 and four nonnegative projected history gains initialized to zero. Its adaptive event threshold is the source threshold plus `kappa * s * history`, where `s` must be a fixed positive training-only calibration. The helper does not change membrane integration or reset coefficients. `adaptive_rollout.py` is a separate SW0134-shaped wrapper pinned to the source rollout, phase integration model, native membrane activation, and gating source hashes. It copies the frozen rollout equations and substitutes only the event/history step; emitted adaptive spikes are fed back through the existing dendrite and membrane reset paths.

The helper matches the native float32 event path and rejects other membrane/gate dtypes rather than silently converting them. At initialization (`kappa=0`), event and emitted-spike outputs match the native event rule for any supplied source threshold. A zero gate emits zero and holds both auxiliary recurrent states. Recurrent state stays connected across live steps by default; the caller must explicitly detach state at the registered truncation boundary. The wrapper has not been integrated into a registered training or evaluation pipeline. CPU parity tests use a small synthetic native core; they do not establish parity on a trained native32 checkpoint, GPU performance, learning feasibility, or improved segmentation. In an integrated rollout, changed emitted spikes can affect later native reset inputs through the existing reset path; that native reset rule itself must remain unchanged.

Run the CPU tests from the repository root:

```powershell
python -m unittest collaborative_test.SW_0151_source_preserving_event_history.test_event_history collaborative_test.SW_0151_source_preserving_event_history.test_adaptive_rollout collaborative_test.SW_0151_source_preserving_event_history.test_preflight -v
```

The preflight driver reuses SW0135's strict source loader and saved seed1 parity receipt, then streams the first64 registered source1 TRAIN RGB images in microbatches. It compares native and helper traces plus SW0145's production Q at exact `kappa=0`, computes a fixed per-component float64 RMS threshold scale from those same native traces, and records descriptive event/history statistics and timing. It is read-only with respect to models and data and leaves `training_admission=false`. It has not been executed on a GPU or trained source checkpoint.

No GPU, dataset files, ground-truth masks, or training artifacts are used by the CPU tests.
