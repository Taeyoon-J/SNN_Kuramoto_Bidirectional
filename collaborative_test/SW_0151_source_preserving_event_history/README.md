# SW0151: source-preserving adaptive event history (candidate)

This folder contains an isolated CPU-tested event/spike helper for a prospective experiment. It is not registered or admitted for training, and it makes no performance or mechanism claim.

The existing SW0134 rollout computes an integrated membrane value, uses the source threshold `0.06 * exp(b)` for both the event decision and the unchanged membrane reset equation, then emits `event * gate`. The helper here consumes that already-computed membrane, gate, and existing positive threshold. It adds four learned decay values initialized to 0.95 and four nonnegative projected history gains initialized to zero. Its adaptive event threshold is the source threshold plus `kappa * s * history`, where `s` must be a fixed positive training-only calibration. The helper does not change membrane integration or reset coefficients.

The helper matches the native float32 event path and rejects other membrane/gate dtypes rather than silently converting them. At initialization (`kappa=0`), event and emitted-spike outputs match the native event rule for any supplied source threshold. A zero gate emits zero and holds both auxiliary recurrent states. Recurrent state stays connected across live steps by default; the caller must explicitly detach state at the registered truncation boundary. The eventual rollout integration is intentionally not implemented here, so helper tests do not establish whole-rollout parity, learning feasibility, or improved segmentation. In an integrated rollout, changed emitted spikes can affect later native reset inputs through the existing reset path; that native reset rule itself must remain unchanged.

Run the CPU tests from the repository root:

```powershell
python -m unittest collaborative_test.SW_0151_source_preserving_event_history.test_event_history -v
```

No GPU, dataset, ground-truth masks, or training artifacts are used by these tests.
