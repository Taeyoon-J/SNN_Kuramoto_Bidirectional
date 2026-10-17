# SW0191 — production-partition error feedback

Implementation/preflight in progress; no GPU training started.
Keep the SW0186 incumbent and production masks, adding a class-balanced loss
on actual split/merged TRAIN patch pairs. Mined labels stay fixed across each
proposal's gradient and acceptance measurements. Separately check that this
auxiliary loss reaches emitted spikes, beyond the existing QCE gradient.

See [registered design](DESIGN_KO.md). Native32 joint70k, FG.87, automatic count
prediction and gate-causality proof remain outstanding. No new performance claim.
