# PV2_0132 — 253 at grid 64 on CLEVR, 10,000 images x 10 epochs

Status: **completed**, one seed. A candidate, not an achievement: the goal is judged on
3-seed means and this is seed 0 alone.

## The three metrics

Pixel protocol, 320 held-out images, unsupervised, no oracle.

| | peer 3,500 x 10 | **this, 10,000 x 10** | absolute | relative |
| --- | ---: | ---: | ---: | ---: |
| FG-ARI | 0.4281 | **0.7794** | +0.3513 | +82.1% |
| foreground IoU | 0.2778 | **0.7308** | +0.4530 | +163.1% |
| matched-object IoU | 0.2407 | **0.5807** | +0.3400 | +141.3% |

The historical fixed64 patch contract, reported alongside as the engine does:
**0.8332 / 0.7433 / 0.5990**.

## Validation says the same thing

Scored afterwards on the 320 validation images (ids 1000-1319), which nothing in this
benchmark had ever scored:

| | fixed320 | validation | difference |
| --- | ---: | ---: | ---: |
| FG-ARI | 0.7794 | 0.7862 | +0.0068 |
| foreground IoU | 0.7308 | 0.7513 | +0.0205 |
| matched-object IoU | 0.5807 | 0.5808 | +0.0001 |

Two disjoint 320-image sets agreeing to 0.02 at worst, and to 0.0001 on object IoU. **This
is what makes validation usable for selection from here on**: a configuration chosen there
should carry to fixed320. Had they disagreed, selection on validation would have been
worthless.

That matters because `data.py:19` labels fixed320 itself as
`'fixed320 reserved evaluation, not an independent original-paper TEST split'`, and
`bootstrap.py:150` as `'reused fixed320 validation, not untouched test'`. **The peer does
not treat those 320 images as a pristine test split**, so there is no untouched test set
here to protect -- but there is no reason to keep spending fixed320 on configuration
choices either.

## Where the masks come from

`evaluation.py:21-26`. 253 has no head, so `predict` falls through to:

```python
groups = spike_synchrony_components(trace['spikes'].detach().cpu(),
    synchrony_threshold=.5, affinity_mode='spike', ...)
```

**The masks are built from `trace['spikes']`**, not from theta or PLV. That satisfies the
standing rule that a phase readout is never the score. Stated precisely: this is spike
affinity *clustering*, not a trained classifier -- it runs on spikes, which is the part the
rule is about, but the distinction is not blurred here.

`mask_opened_during_training: false` and `GT_used_for_prediction: False`: unsupervised, no
oracle.

## The signal flow is alive end to end

Gradient coverage from `training_report.json`, every group non-zero:

```
gamma_phase_gain 2410 -> graph_generator (projection 26792, coupling_gain 661, temperature 285)
-> kuramoto (omega 7033, kappa 2037, direction_learner 19.9, freq_gain 1024)
-> dendric_layer (oscillator_dense 16045, tau_n 291) -> membrane_layer (tau_m 49)
```

Parameter change 0.09-0.20 for most tensors. `direction_learner` moved 0.0021, small in
relative terms because it is the 4096 x 4096 coupling matrix -- 16.8M parameters, so a
large absolute change spread thin.

Loss 82.4 -> 14.76 over the run; the spike term `positive_spike` 15.46 -> 2.50, which is
the one that has to move if the score comes from spikes.

## Three things this does not show

1. **One seed.** Not a 3-seed mean, so not a result under the standing rule.
2. **The 0.4281 baseline is the peer's reported number, not one we reproduced.** If it came
   from the same evaluation code on the same 320 ids, the comparison holds; that has not
   been checked. **"+82%" must not be written down until it is.**
3. **Two variables moved together.** 3,500 x 10 is 35,000 exposures and 2,190 updates;
   10,000 x 10 is 100,000 and 6,250. Unique images and gradient steps both rose 2.86x, so
   this does not say which one bought the improvement. The arm that would separate them is
   3,500 unique x 29 epochs -- same steps, peer's image count.

## Reproduction

```
peer bundle 5372351, 3 vendored files modified (fused coupling, TRAIN 10,000, exposure
budget); see MODIFIED.txt
SW_TRAIN_PRESET unset -> 10,000 drawn with RandomState(2553500) from the 35,000 parent
python -u -B run.py --config configs/cloud.json --output <out> \
  train --model 253 --dataset clevr --seed 0
```

Trained on a rented RTX 4090 (Vast), 18.50 h at 10.714 s/update, $8.03 -- **not on
frontier**, where the same job measured 117 h while sharing a card with the peer's jobs.
Checkpoint, predictions and logs recovered to frontier; `resume.pt` sha256 matches
`committed.json`.
