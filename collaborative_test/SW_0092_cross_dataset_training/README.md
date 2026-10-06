# SW0092 — reciprocal cross-dataset training

This experiment runs both directions requested after obtaining the released
Slot Attention TFDS data.

1. **Our model on released Slot data.** TFDS CLEVR 3.1.0 train is processed in
   released-code order: skip the first 512 rows, keep scenes with at most six
   objects, center-crop, and bilinear-resize to 128x128. This yields 34,766
   unique training images. The strongest fixed-feature/frozen-graph SNN recipe
   trains for 1/3/10 epochs and is evaluated on our fixed mask-bearing HDF5
   validation IDs1320-1639.
2. **Slot Attention on our data.** The official architecture is trained from
   scratch on our 70,000 unique training IDs for ten exact passes. It uses 11
   slots for the up-to-ten-object HDF5 scenes. A real batch64 backward preflight
   exceeded 24GB; batch32 passed, so all registered runs use batch32. Epochs
   1/3/10 are evaluated on the same validation IDs and three metrics.

The first direction must use our validation domain for mask scores because TFDS
CLEVR 3.1.0 does not provide instance masks. Consequently it is explicitly a
cross-domain transfer test, while the second direction is a matched-data test.

