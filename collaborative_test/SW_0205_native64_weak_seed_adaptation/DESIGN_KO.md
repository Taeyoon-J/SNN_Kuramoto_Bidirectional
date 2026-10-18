# Native64 adaptation of independent weak baseline seeds1/2

SW0197 fixed32 baseline mean .8514, with seed0 .9095 but seed1 .8175 and seed2
.8272. The weaker two have not received the SW0192-style native64 TRAIN8 probe.
Test exactly that adaptation on each independent native32 incumbent mapped64.
Two B4 guarded-descent Adam proposals on first8 registered TRAIN IDs; all five
parameter families enabled, original QCE .1, live64, original inference.
Use the same transduction and fresh empty native64 Adam as192; do not copy seed0
weights into the other runs. Initial models must equal completed197 model hashes.
This tests adaptation, not additional unique-data scaling or a new loss.

Fixed16 development offsets0,2,...30 per seed; read GT after each candidate is
frozen. Compare to each corresponding validated197 baseline on identical IDs/RNG.
Require new applied learning and changed source CNN, paired FG gain >=.01 with
CI lower bound >0, and nonnegative IoU changes for extension. No full320/.90
claim. Rejecting both proposals remains a valid unchanged-model outcome.
All GPU jobs use shared exclusive leases across0..3 and checkpoint resume.
