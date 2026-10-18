# SW0198 native64 graph weighting at readout

One independent all-GPU ready job. Two frozen models, TRAIN8 and TRAIN64,
each use eight predetermined development images. One original forward supplies
both the registered1024 and actual4096 graph-weighted signed partitions.
Reference masks must exactly replay frozen193/194 masks and foreground support
must remain identical. No optimizer updates or data changes.

This tests resolution normalization at the decoder. Separately, a CPU regression
case demonstrates the relative-QCE/absolute-0.5 readout ambiguity: shifting all
off-diagonal affinities can preserve QCE but split a correct partition. That
example does not prove it caused the actual194 decline; normalization alone
cannot guarantee calibration of Q around0.5.

Do not present this screen as full320/three-seed superiority or gate causality.
The existing shared classifier and running jobs remain unchanged.

Completed; actual32 masks (8images x2models x2readouts), frozen-reference replay,
foreground equality and all three metrics/bootstrap intervals independently
verified, maximum metric error2.22e-16.

| Frozen model | Reference1024 FG / FIoU / objectIoU | Actual4096 FG / FIoU / objectIoU |
| --- | --- | --- |
| TRAIN8 | .954783 / .883223 / .799757 | .955533 / .883223 / .800246 |
| TRAIN64 | .878008 / .865682 / .766942 | .878008 / .865682 / .766942 |

Primary FG delta+.000749,95%CI[0,.002248], below the registered+.01/positive-lower
gate. ObjectIoU delta+.000489,CI[-.000156,.001623]. TRAIN64 masks/metrics did not
change. The normalizer alone did not rescue the continuation decline; close this
screen without intermediate-coefficient tuning or changing the production code.
These are eight predetermined images, not the fixed32 or full320 scores; the .9555
number does not establish the native64 three-seed .90 objective. Incumbent unchanged.
