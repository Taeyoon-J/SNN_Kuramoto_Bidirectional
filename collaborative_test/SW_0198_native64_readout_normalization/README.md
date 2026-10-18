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
