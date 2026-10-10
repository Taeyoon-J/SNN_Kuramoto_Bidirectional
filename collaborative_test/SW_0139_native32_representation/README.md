# SW0139 native32 representation pilot

Compare old-loss control, trainable context residual encoder, and cross-view spike consistency. All arms use genuine 32x32 patches, the same ordered4096 TRAIN images and256 logical updates. Encoder and learned graph train together; oscillator, dendrite, membrane and integration parameters are frozen. No TRAIN masks, object counts or learned assignment head. Cross-view doubles view exposures and is not FLOP matched.

Local/server CPU tests18/18, actual source RGB gamma parity and optimizer groups verified. Supervisor2711934 and three GPU disposable preflight/calibration workers verified live. Fresh training follows successful disposable preflights; fixed320 endpoint predictions precede GT scoring. This is a seed1 pilot, not fresh70000-image training or evidence of gating necessity. Publication remains pending scope approval.

Control training PID2712399 GPU1 and context training PID2712528 GPU2 subsequently verified live after disposable preflights passed. Cross-view calibration remains running, its training queued. See training_start_20261010.json.
