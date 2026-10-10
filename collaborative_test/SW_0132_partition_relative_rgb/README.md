# SW0132 partition-relative full RGB reconstruction

This experiment replaces the weak absolute-XY patch-average RGB objective diagnosed in SW0131 with full128x128 RGB reconstruction in each predicted partition's relative coordinate frame. Source97, zero-initialized SW0130 state integration and native actual-spike QCC remain unchanged. Both matched arms receive the same new supervision.

Geometry and content pooling keep assignment credit; gamma content is detached. The decoder receives pooled8D content and relative2D coordinates, with no separate absoluteXY, center, scale or alpha head. All three source seeds must pass fixed TRAIN-only checks before the seed1 paired pilot. No training or new score exists yet.

See protocol.json for exact prospective equations, budget, objective calibration, screens and claim limits. Sol6.1 reviewed the design and clarified live W pooling; implementation is assigned to Luna.
