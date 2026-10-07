# Experiment continuation policy

The final goal remains a **three-seed mean above the saved Slot Attention
reference on all three shared patch metrics**, using spike- or membrane-derived
masks. However, a candidate does **not** need to improve all three metrics in
one step to remain useful.

Maintain multiple validation tracks:

- **FG-ARI-first track:** preserve and extend methods that substantially
  improve foreground object separation, even if foreground IoU or matched
  object IoU is temporarily lower. These remain open candidates for later
  foreground/background or mask-quality improvements.
- **Foreground and mask-quality tracks:** retain methods that materially
  improve either IoU, including as components to combine with an ARI-leading
  method, while recording the full three-metric tradeoff.
- **Balanced track:** prioritize any method that improves all three metrics
  against a matched baseline. Test only frozen validation-selected settings
  on the reference split; never choose thresholds from reference-test scores.

Use the same checkpoint, images, grid, readout, and seeds for a controlled
comparison whenever possible. Label pilots versus full validation and do
not claim final success until the exact three-seed contract is verified.
Do not silently discard a useful one-metric result; explain which track it
advances and what must improve next.

## GPU allocation ??user update 2026-10-07

Before launching a job, check GPU compute PIDs and their actual Unix owners.
If either GPU2 or GPU3 is occupied by tkim1 (the user's colleague, referred to
as tkim), our new jobs may use GPU0/1 only. Do not migrate, terminate or share
the colleague's jobs. Existing SW0095 seed2 is already on GPU0 and remains
running. Otherwise GPU2 retains its earlier authorization subject to the
usual ownership/availability checks; GPU3 remains reserved for the colleague.

Requested reasoning/work split, as authorized by the user: important design decisions on6.1 Sol,
ordinary analysis on5.6 Sol, and implementation/execution on6 Luna (replacing
unavailable5.6 Luna with user approval). The implementation operator is6 Luna;
analysis is handled by5.6 Sol. Use6.1 Sol only when a major branch decision
requires it. Main-agent model switching is not exposed as a tool in this
session, so do not claim automatic routing or that the main-agent model changed.
