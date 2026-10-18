# SW0219 native64 actual-Q readout ablation

CPU reuse of all48 private native rows from217. Bitwise replay the original
Q/A baseline, then remove only A weighting at readout using the same production
agglomerator. No new core forward, GPU work, fitted threshold or model update.
The native graph remains inside the actual spike dynamics. This is a readout
ablation, not removal of a physical gate. See DESIGN_KO.md.
