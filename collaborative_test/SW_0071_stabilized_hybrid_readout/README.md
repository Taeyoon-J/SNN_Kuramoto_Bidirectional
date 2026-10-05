# SW0071 - stabilized spike foreground with membrane/spike partition

SW0070 establishes the new full three-seed spike baseline
`.6774/.4354/.4799`. SW0057 shows a stronger, more seed-stable grouping signal
in membrane activity, but its foreground selection is poor. SW0051 already
provides the causal composition needed here: freeze the spike-CC foreground
mask, then repartition only those patches from membrane or component-spike
affinity.

SW0071 applies that readout to all graph-init0 checkpoints on the full fixed
long contract. The frozen foreground uses the registered primary spike
threshold `.50`, so every hybrid must preserve SW0070 foreground IoU exactly.
No masks or object counts enter prediction. A readout is accepted only if its
three-seed mean raises FG-ARI without lowering matched-object IoU relative to
SW0070; foreground IoU is asserted equal.

## Result

No hybrid is accepted. Against the spike-CC baseline recomputed in the same
run (`.676744/.435449/.479205`), spike-freeze raises FG-ARI by `.022007` to
`.698751` and preserves foreground IoU exactly, but lowers matched-object IoU
by `.033346` to `.445859`. The closest balanced row, spike-restrict-dynamic,
changes the metrics by `+.001059/0/-.010966`. This confirms useful continuous
grouping information, but replacing whole spike components damages object
overlap. Stop this readout direction and retain SW0070 as the formal reference.
