# SW0236: TRAIN512 exact continuation at matched512-update budget

Restore actual SW0226 TRAIN512/128 weights, Adam, history and Torch RNG exactly;
add384 updates to cumulative512. Compare with SW0232 TRAIN64/512 on fixed16,
three seeds. Same teacher objective and encoder+actual gamma projection; other
core states and spike/membrane readout frozen. CPU restore/gradient admissions
precede GPU queues. Wait until every SW0235 seed has had its first GPU attempt.
No result yet. No fresh70000 training, full320 promotion or gating causal claim.
