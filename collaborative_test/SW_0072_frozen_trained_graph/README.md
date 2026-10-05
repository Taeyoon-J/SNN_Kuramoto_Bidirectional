# SW0072 - frozen trained seed0 graph

SW0065 shows that replacing only seed2's graph generator with the trained
seed0 graph rescues the 32-image result from `.3895/.2003/.1822` to
`.7036/.3081/.5376`. SW0066 fixed only its random initialization; training was
still free to move weak seeds back to different graph solutions.

SW0072 makes the causal intervention direct. Seeds 1 and 2 retain their own
non-graph initialization and data order, but load the completed SW0055 seed0
graph generator and keep it frozen for all ten epochs. Data, loss, core LR,
epoch count, and evaluation contract remain unchanged. The runner asserts that
every graph tensor is bitwise equal to the source after training.

