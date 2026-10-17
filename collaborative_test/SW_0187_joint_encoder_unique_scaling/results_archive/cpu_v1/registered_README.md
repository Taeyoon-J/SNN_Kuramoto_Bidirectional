# SW0187 — joint encoder, unique1024 versus repeated512

Preparing CPU and runtime admission. No GPU training has been started.
Both arms share a contemporaneous32-proposal prefix and then fork exact
model/Adam/RNG state. Only the second512 image identities differ.
The original RGB CNN remains trainable exactly as in SW0186.
See [the registered design](DESIGN_KO.md).
