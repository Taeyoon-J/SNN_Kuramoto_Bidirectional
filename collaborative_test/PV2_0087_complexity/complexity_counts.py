# Exact MAC / parameter counts for the two models, from the architectures read
# out of the code. One image, one forward pass.

def ours(N, T, C=8, d=16, D=4, S=3, k=32, img=128, kern=3, nk=8):
    enc  = (img - kern + 1) ** 2 * kern * kern * 3 * nk        # one valid 3x3 conv
    proj = N * C * d                                            # graph projection
    sim  = N * N * d                                            # z z^T
    geo  = S * N ** 3                                           # min-plus relaxation
    topk = N * N                                                # top-k + softmax + symmetrise
    kura = T * 4 * N * N * D                                    # 4 bmm [N,N]x[N,D] per step
    spik = T * N * (D * 2 + 4)                                  # gate+dendrite+membrane+spike
    aff  = N * N * T                                            # spike correlation
    spec = N ** 3                                               # eigendecomposition
    return {"encoder": enc, "graph proj+sim": proj + sim, "geodesic O(N^3)": geo,
            "top-k": topk, "Kuramoto O(T N^2 D)": kura, "gate/dendrite/membrane": spik,
            "spike affinity O(T N^2)": aff, "spectral O(N^3)": spec}

def slot(K=7, I=3, Ds=64, H=128, Cc=64, mlp=128):
    P = H * H
    enc = P * 25 * 3 * Cc + 3 * (P * 25 * Cc * Cc)              # 4 conv 5x5 SAME
    pos = P * 4 * Cc
    mlp_in = P * (Cc * Cc + Cc * Cc)
    kv = 2 * P * Cc * Ds                                        # k,v once
    per_iter = (K * Ds * Ds                                     # q
                + K * P * Ds                                    # attention logits
                + K * P * Ds                                    # weighted mean
                + K * 6 * Ds * Ds                               # GRU
                + K * 2 * Ds * mlp)                             # slot MLP
    sa = kv + I * per_iter
    dec = 0                                                     # per slot, in-pixels x k^2 x Cin x Cout
    for inp, kk, cin, cout in ((8, 25, Cc, Cc), (16, 25, Cc, Cc), (32, 25, Cc, Cc),
                               (64, 25, Cc, Cc), (128, 25, Cc, Cc), (128, 9, Cc, 4)):
        dec += inp * inp * kk * cin * cout
    dec += 8 * 8 * 4 * Cc                                       # decoder position embed
    return {"CNN encoder": enc, "position+MLP": pos + mlp_in,
            "slot attention": sa, "broadcast decoder x%d slots" % K: K * dec}

def show(name, parts):
    tot = sum(parts.values())
    print(name)
    for k, v in sorted(parts.items(), key=lambda x: -x[1]):
        print("    %-30s %10.3e  %5.1f%%" % (k, v, 100 * v / tot))
    print("    %-30s %10.3e" % ("TOTAL", tot))
    print()
    return tot

print("=" * 66, "\nMACs per image, one forward pass\n" + "=" * 66)
o16 = show("ours, 16x16 grid (N=256, T=1024 eval)", ours(256, 1024))
o32 = show("ours, 32x32 grid (N=1024, T=1024 eval)", ours(1024, 1024))
sa  = show("Slot Attention (128x128, 7 slots, 3 iters)", slot())

print("ours 16x16 vs Slot : Slot is %.1fx our cost" % (sa / o16))
print("ours 32x32 vs Slot : Slot is %.2fx our cost" % (sa / o32))
print("our 16->32 grid    : %.1fx more expensive for a 4x finer grid" % (o32 / o16))
print()
t16 = sum(ours(256, 64).values()); t32 = sum(ours(1024, 64).values())
print("training-time (T=64): N=256 %.3e   N=1024 %.3e  (%.1fx)" % (t16, t32, t32 / t16))
print()
print("geodesic peak intermediate, batch 16, float32:")
for N in (256, 1024):
    print("    N=%4d  B*N^3*4B = %6.2f GB" % (N, 16 * N ** 3 * 4 / 1e9))

print()
print("=" * 66, "\nparameters\n" + "=" * 66)
p = 0
for inp, kk, cin, cout in ((0,25,3,64),(0,25,64,64),(0,25,64,64),(0,25,64,64)):
    p += kk * cin * cout + cout
p += 4 * 64 + 64                      # encoder_pos Dense(4->64)
p += 2 * 64 + 64 * 64 + 64 + 64 * 64 + 64   # layernorm + MLP
p += 3 * 64 * 64                      # q,k,v no bias
p += 3 * (64 * 64 + 64 * 64 + 64)     # GRUCell
p += 64 * 128 + 128 + 128 * 64 + 64   # slot MLP
p += 2 * 2 * 64                       # slots_mu, slots_log_sigma + layernorms
dec_p = 0
for kk, cin, cout in ((25,64,64),(25,64,64),(25,64,64),(25,64,64),(25,64,64),(9,64,4)):
    dec_p += kk * cin * cout + cout
p += dec_p + 4 * 64 + 64
print("Slot Attention      ~%d parameters" % p)
print("ours                 69042 trainable (200114 tensors - 2 x 65536 buffers)")
print("                     of which PV2_0066 found 1325 matter")
