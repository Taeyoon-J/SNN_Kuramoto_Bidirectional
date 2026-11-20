"""Train an input encoder per benchmark dataset, the way MOVi-A got one (PV2_0086).

Contract v1's 8-channel gamma comes from a 224-parameter 3x3 convolution trained by
reconstruction (train_input_layer_generator), then parameterless patch pooling. MOVi-A
got its own encoder rather than borrowing CLEVR's, because PV2_0085's control reached
only .3042 against .7250 when its gamma was raw RGB patch means instead of a trained
encoder. Tetrominoes and Multi-dSprites get the same treatment: same recipe, own
statistics, so a number from one dataset can stand beside a number from another.

TRAIN frames only -- the manifest's [0, 6000) -- so validation and test stay unseen.
Reconstruction is unsupervised; no mask is read.
"""
import argparse, json, sys
import torch
sys.path.insert(0, "/export_home/tkim1")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")
from training.train_input_layer_generator import train_input_layer_generator, save_input_encoder

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--src", default="/work/USERS/tkim1/mod")
ap.add_argument("--num-kernels", type=int, default=8)
ap.add_argument("--kernel-size", type=int, default=3)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

man = json.load(open("%s/split_manifest_%s.json" % (args.src, args.dataset)))
lo, hi = man["ranges"]["train"]
img = torch.load("%s/images_%s.pt" % (args.src, args.dataset), map_location="cpu")
images = img[lo:hi].permute(0, 3, 1, 2).float() / 255.0      # [N, 3, H, W] in [0,1]
print("%s: training on %s TRAIN frames, ids [%d, %d)"
      % (args.dataset, tuple(images.shape), lo, hi), flush=True)

encoder, _, history = train_input_layer_generator(
    images, num_kernels=args.num_kernels, kernel_size=args.kernel_size,
    epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=args.device)
print("reconstruction loss %.6f -> %.6f over %d epochs"
      % (history[0], history[-1], len(history)))
params = sum(p.numel() for p in encoder.parameters())
print("encoder parameters: %d (contract v1's is 224 at 8 kernels of 3x3)" % params)
out = "%s/input_encoder_%s_k%d.pt" % (args.src, args.dataset, args.num_kernels)
save_input_encoder(encoder, out)
print("saved %s" % out)
