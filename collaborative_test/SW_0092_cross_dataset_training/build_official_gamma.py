import argparse, hashlib, json, sys
from pathlib import Path
import h5py
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from snn_kuramoto_bidirectional.gamma_initializer import feature_maps_to_patch_gamma
from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_input_encoder

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--output", required=True)
p.add_argument("--manifest", required=True)
p.add_argument("--encoder", default="/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/input_encoder/input_layer_encoder.pt")
p.add_argument("--stats", default="/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/feature_preprocessing.pt")
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--device", default="cuda")
a = p.parse_args()
out, manifest = Path(a.output), Path(a.manifest)
if out.exists() or manifest.exists(): raise FileExistsError("refusing overwrite")
stats = torch.load(a.stats, map_location="cpu", weights_only=True)
mean, std, clip = stats["mean"].float(), stats["std"].float(), float(stats.get("clip", 3.0))
device = torch.device(a.device)
encoder = load_input_encoder(a.encoder, num_kernels=8, kernel_size=3, channels=3, device=device).eval()
rows = []
with h5py.File(a.dataset, "r") as h5, torch.no_grad():
    n = len(h5["image"])
    for start in range(0, n, a.batch_size):
        x = torch.from_numpy(h5["image"][start:start+a.batch_size].astype("float32")).permute(0,3,1,2).to(device)
        features = encoder(x).cpu()
        rows.append(feature_maps_to_patch_gamma(((features-mean)/std).clamp(-clip,clip), grid_size=16, device="cpu").float())
gamma = torch.cat(rows)
if tuple(gamma.shape) != (n,8,256) or not torch.isfinite(gamma).all(): raise AssertionError(gamma.shape)
out.parent.mkdir(parents=True, exist_ok=True); torch.save(gamma, out)
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
manifest.write_text(json.dumps({"dataset":a.dataset,"shape":list(gamma.shape),"encoder":a.encoder,
                                "stats":a.stats,"gamma_sha256":sha(out)},indent=2)+"\n")
print(json.dumps({"shape":list(gamma.shape),"output":str(out)},indent=2))

