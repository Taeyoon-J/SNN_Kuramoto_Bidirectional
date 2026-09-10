"""
Score a trained core's grouping against CLEVR object masks.

This lived only in ad-hoc scripts under /tmp on the cluster and was rewritten
from scratch every session, which is how the readout comparison ended up being
run against the wrong baseline more than once. It is a file now.

Every readout the project has is scored side by side, at the oracle cluster
count and at fixed counts, because the choice of k turned out to matter more
than the choice of readout: on the phase route a fixed k=3 beats the oracle.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# CLEVR names its object colours, and the renderer paints them from a fixed
# palette, so an object mask can be recovered from the image without any
# segmentation ground truth. This captures about 78% of the scene objects:
# objects that are occluded, in shadow, or made of the reflective material lose
# too many pixels to the coverage test.
PALETTE = {
    "gray": (87, 87, 87), "red": (173, 35, 35), "blue": (42, 75, 215),
    "green": (29, 105, 20), "brown": (129, 74, 25), "purple": (129, 38, 192),
    "cyan": (41, 208, 208), "yellow": (255, 238, 51),
}
MIN_COVER = 0.30


def build_masks(names, objects_by_name, image_dir, grid):
    """
    Object id per patch, -1 for background. [len(names), grid*grid]

    Ported here from a script that lived outside the repository with the grid
    size fixed at 16, which is what made a 32x32 comparison impossible to run.
    """
    image_dir = Path(image_dir)
    labels = torch.full((len(names), grid * grid), -1, dtype=torch.long)
    for bi, name in enumerate(names):
        image = Image.open(image_dir / name).convert("RGB")
        width, height = image.size
        pixels = torch.as_tensor(list(image.getdata()), dtype=torch.float32).view(height, width, 3)
        yy, xx = torch.meshgrid(torch.arange(height).float(),
                                torch.arange(width).float(), indexing="ij")
        best = torch.zeros(grid * grid)
        for oi, obj in enumerate(objects_by_name[name]):
            colour = torch.tensor(PALETTE[obj["color"]], dtype=torch.float32)
            cx, cy = obj["pixel_coords"][0], obj["pixel_coords"][1]
            near = ((xx - cx) ** 2 + (yy - cy) ** 2).sqrt() < 60.0
            mask = ((pixels - colour).norm(dim=-1) < 60.0) & near
            if mask.sum() < 20:
                continue
            cover = F.adaptive_avg_pool2d(mask.float().view(1, 1, height, width),
                                          (grid, grid)).flatten()
            take = (cover >= MIN_COVER) & (cover > best)
            labels[bi][take] = oi
            best[take] = cover[take]
    return labels

from hyperparameter import S2NetHyperparameters
from s2net_cls import S2NetCore
from loss_function import phase_locking_value, signal_synchrony


def adjusted_rand_index(a, b):
    """ARI between two label vectors, without sklearn."""
    a = torch.unique(a, return_inverse=True)[1]
    b = torch.unique(b, return_inverse=True)[1]
    table = torch.zeros(int(a.max()) + 1, int(b.max()) + 1)
    table.index_put_((a, b), torch.ones(a.numel()), accumulate=True)
    comb = lambda x: (x * (x - 1) / 2).sum()
    index = comb(table)
    expected = comb(table.sum(1)) * comb(table.sum(0)) / max(comb(torch.tensor([a.numel()])), 1e-12)
    maximum = (comb(table.sum(1)) + comb(table.sum(0))) / 2
    return float((index - expected) / (maximum - expected).clamp_min(1e-12))


def kmeans(x, k, iters=30):
    """Spherical k-means with farthest-point seeds, on the rows of x."""
    z = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    centres = z[(z @ z.mean(0, keepdim=True).t()).squeeze(1).argmin()].unsqueeze(0)
    while centres.size(0) < k:
        centres = torch.cat([centres, z[(z @ centres.t()).max(1).values.argmin()].unsqueeze(0)])
    for _ in range(iters):
        assign = (z @ centres.t()).argmax(1)
        for j in range(k):
            members = assign == j
            if members.any():
                centres[j] = torch.nn.functional.normalize(z[members].mean(0), dim=0)
    return (z @ centres.t()).argmax(1)


def spectral_cluster(affinity, k, jitter=1e-6):
    """Normalized-cut spectral clustering. The jitter is not cosmetic: without it
    eigh fails to converge on the degenerate affinities this model produces."""
    affinity = affinity + jitter * torch.eye(affinity.size(0), dtype=affinity.dtype)
    degree = affinity.sum(1).clamp_min(1e-8).rsqrt()
    laplacian = degree.unsqueeze(1) * affinity * degree.unsqueeze(0)
    try:
        _, vectors = torch.linalg.eigh(laplacian)
    except Exception:
        _, vectors = torch.linalg.eigh(laplacian.double())
        vectors = vectors.float()
    embedding = vectors[:, -k:]
    return kmeans(embedding, k)


def foreground_iou(pred, truth):
    """Mean over true objects of the best IoU against any predicted group."""
    background = int(torch.bincount(truth).argmax())
    scores = []
    for gid in truth.unique().tolist():
        if gid == background:
            continue
        gm = truth == gid
        scores.append(max(
            float((gm & (pred == pid)).sum()) / max(float((gm | (pred == pid)).sum()), 1)
            for pid in pred.unique().tolist()
        ))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-seq-path", required=True)
    parser.add_argument("--masks", required=True, help="blob supplying the image name list")
    parser.add_argument("--image-dir", required=True, help="CLEVR train images")
    parser.add_argument("--scenes", required=True, help="CLEVR_train_scenes.json")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--skip", type=int, default=200, help="images to skip, to score held-out ones")
    parser.add_argument("--num-regions", type=int, default=256)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--num-time-steps", type=int, default=256)
    parser.add_argument("--settle", type=int, default=64)
    parser.add_argument(
        "--graph-top-k",
        type=int,
        default=8,
        help=(
            "Must match training. top_k selects edges rather than parameterising "
            "them, so a mismatch loads without error and silently evaluates a "
            "different coupling graph than the one that was trained."
        ),
    )
    parser.add_argument("--freq-gain", type=float, default=2.0)
    parser.add_argument("--readout-slots", type=int, default=0)
    parser.add_argument("--readout-source", choices=["phase", "signal"], default="signal")
    parser.add_argument("--readout-temperature", type=float, default=0.05)
    parser.add_argument("--fixed-k", type=int, nargs="*", default=[3, 4, 6, 8])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    masks_blob = torch.load(args.masks)
    names = masks_blob["names"][args.skip:args.skip + args.num_images]
    scenes = json.load(open(args.scenes))["scenes"]
    objects_by_name = {s["image_filename"]: s["objects"]
                       for s in scenes if s["image_filename"] in set(names)}
    labels = build_masks(names, objects_by_name, args.image_dir, args.grid)

    gamma = torch.load(args.gamma_seq_path, map_location="cpu").float()
    gamma = gamma[args.skip:args.skip + args.num_images].to(args.device)

    hp = S2NetHyperparameters(
        num_feature_maps=gamma.size(1), num_regions=args.num_regions, sc=None,
        gamma_drive_mode="static", num_time_steps=args.num_time_steps,
        theta_init="gamma", gamma_phase_mode="standardize_tanh", osc_dim=4,
        freq_gain=args.freq_gain,
        graph_mode="learned", graph_top_k=args.graph_top_k, k=256.,
        graph_spatial_decay=0.861,
        low_n=-4., high_n=0., membrane_vth=0.06, membrane_low_m=-4., membrane_high_m=0.,
        gate_mode="phase_mean", spike_classify_method="spatial_components",
        spike_spatial_grid_size=(args.grid, args.grid),
        readout_slots=args.readout_slots, readout_source=args.readout_source,
        readout_temperature=args.readout_temperature,
        readout_signal_dim=(args.num_time_steps - args.settle
                            if args.readout_source == "signal" else None),
    ).validate()
    core = S2NetCore(hp, device=args.device).to(args.device)
    core.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    core.eval()

    plv, spike_sync, trains = [], [], []
    with torch.no_grad():
        for start in range(0, gamma.size(0), 25):
            _, _, membrane, theta = core(gamma[start:start + 25],
                                         return_core_out=True, return_theta=True)
            spikes = (membrane > torch.quantile(
                membrane[:, :, args.settle:], 0.50, dim=2, keepdim=True)).float()
            plv.append(phase_locking_value(theta, settle=args.settle).cpu())
            spike_sync.append(signal_synchrony(spikes, settle=args.settle).clamp_min(0).cpu())
            trains.append(spikes[:, :, args.settle:].cpu())
    plv, spike_sync, trains = torch.cat(plv), torch.cat(spike_sync), torch.cat(trains)

    readout_labels = None
    if core.cluster_readout is not None:
        with torch.no_grad():
            readout_labels = core.cluster_readout(
                signal=trains.to(args.device), settle=0
            ).argmax(-1).cpu() if args.readout_source == "signal" else None

    def truth(i):
        l = labels[i]
        if l.max() < 1:
            return None
        g = l.clone(); g[g < 0] = g.max() + 1
        return torch.unique(g, return_inverse=True)[1]

    def report(name, pred_of):
        aris, ious = [], []
        for i in range(len(names)):
            g = truth(i)
            if g is None:
                continue
            p = pred_of(i)
            aris.append(adjusted_rand_index(p, g))
            ious.extend(foreground_iou(p, g))
        print("%-46s %-8.4f %.4f" % (name, sum(aris) / len(aris), sum(ious) / len(ious)))

    counts = [int(truth(i).max()) + 1 for i in range(len(names)) if truth(i) is not None]
    print("checkpoint %s  (graph_top_k=%d)" % (Path(args.checkpoint), args.graph_top_k))
    print("oracle cluster count: mean %.1f, range %d-%d\n"
          % (sum(counts) / len(counts), min(counts), max(counts)))
    print("%-46s %-8s %s" % ("READOUT", "ARI", "fgIoU"))
    print("-" * 66)
    report("phases -> PLV -> spectral, oracle k",
           lambda i: spectral_cluster(plv[i], int(truth(i).max()) + 1))
    report("spikes -> correlation -> spectral, oracle k",
           lambda i: spectral_cluster(spike_sync[i], int(truth(i).max()) + 1))
    for k in args.fixed_k:
        report("phases -> spectral, k=%d" % k, lambda i, k=k: spectral_cluster(plv[i], k))
    for k in args.fixed_k:
        report("spikes -> k-means on trains, k=%d" % k, lambda i, k=k: kmeans(trains[i], k))
    if readout_labels is not None:
        report("spikes -> trained readout, k=%d" % args.readout_slots,
               lambda i: readout_labels[i])


if __name__ == "__main__":
    main()
