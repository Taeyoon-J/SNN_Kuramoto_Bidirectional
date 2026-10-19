"""Score a trained core under the evaluation contract.

The final number comes from masks a classifier builds out of spikes, not from
clustering theta. Phase readouts are computed too, clearly labelled, because the
contract keeps them as diagnostics.

Nothing here uses the true object count, the true masks, or anything else from
the ground truth to choose a threshold or a group count: the classifier's two
thresholds are arguments, chosen on validation.
"""
import argparse, json, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "snn_kuramoto_bidirectional"))
from evaluation import evaluate_patch_masks, spatial_components_to_patch_labels
from spike_classifier import spike_synchrony_components, spike_spatial_components
from hyperparameter import S2NetHyperparameters
from s2net_cls import S2NetCore
from loss_function import phase_locking_value


def _kmeans(x, k, iters=30):
    """Spherical k-means with farthest-point seeds, copied from
    training/evaluate_binding.py so the peer's spectral readout reproduces
    exactly, without importing that module's side effects."""
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


def _spectral_labels(affinity, count, grid, background_count=1):
    """Spectral clustering of one image's affinity into `count` groups.

    Ported from the peer branch (SW_0027). Normalised affinity, top eigenvectors,
    spherical k-means, and the largest cluster relabelled as background so the
    output matches what the connected-component readout produces.
    """
    nodes = affinity.shape[0]
    a = affinity + 1e-6 * torch.eye(nodes, dtype=affinity.dtype, device=affinity.device)
    inv_deg = a.sum(dim=1).clamp_min(1e-8).rsqrt()
    operator = inv_deg[:, None] * a * inv_deg[None, :]
    try:
        _, vectors = torch.linalg.eigh(operator)
    except RuntimeError:
        _, vectors = torch.linalg.eigh(operator.double())
        vectors = vectors.float()
    raw = _kmeans(vectors[:, -count:], count)
    sizes = torch.bincount(raw, minlength=count)
    # The n largest clusters are background. n=1 is the peer's rule.
    n_bg = max(1, min(int(background_count), count - 1))
    background = set(torch.argsort(sizes, descending=True)[:n_bg].tolist())
    labels = torch.zeros_like(raw, dtype=torch.int64)
    next_id = 1
    for group in range(count):
        if group not in background:
            labels[raw == group] = next_id
            next_id += 1
    return labels.reshape(grid, grid)


def _spike_affinity(spikes, components, args):
    """The same affinity the classifier would build, so a spatial kernel can be
    applied to it here without changing the classifier."""
    def correlate(x):
        x = x[:, :, int(args.settle):].float()
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return (x @ x.transpose(-1, -2)).clamp(-1.0, 1.0)
    if components is None:
        return correlate(spikes)
    per = [correlate(components[:, d]) for d in range(components.size(1))]
    return torch.stack(per).clamp_min(0.0).prod(dim=0)


def _checkpoint_channels(path, device):
    """Gamma widths this checkpoint was trained with: (total, graph-only extras).

    The graph projection carries every channel; the phase projection carries the
    appearance channels alone. Their difference is graph_extra_channels, so a core
    built from these two numbers always matches the weights being loaded. Equal
    widths give 0 extras, which is the single-width path.
    """
    state = torch.load(path, map_location=device)
    graph = state.get("graph_generator.projection.weight")
    phase = state.get("gamma_channel_proj.weight")
    total = int(graph.shape[1]) if graph is not None else (
        int(phase.shape[1]) if phase is not None else 8)
    extra = 0
    if graph is not None and phase is not None:
        extra = int(graph.shape[1]) - int(phase.shape[1])
        if extra < 0:
            raise ValueError(
                f"{path}: phase projection is wider than the graph projection "
                f"({phase.shape[1]} > {graph.shape[1]}), which no configuration builds."
            )
    return total, extra


def build_core(args, device):
    torch.manual_seed(0)
    channels, graph_extra = _checkpoint_channels(args.checkpoint, device)
    hp = S2NetHyperparameters(
        num_feature_maps=channels, graph_extra_channels=graph_extra,
        num_regions=args.num_regions, sc=None,
        gamma_drive_mode="static", num_time_steps=args.num_time_steps,
        theta_init="gamma", gamma_phase_mode="standardize_tanh", osc_dim=args.osc_dim,
        freq_gain=args.freq_gain, graph_mode="learned", graph_top_k=args.graph_top_k,
        k=float(args.num_regions), graph_spatial_decay=args.graph_spatial_decay,
        low_n=-4., high_n=0., membrane_vth=0.06, membrane_low_m=-4., membrane_high_m=0.,
        gate_mode=args.gate_mode, spike_per_component=args.spike_per_component,
        dendrite_per_region=args.dendrite_per_region,
        membrane_threshold_mode=args.membrane_threshold_mode,
        membrane_threshold_k=args.membrane_threshold_k,
        spike_pulse_gain=args.spike_pulse_gain, center_pulse=not args.no_center_pulse,
        geodesic_steps=args.geodesic_steps, geodesic_radius=args.geodesic_radius,
        geodesic_chunk=args.geodesic_chunk,
        coupling_balance=args.coupling_balance,
        geodesic_contrast=args.geodesic_contrast,
        geodesic_power=args.geodesic_power,
        geodesic_base=args.geodesic_base,
        spike_classify_method="spatial_components",
        spike_spatial_grid_size=(args.grid, args.grid),
    ).validate()
    core = S2NetCore(hp, device=device).to(device)
    core.load_state_dict(torch.load(args.checkpoint, map_location=device))
    core.gamma_channels = channels
    core.gamma_graph_extra = graph_extra
    core.eval()
    return core


def _merge_groups(groups, affinity, threshold, max_rounds=20):
    """Average-linkage agglomeration of the groups the synchrony readout returned.

    PV2_0105's oracle showed that merging the right pairs is worth more than any
    dial on the first stage, so this is a second stage rather than a retuning of
    the first. It uses the same affinity the first stage used and never sees the
    truth: the pair with the highest mean inter-group affinity is merged while
    that mean is above `threshold`.

    Components come back as tuples of node indices, not tensors, so the blocks are
    taken with index_select rather than A[gi][:, gj].
    """
    if affinity is None or threshold is None:
        return groups
    import torch as _t
    out = []
    for b, per_image in enumerate(groups):
        current = [list(g) for g in per_image]
        A = affinity[b]
        for _ in range(int(max_rounds)):
            n = len(current)
            if n < 2:
                break
            best, pair = None, None
            for i in range(n):
                gi = _t.as_tensor(current[i], device=A.device, dtype=_t.long)
                for j in range(i + 1, n):
                    gj = _t.as_tensor(current[j], device=A.device, dtype=_t.long)
                    if gi.numel() == 0 or gj.numel() == 0:
                        continue
                    score = float(A.index_select(0, gi).index_select(1, gj).mean())
                    if best is None or score > best:
                        best, pair = score, (i, j)
            if best is None or best <= float(threshold):
                break
            i, j = pair
            current[i] = current[i] + current[j]
            current.pop(j)
        out.append([tuple(sorted(g)) for g in current])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gamma-seq-path", required=True)
    ap.add_argument("--targets", required=True, help="targets_v1.pt from make_targets.py")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--limit", type=int, default=None, help="score the first N of the split")
    ap.add_argument("--num-regions", type=int, default=256)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--osc-dim", type=int, default=4)
    ap.add_argument("--graph-top-k", type=int, default=32)
    ap.add_argument("--graph-spatial-decay", type=float, default=0.55)
    ap.add_argument("--freq-gain", type=float, default=2.0)
    ap.add_argument("--gate-mode", default="raw")
    ap.add_argument("--spike-per-component", action="store_true")
    ap.add_argument("--membrane-threshold-mode", default="absolute",
                    choices=["absolute", "population"],
                    help="where the spiking threshold sits. 'population' places it "
                         "membrane-threshold-k standard deviations above the batch's "
                         "mean membrane, as one global scalar, so the membrane's "
                         "absolute scale stops deciding whether anything fires at "
                         "all. With the fixed v_th, four seeds of one configuration "
                         "gave membrane means +1.14, +2.64, +0.26 and -0.65, the "
                         "last never reaching 0.06 and firing nothing.")
    ap.add_argument("--membrane-threshold-k", type=float, default=0.5)
    ap.add_argument("--dendrite-per-region", action="store_true",
                    help="must match training. Gives each oscillator its own map "
                         "into its dendritic branches; the checkpoint will not load "
                         "without it.")
    ap.add_argument("--geodesic-steps", type=int, default=0, help="must match training")
    ap.add_argument("--coupling-balance", type=float, default=0.0,
                    help="must match training")
    ap.add_argument("--geodesic-chunk", type=int, default=0,
                    help="rows of the min-plus relaxation formed at once; "
                         "bit-identical, only the peak memory changes")
    ap.add_argument("--geodesic-radius", type=float, default=1.5, help="must match training")
    ap.add_argument("--geodesic-contrast", type=float, default=2.0, help="must match training")
    ap.add_argument("--spike-pulse-gain", type=float, default=0.0)
    ap.add_argument("--no-center-pulse", action="store_true")
    ap.add_argument("--num-time-steps", type=int, default=256)
    ap.add_argument("--settle", type=int, default=64)
    ap.add_argument("--foreground-threshold", type=float, default=0.15)
    ap.add_argument("--synchrony-threshold", type=float, default=0.5)
    ap.add_argument("--min-group-size", type=int, default=2)
    ap.add_argument("--membrane-threshold", type=float, default=0.06,
                    help="used when --use-model-spikes is off; the core's own v_th")
    ap.add_argument("--target-foreground", type=float, default=None,
                    help=("calibrate the synchrony threshold per image so the groups "
                          "cover this fraction of patches; chosen on validation"))
    ap.add_argument("--background", default="largest_component",
                    choices=["largest_component", "activity", "hybrid"])
    ap.add_argument("--merge-threshold", type=float, default=None,
                    help="Second readout stage: agglomerate groups whose mean "
                         "inter-group affinity exceeds this. The affinity is spike "
                         "correlation TIMES the spatial Gaussian, so its real scale is "
                         "~1e-4, not ~1e-1 -- a threshold of 0.05 merges nothing.")
    ap.add_argument("--merge-rounds", type=int, default=20,
                    help="Maximum agglomeration rounds per image.")
    ap.add_argument("--synchrony-quantile", type=float, default=0.35,
                    help="hybrid background: fraction of units the synchrony ranking admits")
    ap.add_argument("--membrane-quantile", type=float, default=None,
                    help="threshold the membrane at a per-image GLOBAL quantile "
                         "instead of the fixed v_th, so the firing rate is set by "
                         "the quantile and the membrane's absolute scale stops "
                         "mattering. v_th is a constant while nothing constrains "
                         "that scale: across seeds of one configuration the "
                         "membrane sits at mean +1.14 (91.7%% above v_th, rate "
                         "0.300), +2.64 (100%% above, rate 0.468), +0.26 (72.5%%) "
                         "and -0.65 with a maximum of -0.046, which never reaches "
                         "0.06 at all and fires nothing. One GLOBAL quantile per "
                         "image, not one per unit: a per-unit quantile would force "
                         "every unit to the same rate and destroy the between-unit "
                         "differences synchrony is made of, which is a mistake "
                         "already made here with a per-unit median.")
    ap.add_argument("--membrane-spikes", action="store_true",
                    help=(
                        "Binarise the membrane at --membrane-threshold instead of "
                        "reading the layer's own spikes. Off by default: an earlier "
                        "version thresholded at the median, which makes every unit "
                        "fire exactly half the time and left foreground IoU at "
                        "chance."
                    ))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--spectral-k", type=int, default=None,
                    help="group by spectral clustering into this many clusters "
                         "instead of connected components, ported from the peer "
                         "branch's SW_0024/0027 where it raised their FG-ARI 0.357 "
                         "to 0.394. The largest cluster is background, as there. "
                         "Unlike connected components it emits k groups whatever the "
                         "affinity contains, which is also what makes the peer's "
                         "SW_0033 spatial-only control runnable: that control "
                         "degenerates to zero groups under a connected-component "
                         "readout, leaving open how much of this branch's score is "
                         "grid structure rather than binding.")
    ap.add_argument("--spectral-oracle-k", action="store_true",
                    help="set the spectral cluster count per image from the TRUE "
                         "object count plus one for background. ORACLE: it uses the "
                         "ground-truth masks, so it is a diagnostic ceiling reported "
                         "separately and never a score. It isolates what count "
                         "inference costs: fg_ari is computed over foreground patches "
                         "only, so an oracle foreground mask leaves it untouched (the "
                         "peer measured exactly that, 0.357 to 0.357), and the count "
                         "is the one oracle that can move it.")
    ap.add_argument("--permute-spatial", action="store_true",
                    help="randomly permute which patch each row of the spatial "
                         "kernel corresponds to, keeping the kernel's values. "
                         "Ported from the peer branch's SW0047: correctly aligned "
                         "they scored .5993/.6085/.4637 and permuted .3192/.4945/"
                         ".2543, so the kernel's benefit depends on real grid "
                         "geometry rather than on smoothing alone. CONTROL: the drop "
                         "this causes is the part of a spatial-kernel gain that is "
                         "actually positional.")
    ap.add_argument("--spectral-background-count", type=int, default=1,
                    help="how many of the largest spectral clusters become "
                         "background. The peer's rule keeps one, which leaves k-1 "
                         "clusters as foreground: at k=10 that predicted 0.398 of "
                         "patches foreground against a target of 0.098, and their "
                         "SW_0031 hit the same wall at 0.659 against 0.217. "
                         "Foreground extent is the IoU bottleneck for this readout "
                         "while the grouping itself is strong -- matched-object IoU "
                         "went 0.490 to 0.611 on the pilot.")
    ap.add_argument("--spatial-sigma", type=float, default=None,
                    help="multiply the synchrony affinity by exp(-d^2 / 2 sigma^2) on "
                         "the patch grid, ported from the peer branch's SW_0028, "
                         "where it raised their FG-ARI 0.3940 to 0.4946.")
    ap.add_argument("--readout", default="spikes",
                    choices=["spikes", "plv", "spatial_only"],
                    help="what the classifier groups on. 'plv' feeds it the phase "
                         "locking matrix instead of spike synchrony, through the same "
                         "grouping rule and the same evaluation, so nothing else "
                         "differs. DIAGNOSTIC: a phase readout is never the success "
                         "score; it measures what the spiking path loses. "
                         "'spatial_only' is the control the peer branch's SW_0033 "
                         "ran: group on the spatial kernel ALONE, with no spike or "
                         "membrane information at all. On their data it scored "
                         "0.4913 against 0.4946 for membrane-times-spatial, and a "
                         "random permutation of the membrane affinity scored 0.4814 "
                         "-- so nearly all of that FG-ARI came from the spatial prior "
                         "rather than learned binding. Anything this control "
                         "reproduces is not evidence of binding by synchrony.")
    ap.add_argument("--oracle-graph", action="store_true",
                    help="couple only within a ground-truth patch group. ORACLE: it "
                         "uses the true masks, so the result is a diagnostic ceiling "
                         "and is never reported as a score.")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--expect-contract", default="1",
                    help="contract version the targets must declare. Stays 1 so every "
                         "existing command keeps the same protection; a different "
                         "contract, such as the MOVi-A pilot's movi-1, has to be named "
                         "here (PV2_0085)")
    ap.add_argument("--geodesic-base", type=float, default=1.0,
                    help="per-step toll in the geodesic step cost. 1.0 is the "
                         "original, where path cost is dominated by step count; "
                         "lowering it lets dissimilarity dominate (PV2_0078)")
    ap.add_argument("--geodesic-power", type=float, default=1.0,
                    help="exponent on the dissimilarity in the geodesic step cost. "
                         "1.0 is the original linear cost; above it the boundary "
                         "crossing is charged selectively rather than the whole cost "
                         "being scaled, which is what PV2_0074 could not do")
    ap.add_argument("--phase-gamma-path", default=None,
                    help="gamma for the phase path only; --gamma-seq-path then feeds "
                         "the coupling graph alone. Splits the two routes features "
                         "take into the dynamics (PV2_0066)")
    ap.add_argument("--save-predictions", default=None,
                    help="write the patch labels and image names to this npz, for "
                         "comparing one readout's partition against another's "
                         "without ground truth (PV2_0056)")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    start, end = manifest["ranges"][args.split]
    if args.limit is not None:
        end = min(end, start + args.limit)
    blob = torch.load(args.targets)
    version = blob.get("contract_version", 1)
    if str(version) != str(args.expect_contract):
        raise ValueError(
            f"targets declare contract version {version!r} but --expect-contract is "
            f"{args.expect_contract!r}. The default is 1, so a different contract has "
            f"to be named explicitly.")
    target = blob["patch_labels"][start:end]
    names = blob["names"][start:end]
    if names != manifest["splits"][args.split][: len(names)]:
        raise ValueError("target order does not match the split manifest.")
    gamma = torch.load(args.gamma_seq_path, map_location="cpu").float()[start:end]
    phase_gamma = None
    if args.phase_gamma_path:
        phase_gamma = torch.load(args.phase_gamma_path,
                                 map_location="cpu").float()[start:end]
        if phase_gamma.shape[0] != gamma.shape[0]:
            raise ValueError(
                f"phase gamma has {phase_gamma.shape[0]} rows for {gamma.shape[0]} "
                "graph rows; the two tensors must cover the same images in order.")
        if phase_gamma.shape[2] != gamma.shape[2]:
            raise ValueError(
                f"phase gamma has {phase_gamma.shape[2]} regions, graph gamma "
                f"{gamma.shape[2]}.")
        print(f"phase path reads {args.phase_gamma_path} "
              f"({tuple(phase_gamma.shape)}); the graph reads {args.gamma_seq_path}")
    if gamma.size(0) != target.size(0):
        raise ValueError(f"gamma has {gamma.size(0)} rows for {target.size(0)} targets.")
    print(f"{args.split} split: {gamma.size(0)} images, ids {start}..{end - 1}", flush=True)

    core = build_core(args, args.device)
    spikes_all, comp_all, plv_all, rates = [], [], [], []
    with torch.no_grad():
        for s in range(0, gamma.size(0), 25):
            # The oracle ceiling: couple only within a true group, built exactly
            # as training builds it. A model trained on the forced graph never
            # trained its graph generator, so scoring it on the learned graph
            # would measure nothing.
            if args.oracle_graph:
                rows = target[s:s + 25].reshape(target[s:s + 25].size(0), -1).long()
                same = (rows.unsqueeze(2) == rows.unsqueeze(1)).float()
                same = same * (1.0 - torch.eye(same.size(-1)).unsqueeze(0))
                core.forced_graph = (same / same.sum(-1, keepdim=True).clamp_min(1e-8)
                                     ).to(args.device)
            if phase_gamma is not None:
                # The core strips graph-only channels after building the graph, so
                # the same hook serves here: hand it the full-width graph tensor and
                # let it read the phase tensor for everything downstream.
                core.phase_gamma_override = phase_gamma[s:s + 25].to(args.device)
            _, model_spikes, membrane, theta = core(gamma[s:s + 25].to(args.device),
                                                    return_core_out=True, return_theta=True)
            # The model's own spikes, from its membrane threshold. Re-binarising
            # the membrane at its median destroyed the foreground signal: it makes
            # every unit fire exactly half the time by construction, so every patch
            # looks active and foreground IoU collapses to the fraction of the grid
            # that is genuinely foreground.
            if args.membrane_quantile is not None:
                # One threshold per image over all units and timesteps, so each
                # image fires at the chosen rate while the ordering between its
                # units is untouched.
                flat = membrane.reshape(membrane.size(0), -1)
                cut = torch.quantile(flat.float(), 1.0 - args.membrane_quantile,
                                     dim=1)
                spike = (membrane > cut.view(-1, *([1] * (membrane.dim() - 1)))).float()
            else:
                spike = ((membrane > args.membrane_threshold).float()
                         if args.membrane_spikes else model_spikes.float())
            spikes_all.append(spike.cpu())
            rates.append(float(spike[:, :, args.settle:].mean()))
            comp_all.append(core.last_component_spikes.cpu()
                            if core.last_component_spikes is not None else None)
            plv_all.append(phase_locking_value(theta, settle=args.settle, combine="product").cpu())
    spikes = torch.cat(spikes_all)
    components = torch.cat([c for c in comp_all]) if comp_all[0] is not None else None
    plv = torch.cat(plv_all)

    # Spatial kernel on the patch grid, shared by --spatial-sigma and the
    # spatial_only control.
    if args.spatial_sigma is not None or args.readout == "spatial_only":
        g = args.grid
        yy, xx = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
        pos = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1).float()
        d2 = (pos.unsqueeze(1) - pos.unsqueeze(0)).pow(2).sum(-1)
        sigma = float(args.spatial_sigma if args.spatial_sigma is not None else 1.5)
        kernel = torch.exp(-d2 / (2.0 * sigma * sigma))
        if args.permute_spatial:
            # Same kernel values, wrong patch correspondence. Symmetric, so the
            # same permutation is applied to rows and columns.
            perm = torch.randperm(kernel.size(0), generator=torch.Generator().manual_seed(0))
            kernel = kernel[perm][:, perm]
    if args.readout == "spatial_only":
        # No model information whatsoever: the affinity IS the kernel.
        affinity_in = kernel.unsqueeze(0).expand(spikes.size(0), -1, -1).clone()
        components = None
    elif args.readout == "plv":
        affinity_in = plv
    else:
        affinity_in = None
    if args.spatial_sigma is not None and args.readout != "spatial_only":
        base = plv if args.readout == "plv" else _spike_affinity(spikes, components, args)
        affinity_in = base * kernel.unsqueeze(0)
        components = None
    if args.spectral_k is not None:
        # A different grouping rule, so it bypasses the connected-component
        # classifier entirely and produces patch labels directly.
        base = (affinity_in if affinity_in is not None
                else _spike_affinity(spikes, components, args))
        base = base.abs()
        if args.spectral_oracle_k:
            # True object count per image, plus one cluster for background.
            per_image_k = [
                int(target[i].max().item()) + 1 for i in range(base.size(0))
            ]
        else:
            per_image_k = [int(args.spectral_k)] * base.size(0)
        labels = torch.stack([
            _spectral_labels(base[i].cpu(), max(2, per_image_k[i]), args.grid,
                             int(args.spectral_background_count))
            for i in range(base.size(0))
        ])
        if args.save_predictions:
            import numpy as _np
            _np.savez_compressed(
                args.save_predictions,
                patch_labels=labels.cpu().numpy(),
                names=_np.array(names[: labels.size(0)], dtype=object),
                readout=args.readout, grid=args.grid,
                spectral_k=(-1 if args.spectral_oracle_k else int(args.spectral_k)),
                spectral_background_count=int(args.spectral_background_count),
            )
            print("wrote predictions to", args.save_predictions)
        scored = evaluate_patch_masks(labels, target)
        counts = [int(labels[i].max()) for i in range(labels.size(0))]
        report = {
            "split": args.split, "images": int(gamma.size(0)),
            "oracle_graph": bool(args.oracle_graph),
            "membrane_quantile": args.membrane_quantile,
            "readout": args.readout, "spatial_sigma": args.spatial_sigma, "permute_spatial": bool(args.permute_spatial),
            "spectral_k": (None if args.spectral_oracle_k else int(args.spectral_k)),
            "spectral_oracle_k": bool(args.spectral_oracle_k),
            "spectral_background_count": int(args.spectral_background_count),
            # evaluate_patch_masks returns "mean", not "metrics" -- the key was
            # assumed rather than checked, and every spectral row died on
            # KeyError before any of them produced a number.
            "metrics": {k: float(v) for k, v in scored["mean"].items()},
            "valid_count": {k: int(v) for k, v in scored["valid_count"].items()},
            "diagnostics": {
                "spike_rate": float(sum(rates) / max(len(rates), 1)),
                "groups_per_image_mean": float(sum(counts) / max(len(counts), 1)),
                "images_with_no_group": int(sum(1 for c in counts if c == 0)),
                "plv_mean": float(plv.mean()),
                "predicted_foreground_fraction": float((labels > 0).float().mean()),
                "target_foreground_fraction": float((target > 0).float().mean()),
            },
        }
        for k, v in report["metrics"].items():
            print("%-26s %.4f   (n=%d)" % (k, v, report["valid_count"][k]))
        for k, v in report["diagnostics"].items():
            print("%-26s %.4f" % (k, v))
        if args.json_out:
            json.dump(report, open(args.json_out, "w"), indent=2)
        return
    groups = spike_synchrony_components(
        spikes, foreground_threshold=args.foreground_threshold,
        synchrony_threshold=args.synchrony_threshold,
        min_group_size=args.min_group_size, settle=args.settle, components=components,
        background=args.background, target_foreground=args.target_foreground,
        synchrony_quantile=args.synchrony_quantile,
        affinity=affinity_in,
    )
    if args.merge_threshold is not None:
        before = sum(len(g) for g in groups) / max(len(groups), 1)
        groups = _merge_groups(groups, affinity_in, args.merge_threshold, args.merge_rounds)
        after = sum(len(g) for g in groups) / max(len(groups), 1)
        print("merge stage: %.2f -> %.2f groups per image (threshold %.4f)"
              % (before, after, args.merge_threshold))
    prediction = spatial_components_to_patch_labels(groups, args.grid)
    if args.save_predictions:
        import numpy as _np
        _np.savez_compressed(
            args.save_predictions,
            patch_labels=prediction.cpu().numpy(),
            names=_np.array(names[: len(prediction)], dtype=object),
            readout=args.readout, grid=args.grid,
            synchrony_threshold=args.synchrony_threshold,
        )
        print("wrote predictions to", args.save_predictions)
    scored = evaluate_patch_masks(prediction, target)

    counts = [len(g) for g in groups]
    report = {
        "split": args.split, "images": int(gamma.size(0)),
        "oracle_graph": bool(args.oracle_graph), "membrane_quantile": args.membrane_quantile, "readout": args.readout, "spatial_sigma": args.spatial_sigma, "permute_spatial": bool(args.permute_spatial),
        "contract_version": args.expect_contract,
            "manifest_version": manifest.get("version", "unversioned"),
        "checkpoint": str(args.checkpoint),
        "classifier": "spike_synchrony_components",
        "background": args.background,
        "foreground_threshold": args.foreground_threshold,
        "synchrony_threshold": args.synchrony_threshold,
        "target_foreground": args.target_foreground,
        "min_group_size": args.min_group_size,
        "metrics": {k: float(v) for k, v in scored["mean"].items()},
        "valid_count": {k: int(v) for k, v in scored["valid_count"].items()},
        "diagnostics": {
            "spike_rate": sum(rates) / len(rates),
            "groups_per_image_mean": sum(counts) / len(counts),
            "images_with_no_group": sum(1 for c in counts if c == 0),
            "plv_mean": float(plv.mean()),
            "predicted_foreground_fraction": float((prediction != 0).float().mean()),
            "target_foreground_fraction": float((target != 0).float().mean()),
        },
    }
    print()
    print("%-26s %s" % ("SCORED FROM SPIKE MASKS", "mean"))
    print("-" * 40)
    for k, v in report["metrics"].items():
        print("%-26s %.4f   (n=%d)" % (k, v, report["valid_count"][k]))
    print()
    for k, v in report["diagnostics"].items():
        print("  %-34s %.4f" % (k, v))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.json_out, "w"), indent=1)
        print("\nwrote", args.json_out)


if __name__ == "__main__":
    main()
