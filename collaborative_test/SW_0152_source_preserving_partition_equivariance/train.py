"""Checkpointable SW0152 cross-view partition pilot (training only)."""
from __future__ import annotations

import hashlib, json, os, sys, uuid
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"), str(ROOT / "snn_kuramoto_bidirectional")]

import numpy as np
import torch

from collaborative_test.SW_0130_phase_state_integration import run as sw130
from collaborative_test.SW_0135_native32_spike_binding import foundation as sw135
from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint as durable
from collaborative_test.SW_0150_spike_spatial_mixture import preflight as sw150_preflight
from collaborative_test.SW_0150_spike_spatial_mixture.integration import forward_assignment
from collaborative_test.SW_0152_source_preserving_partition_equivariance import preflight, training

NAME = preflight.EXPERIMENT
ARMS = ("equivariance", "mass_control")
SEED, CORE_INIT_SEED, RESIDUAL_INIT_SEED = 1, 149001, 152001
BATCH, MICRO, UPDATES, STEPS, SETTLE, TAIL = 16, 4, 128, 1024, 512, 64
TRAIN = HERE / "training"
WORK = HERE / "work"
PREFLIGHT = HERE / "results_archive" / "preflight_seed1.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def implementation_fingerprint():
    result = dict(sw135.implementation_fingerprint())
    paths = [HERE / n for n in ("protocol.json", "deep_encoder.py", "view_augmentation.py",
             "partition_objective.py", "training.py", "preflight.py", "preflight_queue.py", "train.py")]
    paths += [ROOT / p for p in (
        "collaborative_test/SW_0150_spike_spatial_mixture/integration.py",
        "collaborative_test/SW_0150_spike_spatial_mixture/spatial_diffusion.py",
        "collaborative_test/SW_0150_spike_spatial_mixture/seed_selection.py",
        "collaborative_test/SW_0145_native32_boundary_credit_probe/run.py",
        "collaborative_test/SW_0136_native32_transfer/evaluate.py",
        "collaborative_test/SW_0148_checkpointable_boundary_q/checkpoint.py")]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("SW0152 training dependency is missing")
    result.update({p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
    return result


def implementation_sha():
    raw = json.dumps(implementation_fingerprint(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def source_contract():
    checkpoint, manifest_path, manifest, rows, ids, source_sha = sw130.source_contract(SEED)
    rows = np.asarray(rows, dtype=np.int64); ids = [int(x) for x in ids]
    if len(ids) != 4096 or len(set(ids)) != 4096 or rows.shape != (4096,) or len(np.unique(rows)) != 4096:
        raise ValueError("SW0152 requires the registered ordered 4096-image TRAIN source")
    return {"checkpoint": Path(checkpoint), "manifest_path": Path(manifest_path), "manifest": manifest,
            "rows": rows, "ids": ids, "source_sha256": source_sha}


def validate_preflight(path=PREFLIGHT):
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    src = source_contract(); assets = sw130.validate_rgb_assets()
    expected = {"checkpoint_sha256": src["source_sha256"], "manifest_sha256": sha(src["manifest_path"])}
    saved_parity, parity_sha = sw150_preflight.load_and_validate_parity(
        sw150_preflight.PARITY_PATH, source_sha=src["source_sha256"],
        source_manifest_sha=expected["manifest_sha256"], training_ids=src["ids"],
        asset_validation=assets)
    if not preflight_queue_valid(report, src, expected, assets):
        raise ValueError("SW0152 source/preflight provenance or selected-family evidence failed")
    report_parity = report.get("saved_source_parity", {})
    if (report_parity.get("path") != str(sw150_preflight.PARITY_PATH)
            or report_parity.get("sha256") != parity_sha
            or report_parity.get("status") != saved_parity.get("status")):
        raise ValueError("SW0152 preflight saved parity differs from current registered receipt")
    if report.get("implementation_fingerprint") != preflight.implementation_fingerprint():
        raise ValueError("SW0152 preflight implementation changed")
    delta = require_color_permutation_gate(report)
    return {"report": report, "report_sha256": sha(path), "source": src, "assets": assets,
            "permutation_delta": delta}


def require_color_permutation_gate(report):
    diag = report.get("within_color_permutation_diagnostic", {})
    try:
        delta = float(diag["within_color_permuted_alignment_js_mean"])-float(diag["baseline_alignment_js_mean"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("preflight lacks a finite color-permutation diagnostic") from exc
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("registered first64 within-color permutation admission did not pass")
    return delta


def preflight_queue_valid(report, src, expected, assets):
    # The CPU supervisor intentionally hides CUDA at import. Restore the
    # trainer's inherited mapping before the first CUDA runtime query.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    existed = "CUDA_VISIBLE_DEVICES" in os.environ
    try:
        from collaborative_test.SW_0152_source_preserving_partition_equivariance import preflight_queue
    finally:
        if existed: os.environ["CUDA_VISIBLE_DEVICES"] = visible
        else: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    return preflight_queue.validate_report(report, expected_ids=src["ids"], expected_source=expected,
        expected_assets=assets, expected_parity_sha=report.get("saved_source_parity", {}).get("sha256"),
        expected_fingerprint=preflight.implementation_fingerprint())


def _load_foundation(device):
    d = torch.device(device); index = d.index if d.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[index]):
        torch.random.default_generator.manual_seed(CORE_INIT_SEED); torch.cuda.manual_seed(CORE_INIT_SEED)
        return sw135.load_native32_foundation(SEED, device, verify_rgb_assets=True)


def _read_batch(cache, rows, start, count, device):
    values = np.asarray(cache[rows[start:start + count]]).copy()
    if values.dtype != np.uint8 or values.shape != (count, 128, 128, 3):
        raise ValueError("registered source RGB cache row schema changed")
    return torch.from_numpy(values).to(device)


def _source_masses(foundation, cache, rows, count, device):
    masses = []
    with torch.no_grad():
        for start in range(0, count, MICRO):
            rgb = _read_batch(cache, rows, start, MICRO, device)
            p = forward_assignment(foundation, rgb, live_tail_steps=TAIL)["p"]
            masses.append(preflight.padded_mass(p).detach().cpu())
    result = torch.cat(masses, dim=0).contiguous()
    return result, _tensor_sha(result)


def _source_bindings(f, admission, arm, mass_sha, parity):
    src = admission["source"]
    return {"experiment": NAME, "arm": arm, "seed": SEED,
        "source_checkpoint_sha256": src["source_sha256"],
        "source_manifest_sha256": sha(src["manifest_path"]),
        "source_training_ids_sha256": ids_sha(src["ids"]),
        "ordered_train_ids_sha256": ids_sha(src["ids"][:BATCH*UPDATES]),
        "rgb_asset_validation": admission["assets"],
        "preflight_sha256": admission["report_sha256"],
        "preflight_implementation_fingerprint": admission["report"]["implementation_fingerprint"],
        "source_mass_sha256": mass_sha, "initial_source_parity": parity,
        "initial_parameter_sha256": {k: training.parameter_hash(v)
                                      for k, v in training.trainable_families(f, arm).items()},
        "batch": BATCH, "microbatch": MICRO, "updates": UPDATES,
        "steps": STEPS, "settle": SETTLE, "live_tail": TAIL,
        "optimizer_rates": {"spatial_residual": 3e-6, "graph": 3e-5,
            "native_core": 3e-5, "integration_scalars": 1e-3},
        "implementation_fingerprint": implementation_fingerprint(),
        "implementation_sha256": implementation_sha(), "ground_truth_used": False}


def _save_boundary(path, completed, foundation, optimizer, history, bindings, runtime, device):
    return durable.save_boundary(path, completed_updates=completed, wrapped=foundation.wrapped,
        encoder=foundation.encoder, optimizer=optimizer, history=history, bindings=bindings,
        runtime=runtime, device=device)


def _finalize(arm, *, foundation, optimizer, family_map, history, bindings,
              output_dir, work_dir, initial_hashes, frozen_base_sha):
    if len(history) != UPDATES or output_dir.exists():
        raise ValueError("cannot finalize an incomplete or pre-existing SW0152 output")
    base_after = training.state_sha(foundation.encoder.base_encoder)
    if base_after != frozen_base_sha:
        raise AssertionError("frozen source RGB encoder changed")
    final_hashes = {name: training.parameter_hash(params) for name, params in family_map.items()}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        checkpoint, optim, hist = staging / "checkpoint.pt", staging / "optimizer.pt", staging / "history.json"
        for path, payload in ((checkpoint, {"wrapped": foundation.wrapped.state_dict(),
                                            "encoder": foundation.encoder.state_dict()}),
                              (optim, optimizer.state_dict())):
            with path.open("xb") as stream:
                torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        with hist.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(history, stream, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        manifest = {"status": "training_complete", "experiment": NAME, "arm": arm, "seed": SEED,
            "updates": UPDATES, "completed_updates": UPDATES, "logical_batch": BATCH,
            "microbatch": MICRO, "exposures": BATCH*UPDATES,
            "train_ids": [x for row in history for x in row["image_ids"]],
            "bindings": bindings, "initial_parameter_sha256": initial_hashes,
            "source_mass_sha256": bindings["source_mass_sha256"],
            "final_parameter_sha256": final_hashes,
            "frozen_base_encoder_sha256_before_after": frozen_base_sha,
            "checkpoint_sha256": sha(checkpoint), "optimizer_sha256": sha(optim),
            "history_sha256": sha(hist), "durable_boundary_sha256": sha(Path(work_dir)/"boundary.ptx"),
            "optimizer_step_summary": {name: sw150_preflight._optimizer_steps(optimizer, params)
                                        for name, params in family_map.items()},
            "optimizer_schema": durable.optimizer_schema(optimizer),
            "implementation_fingerprint": bindings["implementation_fingerprint"],
            "implementation_sha256": bindings["implementation_sha256"], "ground_truth_used": False}
        with (staging/"manifest.json").open("x",encoding="utf-8",newline="\n") as stream:
            json.dump(manifest,stream,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
        os.rename(staging, output_dir)
        return manifest
    except BaseException:
        raise


def train_arm(arm, *, device="cuda:0", preflight_path=PREFLIGHT, work_root=WORK, output_root=TRAIN):
    if arm not in ARMS: raise ValueError("unregistered SW0152 arm")
    admission = validate_preflight(preflight_path)
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SW0152 main training requires assigned CUDA")
    torch.cuda.set_device(dev)
    src = admission["source"]
    cache = np.load(sw130.TRAIN_RGB, mmap_mode="r", allow_pickle=False)
    if cache.dtype != np.uint8 or cache.shape != (70000,128,128,3):
        raise ValueError("registered TRAIN RGB cache schema changed")
    foundation = _load_foundation(dev)
    if (foundation.provenance.get("source_core_sha256") != src["source_sha256"]
            or foundation.provenance.get("source_manifest_sha256") != sha(src["manifest_path"])
            or foundation.provenance.get("rgb_asset_validation") != admission["assets"]
            or list(map(int,foundation.image_ids)) != src["ids"]
            or not np.array_equal(foundation.pool_indices,src["rows"])):
        raise ValueError("fresh foundation differs from registered SW0152 source order/assets")
    # Source areas are calculated on the untouched source model before adding the zero-head encoder.
    masses, mass_sha = _source_masses(foundation,cache,src["rows"],BATCH*UPDATES,dev)
    source_rgb = _read_batch(cache,src["rows"],0,MICRO,dev)
    with torch.no_grad(): source_ref = forward_assignment(foundation,source_rgb,live_tail_steps=TAIL)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(RESIDUAL_INIT_SEED)
        training.install_deep_encoder(foundation)
    foundation.encoder.to(dev)
    student_ref = forward_assignment(foundation,source_rgb,live_tail_steps=TAIL)
    parity_names = ("q","qcc","p","gamma")
    parity_exact = all(torch.equal(source_ref[k].detach(),student_ref[k].detach()) for k in parity_names)
    for key in ("theta","membrane","spikes","component_membrane","component_spikes","component_gates"):
        parity_exact = parity_exact and torch.equal(source_ref["trace"][key],student_ref["trace"][key])
    if not parity_exact: raise AssertionError("zero-head SW0152 source parity failed")
    parity = {"exact":True,"source_mass_sha256":mass_sha,"checked_fields":[*parity_names,
        "theta","membrane","spikes","component_membrane","component_spikes","component_gates"]}
    del source_ref,student_ref,source_rgb
    family_map = training.configure_trainability(foundation,arm)
    optimizer = training.make_optimizer(family_map)
    initial_hashes = {name:training.parameter_hash(params) for name,params in family_map.items()}
    base_sha = training.state_sha(foundation.encoder.base_encoder)
    bindings = _source_bindings(foundation,admission,arm,mass_sha,parity)
    runtime = durable.runtime_signature(dev)
    work_dir=Path(work_root)/f"{arm}_seed1"/"checkpoints"
    out=Path(output_root)/f"{arm}_seed1"
    if out.exists(): raise FileExistsError("preserving existing completed training output")
    ids=src["ids"][:BATCH*UPDATES]; rows=src["rows"][:BATCH*UPDATES]
    if work_dir.exists() and (work_dir/"boundary.ptx").exists():
        meta=durable.inspect_committed_metadata(work_dir)
        if meta.get("bindings")!=bindings or meta.get("runtime")!=runtime:
            raise ValueError("SW0152 checkpoint lineage/runtime mismatch")
        loaded=durable.load_latest(work_dir,wrapped=foundation.wrapped,encoder=foundation.encoder,
            optimizer=optimizer,expected_bindings=bindings,expected_runtime=runtime,device=dev,max_updates=UPDATES)
        completed,history=loaded["completed_updates"],loaded["history"]
    else:
        if work_dir.exists() and not durable.is_only_uncommitted_temp(work_dir):
            raise ValueError("preserving unknown SW0152 work artifacts")
        work_dir.mkdir(parents=True,exist_ok=True); completed,history=0,[]
        _save_boundary(work_dir,0,foundation,optimizer,history,bindings,runtime,dev)
    if completed==UPDATES:
        return _finalize(arm,foundation=foundation,optimizer=optimizer,family_map=family_map,
            history=history,bindings=bindings,output_dir=out,work_dir=work_dir,
            initial_hashes=initial_hashes,frozen_base_sha=base_sha)
    for update in range(completed,UPDATES):
        batch_ids=ids[update*BATCH:(update+1)*BATCH]
        augmentation_records=[]
        def compute_micro(micro):
            start=update*BATCH+micro*MICRO
            clean_cpu=_read_batch(cache,rows,start,MICRO,"cpu")
            clean,aug,records=training.make_paired_views(clean_cpu,batch_ids[micro*MICRO:(micro+1)*MICRO],source_seed=SEED)
            augmentation_records.extend(records)
            canonical,transformed=training.paired_assignments(foundation,clean.to(dev),aug.to(dev),live_tail_steps=TAIL)
            result=training.pair_loss(canonical,transformed,masses[start:start+MICRO].to(dev),arm=arm)
            return result
        row=training.logical_update(optimizer=optimizer,family_map=family_map,compute_micro=compute_micro,wrapped=foundation.wrapped)
        row.update(update=update+1,image_ids=batch_ids)
        if [r.get("image_id") for r in augmentation_records] != batch_ids:
            raise AssertionError("augmentation metadata does not preserve registered update order")
        row["augmentation_records"]=augmentation_records
        history.append(row)
        _save_boundary(work_dir,update+1,foundation,optimizer,history,bindings,runtime,dev)
        if (update+1)%16==0:
            print(json.dumps({"arm":arm,"completed_updates":update+1,"total_updates":UPDATES},allow_nan=False),flush=True)
    return _finalize(arm,foundation=foundation,optimizer=optimizer,family_map=family_map,
        history=history,bindings=bindings,output_dir=out,work_dir=work_dir,
        initial_hashes=initial_hashes,frozen_base_sha=base_sha)


def validate_training(arm, *, output_root=TRAIN, work_root=WORK):
    if arm not in ARMS: raise ValueError("unregistered SW0152 arm")
    admission=validate_preflight()
    out=Path(output_root)/f"{arm}_seed1"; manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    paths={name:out/name for name in ("checkpoint.pt","optimizer.pt","history.json")}
    history=json.loads(paths["history.json"].read_text(encoding="utf-8"))
    family_names={"spatial_residual","graph","native_core","integration_scalars"}
    expected_ids=admission["source"]["ids"][:BATCH*UPDATES]
    step_summary=manifest.get("optimizer_step_summary",{})
    schema=manifest.get("optimizer_schema",[])
    boundary=Path(work_root)/f"{arm}_seed1"/"checkpoints"/"boundary.ptx"
    if (manifest.get("status")!="training_complete" or manifest.get("experiment")!=NAME
            or manifest.get("arm")!=arm or manifest.get("completed_updates")!=UPDATES
            or manifest.get("updates")!=UPDATES or manifest.get("logical_batch")!=BATCH
            or manifest.get("microbatch")!=MICRO or manifest.get("exposures")!=BATCH*UPDATES
            or manifest.get("train_ids")!=expected_ids or len(history)!=UPDATES
            or manifest.get("bindings",{}).get("preflight_sha256")!=admission["report_sha256"]
            or manifest.get("bindings",{}).get("source_training_ids_sha256")!=ids_sha(admission["source"]["ids"])
            or manifest.get("bindings",{}).get("source_checkpoint_sha256")!=admission["source"]["source_sha256"]
            or manifest.get("bindings",{}).get("source_manifest_sha256")!=sha(admission["source"]["manifest_path"])
            or manifest.get("bindings",{}).get("rgb_asset_validation")!=admission["assets"]
            or manifest.get("bindings",{}).get("source_mass_sha256")!=manifest.get("source_mass_sha256")
            or not isinstance(manifest.get("source_mass_sha256"),str)
            or len(manifest["source_mass_sha256"])!=64
            or manifest.get("implementation_fingerprint")!=implementation_fingerprint()
            or manifest.get("bindings",{}).get("implementation_fingerprint")!=implementation_fingerprint()
            or manifest.get("ground_truth_used") is not False
            or any(manifest.get(f"{key}_sha256")!=sha(path) for key,path in
                   (("checkpoint", paths["checkpoint.pt"]), ("optimizer", paths["optimizer.pt"]),
                    ("history", paths["history.json"])) )
            or not boundary.is_file()
            or manifest.get("durable_boundary_sha256")!=sha(boundary)):
        raise ValueError("SW0152 completed training manifest/hash lineage failed")
    rates={"spatial_residual":3e-6,"graph":3e-5,"native_core":3e-5,"integration_scalars":1e-3}
    if (set(step_summary)!=family_names or {row.get("name") for row in schema}!=family_names
            or manifest["bindings"].get("optimizer_rates")!=rates
            or any(int(step_summary[n].get("maximum") or 0)!=UPDATES
                   or int(step_summary[n].get("minimum") or 0)<1
                   or int(step_summary[n].get("count") or 0)<1 for n in family_names)):
        raise ValueError("SW0152 optimizer group step summary is incomplete")
    if manifest.get("frozen_base_encoder_sha256_before_after") != admission["report"]["source"]["source_encoder_sha256"]:
        raise ValueError("SW0152 frozen source encoder digest changed")
    for index,row in enumerate(history):
        aug=row.get("augmentation_records",[])
        if (row.get("update")!=index+1 or row.get("image_ids")!=expected_ids[index*BATCH:(index+1)*BATCH]
                or len(aug)!=BATCH or [item.get("image_id") for item in aug]!=row.get("image_ids")
                or any(item.get("source_seed")!=SEED or len(item.get("gain",[]))!=3
                       or len(item.get("offset",[]))!=3
                       or any(not 0.6<=float(v)<=1.4 for v in item["gain"])
                       or any(not -0.1<=float(v)<=0.1 for v in item["offset"]) for item in aug)
                or set(row.get("family_gradient_norms",{}))!=family_names
                or any(not np.isfinite(float(row["family_gradient_norms"][n]))
                       or float(row["family_gradient_norms"][n])<=0 for n in family_names)):
            raise ValueError("SW0152 update history IDs/gradient/augmentation record failed")
    metadata=durable.inspect_committed_metadata(boundary.parent)
    if metadata.get("completed_updates")!=UPDATES or metadata.get("bindings")!=manifest.get("bindings"):
        raise ValueError("SW0152 durable endpoint differs from completed manifest lineage")
    return {"manifest":manifest,"manifest_path":out/"manifest.json","manifest_sha256":sha(out/"manifest.json"),
            "checkpoint":paths["checkpoint.pt"],"optimizer":paths["optimizer.pt"],"history":paths["history.json"]}


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--arm",choices=ARMS,required=True)
    parser.add_argument("--device",default="cuda:0");parser.add_argument("--preflight",type=Path,default=PREFLIGHT)
    args=parser.parse_args(argv)
    result=train_arm(args.arm,device=args.device,preflight_path=args.preflight)
    print(json.dumps({"status":result["status"],"arm":args.arm,"ground_truth_used":False},allow_nan=False),flush=True)
    return 0


if __name__=="__main__": raise SystemExit(main())
