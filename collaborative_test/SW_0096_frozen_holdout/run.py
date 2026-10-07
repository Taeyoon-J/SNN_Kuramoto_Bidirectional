"""No-training, frozen-checkpoint comparison on preregistered holdouts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "trained_models/SW0096_frozen_holdout"
DATA = Path("/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5")
PY = Path("/Data0/kevinswk/envs/snn/bin/python")
TFPY = Path("/Data0/kevinswk/envs/slot_attention_gpu_tf215/bin/python")
ASSETS = Path("/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002")
SPLITS = {"independent": 90000, "reference_previously_inspected": 1000}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def slot_folder(seed):
    return ROOT / f"trained_models/SW0092_slot_our70000_s{seed}{'_final' if seed == 0 else ''}"


def prepare():
    import h5py
    import torch
    sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional")]
    from snn_kuramoto_bidirectional.gamma_initializer import FeaturePatchGammaInitializer
    from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_input_encoder

    torch.set_num_threads(2)
    excluded = set(range(1000, 1320)) | set(range(90000, 90320))
    train_ids = set(range(1000)) | set(range(1640, 70640))
    sources = {}
    for seed in range(3):
        ours = ROOT / f"trained_models/SW0095_full70k_aligned_loss/seed{seed}"
        m = json.loads((ours / "manifest.json").read_text())
        assert (ours / "COMPLETED").exists() and m["status"] == "complete"
        assert m["source_model_seed"] == seed and m["arm"] == "positive_frozen"
        assert len(m["training_ids"]) == 70000 and set(m["training_ids"]) == train_ids
        assert not train_ids.intersection(excluded) and not m["ground_truth_used_for_training"]
        slot = slot_folder(seed)
        p = json.loads((slot / "training_protocol.json").read_text())
        assert p["seed"] == seed and p["unique_training_images"] == 70000
        assert p["training_id_segments_inclusive"] == [[0, 999], [1640, 70639]]
        assert p["epochs"] == 10 and not p["ground_truth_used_for_training"]
        assert (slot / "checkpoint/ckpt-21880.index").is_file()
        sources[str(seed)] = {"our_core_sha256": sha(ours / "core.pt"),
                              "slot_checkpoint_index_sha256": sha(slot / "checkpoint/ckpt-21880.index")}
    encoder_path = ASSETS / "input_encoder/input_layer_encoder.pt"
    stats_path = ASSETS / "feature_preprocessing.pt"
    encoder = load_input_encoder(str(encoder_path), num_kernels=8, kernel_size=3, channels=3, device="cpu").eval()
    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    assert stats["mode"] == "standardize" and float(stats["std"]) > 0
    for seed in range(3):
        ours = ROOT / f"trained_models/SW0095_full70k_aligned_loss/seed{seed}"
        saved = torch.load(ours / "encoder.pt", map_location="cpu", weights_only=True)
        assert saved.keys() == encoder.state_dict().keys()
        assert all(torch.equal(saved[k], v) for k, v in encoder.state_dict().items())
        saved_stats = torch.load(ours / "feature_preprocessing.pt", map_location="cpu", weights_only=True)
        assert all(torch.equal(saved_stats[k], stats[k]) for k in ("mean", "std"))
        assert float(saved_stats.get("clip", 3.)) == float(stats.get("clip", 3.))
    patcher = FeaturePatchGammaInitializer(grid_size=16)

    def encode(rgb):
        features = encoder(torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.)
        return patcher(((features - stats["mean"]) / stats["std"]).clamp(-float(stats.get("clip", 3.)), float(stats.get("clip", 3.))))

    with h5py.File(DATA, "r") as dataset, torch.no_grad():
        assert len(dataset["image"]) >= 90320
        cached = torch.load(ROOT / "data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt", weights_only=True)
        error = float((encode(dataset["image"][1320:1328]) - cached[:8]).abs().max())
        assert error < 2e-5, error
        for name, start in SPLITS.items():
            folder = OUT / name
            folder.mkdir(exist_ok=True)
            gamma = torch.cat([encode(dataset["image"][i:i + 16]) for i in range(start, start + 320, 16)])
            assert tuple(gamma.shape) == (320, 8, 256) and torch.isfinite(gamma).all()
            torch.save(gamma, folder / "gamma.pt")
            write(folder / "gamma_manifest.json", {"dataset": str(DATA), "image_ids": [start, start + 319],
                  "encoder": str(encoder_path), "encoder_sha256": sha(encoder_path),
                  "preprocessing": str(stats_path), "preprocessing_sha256": sha(stats_path),
                  "gamma_sha256": sha(folder / "gamma.pt"), "shape": [320, 8, 256]})
    write(OUT / "manifest.json", {"sources": sources, "splits": SPLITS, "count_per_split": 320,
          "selection_frozen_before_holdout_scoring": True, "training": False,
          "validation_gamma_reproduction_max_error": error,
          "prior_reference_exposure_disclosed": True, "unrecorded_prior_use_cannot_be_excluded": True})
    (OUT / "PREFLIGHT_COMPLETED").write_text("asset, split-exclusion and gamma checks passed\n")


def invoke(command, folder, gpu, slot=False):
    probe = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, check=True)
    while any(c.isdigit() for c in probe.stdout):
        time.sleep(55)
        probe = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, check=True)
    folder.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", SW0092_ALLOW_GPU="1", TF_FORCE_GPU_ALLOW_GROWTH="true")
    with (folder / "run.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        write(OUT / "state.json", {"status": "running", "pid": process.pid, "folder": str(folder), "command": command, "gpu": gpu})
        if process.wait():
            raise RuntimeError(f"evaluation failed: {folder}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--gpu", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.daemon:
        if (OUT / "queue.pid").exists():
            raise FileExistsError("inspect existing queue; do not duplicate")
        with (OUT / "queue.log").open("w") as log:
            process = subprocess.Popen([str(PY), __file__, "--gpu", str(args.gpu)], stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        (OUT / "queue.pid").write_text(str(process.pid) + "\n")
        print(f"QUEUE_PID={process.pid}")
        return
    lock = (OUT / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not (OUT / "PREFLIGHT_COMPLETED").exists():
        prepare()
    if args.prepare_only:
        print((OUT / "manifest.json").read_text())
        return
    base_command = json.loads((ROOT / "trained_models/SW0094_aligned_joint_pilot/positive_frozen/full320_batch8_launch.json").read_text())["command"]
    try:
        for name, start in SPLITS.items():
            for seed in range(3):
                folder = OUT / name / f"ours_seed{seed}"
                command = list(base_command)
                replacements = {"--checkpoint": str(ROOT / f"trained_models/SW0095_full70k_aligned_loss/seed{seed}/core.pt"),
                                "--gamma-path": str(OUT / name / "gamma.pt"), "--gamma-global-start": str(start),
                                "--gamma-manifest": str(OUT / name / "gamma_manifest.json"),
                                "--output-path": str(folder / "evaluation.json"), "--start": str(start)}
                for flag, value in replacements.items():
                    command[command.index(flag) + 1] = value
                invoke(command, folder, args.gpu)
                folder = OUT / name / f"slot_seed{seed}"
                slot = slot_folder(seed)
                command = [str(TFPY), str(ROOT / "collaborative_test/SW_0092_cross_dataset_training/slot_predict_gpu.py"),
                           "--checkpoint-dir", str(slot / "checkpoint"), "--checkpoint-prefix", "ckpt-21880",
                           "--checkpoint-source", "SW0092 frozen70k epoch10 holdout", "--training-seed", str(seed),
                           "--inference-seed", "0", "--training-protocol", str(slot / "training_protocol.json"),
                           "--tf-intra-threads", "2", "--tf-inter-threads", "1", "--model-py",
                           "/Data0/kevinswk/patch_v2/trained_models/slot_attention_official_patch_eval_20261002/model.py",
                           "--dataset", str(DATA), "--start", str(start), "--count", "320", "--output-dir", str(folder)]
                invoke(command, folder, args.gpu, slot=True)
                with (folder / "score.log").open("w") as log:
                    subprocess.run([str(PY), str(ROOT / "collaborative_test/SW_0092_cross_dataset_training/score_predictions.py"),
                                    "--predictions", str(folder / "predictions.npz"), "--protocol", str(folder / "protocol.json"),
                                    "--dataset", str(DATA), "--start", str(start), "--count", "320", "--output-dir", str(folder)],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
        write(OUT / "state.json", {"status": "complete", "completed": time.time()})
    except Exception as error:
        write(OUT / "state.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
