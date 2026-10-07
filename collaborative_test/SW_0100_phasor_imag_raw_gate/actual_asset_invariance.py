"""CPU actual-source check that gate mode leaves graph/theta/carrier unchanged."""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"),
                str(ROOT / "collaborative_test")]
from SW_0094_aligned_joint_pilot.run import GAMMA
from SW_0100_phasor_imag_raw_gate.coordinator import CONTROL, SOURCE, read_json, sha256
from SW_0094_aligned_joint_pilot.run import hparams
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore


def gamma_rows(training_ids):
    rows = [image_id if image_id < 1000 else image_id - 640 for image_id in training_ids[:2]]
    if any(row < 0 or row >= 70000 for row in rows):
        raise ValueError("control IDs do not map into the registered 70k gamma cache")
    return rows


def run_seed(seed, gamma_all):
    source = SOURCE / f"seed{seed}"
    control = CONTROL / f"seed{seed}_positive_frozen"
    if not (source / "COMPLETED").is_file() or not (control / "COMPLETED").is_file():
        raise FileNotFoundError(f"source/control completion missing for seed {seed}")
    manifest = read_json(control / "manifest.json")
    if manifest.get("source_sha256") != sha256(source / "core.pt"):
        raise AssertionError(f"matched SW0097 source hash mismatch for seed {seed}")
    rows = gamma_rows(manifest["training_ids"])
    gamma = gamma_all[rows].clone()
    state = torch.load(source / "core.pt", map_location="cpu", weights_only=True)
    outputs = {}
    gate_outputs = {}
    import snn_kuramoto_bidirectional.s2net_cls as s2net_module
    original_gating = s2net_module.sinusoidal_gating
    for mode in ("raw", "phasor_imag_raw"):
        hp = hparams(mode)
        hp.num_time_steps = 64
        if hp.graph_feedback_strength != 0 or hp.spike_pulse_gain != 0:
            raise AssertionError("invariance audit requires graph feedback and spike pulse disabled")
        core = S2NetCore(hp.validate(), device="cpu").eval()
        core.load_state_dict(state, strict=True)
        # This audit only compares phases and gate signals. Core's returned
        # object-group helper is unused and can enumerate expensive cliques;
        # bypass it exactly as the registered train/evaluation harnesses do.
        core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
        gates = []
        drives = []

        def capture(history, t, delay, gate_mode="sigmoid"):
            drive_t, gate_t = original_gating(history, t, delay, gate_mode)
            if t == 63:
                gates.append(gate_t.detach().clone())
                drives.append(drive_t.detach().clone())
            return drive_t, gate_t

        s2net_module.sinusoidal_gating = capture
        try:
            with torch.inference_mode():
                graph = core.graph_generator(gamma)
                _, _, _, theta = core(gamma, return_core_out=True,
                                      return_theta=True, num_time_steps=64)
        finally:
            s2net_module.sinusoidal_gating = original_gating
        if len(gates) != 1 or len(drives) != 1:
            raise RuntimeError("could not capture actual final gate/drive")
        outputs[mode] = {"graph": graph.detach().clone(), "theta": theta.detach().clone(),
                         "carrier": theta.sin().detach().clone()}
        gate_outputs[mode] = {"gate": gates[0], "drive": drives[0]}
        if not all(torch.isfinite(v).all() for v in (*outputs[mode].values(),
                                                     *gate_outputs[mode].values())):
            raise FloatingPointError(f"nonfinite actual-source output for {mode} seed{seed}")
        del core

    deltas = {name: float((outputs["raw"][name] - outputs["phasor_imag_raw"][name]).abs().max())
              for name in ("graph", "theta", "carrier")}
    if any(delta != 0.0 for delta in deltas.values()):
        raise AssertionError(f"gate-only mode changed graph/theta/carrier for seed{seed}: {deltas}")
    gate_delta = float((gate_outputs["raw"]["gate"] -
                        gate_outputs["phasor_imag_raw"]["gate"]).abs().max())
    drive_delta = float((gate_outputs["raw"]["drive"] -
                         gate_outputs["phasor_imag_raw"]["drive"]).abs().max())
    return {"seed": seed, "source_checkpoint_sha256": sha256(source / "core.pt"),
            "control_training_ids_first_two": manifest["training_ids"][:2],
            "gamma_cache_rows": rows, "batch": 2, "train_window": 64,
            "graph_feedback_strength": 0.0, "spike_pulse_gain": 0.0,
            "strict_state_load": True, "graph_theta_carrier_max_abs_delta": deltas,
            "actual_step63_gate_max_abs_delta": gate_delta,
            "actual_step63_drive_max_abs_delta": drive_delta}


def main():
    torch.set_num_threads(2)
    gamma_all = torch.load(GAMMA, map_location="cpu", weights_only=True, mmap=True)
    if tuple(gamma_all.shape) != (70000, 8, 256):
        raise AssertionError(f"unexpected registered gamma cache shape: {tuple(gamma_all.shape)}")
    records = [run_seed(seed, gamma_all) for seed in range(3)]
    output = ROOT / "trained_models/SW0100_phasor_imag_raw_gate/actual_asset_invariance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": "complete", "device": "cpu", "ground_truth_read": False,
               "gamma_cache_sha256": sha256(GAMMA),
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "per_seed": records}
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print("SW0100_ACTUAL_SOURCE_GRAPH_THETA_CARRIER_EXACT", flush=True)


if __name__ == "__main__":
    main()
