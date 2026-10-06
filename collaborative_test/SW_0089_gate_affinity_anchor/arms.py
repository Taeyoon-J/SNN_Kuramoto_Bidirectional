"""Coarse gate-affinity preservation controls for SW0089."""

ARMS = {
    "H": {"reconstruction_weight": 0.3, "gate_affinity_anchor_weight": 1.0},
    "I": {"reconstruction_weight": 0.3, "gate_affinity_anchor_weight": 10.0},
    "J": {"reconstruction_weight": 0.3, "gate_affinity_anchor_weight": 100.0},
}


def arm_config(name):
    return dict(ARMS[name])
