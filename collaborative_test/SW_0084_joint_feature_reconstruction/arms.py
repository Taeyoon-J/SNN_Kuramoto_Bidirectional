"""Preregistered SW0084 coarse arm matrix."""

ARMS = {
    "A": {"name": "fresh_joint_reconstruction_1", "reconstruction_weight": 1.0,
          "initialization": "fresh_core_graph_seed0"},
    "B": {"name": "fresh_joint_no_reconstruction_control", "reconstruction_weight": 0.0,
          "initialization": "fresh_core_graph_seed0"},
    "C": {"name": "sw0072_initialized_trainable_graph_reconstruction_1",
          "reconstruction_weight": 1.0, "initialization": "sw0072_seed1_core_trainable"},
    "D": {"name": "fresh_joint_reconstruction_0p3", "reconstruction_weight": 0.3,
          "initialization": "fresh_core_graph_seed0"},
}


def arm_config(arm):
    try:
        return dict(ARMS[arm])
    except KeyError as error:
        raise ValueError(f"unknown SW0084 arm: {arm}") from error
