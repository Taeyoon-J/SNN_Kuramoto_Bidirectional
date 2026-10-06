"""Registered SW0088 reconstruction controls."""


ARMS = {
    "E": {"name": "encoder_graph_no_reconstruction", "reconstruction_weight": 0.0,
          "initialization": "sw0072_seed1_encoder_graph_only"},
    "F": {"name": "encoder_graph_reconstruction_0p3", "reconstruction_weight": 0.3,
          "initialization": "sw0072_seed1_encoder_graph_only"},
    "G": {"name": "encoder_graph_reconstruction_1", "reconstruction_weight": 1.0,
          "initialization": "sw0072_seed1_encoder_graph_only"},
}


def arm_config(name):
    return dict(ARMS[name])
