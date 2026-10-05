import unittest

import torch

from rgb_filter import border_color_component_filter
from snn_kuramoto_bidirectional.spike_classifier import (
    spike_synchrony_affinity,
    spike_synchrony_components,
)


def patch_image(background=(100, 110, 120), object_color=(200, 200, 200)):
    image = torch.empty(3, 128, 128, dtype=torch.uint8)
    image[:] = torch.tensor(background, dtype=torch.uint8)[:, None, None]
    # A 2x2 patch block belongs to a distinct-color component.
    image[:, 48:64, 48:64] = torch.tensor(object_color, dtype=torch.uint8)[:, None, None]
    return image


class RGBBackgroundVetoTests(unittest.TestCase):
    def test_border_like_group_removed_and_foreground_group_preserved_exactly(self):
        groups = [[(0, 1, 16), (102, 103, 118, 119)]]
        kept, diagnostics = border_color_component_filter(
            patch_image()[None], groups, distance_threshold=2.8
        )
        self.assertEqual(kept, [[(102, 103, 118, 119)]])
        self.assertEqual(diagnostics[0]["removed_group_count"], 1)
        self.assertEqual(diagnostics[0]["removed_patch_count"], 3)

    def test_no_group_member_is_split_or_reassigned(self):
        groups = [[(102, 103, 118, 119)]]
        kept, _ = border_color_component_filter(
            patch_image()[None], groups, distance_threshold=2.8
        )
        self.assertEqual(kept, groups)

    def test_invalid_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "distance_threshold"):
            border_color_component_filter(patch_image()[None], [[]], 0)
        with self.assertRaisesRegex(ValueError, "patch index"):
            border_color_component_filter(patch_image()[None], [[(256,)]], 2.8)

    def test_binary_only_affinity_ignores_component_gate(self):
        spikes = torch.ones(1, 3, 6)
        components = torch.tensor([
            [[[0, 2, 1, 1, 3, 2], [2, 4, 3, 1, 3, 1], [3, 2, 2, 0, 1, 2]],
             [[3, 3, 1, 0, 0, 1], [3, 3, 3, 2, 0, 3], [1, 0, 1, 3, 0, 1]]]
        ], dtype=torch.float32)
        binary = spike_synchrony_affinity(
            spikes, components=components, affinity_mode="spike_binary"
        )
        expected = spike_synchrony_affinity(
            spikes, components=(components != 0).float(), affinity_mode="spike"
        )
        gated = spike_synchrony_affinity(
            spikes, components=components, affinity_mode="spike"
        )
        self.assertTrue(torch.equal(binary, expected))
        self.assertFalse(torch.allclose(binary, gated))
        groups = spike_synchrony_components(
            spikes, components=components, affinity_mode="spike_binary",
            synchrony_threshold=0.2, min_group_size=1, background="largest_component",
        )
        self.assertEqual(len(groups), 1)


if __name__ == "__main__":
    unittest.main()
