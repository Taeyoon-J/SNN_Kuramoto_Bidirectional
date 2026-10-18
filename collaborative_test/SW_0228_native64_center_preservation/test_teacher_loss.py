import unittest
import numpy as np
import torch
from .teacher_loss import make_teacher, teacher_objective


class CenterTests(unittest.TestCase):
    def fixture(self):
        gamma = torch.zeros((1, 2, 4), dtype=torch.float64)
        source = torch.tensor([[[0., 1.], [0., 1.], [.2, -.4], [.2, -.4]]], dtype=torch.float64)
        labels = np.array([[[0, 0], [7, 7]]], dtype=np.int64)
        return gamma, source, labels, make_teacher(gamma.numpy(), source.numpy(), labels)

    def test_uniform_center_drift_gets16_times_gradient_and_loss(self):
        gamma, source, _, targets = self.fixture()
        student = source.clone(); student[:, :2] += .1; student.requires_grad_(True)
        old, parts = teacher_objective(gamma, student, targets, center_weight=1.)
        new, _ = teacher_objective(gamma, student, targets)
        old_grad = torch.autograd.grad(old, student, retain_graph=True)[0]
        new_grad = torch.autograd.grad(new, student)[0]
        torch.testing.assert_close(new, 16 * old, atol=1e-12, rtol=0)
        torch.testing.assert_close(new_grad, 16 * old_grad, atol=1e-12, rtol=0)
        self.assertAlmostEqual(float(parts['within_object_variance'].detach()), 0., places=12)

    def test_mean_zero_within_class_noise_keeps_original_coefficient(self):
        gamma, source, _, targets = self.fixture()
        student = source.clone(); student[:, 0] += .1; student[:, 1] -= .1; student.requires_grad_(True)
        old, _ = teacher_objective(gamma, student, targets, center_weight=1.)
        new, parts = teacher_objective(gamma, student, targets)
        torch.testing.assert_close(old, new, atol=1e-12, rtol=0)
        torch.testing.assert_close(torch.autograd.grad(old, student, retain_graph=True)[0],
                                   torch.autograd.grad(new, student)[0], atol=1e-12, rtol=0)
        self.assertAlmostEqual(float(parts['source_center_MSE'].detach()), 0., places=12)

    def test_decomposition_and_class_ID_permutation_not_channel_order(self):
        gamma, source, labels, targets = self.fixture()
        student = source + torch.tensor([[[.1, .2], [-.3, .1], [.2, -.1], [.05, .2]]], dtype=torch.float64)
        loss, parts = teacher_objective(gamma, student, targets)
        torch.testing.assert_close(loss, parts['within_object_variance'] + 16 * parts['source_center_MSE'] + parts['gamma_anchor'], atol=1e-12, rtol=0)
        relabeled = np.where(labels == 0, 9, 3)
        changed, _ = teacher_objective(gamma, student, make_teacher(gamma.numpy(), source.numpy(), relabeled))
        torch.testing.assert_close(loss, changed, atol=1e-12, rtol=0)

    def test_invalid_scope_and_weights_rejected(self):
        gamma, source, _, targets = self.fixture()
        for weight in (.5, float('nan')):
            with self.assertRaises(ValueError):teacher_objective(gamma, source, targets, center_weight=weight)
        targets['class_labels'] = targets['class_labels'][:, :2]
        with self.assertRaises(ValueError):teacher_objective(gamma, source, targets)


if __name__ == '__main__':
    unittest.main()
