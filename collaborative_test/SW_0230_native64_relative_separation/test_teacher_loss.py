import unittest
import numpy as np
import torch
from .teacher_loss import make_teacher,teacher_objective,relative_separation
from collaborative_test.SW_0222_native64_frequency_teacher_denoising.teacher_loss import teacher_objective as original


class LossTests(unittest.TestCase):
    def fixture(self):
        labels=np.array([[[0,0],[3,3]]])
        frequency=torch.tensor([[[-1.,0.],[-1.,0.],[1.,0.],[1.,0.]]],dtype=torch.float64)
        gamma=torch.zeros((1,2,4),dtype=torch.float64)
        return gamma,frequency,make_teacher(gamma.numpy(),frequency.numpy(),labels)

    def test_source_has_zero_extra_loss_and_original_objective(self):
        gamma,frequency,targets=self.fixture()
        loss,parts=teacher_objective(gamma,frequency,targets)
        expected,_=original(gamma,frequency,targets)
        torch.testing.assert_close(loss,expected);self.assertEqual(float(parts['relative_separation']),0.)

    def test_contraction_penalized_with_outward_gradient(self):
        _,frequency,targets=self.fixture();current=(frequency*.5).requires_grad_()
        penalty=relative_separation(current,targets);gradient=torch.autograd.grad(penalty,current)[0]
        self.assertGreater(float(penalty.detach()),0.)
        self.assertGreater(float(gradient[0,0,0]),0.)
        self.assertLess(float(gradient[0,2,0]),0.)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_common_translation_and_orthogonal_component_order_do_not_penalize(self):
        _,frequency,targets=self.fixture()
        self.assertEqual(float(relative_separation(frequency+4.,targets)),0.)
        rotated=frequency.flip(-1)
        self.assertEqual(float(relative_separation(rotated,targets)),0.)
        self.assertEqual(float(relative_separation(frequency*2.,targets)),0.)

    def test_class_and_patch_order_invariant_and_single_class_finite(self):
        gamma,frequency,targets=self.fixture();current=frequency*.5
        expected=relative_separation(current,targets)
        labels=targets['class_labels'].clone();targets['class_labels']=torch.where(labels==0,99,2)
        torch.testing.assert_close(expected,relative_separation(current,targets))
        order=torch.tensor([3,0,2,1]);permuted={**targets,
            'class_labels':targets['class_labels'][:,order],
            'frequency_centers':targets['frequency_centers'][:,order]}
        torch.testing.assert_close(expected,relative_separation(current[:,order],permuted))
        one=make_teacher(gamma.numpy(),frequency.numpy(),np.zeros((1,2,2),np.int64))
        self.assertEqual(float(relative_separation(frequency,one)),0.)

    def test_loss_decomposition_and_invalid_labels(self):
        gamma,frequency,targets=self.fixture();current=frequency*.5
        loss,parts=teacher_objective(gamma,current,targets)
        torch.testing.assert_close(loss,parts['frequency_denoising']+parts['gamma_anchor']+parts['relative_separation'])
        targets['class_labels']=targets['class_labels'].float()
        with self.assertRaises(ValueError):teacher_objective(gamma,current,targets)


if __name__=='__main__':unittest.main()
