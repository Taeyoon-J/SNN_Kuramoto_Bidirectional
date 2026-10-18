import unittest
import numpy as np
import torch
from .teacher_loss import make_teacher,teacher_objective


class TeacherLossTests(unittest.TestCase):
    def test_natural_frequency_centers_and_class_balance(self):
        gamma=np.zeros((1,2,4),np.float64);frequency=np.array([[[.09],[.11],[.1],[.4]]])
        target=make_teacher(gamma,frequency,np.array([[[0,0],[0,1]]]))
        torch.testing.assert_close(target['frequency_centers'],torch.tensor([[[.1],[.1],[.1],[.4]]],dtype=torch.float64))
        torch.testing.assert_close(target['class_weights'],torch.tensor([[1/6,1/6,1/6,1/2]],dtype=torch.float64))
        _,parts=teacher_objective(torch.from_numpy(gamma),torch.from_numpy(frequency),target)
        self.assertEqual(float(parts['gamma_anchor']),0.)
        self.assertAlmostEqual(float(parts['frequency_denoising']),(.01**2+.01**2)/6,places=12)

    def test_unnecessary_expansion_and_latent_drift_are_penalized(self):
        gamma=np.zeros((1,2,4),np.float64);frequency=np.array([[[.1],[.1],[.2],[.2]]]);target=make_teacher(gamma,frequency,np.array([[[0,0],[1,1]]]))
        original=teacher_objective(torch.from_numpy(gamma),torch.from_numpy(frequency),target)[0]
        self.assertEqual(float(original),0.)
        self.assertGreater(float(teacher_objective(torch.from_numpy(gamma),10*torch.from_numpy(frequency),target)[0]),0.)
        self.assertGreater(float(teacher_objective(torch.from_numpy(gamma)+1,torch.from_numpy(frequency),target)[0]),0.)

    def test_gradient_points_to_existing_center_and_matches_finite_difference(self):
        gamma=np.zeros((1,2,4),np.float64);frequency=np.array([[[.09],[.11],[.19],[.21]]]);target=make_teacher(gamma,frequency,np.array([[[0,0],[1,1]]]))
        student=torch.from_numpy(frequency.copy()).requires_grad_();g=torch.from_numpy(gamma.copy()).requires_grad_()
        loss,_=teacher_objective(g,student,target);gradient=torch.autograd.grad(loss,student)[0]
        self.assertLess(float(gradient[0,0,0]),0.);self.assertGreater(float(gradient[0,1,0]),0.)
        delta=1e-6;plus=student.detach().clone();minus=student.detach().clone();plus[0,0,0]+=delta;minus[0,0,0]-=delta
        numeric=(teacher_objective(g,plus,target)[0]-teacher_objective(g,minus,target)[0])/(2*delta)
        self.assertAlmostEqual(float(gradient[0,0,0]),float(numeric.detach()),places=10)
        self.assertFalse(target['frequency_centers'].requires_grad)


if __name__=='__main__':unittest.main()
