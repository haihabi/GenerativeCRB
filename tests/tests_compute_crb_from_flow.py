import unittest

import numpy as np
import torch

from data_model.linear_example import LinearModel
from data_model.pow_1_3_gaussian_variance import Pow1Div3Gaussian
from data_model.gaussian_variance import GaussianVarianceDataModel
import normalizing_flow as nf
import constants
import gcrb
from torch.distributions import MultivariateNormal


class FlowToCRBTest(unittest.TestCase):
    def compare(self, grcb_value, crb_value):
        stauts = np.isclose(crb_value, grcb_value, rtol=1e-1).flatten()
        if not stauts:
            print(f"CRB Value:{crb_value}")
            print(f"GCRB Value:{grcb_value}")
            print(grcb_value, crb_value, crb_value / grcb_value)
        return stauts

    def model_init(self, in_model, in_dim, theta_value):
        theta = theta_value * torch.ones([1])
        prior = MultivariateNormal(torch.zeros(in_dim, device=constants.DEVICE),
                                   torch.eye(in_dim, device=constants.DEVICE))
        crb = in_model.crb(theta)
        return nf.NormalizingFlowModel(prior, [in_model._get_optimal_model()]), theta, crb

    def run_test_dual(self, in_model, in_dim, theta_value):
        model_opt, theta, crb = self.model_init(in_model, in_dim, theta_value)
        gfim_res = gcrb.compute_fim(model_opt, theta, 8192)
        gcrb_value = torch.linalg.inv(gfim_res).item()
        return self.compare(gcrb_value, crb)

    def run_test_backward(self, in_model, in_dim, theta_value):
        model_opt, theta, crb = self.model_init(in_model, in_dim, theta_value)
        gfim_res_back = gcrb.compute_fim_backward(model_opt, theta, 8192)
        gcrb_res = torch.linalg.inv(gfim_res_back)
        gcrb_value = gcrb_res.item()

        return self.compare(gcrb_value, crb)

    def test_flow_linear(self):
        dim = 2
        lm = LinearModel(dim, -10.0, 10.0, 0.1)
        status = self.run_test_dual(lm, dim, 2)
        self.assertTrue(status)

    def test_backward_flow_linear(self):
        dim = 2
        lm = LinearModel(dim, -10.0, 10.0, 0.1)
        status = self.run_test_backward(lm, dim, 2)
        self.assertTrue(status)

    def test_backward_flow_pow_1_3(self):
        dim = 2
        lm = Pow1Div3Gaussian(dim, 0.3, 10.0)
        status = self.run_test_backward(lm, dim, 2)
        self.assertTrue(status)

    def test_flow_pow_1_3(self):
        dim = 4
        lm = Pow1Div3Gaussian(dim, 0.3, 10.0)
        status = self.run_test_dual(lm, dim, 0.1)
        self.assertTrue(status)

    def test_backward_flow_gaussian_variance(self):
        dim = 2
        lm = GaussianVarianceDataModel(dim, 0.3, 10.0)
        status = self.run_test_backward(lm, dim, 2)
        self.assertTrue(status)

    def test_flow_gaussian_variance(self):
        dim = 4
        lm = GaussianVarianceDataModel(dim, 0.3, 10.0)
        status = self.run_test_dual(lm, dim, 0.1)
        self.assertTrue(status)


if __name__ == '__main__':
    unittest.main()
