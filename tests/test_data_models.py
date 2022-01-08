import unittest
import torch
import data_model as dm
import normflowpy as nf
from experiments import constants


def generate_model_dict():
    return {constants.DIM: 4,
            constants.THETA_MIN: 0.3,
            constants.THETA_MAX: 10,
            constants.SIGMA_N: 0.1,
            }


class FlowToCRBTest(unittest.TestCase):
    def run_model_check(self, model_enum):
        model = dm.get_model(model_enum, generate_model_dict())
        self.assertTrue(str(4) in model.name)  # Check that dim size in model name
        self.assertTrue(model.__class__.__name__ in model.name)  # Check that class name is in model
        flow = model.get_optimal_model()
        self.assertTrue(isinstance(flow, nf.NormalizingFlowModel))
        crb_value = model.crb(2.0)
        self.assertTrue(isinstance(crb_value, torch.Tensor))

    def test_flow_linear(self):
        self.run_model_check(dm.ModelType.Linear)

    def test_gaussian_variance(self):
        self.run_model_check(dm.ModelType.GaussianVariance)

    def test_mult_1_3(self):
        self.run_model_check(dm.ModelType.Pow1Div3Gaussian)

    def test_mult_3(self):
        self.run_model_check(dm.ModelType.Pow3Gaussian)
