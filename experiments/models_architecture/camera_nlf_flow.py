import torch
import numpy as np
from experiments import constants
import normflowpy as nfp
from torch.distributions import MultivariateNormal
from experiments.models_architecture.noise_level_function import NoiseLevelFunction


class ImageFlowStep(nfp.ConditionalBaseFlowLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x, cond):
        clean_image = cond[0]
        return x - clean_image, 0

    def backward(self, z, cond):
        clean_image = cond[0]
        return z + clean_image, 0


def generate_nlf_flow(in_input_shape, in_trained_alpha, noise_only=True):
    dim = int(np.prod(in_input_shape))
    prior = MultivariateNormal(torch.zeros(dim, device=constants.DEVICE),
                               torch.eye(dim, device=constants.DEVICE))
    flows = []
    if not noise_only:
        flows.append(ImageFlowStep())
    flows.extend([NoiseLevelFunction(trained_alpha=in_trained_alpha), nfp.flows.Tensor2Vector(in_input_shape)])
    return nfp.NormalizingFlowModel(prior, flows).to(constants.DEVICE)
