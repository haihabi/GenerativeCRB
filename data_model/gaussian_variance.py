import torch
import common
import constants
import numpy as np
from tqdm import tqdm
from torch import nn
from data_model.base_mode import BaseModel


class MultiplicationFlow(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, cond=None):
        z = x / cond
        log_det = torch.log(1/torch.pow(cond,self.dim)).squeeze(dim=1)
        return z, log_det

    def backward(self, z, cond=None):
        x = z * cond
        log_det = self.dim*torch.log(cond).squeeze(dim=-1)
        return x, log_det


class MultiplicationSimple(BaseModel):
    def __init__(self, dim: int, theta_min: float, theta_max: float):
        super().__init__(theta_min, theta_max)
        self.dim = dim

    def get_optimal_model(self):
        return MultiplicationFlow(self.dim)

    @property
    def parameter_vector_length(self):
        return 1

    def parameter_range(self, n_steps):
        return self.theta_min + (self.theta_max - self.theta_min) * torch.linspace(0, 1, n_steps,
                                                                                   device=constants.DEVICE)

    def generate_data(self, n_samples, theta):
        z = torch.randn([n_samples, self.dim], device=constants.DEVICE)
        return z * theta

    def ml_estimator(self, r):
        return torch.pow(torch.mean(torch.pow(torch.abs(r), 6), dim=1), 1 / 6)

    def crb(self, theta):
        return  torch.pow(theta, 2.0) / (2*self.dim)  # Check CRB in the case of dim>1
