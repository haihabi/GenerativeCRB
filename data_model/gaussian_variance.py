import torch
import common
import constants
from torch import nn
from data_model.base_mode import BaseModel


class GaussianVarianceOptimalFlow(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, cond=None):
        z = x / cond
        log_det = torch.log(1 / torch.pow(cond, self.dim)).squeeze(dim=1)
        return z, log_det

    def backward(self, z, cond=None):
        x = z * cond
        log_det = self.dim * torch.log(cond).squeeze(dim=-1)
        return x, log_det


class GaussianVarianceDataModel(BaseModel):
    def __init__(self, dim: int, theta_min: float, theta_max: float):
        super().__init__(dim, theta_min, theta_max)

    def save_data_model(self, folder):
        pass

    def load_data_model(self, folder):
        pass

    def _get_optimal_model(self):
        return GaussianVarianceOptimalFlow(self.dim)

    def generate_data(self, n_samples, theta):
        z = torch.randn([n_samples, self.dim], device=constants.DEVICE)
        return z * theta

    @staticmethod
    def ml_estimator(r):
        return torch.pow(torch.mean(torch.pow(torch.abs(r), 6), dim=1), 1 / 6)

    def crb(self, theta):
        theta = common.change2tensor(theta)
        return torch.pow(theta, 2.0) / (2 * self.dim)  # Check CRB in the case of dim>1
