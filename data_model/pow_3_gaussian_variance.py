import torch
import constants
import common
from torch import nn
from data_model.base_mode import BaseModel


class OptPow3Flow(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, cond=None):
        z = torch.sign(x) * torch.pow(torch.abs(x / cond), 1 / 3)
        log_det = torch.log(torch.prod((1 / 3) * torch.pow(cond, -1 / 3) * torch.pow(torch.pow(x, 2.0), -1 / 3), dim=1))
        return z, log_det

    def backward(self, z, cond=None):
        x = torch.pow(z, 3.0) * cond
        log_det = torch.log(torch.prod(3 * torch.pow(z, 2.0) * cond, dim=1))
        return x, log_det


class Pow3Gaussian(BaseModel):
    def __init__(self, dim: int, theta_min: float, theta_max: float):
        super().__init__(dim, theta_min, theta_max)

    def save_data_model(self, folder):
        pass

    def load_data_model(self, folder):
        pass

    def _get_optimal_model(self):
        return OptPow3Flow(self.dim)

    def generate_data(self, n_samples, theta):
        return torch.pow(torch.randn([n_samples, self.dim], device=constants.DEVICE), 3.0) * theta

    @staticmethod
    def ml_estimator(r):
        return torch.pow(torch.mean(torch.pow(torch.abs(r), 2 / 3), dim=1), 3 / 2)

    def crb(self, theta):
        theta = common.change2tensor(theta)
        return torch.pow(theta, 2.0) * 9 / (2 * self.dim)  # Check CRB in the case of dim>1
