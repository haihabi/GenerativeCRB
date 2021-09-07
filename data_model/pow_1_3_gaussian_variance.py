import torch
import constants
import common
from torch import nn
from data_model.base_mode import BaseModel


class Pow1Div3OptimalFlow(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, cond=None):
        z = torch.sign(x) * torch.pow(torch.abs(x / cond), 3)
        log_det = torch.log(torch.prod(3 * torch.pow(cond, - 3) * torch.pow(x, 2.0), dim=1))
        return z, log_det

    def backward(self, z, cond=None):
        x = torch.sign(z) * torch.pow(torch.abs(z), 1 / 3) * cond
        log_det = torch.log(torch.prod((1 / 3) * torch.pow(torch.abs(z), -2 / 3) * cond, dim=1))
        return x, log_det


class Pow1Div3Gaussian(BaseModel):
    def __init__(self, dim: int, theta_min: float, theta_max: float):
        super().__init__(dim, theta_min, theta_max)

    def save_data_model(self, folder):
        pass

    def load_data_model(self, folder):
        pass

    def _get_optimal_model(self):
        return Pow1Div3OptimalFlow(self.dim)

    def generate_data(self, n_samples, theta):
        z = torch.randn([n_samples, self.dim], device=constants.DEVICE)
        return torch.pow(torch.abs(z), 1 / 3) * theta * torch.sign(z)

    @staticmethod
    def ml_estimator(r):
        return torch.pow(torch.mean(torch.pow(torch.abs(r), 6), dim=1), 1 / 6)

    def crb(self, theta):
        theta = common.change2tensor(theta)
        return torch.pow(theta, 2.0) / (18 * self.dim)  # Check CRB in the case of dim>1
