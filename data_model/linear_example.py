import torch
import common
import constants
import numpy as np
from tqdm import tqdm
from torch import nn
from data_model.base_mode import BaseModel


class LinearFlow(nn.Module):

    def __init__(self, dim, parameter_vector_size, sigma_n):
        super().__init__()
        a = torch.randn([dim, parameter_vector_size])
        a_norm = a / torch.sqrt(torch.pow(torch.abs(a), 2.0).sum())
        self.a = nn.Parameter(a_norm, requires_grad=False)
        b = torch.randn([dim, dim])
        b = b / torch.norm(b)
        bbt = torch.matmul(b.transpose(dim0=0, dim1=1), b)
        self.sigma_n = sigma_n
        c_xx = torch.eye(dim) * (self.sigma_n ** 2) + bbt
        l_matrix = torch.linalg.cholesky(c_xx)
        self.l_matrix = nn.Parameter(l_matrix, requires_grad=False)
        self.l_matrix_inv = nn.Parameter(torch.linalg.inv(l_matrix), requires_grad=False)
        self.l_log_det = torch.log(torch.linalg.det(self.l_matrix))
        self.l_inv_log_det = torch.log(torch.linalg.det(self.l_matrix_inv))
        self.dim = dim

    def forward(self, x, cond=None):
        z = torch.matmul(self.l_matrix_inv,
                         x.transpose(dim0=0, dim1=1) - torch.matmul(self.a, cond.transpose(dim0=0, dim1=1))).transpose(
            dim0=0, dim1=1)
        return z, self.l_inv_log_det

    def backward(self, z, cond=None):
        x = torch.matmul(self.l_matrix, z.transpose(dim0=0, dim1=1)) + torch.matmul(self.a,
                                                                                    cond.transpose(dim0=0, dim1=1))
        x = x.transpose(dim0=0, dim1=1)
        return x, self.l_log_det


class LinearModel(BaseModel):
    def __init__(self, dim: int, theta_min: float, theta_max: float, sigma_n):
        super().__init__(theta_min, theta_max)
        self.dim = dim
        self.sigma_n = sigma_n
        self.optimal_flow = LinearFlow(self.dim, self.parameter_vector_length, self.sigma_n)

    def get_optimal_model(self):
        return self.optimal_flow

    def save_data_model(self, folder):
        pass

    def load_data_model(self, folder):
        pass

    @property
    def parameter_vector_length(self):
        return 1

    def parameter_range(self, n_steps):
        return self.theta_min + (self.theta_max - self.theta_min) * torch.linspace(0, 1, n_steps,
                                                                                   device=constants.DEVICE)

    def generate_data(self, n_samples, theta):
        cond = torch.ones([n_samples, 1], device=constants.DEVICE) * theta
        z = torch.randn([n_samples, self.dim], device=constants.DEVICE)
        return self.optimal_flow.backward(z, cond=cond)[0]

    def ml_estimator(self, r):
        return torch.pow(torch.mean(torch.pow(torch.abs(r), 6), dim=1), 1 / 6)

    def crb(self, theta):
        a = self.optimal_flow.a
        l = self.optimal_flow.l_matrix
        fim = torch.matmul(
            torch.matmul(a.transpose(dim0=0, dim1=1), torch.linalg.inv(torch.matmul(l, l.transpose(dim0=0, dim1=1)))),
            a)
        return torch.linalg.inv(fim)
