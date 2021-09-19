import torch
import constants
from torch import nn
from data_model.base_mode import BaseModel
import os


class MeanOptimalFlow(nn.Module):
    def __init__(self, dim, parameter_vector_size, sigma_n):
        super().__init__()
        a = torch.randn([dim, parameter_vector_size])
        a_norm = a / torch.sqrt(torch.pow(torch.abs(a), 2.0).sum())
        self.a = nn.Parameter(a_norm, requires_grad=False)
        # b = torch.randn([dim, dim])
        # b = b / torch.norm(b)
        # bbt = torch.matmul(b.transpose(dim0=0, dim1=1), b)
        self.sigma_n = sigma_n
        c_xx = torch.eye(dim) * self.sigma_n
        l_matrix = torch.linalg.cholesky(c_xx)
        self.l_matrix = l_matrix
        self.l_matrix_inv = torch.linalg.inv(l_matrix)
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


class MeanModel(BaseModel):
    def __init__(self, dim: int, theta_min: float, theta_max: float, sigma_n):
        super().__init__(dim, theta_min, theta_max)
        self.sigma_n = sigma_n
        self.optimal_flow = MeanOptimalFlow(self.dim, self.parameter_vector_length, self.sigma_n)

    @property
    def name(self) -> str:
        return f"{super().name}_{self.sigma_n}"  # Append Sigma N to Name

    def _get_optimal_model(self):
        return self.optimal_flow

    def save_data_model(self, folder):
        torch.save(self.optimal_flow.state_dict(), os.path.join(folder, f"{self.model_name}_model.pt"))

    def load_data_model(self, folder):
        data = torch.load(os.path.join(folder, f"{self.model_name}_model.pt"))
        self.optimal_flow.load_state_dict(data)

    def generate_data(self, n_samples, theta):
        cond = torch.ones([n_samples, 1], device=constants.DEVICE) * theta
        z = torch.randn([n_samples, self.dim], device=constants.DEVICE)
        return self.optimal_flow.backward(z, cond=cond)[0]

    @staticmethod
    def ml_estimator(r):
        return torch.pow(torch.mean(torch.pow(torch.abs(r), 6), dim=1), 1 / 6)

    def crb(self, theta):
        a = self.optimal_flow.a
        l = self.optimal_flow.l_matrix
        fim = torch.matmul(
            torch.matmul(a.transpose(dim0=0, dim1=1), torch.linalg.inv(torch.matmul(l, l.transpose(dim0=0, dim1=1)))),
            a)
        return torch.linalg.inv(fim)
