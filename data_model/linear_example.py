import torch
import constants
from torch import nn
from data_model.base_mode import BaseModel
import os


class LinearOptimalFlow(nn.Module):
    def __init__(self, dim, parameter_vector_size, sigma_n):
        super().__init__()
        a = torch.randn([dim, parameter_vector_size], device=constants.DEVICE)
        a_norm = a / torch.sqrt(torch.pow(torch.abs(a), 2.0).sum())
        self.a = nn.Parameter(a_norm, requires_grad=False)
        b = torch.randn([dim, dim], device=constants.DEVICE)
        b = b / torch.norm(b)
        bbt = torch.matmul(b.transpose(dim0=0, dim1=1), b)
        self.bbt = nn.Parameter(bbt, requires_grad=False)
        self.sigma_n = sigma_n
        self.dim = dim
        self.calculate_l_matrix()

    def calculate_l_matrix(self):
        c_xx = torch.eye(self.dim, device=constants.DEVICE) * self.sigma_n + self.bbt
        l_matrix = torch.linalg.cholesky(c_xx)
        self.l_matrix = l_matrix
        self.l_matrix_inv = torch.linalg.inv(l_matrix)
        self.l_log_det = torch.log(torch.linalg.det(self.l_matrix))
        self.l_inv_log_det = torch.log(torch.linalg.det(self.l_matrix_inv))

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
    def __init__(self, dim: int, theta_dim, theta_min: float, theta_max: float, sigma_n):
        super().__init__(dim, theta_min, theta_max, theta_dim=theta_dim)
        self.sigma_n = sigma_n
        self.optimal_flow = LinearOptimalFlow(self.dim, self.parameter_vector_length, self.sigma_n)

    @property
    def name(self) -> str:
        return f"{super().name}_{self.sigma_n}"  # Append Sigma N to Name

    def _get_optimal_model(self):
        return self.optimal_flow

    def save_data_model(self, folder):
        torch.save(self.optimal_flow.state_dict(), os.path.join(folder, f"{self.model_name}_model.pt"))

    def load_data_model(self, folder):
        data = torch.load(os.path.join(folder, f"{self.model_name}_model.pt"), map_location="cpu")
        self.optimal_flow.load_state_dict(data)
        self.optimal_flow.calculate_l_matrix()

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


if __name__ == '__main__':
    import gcrb
    import numpy as np

    dm = LinearModel(10, 2, -10, 10, 0.1)
    theta_array = dm.parameter_range(5)
    model_opt = dm.get_optimal_model()
    crb_list = [dm.crb(theta) for theta in theta_array]
    gcrb_list = [torch.inverse(gcrb.adaptive_sampling_gfim(model_opt, theta.reshape([-1]))) for theta in theta_array]

    theta_array = theta_array.cpu().detach().numpy()
    crb_array = torch.stack(crb_list).cpu().detach().numpy()
    gcrb_array = torch.stack(gcrb_list).cpu().detach().numpy()
    from matplotlib import pyplot as plt

    plt.plot(theta_array[:, 0], np.diagonal(crb_array, axis1=1, axis2=2).sum(axis=-1))
    plt.plot(theta_array[:, 0], np.diagonal(gcrb_array, axis1=1, axis2=2).sum(axis=-1))
    plt.show()
    # print("a")
