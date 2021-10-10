import torch
import constants
from torch import nn
from data_model.base_mode import BaseModel
import os
import math


class FrequencyOptimalFlow(nn.Module):
    def __init__(self, n_samples, sigma_n):
        super().__init__()
        self.sigma_n = sigma_n
        self.n_samples = n_samples
        self.n = nn.Parameter(
            torch.linspace(0, self.n_samples - 1, self.n_samples, device=constants.DEVICE).reshape([1, -1]),
            requires_grad=False)

    def _sine(self, f_0, phase=0):
        return torch.cos(2 * math.pi * f_0 * self.n + phase)

    def forward(self, x, cond=None):
        z_tilde = x - self._sine(cond)
        z = z_tilde / self.sigma_n
        return z, 0

    def backward(self, z, cond=None):
        x = self._sine(cond) + z * self.sigma_n
        return x, 0


class FrequencyModel(BaseModel):
    def __init__(self, dim: int, sigma_n):
        super().__init__(dim, 0.01, 0.49)
        self.sigma_n = sigma_n
        self.optimal_flow = FrequencyOptimalFlow(self.dim, self.sigma_n)

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
        n = self.optimal_flow.n
        fim = torch.pow(2 * math.pi * n * torch.sin(2 * math.pi * theta * n), 2.0).sum().reshape([1, 1]) / (
                self.sigma_n ** 2)
        return torch.linalg.inv(fim)


if __name__ == '__main__':
    import gcrb

    dm = FrequencyModel(10, 1.0)
    theta_array = dm.parameter_range(1000)
    print(theta_array)
    model_opt = dm.get_optimal_model()
    crb_list = [dm.crb(theta) for theta in theta_array]
    # gcrb_list = [gcrb.adaptive_sampling_gfim(model_opt, theta.reshape([1])) for theta in theta_array]

    theta_array = theta_array.numpy()
    crb_array = torch.stack(crb_list).flatten().numpy()
    # gcrb_array = torch.stack(gcrb_list).flatten().numpy()
    from matplotlib import pyplot as plt
    from analysis.analysis_helpers import db
    plt.plot(theta_array, crb_array)
    # plt.plot(theta_array, 1/gcrb_array)
    plt.show()
    # print("a")
