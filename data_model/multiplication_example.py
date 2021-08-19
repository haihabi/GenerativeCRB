import torch
import common
import constants
import numpy as np
from tqdm import tqdm


class MultiplicationModel(object):
    def __init__(self, dim: int, theta_min: float, theta_max: float):
        self.dim = dim
        self.theta_min = theta_min
        self.theta_max = theta_max

    @property
    def parameter_vector_length(self):
        return 1

    def pdf(self, r, theta):
        scale = 3 * np.sqrt(2 * np.pi)
        r23 = np.power(np.power(r, 2.0), 1 / 3)
        theta_r_factor = np.power(theta, -1 / 3) / r23
        exp_value = np.exp(-0.5 * r23 / (2 * np.power(theta, 2 / 3)))
        return exp_value * theta_r_factor / scale

    def parameter_range(self, n_steps):
        return self.theta_min + (self.theta_max - self.theta_min) * torch.linspace(0, 1, n_steps,
                                                                                   device=constants.DEVICE)

    def generate_data(self, n_samples, theta):
        return torch.pow(torch.randn([n_samples, self.dim], device=constants.DEVICE), 3.0) * theta

    def ml_estimator(self, r):
        return torch.pow(torch.mean(torch.pow(torch.abs(r), 2 / 3), dim=1), 3 / 2)

    def crb(self, theta):
        return torch.pow(theta, 2.0) * 9 / (2 * self.dim)  # Check CRB in the case of dim>1

    def build_dataset(self, dataset_size):
        print("Start Dataset Generation")
        data = []
        label = []
        for _ in tqdm(range(dataset_size)):
            theta = self.theta_min + (self.theta_max - self.theta_min) * torch.rand([1, 1], device=constants.DEVICE)
            signal = self.generate_data(1, theta)

            data.append(signal.detach().cpu().numpy().flatten())

            label.append(theta.detach().cpu().numpy().flatten())

        return common.NumpyDataset(data, label)
