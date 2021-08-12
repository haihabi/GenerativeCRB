import torch
import common
import constants
from tqdm import tqdm


class MultiplicationModel(object):
    def __init__(self, dim: int, theta_min: float, theta_max: float):
        self.dim = dim
        self.theta_min = theta_min
        self.theta_max = theta_max

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
