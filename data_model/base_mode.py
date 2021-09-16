import os

import common
from tqdm import tqdm
import constants
import torch
import normalizing_flow as nf
from torch.distributions import MultivariateNormal


class BaseModel(object):
    def __init__(self, dim: int, theta_min: float, theta_max: float):
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.dim = dim

    @property
    def parameter_vector_length(self):
        return 1

    @property
    def name(self) -> str:
        return f"{type(self).__name__}_{self.dim}"

    @property
    def model_name(self) -> str:
        return f"{type(self).__name__}_{self.dim}"

    def model_exist(self,folder):
        return os.path.isfile(os.path.join(folder, f"{self.model_name}_model.pt"))

    def parameter_range(self, n_steps):
        return self.theta_min + (self.theta_max - self.theta_min) * torch.linspace(0, 1, n_steps,
                                                                                   device=constants.DEVICE)

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

    def save_data_model(self, folder):
        pass

    def load_data_model(self, folder):
        pass

    def _get_optimal_model(self):
        raise NotImplemented

    def get_optimal_model(self):
        prior = MultivariateNormal(torch.zeros(self.dim, device=constants.DEVICE),
                                   torch.eye(self.dim, device=constants.DEVICE))
        return nf.NormalizingFlowModel(prior, [self._get_optimal_model()])

    def generate_data(self, n_samples, theta):
        raise NotImplemented

    def crb(self, param):
        raise NotImplemented
