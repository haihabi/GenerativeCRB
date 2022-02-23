import os

from experiments import common
from tqdm import tqdm
from experiments import constants
import torch
import normflowpy as nf
from torch.distributions import MultivariateNormal


class BaseModel(object):
    def __init__(self, dim: int, theta_min, theta_max, theta_dim=1):
        self.theta_min = theta_min * torch.ones([1, theta_dim], device=constants.DEVICE)
        self.theta_max = theta_max * torch.ones([1, theta_dim], device=constants.DEVICE)
        self.dim = dim
        self.theta_dim = theta_dim

    @property
    def parameter_vector_length(self):
        return self.theta_dim

    @property
    def name(self) -> str:
        return f"{type(self).__name__}_{self.dim}_{self.theta_dim}"

    @property
    def model_name(self) -> str:
        return f"{type(self).__name__}_{self.dim}_{self.theta_dim}"

    def model_exist(self, folder):
        return os.path.isfile(os.path.join(folder, f"{self.model_name}_model.pt"))

    def parameter_range(self, n_steps, theta_scale_min=None, theta_scale_max=None):
        theta_min = self.theta_min if theta_scale_min is None else theta_scale_min * self.theta_min
        theta_max = self.theta_max if theta_scale_max is None else theta_scale_max * self.theta_max
        return theta_min + (theta_max - theta_min) * torch.linspace(0, 1, n_steps,
                                                                    device=constants.DEVICE).reshape(
            [-1, 1])

    def build_dataset(self, dataset_size):
        print("Start Dataset Generation")
        data = []
        label = []
        for _ in tqdm(range(dataset_size)):
            theta = self.theta_min + (self.theta_max - self.theta_min) * torch.rand([1, self.parameter_vector_length],
                                                                                    device=constants.DEVICE)
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
        return nf.NormalizingFlowModel(prior, [self._get_optimal_model()]).to(constants.DEVICE)

    def generate_data(self, n_samples, theta):
        raise NotImplemented

    def crb(self, param):
        raise NotImplemented
