import common
from tqdm import tqdm
import constants
import torch


class BaseModel(object):
    def __init__(self, theta_min: float, theta_max: float):
        self.theta_min = theta_min
        self.theta_max = theta_max

    def save_data_model(self, folder):
        pass

    def load_data_model(self, folder):
        pass

    def get_optimal_model(self):
        raise NotImplemented

    def generate_data(self, n_samples, theta):
        raise NotImplemented

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
