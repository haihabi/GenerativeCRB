import numpy as np
from torch.utils.data.dataset import Dataset


class NumpyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.n = len(data)

    def __getitem__(self, index):
        d = self.data[index]
        l = self.label[index]
        return d, l

    def __len__(self):
        return self.n

    def get_second_order_stat(self):
        x = np.stack(self.data)
        return np.mean(x, axis=0), np.std(x, axis=0)
