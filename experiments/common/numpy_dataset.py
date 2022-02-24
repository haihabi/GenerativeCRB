import numpy as np
from torch.utils.data.dataset import Dataset


class NumpyDataset(Dataset):
    def __init__(self, data, label, transform):
        self.data = data
        self.label = label
        self.transform = transform
        self.n = len(data)

    def __getitem__(self, index):
        d = self.data[index]
        if self.transform is not None:
            d = self.transform(d)
        l = self.label[index]
        return d, l

    def __len__(self):
        return self.n

    def get_min_max_vector(self):
        x = np.stack(self.data)
        return np.min(x, axis=0).astype("float32"), (np.max(x, axis=0) - np.min(x, axis=0)).astype("float32")
