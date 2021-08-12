from torch.utils.data.dataset import Dataset
from tqdm import tqdm


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
