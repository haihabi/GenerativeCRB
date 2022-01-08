import torch


def change2tensor(x):
    if isinstance(x, torch.Tensor): # If is tensor return
        return x
    if isinstance(x, (float, int)):  # Change float to tensor
        x = [x]
    return torch.Tensor(x)
