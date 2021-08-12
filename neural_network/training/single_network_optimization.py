import torch


class SingleNetworkOptimization(object):
    def __init__(self, network: torch.nn.Module):
        self.opt = torch.optim.SGD(network.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
