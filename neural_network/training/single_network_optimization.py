import torch
from enum import Enum


class OptimizerType(Enum):
    SGD = 0
    Adam = 1


class SingleNetworkOptimization(object):
    def __init__(self, network: torch.nn.Module, n_epochs: int,
                 lr=1e-4, weight_decay=1e-3, optimizer_type: OptimizerType = OptimizerType.SGD):
        self.n_epochs = n_epochs
        self.optimizer_type = optimizer_type
        if self.optimizer_type == OptimizerType.SGD:
            self.opt = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.0, nesterov=False,
                                       weight_decay=weight_decay)
        elif self.optimizer_type == OptimizerType.Adam:
            self.opt = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplemented
