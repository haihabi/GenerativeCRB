import torch
from torch import nn
import constants
import numpy as np


class NormalPrior(object):
    """Prior for a standard Normal distribution."""

    def __init__(self, dim):
        self.dim = dim

    def sample(self, n):
        return torch.randn(n, self.dim, device=constants.DEVICE)

    def nll(self, u):
        return .5 * (self.dim * np.log(2 * np.pi) + (u ** 2).sum(dim=1))


class FlowLayer(nn.Module):
    def forward(self, t, cond=None):
        raise NotImplemented

    def invert(self, u, cond=None):
        pass


class SequentialFlow(FlowLayer):
    def __init__(self, flow_steps_list):
        super(SequentialFlow, self).__init__()
        self.flow_steps_list = flow_steps_list
        for i, fs in enumerate(self.flow_steps_list):
            self.add_module(f'flow_step_{i}', fs)

    def forward(self, t, cond=None):
        ld = 0
        for flow_step in self.flow_steps_list:
            t, ld_ = flow_step(t, cond)
            ld += ld_
        return t, ld

    def invert(self, u, cond=None):
        ld = 0
        for flow_step in reversed(self.flow_steps_list):
            u, ld_ = flow_step(u)
            ld += ld_
        return u, ld


class NormalizingFlow(SequentialFlow):
    def __init__(self, dim, flow_steps_list, flow_prior=NormalPrior):
        super(NormalizingFlow, self).__init__(flow_steps_list)
        self.dim = dim
        self.prior = flow_prior(dim)

    def nll(self, t, cond=None):
        u, log_det = self(t, cond)
        return self.prior.nll(u) - log_det
