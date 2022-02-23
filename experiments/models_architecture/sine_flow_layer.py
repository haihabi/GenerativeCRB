import torch
import normflowpy as nfp
from torch import nn
import math


class SineFlowLayer(nfp.ConditionalBaseFlowLayer):
    def __init__(self, x_shape):
        super().__init__()
        self.n_samples = x_shape[0]
        self.n = nn.Parameter(torch.linspace(0, self.n_samples - 1, self.n_samples).reshape([1, -1]),
                              requires_grad=False)

    def _generate_sine(self, cond):
        a = cond[:, 0].reshape([-1, 1])
        phase = cond[:, 2].reshape([-1, 1])
        f_0 = cond[:, 1].reshape([-1, 1])
        out = a * torch.cos(2 * math.pi * f_0 * self.n.reshape([1, -1]) + phase)
        return out

    def forward(self, x, cond):
        a = self._generate_sine(cond)
        return x - a, 0

    def backward(self, z, cond):
        a = self._generate_sine(cond)
        return a + z, 0
