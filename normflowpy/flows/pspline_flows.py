"""
Neural Spline Flows, coupling and autoregressive

Paper reference: Durkan et al https://arxiv.org/abs/1906.04032
Code reference: slightly modified https://github.com/tonyduan/normalizing-flows/blob/master/nf/flows.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from normflowpy.flows.spline_flows import unconstrained_RQS


class PNSF(nn.Module):
    """ A Modified version of  Neural spline flow, coupling layer, [Durkan et al. 2019] where the spline parameters
        are constant and not data dependant.
    """

    def __init__(self, dim, K=5, B=3):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.p1 = nn.Parameter(torch.randn((3 * K - 1) ), requires_grad=True)
        self.p2 = nn.Parameter(torch.randn((3 * K - 1) ), requires_grad=True)

    def forward(self, x, cond=None):
        log_det = torch.zeros(x.shape[0], device=x.device)
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]
        # out = self.f1(lower).reshape(-1, self.dim // 2, 3 * self.K - 1)
        w1 = self.p1.reshape([1, -1]).repeat([x.shape[0], 1]).reshape(-1, 1, 3 * self.K - 1).repeat(
            [1, self.dim // 2, 1])
        W, H, D = torch.split(w1, self.K, dim=2)
        W, H = torch.softmax(W, dim=2), torch.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(upper, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)

        # out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        w2 = self.p2.reshape([1, -1]).repeat([x.shape[0], 1]).reshape(-1, 1, 3 * self.K - 1).repeat(
            [1, self.dim // 2, 1])
        W, H, D = torch.split(w2, self.K, dim=2)
        W, H = torch.softmax(W, dim=2), torch.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(lower, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)

        return torch.cat([lower, upper], dim=1), log_det

    def backward(self, z, cond=None):
        log_det = torch.zeros(z.shape[0], device=x.device)
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]
        w2 = self.p2.reshape([1, -1]).repeat([x.shape[0], 1]).reshape(-1, 1, 3 * self.K - 1).repeat(
            [1, self.dim // 2, 1])
        W, H, D = torch.split(w2, self.K, dim=2)
        W, H = torch.softmax(W, dim=2), torch.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(lower, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)
        w1 = self.p1.reshape([1, -1]).repeat([x.shape[0], 1]).reshape(-1, 1, 3 * self.K - 1).repeat(
            [1, self.dim // 2, 1])
        W, H, D = torch.split(w1, self.K, dim=2)
        W, H = torch.softmax(W, dim=2), torch.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(upper, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)
        return torch.cat([lower, upper], dim=1), log_det
