import torch
from torch import nn


class Invertible1x1Conv(nn.Module):
    """
    As introduced in Glow paper.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = P  # remains fixed during optimization
        self.L = nn.Parameter(L)  # lower triangular portion
        self.S = nn.Parameter(U.diag())  # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1))  # "crop out" diagonal, stored in S

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x, cond=None):
        W = self._assemble_W()
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def backward(self, z, cond=None):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det
