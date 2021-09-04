import torch
from torch import nn
from normalizing_flow.base_nets import MLP


class AffineConstantFlow(nn.Module):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """

    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None

    def forward(self, x, cond=None):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z, cond=None):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class AffineHalfFlow(nn.Module):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, dim, parity, net_class=MLP, nh=24, scale=True, shift=True, condition_vector_size=1):
        super().__init__()
        self.dim = dim
        self.parity = parity
        self.condition_vector_size = condition_vector_size
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        if scale:
            self.s_cond = net_class(self.condition_vector_size + (self.dim // 2), self.dim // 2, nh)
        if shift:
            self.t_cond = net_class(self.condition_vector_size + (self.dim // 2), self.dim // 2, nh)

    def forward(self, x, cond=None):
        x0, x1 = x[:, ::2], x[:, 1::2]
        if self.parity:
            x0, x1 = x1, x0
        if self.condition_vector_size > 0:
            s = self.s_cond(torch.cat([x0, cond], dim=-1))
            t = self.t_cond(torch.cat([x0, cond], dim=-1))
        else:
            s = self.s_cond(x0)
            t = self.t_cond(x0)

        z0 = x0  # untouched half
        z1 = torch.exp(s) * x1 + t  # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z, cond=None):
        z0, z1 = z[:, ::2], z[:, 1::2]
        if self.parity:
            z0, z1 = z1, z0
        if self.condition_vector_size > 0:
            s = self.s_cond(torch.cat([z0, cond], dim=-1))
            t = self.t_cond(torch.cat([z0, cond], dim=-1))
        else:
            s = self.s_cond(z0)
            t = self.t_cond(z0)
        x0 = z0  # this was the same
        x1 = (z1 - t) * torch.exp(-s)  # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class AffineInjector(nn.Module):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, dim, net_class=MLP, nh=24, scale=True, shift=True, condition_vector_size=1):
        super().__init__()
        self.dim = dim
        # self.parity = parity
        self.condition_vector_size = condition_vector_size
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim)
        if scale:
            self.s_cond = net_class(self.condition_vector_size, self.dim, nh)
        if shift:
            self.t_cond = net_class(self.condition_vector_size, self.dim, nh)

    def forward(self, x, cond=None):
        # x0, x1 = x[:, ::2], x[:, 1::2]
        # if self.parity:
        #     x0, x1 = x1, x0
        if self.condition_vector_size > 0:
            s = self.s_cond(cond)
            t = self.t_cond(cond)
        else:
            raise Exception("")

        # z0 = x0  # untouched half
        z = torch.exp(s) * x + t  # transform this half as a function of the other
        # if self.parity:
        #     z0, z1 = z1, z0
        # z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z, cond=None):
        # z
        if self.condition_vector_size > 0:
            s = self.s_cond(cond)
            t = self.t_cond(cond)
        else:
            raise Exception("")
        # x0 = z0  # this was the same
        x = (z - t) * torch.exp(-s)  # reverse the transform on this half
        # if self.parity:
        #     x0, x1 = x1, x0
        # x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det
