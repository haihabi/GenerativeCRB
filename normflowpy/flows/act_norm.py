import torch
from normflowpy.flows.affine import AffineConstantFlow
from torch import nn


class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False

    def forward(self, x, cond=None):
        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None  # for now
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        return super().forward(x)


class InputNorm(nn.Module):
    def __init__(self, mu, std):
        super().__init__()
        self.t = nn.Parameter(mu, requires_grad=False)
        self.s = nn.Parameter(std, requires_grad=False)

    def forward(self, x, cond=None):
        z = (x - self.t) / self.s
        log_det = -torch.sum(torch.log(self.s), dim=1)
        return z, log_det

    def backward(self, z, cond=None):
        x = self.s * z + self.t
        log_det = torch.sum(torch.log(self.s), dim=1)
        return x, log_det
