import torch
from torch import nn
from neural_network.architecture.normalizing_flows.base_flow import FlowLayer


class BatchNormFlow(FlowLayer):
    """Perform BatchNormalization as a Flow class.
    If not affine, just learns batch statistics to normalize the input.
    """

    @property
    def affine(self):
        return self._affine.item()

    def __init__(self, dim, affine=True, momentum=.1, eps=1e-5):
        """
        Args:
            affine (bool): whether to learn parameters loc/scale.
            momentum (float): value used for the moving average
                of batch statistics. Must be between 0 and 1.
            eps (float): lower-bound for the scale tensor.
        """
        super().__init__()
        self.dim = dim

        assert 0 <= momentum and momentum <= 1

        self.register_buffer('eps', torch.tensor(eps))
        self.register_buffer('momentum', torch.tensor(momentum))

        self.register_buffer('updates', torch.tensor(0))

        self.register_buffer('batch_loc', torch.zeros(1, self.dim))
        self.register_buffer('batch_scale', torch.ones(1, self.dim))

        assert isinstance(affine, bool)
        self.register_buffer('_affine', torch.tensor(affine))

        self.loc = nn.Parameter(torch.zeros(1, self.dim))
        self.log_scale = nn.Parameter(torch.zeros(1, self.dim))

    def _activation(self, x=None):
        if self.training:
            assert x is not None and x.size(0) >= 2, \
                'If training BatchNorm, pass more than 1 sample.'

            bloc = x.mean(0, keepdim=True)
            bscale = x.std(0, keepdim=True) + self.eps

            # Update self.batch_loc, self.batch_scale
            with torch.no_grad():
                if self.updates.data == 0:
                    self.batch_loc.data = bloc
                    self.batch_scale.data = bscale
                else:
                    m = self.momentum
                    self.batch_loc.data = (1 - m) * self.batch_loc + m * bloc
                    self.batch_scale.data = \
                        (1 - m) * self.batch_scale + m * bscale

                self.updates += 1
        else:
            bloc, bscale = self.batch_loc, self.batch_scale

        loc, scale = self.loc, self.log_scale

        scale = torch.exp(scale) + self.eps
        # Note that batch_scale does not use activation,
        # since it is already in scale units.

        return bloc, bscale, loc, scale

    def _log_det(self, bscale):
        if self.affine:
            return (self.log_scale - torch.log(bscale)).sum(dim=1)
        else:
            return -torch.log(bscale).sum(dim=1)

    def forward(self, x, cond=None):
        bloc, bscale, loc, scale = self._activation(x)
        u = (x - bloc) / bscale
        if self.affine:
            u = u * scale + loc
        log_det = self._log_det(bscale)
        return u, log_det

    def invert(self, u, cond=None):
        assert not self.training, (
            'If using BatchNorm in reverse training mode, '
            'remember to call it reversed: inv_flow(BatchNorm)(dim=dim)'
        )

        bloc, bscale, loc, scale = self._activation()
        if self.affine:
            x = (u - loc) / scale * bscale + bloc
        else:
            x = u * bscale + bloc

        log_det = -self._log_det(bscale)
        return x, log_det
