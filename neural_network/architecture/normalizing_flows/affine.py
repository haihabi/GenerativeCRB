import torch
from neural_network.architecture.normalizing_flows.base_flow import FlowLayer


class Permute(FlowLayer):
    """
    Permutation features along the channel dimension
    """

    def __init__(self, num_channels, mode='shuffle'):
        """
        Constructor
        :param num_channel: Number of channels
        :param mode: Mode of permuting features, can be shuffle for
        random permutation or swap for interchanging upper and lower part
        """
        super().__init__()
        self.mode = mode
        self.num_channels = num_channels
        if self.mode == 'shuffle':
            perm = torch.randperm(self.num_channels)
            inv_perm = torch.empty_like(perm).scatter_(dim=0, index=perm,
                                                       src=torch.arange(self.num_channels))
            self.register_buffer("perm", perm)
            self.register_buffer("inv_perm", inv_perm)

    def forward(self, z, cond=None):
        if self.mode == 'shuffle':
            z = z[:, self.perm, ...]
        elif self.mode == 'swap':
            z1 = z[:, :self.num_channels // 2, ...]
            z2 = z[:, self.num_channels // 2:, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError('The mode ' + self.mode + ' is not implemented.')
        log_det = 0
        return z, log_det

    def inverse(self, z, cond=None):
        if self.mode == 'shuffle':
            z = z[:, self.inv_perm, ...]
        elif self.mode == 'swap':
            z1 = z[:, :(self.num_channels + 1) // 2, ...]
            z2 = z[:, (self.num_channels + 1) // 2:, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError('The mode ' + self.mode + ' is not implemented.')
        log_det = 0
        return z, log_det


class AffineCoupling(FlowLayer):
    """Affine Transformer.
    """

    def __init__(self, dim, mlp_addition, mlp_scale, scale_function='exp'):
        """
        Args:
            eps (float): lower-bound for scale parameter.
        """

        super().__init__()
        self.dim = dim
        self.dim_split = dim // 2
        self.mlp_add = mlp_addition
        self.mlp_scale = mlp_scale
        self.scale_function = scale_function

    def split(self, x):
        x_a, x_b = x[:, :self.dim_split], x[:, self.dim_split:]
        return x_a, x_b

    def get_s_t_parameters(self, b_part, cond=None):
        if cond is None:
            x = b_part
        else:
            x = torch.cat([b_part, cond], dim=-1)

        t = self.mlp_add(x)
        logs = self.mlp_scale(x)
        return logs, t

    def forward(self, x, cond=None):
        x_a, x_b = self.split(x)
        logs, t = self.get_s_t_parameters(x_b, cond)
        if self.scale_function == 'sigmoid':
            s = torch.sigmoid(logs + 2)
            log_det = -torch.sum(torch.log(s))
            y_a = x_a / s + t
        else:
            log_det = torch.sum(logs)
            s = torch.exp(logs)

            y_a = s * x_a + t
        y_b = x_b
        return torch.cat([y_a, y_b], dim=-1), log_det

    def invert(self, y, cond=None):
        y_a, y_b = self.split(y)
        logs, t = self.get_s_t_parameters(y_b, cond)
        if self.scale_function == 'sigmoid':
            s = torch.sigmoid(logs + 2)
            log_det = torch.sum(torch.log(s))
            x_a = (y_a - t) * s
        else:
            log_det = torch.sum(logs)
            s = -torch.exp(logs)
            x_a = (y_a - t) / s
        x_b = y_b
        return torch.cat([x_a, x_b], dim=-1), log_det
