class BatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1.0e-5, affine=True):
        super(BatchNorm, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.dimensions = [1] + [1 for _ in num_features]
        self.dimensions[1] = num_features[0]
        log_gamma = torch.zeros(self.dimensions)
        beta = torch.zeros(self.dimensions)
        if affine:
            self.register_parameter('log_gamma', nn.Parameter(log_gamma))
            self.register_parameter('beta', nn.Parameter(beta))
        else:
            self.register_buffer('log_gamma', log_gamma)
            self.register_buffer('beta', beta)

        self.register_buffer('running_mean', torch.zeros(self.dimensions))
        self.register_buffer('running_var', torch.ones(self.dimensions))
        self.register_buffer('batch_mean', torch.zeros(self.dimensions))
        self.register_buffer('batch_var', torch.ones(self.dimensions))

    def forward(self, x, log_det_jacob):
        if self.training:
            x_reshape = x.view(x.size(0), self.num_features[0], -1)
            x_mean = torch.mean(x_reshape, dim=[0, 2], keepdim=True)
            x_var = torch.mean((x_reshape - x_mean).pow(2), dim=[0, 2], keepdim=True) + self.eps
            self.batch_mean.data.copy_(x_mean.view(self.dimensions))
            self.batch_var.data.copy_(x_var.view(self.dimensions))

            self.running_mean.mul_(1.0 - self.momentum)
            self.running_var.mul_(1.0 - self.momentum)
            self.running_mean.add_(self.batch_mean.detach() * self.momentum)
            self.running_var.add_(self.batch_var.detach() * self.momentum)

            mean, var = self.batch_mean, self.batch_var
        else:
            mean, var = self.running_mean, self.running_var

        x = (x - mean) / torch.sqrt(var)
        x = x * torch.exp(self.log_gamma) + self.beta

        num_pixels = np.prod(x.size()) // (x.size(0) * x.size(1))
        log_det = self.log_gamma - 0.5 * torch.log(var)
        log_det_jacob = torch.sum(log_det) * num_pixels

        return x, log_det_jacob

    def backward(self, x, log_det_jacob):
        if self.training:
            mean, var = self.batch_mean, self.batch_var
        else:
            mean, var = self.running_mean, self.running_var

        x = (x - self.beta) / torch.exp(self.log_gamma)
        x = x * torch.sqrt(var) + mean

        num_pixels = np.prod(x.size()) // (x.size(0) * x.size(1))
        log_det = -self.log_gamma + 0.5 * torch.log(var)
        log_det_jacob += torch.sum(log_det) * num_pixels

        return x, log_det_jacob
