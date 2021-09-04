import torch
from torch import nn
import constants


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x, cond=None):
        m, _ = x.shape
        log_det = torch.zeros(m, device=constants.DEVICE)
        zs = [x]
        for flow in self.flows:
            # print(type(flow))
            x, ld = flow.forward(x, cond=cond)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z, cond=None):
        m, _ = z.shape
        log_det = torch.zeros(m, device=constants.DEVICE)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z, cond)
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, prior, flows, condition_network=None):
        super().__init__()
        self.prior = prior
        self.condition_network = condition_network
        self.flow = NormalizingFlow(flows)

    def forward(self, x, cond=None):
        if self.condition_network is not None:
            cond = self.condition_network(cond)
        zs, log_det = self.flow.forward(x, cond=cond)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z, cond=None):
        if self.condition_network is not None:
            cond = self.condition_network(cond)
        xs, log_det = self.flow.backward(z, cond)
        return xs, log_det

    def nll(self, x, cond=None):
        zs, prior_logprob, log_det = self(x, cond=cond)
        logprob = prior_logprob + log_det  # Log-likelihood (LL)
        return -logprob  # Negative LL

    def nll_mean(self, x, cond=None):
        return torch.mean(self.nll(x, cond))  # NLL

    def sample(self, num_samples, cond=None):
        z = self.prior.sample((num_samples,))
        xs, _ = self.backward(z, cond)
        return xs

    def sample_nll(self, num_samples, cond=None):
        y = self.sample(num_samples, cond=cond)[-1]
        y = y.detach()
        return self.nll(y, cond)
