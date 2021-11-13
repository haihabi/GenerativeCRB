from torch import nn

import torch
from scipy.optimize import minimize, NonlinearConstraint
import numpy as np


def cons_f(v):
    return np.power(v, 2.).sum()


nonlinear_constraint = NonlinearConstraint(cons_f, 1, 1)  # Unit sphere constraint


def empirical_v_sphere_orlicz_norm(in_x):
    if isinstance(in_x, torch.Tensor):
        in_x = in_x.detach().cpu().numpy()
    in_x = np.squeeze(in_x, axis=1)
    m = in_x.shape[-1]

    def _loss(v):
        x_hat = np.sum(v.reshape([1, -1]) * in_x, axis=1)
        x_hat2 = np.power(x_hat, 2.0)
        x_hat2_max = np.max(x_hat2)
        mean_exp = np.mean(np.exp(x_hat2 - x_hat2_max)) / 2
        val_log = x_hat2_max + np.log(mean_exp)
        val_log = 0 if val_log < 0 else val_log
        return -np.sqrt(val_log)

    x0 = np.ones(m) / np.sqrt(m)
    res = minimize(_loss, x0, method='trust-constr',
                   constraints=[nonlinear_constraint])
    return res.x


class FisherInformationMatrixCollector(nn.Module):
    def __init__(self, m_parameters):
        super().__init__()
        self.score_exp_sum = nn.Parameter(torch.zeros(m_parameters), requires_grad=False)
        self.fim_mean = nn.Parameter(torch.zeros(m_parameters, m_parameters), requires_grad=False)
        self.fim_mean_p2 = nn.Parameter(torch.zeros(m_parameters, m_parameters), requires_grad=False)
        self.i = 0
        self.v = nn.Parameter(torch.zeros(m_parameters), requires_grad=False)
        self.v_set = False
        self.score_v_list = []
        self.score_norm_list = []

    def append_fim(self, batch_fim):
        with torch.no_grad():
            batch_fim = batch_fim[torch.logical_not(torch.any(torch.any(batch_fim.isnan(), dim=2), dim=1)),
                        :]  # Clear non values
            if batch_fim.shape[0] > 0:
                self.i += batch_fim.shape[0]
                self.fim_mean += batch_fim.sum(dim=0)
                self.fim_mean_p2 += torch.pow(batch_fim, 2.0).sum(dim=0)
        return batch_fim.shape[0]

    def append_score(self, batch_score_vector):
        with torch.no_grad():

            score_vector = torch.squeeze(batch_score_vector, dim=1)

            self.score_norm_list.append(torch.sum(torch.pow(score_vector, 2.0), dim=-1))

    @property
    def size(self):
        return self.i

    @property
    def mean(self):
        return self.fim_mean / self.i

    @property
    def power(self):
        return self.fim_mean_p2 / self.i

    @property
    def varinace(self):
        return self.power - torch.pow(self.mean, 2.0)
