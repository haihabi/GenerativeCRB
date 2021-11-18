from torch import nn

import torch


class FisherInformationMatrixCollector(nn.Module):
    def __init__(self, m_parameters):
        super().__init__()

        self.score_sum = nn.Parameter(torch.zeros(m_parameters), requires_grad=False)
        self.fim_mean = nn.Parameter(torch.zeros(m_parameters, m_parameters), requires_grad=False)
        self.fim_mean_p2 = nn.Parameter(torch.zeros(m_parameters, m_parameters), requires_grad=False)
        self.i = 0
        self.score_norm_list = []

    def append_fim(self, batch_fim, batch_score_vector):
        with torch.no_grad():
            index = torch.logical_not(torch.any(torch.any(batch_fim.isnan(), dim=2), dim=1))
            batch_fim = batch_fim[index, :]  # Clear non values
            batch_score_vector = torch.squeeze(batch_score_vector[index, :], dim=1)
            self.score_norm_list.append(torch.sum(torch.pow(batch_score_vector, 2.0), dim=-1))
            self.score_sum += batch_score_vector.sum(dim=0)
            if batch_fim.shape[0] > 0:
                self.i += batch_fim.shape[0]
                self.fim_mean += batch_fim.sum(dim=0)
                self.fim_mean_p2 += torch.pow(batch_fim, 2.0).sum(dim=0)
        return batch_fim.shape[0]

    def calculate_score_mean(self):
        return self.score_sum / self.i

    def calculate_final_fim(self):
        mean_correction = torch.matmul(torch.unsqueeze(self.calculate_score_mean(), dim=-1),
                                       torch.unsqueeze(self.calculate_score_mean(), dim=-1).T)
        return self.mean - mean_correction

    def calculate_score_norm(self):
        return torch.sqrt(torch.cat(self.score_norm_list).mean() - torch.pow(self.calculate_score_mean(), 2.0).sum())

    @property
    def size(self):
        return self.i

    @property
    def mean(self):
        return self.fim_mean / self.i

