import torch
from torch import nn


class FisherInformationMatrixCollector(nn.Module):
    def __init__(self, m_parameters):
        super().__init__()
        self.fim_sum = nn.Parameter(torch.zeros(m_parameters, m_parameters), requires_grad=False)
        self.i = 0
        self.score_norm_list = []

    def append_fim(self, batch_fim, batch_score_vector):
        with torch.no_grad():
            index = torch.logical_not(
                torch.any(torch.any(torch.logical_or(batch_fim.isinf(), batch_fim.isnan()), dim=2), dim=1))
            batch_fim = batch_fim[index, :]  # Clear non values

            if batch_fim.shape[0] > 0:
                self.i += batch_fim.shape[0]
                self.fim_sum += batch_fim.sum(dim=0)

        return batch_fim.shape[0]

    def calculate_final_fim(self):
        return self.mean

    @property
    def size(self):
        return self.i

    @property
    def mean(self):
        return self.fim_sum / self.i
