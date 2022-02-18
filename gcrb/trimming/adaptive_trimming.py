import torch
import numpy as np
from torch import nn
from enum import Enum

TNAME = "trimming_mean"
BMAX = "trimming_b_max"
BP = "trimming_b_percentile"


class TrimmingType(Enum):
    ALL = 0  # No Trimming at all
    MAX = 1
    PERCENTILE = 2


def calculate_trimming_parameters(in_data, p=0.99):
    in_data = np.reshape(in_data, [in_data.shape[0], -1])  # Assumption that first axis is batch
    data_mean = np.mean(in_data, axis=0, keepdims=True)
    norm_array = np.linalg.norm(in_data - data_mean, axis=1)
    b_max = np.max(norm_array)
    b_p = np.percentile(norm_array, p * 100)
    return TrimmingParameters(data_mean, b_max, b_p)


class TrimmingParameters:
    def __init__(self, mean, b_max, b_percentile):
        self.mean = mean
        self.b_max = b_max
        self.b_percentile = b_percentile

    def as_dict(self) -> dict:
        return {TNAME: self.mean,
                BMAX: self.b_max,
                BP: self.b_percentile}

    @staticmethod
    def from_dict(in_dict: dict):
        return TrimmingParameters(in_dict[TNAME], in_dict[BMAX], in_dict[BP])


class AdaptiveTrimming(torch.nn.Module):
    def __init__(self, trimming_parameters: TrimmingParameters, trimming_type: TrimmingType):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(trimming_parameters.mean), requires_grad=False)
        self.b_max = nn.Parameter(torch.tensor(trimming_parameters.b_max), requires_grad=False)
        self.b_percentile = nn.Parameter(torch.tensor(trimming_parameters.b_percentile), requires_grad=False)
        self.trimming_type = trimming_type

    def get_upper_bound(self):
        if self.trimming_type == TrimmingType.MAX:
            return self.b_max
        elif self.trimming_type == TrimmingType.PERCENTILE:
            return self.b_percentile
        else:
            raise Exception("Unkowen upper bound type")

    def forward(self, in_gamma_tensor: torch.Tensor) -> torch.Tensor:
        m_tilde = in_gamma_tensor.shape[0]  # M tilde is the current batch-size
        if self.trimming_type == TrimmingType.ALL:
            return torch.ones(m_tilde).to(in_gamma_tensor.device).bool()
        gamma_tensor = torch.reshape(in_gamma_tensor, [m_tilde, -1])
        center_gamma_tensor = gamma_tensor - self.mean
        norm_array = torch.linalg.norm(center_gamma_tensor, dim=1)
        state = norm_array < self.get_upper_bound()
        return state
