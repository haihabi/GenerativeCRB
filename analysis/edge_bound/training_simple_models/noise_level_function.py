import torch

import normflowpy as nfp
from torch import nn

ISO2INDEX = {100: 0,
             400: 1,
             800: 2,
             1600: 3,
             3200: 4}


class NoiseLevelFunction(nfp.ConditionalBaseFlowLayer):
    def __init__(self, m_iso=5, n_cam=5):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(m_iso, n_cam), requires_grad=True)
        self.delta = nn.Parameter(torch.ones(m_iso, n_cam))

    def _build_scale(self, clean_image, iso, cam):
        iso_index = ISO2INDEX[iso]
        clean_sqrt = torch.sqrt(clean_image)
        beta1 = torch.abs(self.alpha[iso_index, cam]) * clean_sqrt
        return torch.sqrt(beta1 * clean_image + torch.abs(self.delta))

    def forward(self, x, cond):
        clean_image = cond[0]
        iso = cond[1]
        cam = cond[2]
        scale = self._build_scale(clean_image, iso, cam)
        return x / scale, -torch.sum(torch.log(scale), dim=[1, 2, 3])

    def backward(self, z, cond):
        clean_image = cond[0]
        iso = cond[1]
        cam = cond[2]
        scale = self._build_scale(clean_image, iso, cam)
        return z * scale, torch.sum(torch.log(scale), dim=[1, 2, 3])
