import torch
import constants
import normflowpy as nfp
from torch import nn


class NoiseLevelFunction(nfp.ConditionalBaseFlowLayer):
    def __init__(self, m_iso=5, n_cam=5, trained_alpha=True):
        super().__init__()
        self.alpha = nn.Parameter(1e-6 * torch.ones(m_iso, n_cam) if trained_alpha else torch.zeros(m_iso, n_cam),
                                  requires_grad=trained_alpha)
        self.delta = nn.Parameter(torch.ones(m_iso, n_cam))

    def _build_scale(self, clean_image, iso, cam):
        iso_index = [constants.ISO2INDEX[i.item()] for i in iso]
        beta1 = torch.pow(self.alpha[iso_index, cam], 2.0).reshape([-1, 1, 1, 1])
        beta2 = torch.pow(self.delta[iso_index, cam], 2.0).reshape([-1, 1, 1, 1])
        return torch.sqrt(beta1 * clean_image + beta2)

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
