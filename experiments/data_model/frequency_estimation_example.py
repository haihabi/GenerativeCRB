import torch
from experiments import constants
from torch import nn
from experiments.data_model.base_mode import BaseModel
import os
import math
from matplotlib import pyplot as plt
import numpy as np

MINFREQ = 0.01
MAXFREQ = 0.49
PHASEMIN = 0
PHASEMAX = 2 * math.pi
FREQDELTA = 0.01


def plot_spectrum(x, eps=1e-7):
    assert len(x.shape) == 1
    x_fft = torch.fft.fft(x)
    x_fft_abs = torch.abs(x_fft).cpu().detach().numpy()
    x_fft_angle = torch.angle(x_fft).cpu().detach().numpy()
    f_dig = torch.linspace(-math.pi, math.pi, x.shape[0]).cpu().detach().numpy()
    plt.subplot(2, 1, 1)
    plt.plot(f_dig, 10 * np.log10(x_fft_abs + eps))
    plt.ylabel("dB")
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(f_dig, x_fft_angle)
    plt.grid()
    plt.show()


def power_law_noise(batch_size, dim, scale, h):
    x = torch.randn([batch_size, dim])
    x_fft = torch.fft.fft(x)
    f_dig = torch.linspace(-math.pi, math.pi, dim)
    f_dig2 = torch.pow(f_dig, 2)
    f_dig3 = torch.pow(f_dig, 3)
    f_dig4 = torch.pow(f_dig, 4)
    scale_spectum = scale * ((h[4] / f_dig4) + (h[3] / f_dig3) + (h[2] / f_dig2) + (h[1] / f_dig) + h[0])
    scale_spectum = scale_spectum.reshape([1, -1])
    x_fft_pl = x_fft * scale_spectum
    return torch.fft.ifft(x_fft_pl).real


class FrequencyOptimalFlow(nn.Module):
    def __init__(self, n_samples, sigma_n):
        super().__init__()
        self.sigma_n = sigma_n
        self.n_samples = n_samples
        self.n = nn.Parameter(
            torch.linspace(0, self.n_samples - 1, self.n_samples, device=constants.DEVICE).reshape([1, -1]),
            requires_grad=False)

    def _sine(self, amp, f_0, phase):
        return amp.reshape([-1, 1]) * torch.cos(
            2 * math.pi * f_0.reshape([-1, 1]) * self.n.reshape([1, -1]) + phase.reshape([-1, 1]))

    def forward(self, x, cond=None):
        z_tilde = x - self._sine(cond[:, 0], cond[:, 1], cond[:, 2])
        z = z_tilde / self.sigma_n
        return z, 0

    def backward(self, z, cond=None):
        x = self._sine(cond[:, 0], cond[:, 1], cond[:, 2]) + z * self.sigma_n
        return x, 0


class FrequencyComplexModel(nn.Module):
    def __init__(self, n_samples, sigma_n, quantization_enable=False, q_delta=0.1, q_threshold=1):
        super().__init__()
        self.sigma_n = sigma_n
        self.n_samples = n_samples
        self.n = nn.Parameter(
            torch.linspace(0, self.n_samples - 1, self.n_samples, device=constants.DEVICE).reshape([1, -1]),
            requires_grad=False)
        self.quantization_enable = quantization_enable
        self.q_delta = q_delta
        self.q_threshold = q_threshold

    def quantization(self, x):
        if self.quantization_enable:
            return torch.clamp(self.q_delta*torch.round(x / self.q_delta), -self.q_threshold, self.q_threshold)

        return x

    def _sine(self, amp, f_0, phase, amp_noise, phase_noise):
        x_base = torch.cos(
            2 * math.pi * f_0.reshape([-1, 1]) * self.n.reshape([1, -1]) + phase.reshape([-1, 1]) + phase_noise)
        return (amp.reshape([-1, 1]) + amp_noise) * x_base

    def forward(self, cond):
        batch_size = cond.shape[0]
        amp_noise = 0
        phase_noise = 0
        white_noise = self.sigma_n * torch.randn([batch_size, self.n_samples])
        return self.quantization(self._sine(cond[:, 0], cond[:, 1], cond[:, 2], amp_noise, phase_noise) + white_noise)


class FrequencyModel(BaseModel):
    def __init__(self, dim: int, sigma_n, phase_noise=False, quantization=False, addition_noise_type=None, amp_min=0.8,
                 amp_max=1.2, phase_noise_scale=0.01):
        theta_min = [amp_min, 0.0 + FREQDELTA, PHASEMIN]
        theta_max = [amp_max, 0.5 - FREQDELTA, PHASEMAX]
        super().__init__(dim, torch.tensor(theta_min).reshape([1, -1]), torch.tensor(theta_max).reshape([1, -1]),
                         theta_dim=len(theta_min))
        self.sigma_n = sigma_n
        self.phase_noise = phase_noise
        self.quantization = quantization
        self.addition_noise_type = addition_noise_type
        self.is_optimal_exists = not (self.quantization or self.phase_noise or self.addition_noise_type is not None)
        if self.is_optimal_exists:
            self.optimal_flow = FrequencyOptimalFlow(self.dim, self.sigma_n)
        else:
            pass

    @property
    def name(self) -> str:
        return f"{super().name}_{self.sigma_n}"  # Append Sigma N to Name

    def _get_optimal_model(self):
        if self.is_optimal_exists:
            return self.optimal_flow
        return None

    def save_data_model(self, folder):
        torch.save(self.optimal_flow.state_dict(), os.path.join(folder, f"{self.model_name}_model.pt"))

    def load_data_model(self, folder):
        data = torch.load(os.path.join(folder, f"{self.model_name}_model.pt"))
        self.optimal_flow.load_state_dict(data)

    def generate_data(self, n_samples, theta):
        cond = torch.ones([n_samples, 1], device=constants.DEVICE) * theta
        z = torch.randn([n_samples, self.dim], device=constants.DEVICE)
        return self.optimal_flow.backward(z, cond=cond)[0]

    @staticmethod
    def ml_estimator(r):
        return torch.pow(torch.mean(torch.pow(torch.abs(r), 6), dim=1), 1 / 6)

    def crb(self, theta):
        if self.is_optimal_exists:
            a = theta[0]
            f_0 = theta[1]
            phase = theta[2]
            n = self.optimal_flow.n
            one_over_sigma = 1 / (self.sigma_n ** 2)
            alpha = 2 * math.pi * f_0 * n + phase
            fim11 = one_over_sigma * torch.pow(torch.cos(alpha), 2.0).sum()
            fim12 = -a * math.pi * one_over_sigma * (torch.sin(2 * alpha) * n).sum()
            fim13 = -a * one_over_sigma * (torch.sin(2 * alpha)).sum() / 2
            fim22 = torch.pow(a * 2 * math.pi, 2.0) * one_over_sigma * (torch.pow(n * torch.sin(alpha), 2.0)).sum()
            fim23 = 2 * math.pi * torch.pow(a, 2.0) * one_over_sigma * (n * torch.pow(torch.sin(alpha), 2.0)).sum()
            fim33 = torch.pow(a, 2.0) * one_over_sigma * (torch.pow(torch.sin(alpha), 2.0)).sum()
            fim = torch.zeros([3, 3], device=theta.device)
            fim[0, 0] = fim11
            fim[0, 1] = fim12
            fim[1, 0] = fim12
            fim[2, 0] = fim13
            fim[0, 2] = fim13
            fim[1, 1] = fim22
            fim[1, 2] = fim23
            fim[2, 1] = fim23
            fim[2, 2] = fim33
            return torch.linalg.inv(fim)


if __name__ == '__main__':
    # h = [0, 0, 0, 1, 0]
    # pn = power_law_noise(1, 20, 1e-4, h)[0, :]
    # print(pn)
    fcm = FrequencyComplexModel(20, 0.1, quantization_enable=True, q_delta=1/2**7, q_threshold=1.0)
    cond = [1, 0.2, 0]
    cond = torch.tensor(cond).reshape([1, -1])
    x = fcm(cond)[0, :]
    plt.plot(x.cpu().numpy())
    plt.show()
    # pn = torch.randn([1, 40])[0, :]
    plot_spectrum(x)
