import torch
from experiments import constants
from torch import nn
from experiments.data_model.base_mode import BaseModel
import os
import math
from matplotlib import pyplot as plt
import numpy as np

PHASEMIN = 0
PHASEMAX = 2 * math.pi
FREQDELTA = 0.01
MINAMP = 0.8
MAXAMP = 1.2


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


def winner_phase_noise(batch_size, dim, scale):
    noise = torch.zeros([batch_size, dim], device=constants.DEVICE)
    for i in range(dim - 1):
        noise[:, i + 1] = noise[:, i] + scale * torch.randn([batch_size], device=constants.DEVICE)
    return noise


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
    def __init__(self, n_samples, sigma_n, quantization_enable=False, q_bit_width=8, q_threshold=1,
                 phase_noise_scale=0.0):
        super().__init__()
        self.sigma_n = sigma_n
        self.n_samples = n_samples
        self.n = nn.Parameter(
            torch.linspace(0, self.n_samples - 1, self.n_samples, device=constants.DEVICE).reshape([1, -1]),
            requires_grad=False)
        self.quantization_enable = quantization_enable
        self.q_bit_width = q_bit_width
        self.q_delta = 2 * q_threshold / (2 ** (self.q_bit_width) - 1)
        self.q_threshold = q_threshold
        self.phase_noise_scale = phase_noise_scale

    def quantization(self, x):
        if self.quantization_enable:
            return torch.clamp(self.q_delta * torch.floor(x / self.q_delta) + self.q_delta / 2, -self.q_threshold,
                               self.q_threshold)

        return x

    def _sine(self, amp, f_0, phase, amp_noise, phase_noise):
        x_base = torch.cos(
            2 * math.pi * f_0.reshape([-1, 1]) * self.n.reshape([1, -1]) + phase.reshape([-1, 1]) + phase_noise)
        return (amp.reshape([-1, 1]) + amp_noise) * x_base

    def forward(self, cond):
        batch_size = cond.shape[0]
        amp_noise = 0
        phase_noise = 0

        white_noise = self.sigma_n * torch.randn([batch_size, self.n_samples], device=constants.DEVICE)
        if self.phase_noise_scale > 0:
            phase_noise = winner_phase_noise(batch_size, self.n_samples, self.phase_noise_scale)
        return self.quantization(self._sine(cond[:, 0], cond[:, 1], cond[:, 2], amp_noise, phase_noise) + white_noise)


class FrequencyModel(BaseModel):
    def __init__(self, dim: int, sigma_n, quantization=False, phase_noise=False,
                 bit_width=None, threshold=None,
                 phase_noise_scale=0.01):
        theta_min = [MINAMP, 0.0 + FREQDELTA, PHASEMIN]
        theta_max = [MAXAMP, 0.5 - FREQDELTA, PHASEMAX]
        self.is_optimal_exists = not (quantization or phase_noise)
        super().__init__(dim, torch.tensor(theta_min, device=constants.DEVICE).reshape([1, -1]),
                         torch.tensor(theta_max, device=constants.DEVICE).reshape([1, -1]),
                         theta_dim=len(theta_min), quantized=quantization, has_crb=self.is_optimal_exists)
        self.sigma_n = sigma_n
        self.phase_noise = phase_noise
        self.quantization = quantization
        self.bit_width = bit_width
        self.threshold = threshold
        self.phase_noise_scale = phase_noise_scale
        if self.is_optimal_exists:
            self.optimal_flow = FrequencyOptimalFlow(self.dim, self.sigma_n)
            self.data_generator = None
        else:
            self.data_generator = FrequencyComplexModel(dim, sigma_n, quantization_enable=quantization,
                                                        q_bit_width=bit_width, q_threshold=threshold,
                                                        phase_noise_scale=phase_noise_scale if phase_noise else 0)
            self.optimal_flow = None

    @property
    def name(self) -> str:
        name = f"{super().name}_{self.sigma_n}"
        if self.quantization:
            name = name + f"q_{self.bit_width}_{self.threshold}"
        if self.phase_noise:
            name = name + f"pn_{self.phase_noise_scale}"

        return name  # Append Sigma N to Name

    def _get_optimal_model(self):
        if self.is_optimal_exists:
            return self.optimal_flow
        return None

    def save_data_model(self, folder):
        if self.is_optimal_exists:
            torch.save(self.optimal_flow.state_dict(), os.path.join(folder, f"{self.model_name}_model.pt"))

    def load_data_model(self, folder):
        if self.is_optimal_exists:
            print("Loading optimal model")
            data = torch.load(os.path.join(folder, f"{self.model_name}_model.pt"))
            self.optimal_flow.load_state_dict(data)

    def generate_data(self, n_samples, theta):
        cond = torch.ones([n_samples, 1], device=constants.DEVICE) * theta
        if self.is_optimal_exists:
            z = torch.randn([n_samples, self.dim], device=constants.DEVICE)
            return self.optimal_flow.backward(z, cond=cond)[0]
        else:
            return self.data_generator(cond)

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
    # fcm = FrequencyComplexModel(80, 1.2, quantization_enable=False, q_bit_width=2, q_threshold=1.0)
    # cond = [1, 0.06, 0]
    # cond = torch.tensor(cond).reshape([1, -1])
    # # x = fcm(cond)[0, :]
    # #
    # # plt.plot(x.cpu().numpy())
    # fcm = FrequencyComplexModel(40, 0.2, quantization_enable=True, q_bit_width=3, q_threshold=1.0)
    # x = fcm(cond)[0, :]
    # plt.plot(x.cpu().numpy(), "-o")
    # plt.grid()
    # plt.xlabel("n")
    # plt.ylabel("y[n]")
    # plt.savefig("noisy_sine.svg")
    # plt.show()
    z = np.random.randn(40)
    plt.plot(z, "-o")
    plt.grid()
    plt.xlabel("n")
    plt.ylabel("z[n]")
    plt.savefig("latent.svg")
    plt.show()

    # pn = torch.randn([1, 40])[0, :]
    # plot_spectrum(x)
