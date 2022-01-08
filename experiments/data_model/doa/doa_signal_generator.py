import math
from experiments import constants
import torch
import numpy as np
from enum import Enum
from torch import nn


class SensorsArrangement(Enum):
    ULA = 0
    UCA = 1
    RANDOM = 2


def complex_exp(z):
    return torch.stack([torch.cos(z), torch.sin(z)], dim=0)


def complex_matmul(a, b):
    a_real = a[0, :]
    a_imag = a[1, :]

    b_real = b[0, :]
    b_imag = b[1, :]
    real_part = torch.matmul(a_real, b_real) - torch.matmul(a_imag, b_imag)
    imag_part = torch.matmul(a_imag, b_real) + torch.matmul(a_real, b_imag)
    return torch.stack([real_part, imag_part], dim=0)


def conjugate_transpose(a):
    a_real = a[0, :]
    a_imag = a[1, :]
    a_real = a_real.transpose(dim0=0, dim1=1)
    a_imag = -a_imag.transpose(dim0=0, dim1=1)
    return torch.stack([a_real, a_imag])


def get_nominal_position(m_sensor, sensors_arrangement, distance):
    sensors_index = torch.linspace(0, m_sensor - 1, m_sensor, device=constants.DEVICE).float()
    if sensors_arrangement == SensorsArrangement.UCA:
        angle = np.pi / (m_sensor - 1)
        r = distance / (2 * np.sin(angle))
        angle_vector = 2 * np.pi * sensors_index / (m_sensor)
        x = r * torch.cos(angle_vector)
        y = r * torch.sin(angle_vector)
    elif sensors_arrangement == SensorsArrangement.ULA:
        x = torch.zeros([m_sensor], device=constants.DEVICE)
        y = (sensors_index - m_sensor / 2 + 0.5) * distance
    elif sensors_arrangement == SensorsArrangement.RANDOM:
        d_max = 4 * distance
        x = 2 * d_max * (torch.rand([m_sensor], device=constants.DEVICE) - 0.5)
        y = 2 * d_max * (torch.rand([m_sensor], device=constants.DEVICE) - 0.5)
    else:
        raise Exception('Unknown sensor arrangement')
    return torch.stack([x, y], dim=0)


def gaussian_source_sampler(k_samples, in_n_source, mu_source=0, sigma_source=1):
    return mu_source + sigma_source * torch.randn([2, in_n_source, k_samples], device=constants.DEVICE)


def generate_steering_matrix(source_theta, nominal_position, location_perturbation, wavelength: float = 1):
    """

    :param source_theta: a vector of size N
    :param nominal_position: a vector of size Mx2 (two is for the x and y position)
    :param location_perturbation:  a vec
    :return:
    """
    sensor_x = nominal_position[0, :]
    sensor_y = nominal_position[1, :]
    if location_perturbation is not None:
        m_sensor_pertubation = int(location_perturbation.shape[1] / 2)
        sensor_x[:, :m_sensor_pertubation, 0] += location_perturbation[:, :m_sensor_pertubation]
        sensor_y[:, :m_sensor_pertubation, 0] += location_perturbation[:, m_sensor_pertubation:]

    phi_i = torch.atan(sensor_y / sensor_x) + math.pi * (sensor_x < 0) + 2 * math.pi * (sensor_x > 0) * (sensor_y < 0)
    r = torch.sqrt(torch.pow(sensor_x, 2.0) + torch.pow(sensor_y, 2.0))
    d_nm = torch.unsqueeze(r, dim=-1) * torch.cos(
        torch.unsqueeze(phi_i, dim=-1) - torch.unsqueeze(source_theta, dim=0))  # [M,N]
    return complex_exp(-d_nm / wavelength)


class DOASignalGenerator(nn.Module):
    def __init__(self, m_sensor, k_samples, sensors_arrangement, d_sensors, location_perturbation_scale=0):
        super().__init__()

        self.nominal_position = nn.Parameter(get_nominal_position(m_sensor, sensors_arrangement, d_sensors),
                                             requires_grad=False)
        self.k_samples = k_samples
        self.location_perturbation_scale = location_perturbation_scale

    @property
    def m_sensors(self):
        return self.nominal_position.shape[1]

    def forward(self, source_theta):
        in_n_source = source_theta.shape[0]
        source_tensor = gaussian_source_sampler(self.k_samples, in_n_source)
        a = generate_steering_matrix(source_theta, self.nominal_position, None)
        return complex_matmul(a, source_tensor)
