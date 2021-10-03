import math
import constants
import torch
import numpy as np
from data_model.base_mode import BaseModel
from enum import Enum
from data_model.doa.gaussion_case_bound import batch_fim, batch_func_jacobain


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
    # return torch.stack([real_part, imag_part], dim=0)


def doa_signal_generator(source_theta, k_samples, m_sensor, sensors_arrangement, distance,
                         location_perturbation):
    in_n_source = source_theta.shape[0]
    nominal_position = get_nominal_position(m_sensor, sensors_arrangement, distance)
    source_tensor = gaussian_source_sampler(k_samples, in_n_source)
    a = generate_steering_matrix(source_theta, nominal_position, location_perturbation)
    return complex_matmul(a, source_tensor)
    # pass


class DOAModel(BaseModel):
    def __init__(self, dim: int, sigma_n):
        super().__init__(dim, -math.pi, math.pi)
        self.sigma_n = sigma_n
        # self.optimal_flow = LinearOptimalFlow(self.dim, self.parameter_vector_length, self.sigma_n)

    @property
    def name(self) -> str:
        return f"{super().name}_{self.sigma_n}"  # Append Sigma N to Name

    def _get_optimal_model(self):
        return None

    def save_data_model(self, folder):
        raise NotImplemented
        # torch.save(self.optimal_flow.state_dict(), os.path.join(folder, f"{self.model_name}_model.pt"))

    def load_data_model(self, folder):
        raise NotImplemented
        # data = torch.load(os.path.join(folder, f"{self.model_name}_model.pt"))
        # self.optimal_flow.load_state_dict(data)
        # pass

    def generate_data(self, n_samples, theta):
        cond = torch.ones([n_samples, 1], device=constants.DEVICE) * theta
        z = torch.randn([n_samples, self.dim], device=constants.DEVICE)
        return self.optimal_flow.backward(z, cond=cond)[0]

    @staticmethod
    def ml_estimator(r):
        return torch.pow(torch.mean(torch.pow(torch.abs(r), 6), dim=1), 1 / 6)

    def crb(self, theta):
        a = self.optimal_flow.a
        l = self.optimal_flow.l_matrix
        fim = torch.matmul(
            torch.matmul(a.transpose(dim0=0, dim1=1), torch.linalg.inv(torch.matmul(l, l.transpose(dim0=0, dim1=1)))),
            a)
        return torch.linalg.inv(fim)


if __name__ == '__main__':
    n_source = 3
    m_sensor = 4
    k_samples = 1
    sigma_n = 1

    theta_vector = -math.pi + 2 * math.pi * torch.rand(n_source, device=constants.DEVICE, requires_grad=True)

    nominal_position = get_nominal_position(m_sensor, SensorsArrangement.UCA, 0.1)
    a = generate_steering_matrix(theta_vector, nominal_position, None)
    a = torch.complex(a[0, :], a[1, :])
    ah = a.conj().T
    caa = torch.matmul(a, ah)+torch.eye(m_sensor)


    def func_steering_matrix(in_theta):
        a = generate_steering_matrix(in_theta, nominal_position, None)
        ah = conjugate_transpose(a)
        return complex_matmul(a, ah).unsqueeze(dim=0)


    dr = batch_func_jacobain(func_steering_matrix, theta_vector)
    dr = dr.reshape([2, m_sensor, m_sensor, n_source])
    dr = torch.complex(dr[0, :], dr[1, :])
    fim_list = []
    for i in range(n_source):
        _res = []
        for j in range(n_source):
            _res.append(torch.matmul(torch.matmul(dr[:, :, i], caa), torch.matmul(dr[:, :, j], caa)).trace())
        fim_list.append(torch.stack(_res))
    fim = torch.stack(fim_list).real
    print("a")


    # da = da.diagonal(dim1=1, dim2=2)

    # a_h = a.conj().T
    # p = a @ torch.linalg.solve(a_h @ a, a_h)
    # da_h = da.conj().T
    # h = da_h @ (torch.eye(m_sensor) - p) @ da
    # crb = 2 * torch.linalg.inv(h.real * 2) * sigma_n / k_samples
    # crb = 0.5 * (crb + crb.T)

    def func(in_theta_vector):
        return doa_signal_generator(in_theta_vector, k_samples, 16, SensorsArrangement.UCA, 0.1, None).reshape([1, -1])


    #
    #
    r1 = batch_fim(func, theta_vector, sigma=sigma_n)
    crb_r1 = torch.inverse(r1)
    print("a")
    # r2 = batch_fim(func, theta_vector)
    # print("a")
