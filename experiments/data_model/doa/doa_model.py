import math
import os
import torch
from experiments.data_model.base_mode import BaseModel
from data_model.doa.jacobain_util import batch_func_jacobain
from data_model.doa.doa_signal_generator import DOASignalGenerator, generate_steering_matrix, conjugate_transpose, \
    complex_matmul


class DOAModel(BaseModel):
    def __init__(self, sensors_arrangement, m_sensor, n_sources, k_samples, d_sensors, sigma_n):
        super().__init__(2 * m_sensor * k_samples, -math.pi, math.pi, input_dim=n_sources)
        self.sigma_n = sigma_n
        self.d_sensors = d_sensors
        self.m_sensor = m_sensor
        self.n_sources = n_sources
        self.k_samples = k_samples
        self.sensors_arrangement = sensors_arrangement
        self.signal_generator = DOASignalGenerator(m_sensor, k_samples, sensors_arrangement, d_sensors)

    @property
    def name(self) -> str:
        return f"{super().name}_{self.signal_generator.m_sensors}_{self.k_samples}_{self.sensors_arrangement.name}_{self.d_sensors}_{self.n_sources}_{self.sigma_n}"

    @property
    def model_name(self) -> str:
        return f"{type(self).__name__}_{self.signal_generator.m_sensors}_{self.sensors_arrangement.name}_{self.d_sensors}"

    def _get_optimal_model(self):
        return None

    def save_data_model(self, folder):
        torch.save(self.signal_generator.state_dict(), os.path.join(folder, f"{self.model_name}_model.pt"))

    def load_data_model(self, folder):
        data = torch.load(os.path.join(folder, f"{self.model_name}_model.pt"))
        self.signal_generator.load_state_dict(data)

    def generate_data(self, n_samples, theta):
        _res = []
        for i in range(n_samples):
            _res.append(self.signal_generator(theta))
        return torch.stack(_res, dim=0)

    def crb(self, in_theta_vector):
        n_source = in_theta_vector.shape[0]
        a = generate_steering_matrix(in_theta_vector, self.signal_generator.nominal_position, None)
        a = torch.complex(a[0, :], a[1, :])
        ah = a.conj().T
        caa = torch.matmul(a, ah) + math.pow(self.sigma_n, 2) * torch.eye(self.signal_generator.m_sensors)

        def func_steering_matrix(in_theta):
            a = generate_steering_matrix(in_theta, self.signal_generator.nominal_position, None)
            ah = conjugate_transpose(a)
            return complex_matmul(a, ah).unsqueeze(dim=0)

        dr = batch_func_jacobain(func_steering_matrix, in_theta_vector)
        dr = dr.reshape([2, self.signal_generator.m_sensors, self.signal_generator.m_sensors, n_source])
        dr = torch.complex(dr[0, :], dr[1, :])
        fim_list = []
        for i in range(n_source):
            _res = []
            for j in range(n_source):
                _res.append(torch.matmul(torch.matmul(dr[:, :, i], caa), torch.matmul(dr[:, :, j], caa)).trace())
            fim_list.append(torch.stack(_res))
        fim = torch.stack(fim_list).real
        return torch.linalg.inv(fim)
