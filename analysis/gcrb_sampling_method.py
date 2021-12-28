import constants
import common
import data_model
import torch
import gcrb
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


def generate_model_dict(in_dim, in_theta_dim):
    return {constants.DIM: in_dim,
            constants.THETA_DIM: in_theta_dim,
            constants.THETA_MIN: 0.3,
            constants.THETA_MAX: 10,
            constants.SIGMA_N: 0.1,
            }


def get_gcrb_bounds(in_optimal_flow, in_theta, in_batch_size):
    gfim_res_back = gcrb.compute_fim_backward(in_optimal_flow, in_theta, in_batch_size)
    gcrb_v_back = torch.linalg.inv(gfim_res_back)

    gfim_res = gcrb.compute_fim(in_optimal_flow, in_theta, in_batch_size)
    gcrb_v_dual = torch.linalg.inv(gfim_res)

    return gcrb_v_back, gcrb_v_dual


if __name__ == '__main__':
    n_iter = 2000
    batch_size = 4098
    dim = 16
    theta_dim = 2
    common.set_seed(0)
    theta_value = 0.2
    model_type = data_model.ModelType.Linear

    results = []

    m2r = data_model.get_model(model_type, generate_model_dict(dim, theta_dim))
    optimal_flow = m2r.get_optimal_model()

    theta = theta_value * torch.ones([theta_dim], device=constants.DEVICE)
    fim = np.linalg.inv(m2r.crb(theta).cpu().detach().numpy())
    crb = np.linalg.inv(fim)
    _results_iteration = []
    eps_list = [0.1, 0.05, 0.01, 0.005]
    n_samples = [64e3, 128e3, 256e3, 512e3]
    for i in tqdm(range(n_iter)):
        res_eps = []
        for m in n_samples:
            # fim_rep_mean___ = gcrb.adaptive_sampling_gfim(optimal_flow, theta, batch_size=batch_size, eps=eps)
            fim_rep_mean___ = gcrb.sampling_gfim(optimal_flow, theta, m, batch_size=batch_size)
            gcrb_matrix = torch.linalg.inv(fim_rep_mean___).cpu().detach().numpy()
            re = common.gcrb_empirical_error(gcrb_matrix, crb)
            res_eps.append(re)

        _results_iteration.append(res_eps)
    results_array = np.asarray(_results_iteration)
    results_array = np.squeeze(results_array, axis=-1)

    count_max = 0
    color_list = ["red", "green", "orange", "blue"]
    for i, m in enumerate(n_samples):
        count, bins = np.histogram(results_array[:, i], density=True, bins=20)
        count_max = max(count_max, np.max(count))
        plt.plot(bins[:-1], count, "--", label=r"$m=$" + f"{int(m/1000)}k", color=color_list[i])

    plt.ylabel("PDF")
    plt.xlabel(r"$\mathrm{RE}(\theta)$")
    plt.grid()
    plt.legend()
    plt.show()
