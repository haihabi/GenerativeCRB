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

    theta = theta_value * torch.ones([theta_dim])
    fim = np.linalg.inv(m2r.crb(theta).cpu().detach().numpy())
    _results_iteration = []
    eps_list = [0.1, 0.05, 0.01, 0.005]
    for i in tqdm(range(n_iter)):
        res_eps = []
        for eps in eps_list:
            fim_rep_mean___ = gcrb.adaptive_sampling_gfim(optimal_flow, theta, batch_size=batch_size, eps=eps)
            res_eps.append(fim_rep_mean___.cpu().detach().numpy())

        _results_iteration.append(res_eps)
    results_array = np.asarray(_results_iteration)
    fim_size = results_array.shape[2:]
    n_figures = np.prod(fim_size)
    # count_max = 0
    # for j in range(n_figures):
    # j_x = j % 2
    # j_y = int(j >= 2)
    # plt.subplot(fim_size[0], fim_size[1], j + 1)
    count_max = 0
    results_trace = np.diagonal(results_array, axis1=2, axis2=3).sum(axis=-1) / fim_size[0]
    for i, eps in enumerate(eps_list):
        count, bins = np.histogram(results_trace[:, i], density=True, bins=20)
        count_max = max(count_max, np.max(count))
        plt.plot(bins[:-1], count, "--", label=r"$\epsilon=$" + f"{eps}")

    plt.plot([np.trace(fim) / fim_size[0], np.trace(fim) / fim_size[0]], [0, np.max(count_max)], label="Analytic FIM")
    plt.ylabel("PDF")
    plt.xlabel(r"$\frac{1}{k}\mathrm{Tr}(\overline{\mathrm{GFIM}})$")
    # plt.title(r"$\overline{\mathrm{GFIM}}$" + f"At {j_x}, {j_y}")
    plt.grid()
    plt.legend()
    plt.show()
    print("a")
