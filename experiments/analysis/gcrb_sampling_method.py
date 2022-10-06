from experiments import constants
from experiments import common
from experiments import data_model
from experiments.analysis.bound_validation import gcrb_empirical_error
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


if __name__ == '__main__':
    n_iter = 4000
    batch_size = 4098
    dim = 8
    theta_dim = 2
    common.set_seed(0)
    theta_value = 0.2
    model_type = data_model.ModelType.Linear

    results = []

    m2r = data_model.get_model(model_type, generate_model_dict(dim, theta_dim))
    optimal_flow = m2r.get_optimal_model()
    # m2r.load_data_model("/data/projects/GenerativeCRB/datasets/models")
    m2r.load_data_model(r"C:\work\GenerativeCRB\experiments\analysis")

    theta = theta_value * torch.ones([theta_dim], device=constants.DEVICE)
    fim = np.linalg.inv(m2r.crb(theta).cpu().detach().numpy())
    crb = np.linalg.inv(fim)
    _results_iteration = []
    n_samples = [64e3, 128e3, 256e3, 512e3]
    for i in tqdm(range(n_iter)):
        res_eps = []
        for m in n_samples:
            fim_rep_mean___ = gcrb.sampling_gfim(optimal_flow, theta, m, batch_size=batch_size)
            gcrb_matrix = torch.linalg.inv(fim_rep_mean___).cpu().detach().numpy()
            re = gcrb_empirical_error(gcrb_matrix, crb)
            res_eps.append(re)

        _results_iteration.append(res_eps)
    results_array = np.asarray(_results_iteration)
    results_array = np.squeeze(results_array, axis=-1)

    count_max = 0
    color_list = ["red", "green", "orange", "blue"]
    for i, m in enumerate(n_samples):
        count, bins = np.histogram(results_array[:, i], density=True, bins=20)
        count_max = max(count_max, np.max(count))
        plt.plot(bins[:-1], count, "--", label=r"$m=$" + f"{int(m / 1000)}k", color=color_list[i])

    plt.ylabel("PDF")
    plt.xlabel(r"$\mathrm{RE}(\theta)$")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("sampling_results_spec.svg")
    plt.show()
