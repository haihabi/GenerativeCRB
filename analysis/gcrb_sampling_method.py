import constants
import common
import data_model
import torch
import gcrb
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


def generate_model_dict(in_dim):
    return {constants.DIM: in_dim,
            constants.THETA_MIN: 0.3,
            constants.THETA_MAX: 10,
            constants.SIGMA_N: 0.1,
            }


def get_gcrb_bounds(in_optimal_flow, in_theta, in_batch_size):
    gfim_res_back = gcrb.compute_fim_backward(in_optimal_flow, in_theta, in_batch_size)
    gcrb_v_back = torch.linalg.inv(gfim_res_back)

    gfim_res  = gcrb.compute_fim(in_optimal_flow, in_theta, in_batch_size)
    gcrb_v_dual = torch.linalg.inv(gfim_res)

    return gcrb_v_back, gcrb_v_dual


if __name__ == '__main__':
    n_iter = 10
    batch_size_array = [128, 1024, 4098, 8192, 16384]
    dim_array = [16]
    common.set_seed(0)
    model_type = data_model.ModelType.Mean

    results = []
    results_iteration = []
    for dim in dim_array:

        m2r = data_model.get_model(model_type, generate_model_dict(dim))
        optimal_flow = m2r.get_optimal_model()
        theta_array = [0.2]

        results_dim = []
        results_dim_iteration = []
        for batch_size in batch_size_array:
            print(dim, batch_size)
            results_theta = []
            _results_iteration = []
            for j, theta_value in enumerate(theta_array):
                theta = theta_value * torch.ones([1])
                # gcrb_v_back, gcrb_v_dual = get_gcrb_bounds(optimal_flow, theta, batch_size)
                crb = m2r.crb(theta)

                # error_back = torch.abs(gcrb_v_back - crb).mean().item()
                # error_dual = torch.abs(gcrb_v_dual - crb).mean().item()
                # if j == 10:
                for i in tqdm(range(n_iter)):
                    gcrb_v_back, gcrb_v_dual = get_gcrb_bounds(optimal_flow, theta, batch_size)
                    fim_rep_mean, gcrb_rep_std = gcrb.adaptive_sampling_gfim(optimal_flow, theta, batch_size=128,
                                                                             iteration_step=int(batch_size / 64))
                    gcrb_rep = torch.linalg.inv(fim_rep_mean)
                    error_back = torch.abs(gcrb_v_back - crb).mean().item()
                    error_dual = torch.abs(gcrb_v_dual - crb).mean().item()
                    error_rep = torch.abs(gcrb_rep - crb).mean().item()
                    _results_iteration.append([error_dual, error_back, error_rep])

                # results_theta.append([error_dual, error_back, error_rep])
            # results_dim.append(results_theta)
            results_dim_iteration.append(_results_iteration)
        results.append(results_dim)
        results_iteration.append(results_dim_iteration)
    # print("a")
    print("Finised Run Loop")
    # results = np.asarray(results)
    results_iteration = np.asarray(results_iteration)
    # plt.subplot(1, 3, 1)
    # plt.plot(theta_array, results[0, :, 0], label="Dual")
    # plt.plot(theta_array, results[0, :, 1], label="Backward")
    # plt.subplot(1, 3, 2)
    plt.errorbar(batch_size_array, results_iteration.mean(axis=2)[0, :, 0], yerr=results_iteration.std(axis=2)[0, :, 0],
                 label="Dual")
    plt.errorbar(batch_size_array, results_iteration.mean(axis=2)[0, :, 1], yerr=results_iteration.std(axis=2)[0, :, 1],
                 label="Backward")
    plt.errorbar(batch_size_array, results_iteration.mean(axis=2)[0, :, 2], yerr=results_iteration.std(axis=2)[0, :, 2],
                 label="Rep")
    plt.legend()
    plt.grid()
    plt.show()
    # print("a")
