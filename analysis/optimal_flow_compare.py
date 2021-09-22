import numpy as np
import torch
import data_model
import normalizing_flow as nf
import gcrb
from matplotlib import pyplot as plt
from torch.distributions import MultivariateNormal
from main import generate_flow_model, config
from tqdm import tqdm

if __name__ == '__main__':
    cr = config()
    param = cr.get_user_arguments()
    # dim = 2
    sigma_array = [0.01, 0.1, 1, 10]
    sigma_results = []
    batch_size_list = [128, 256, 512, 1024, 2048, 4096]

    n_iter = 1000
    # theta = 1 * torch.ones([1])
    # for i, sigma in enumerate(sigma_array):
    #     dm = data_model.MeanModel(param.dim, -10, 10, sigma)
    #     prior = MultivariateNormal(torch.zeros(param.dim), torch.eye(param.dim))
    #     model_opt = nf.NormalizingFlowModel(prior, [dm._get_optimal_model()])
    #
    #     crb_value = dm.crb(theta).item()
    #     results_per_batch = []
    #     for batch_size in batch_size_list:
    #         gcrb_iter = []
    #         for i in tqdm(range(n_iter)):
    #             fim = gcrb.compute_fim(model_opt, theta.reshape([1]), batch_size=batch_size)  # 2048
    #             gcrb_matrix = torch.linalg.inv(fim)
    #             gcrb_iter.append(gcrb_matrix.item())
    #         results_per_batch.append(gcrb_iter)
    #     error = np.abs(np.asarray(results_per_batch) - crb_value) / crb_value  # Relative Error
    #     std_array = np.asarray(error).std(axis=1)
    #     mean_array = np.asarray(error).mean(axis=1)
    #     sigma_results.append(np.stack([mean_array, std_array], axis=-1))
    # results = np.stack(sigma_results, axis=0)
    # plt.subplot(1, 2, 1)
    # for i in range(len(sigma_array)):
    #     plt.errorbar(batch_size_list, results[i, :, 0], results[i, :, 1], label=r"$\sigma^2=$" + f"{sigma_array[i]}")
    #
    # plt.grid()
    # plt.legend()
    # plt.xlabel("Batch-size")
    # plt.ylabel("|GCRB-CRB|")
    # plt.subplot(1, 2, 2)
    # for i in range(len(sigma_array)):
    #     plt.plot(batch_size_list, results[i, :, 1], label=r"$\sigma^2=$" + f"{sigma_array[i]}")
    # plt.grid()
    # plt.legend()
    # plt.xlabel("Batch-size")
    # plt.ylabel("|GCRB-CRB|")
    # plt.show()

    # mse_regression_list = []
    # parameter_list = []
    # gcrb_opt_list = []
    # gcrb_list = []
    # crb_list = []
    dm = data_model.Pow3Gaussian(param.dim, 0.3, 100)
    model_opt = dm.get_optimal_model()
    parameter_results = []
    crb_results = []
    param_array = dm.parameter_range(5)
    for theta in param_array:
        # results_per_batch = []
        crb_value = dm.crb(theta).item()
        crb_results.append(1 / crb_value)
        # for batch_size in batch_size_list:
        gfim_iter = []
        for i in tqdm(range(n_iter)):
            gfim = gcrb.compute_fim_tensor(model_opt, theta.reshape([1]), batch_size=128)  # 2048
            gfim_iter.append(gfim.numpy())
        parameter_results.append(np.concatenate(gfim_iter, axis=0))
    crb_results = np.asarray(crb_results).reshape([-1, 1])
    parameter_results = np.stack(parameter_results)


    def cummean(in_x_array):
        cumsum_res = np.cumsum(in_x_array, axis=1)
        n_array = np.linspace(1, in_x_array.shape[1], in_x_array.shape[1]).reshape([1, -1])
        return cumsum_res / n_array


    def cumvar(in_x_array):
        cum_mu = cummean(in_x_array)
        cum_pow2 = cummean(np.power(in_x_array, 2.0))
        return cum_pow2 - np.power(cum_mu, 2.0)


    print("b")
    parameter_results = parameter_results.squeeze()

    ratio = cumvar(parameter_results) / np.power(cummean(parameter_results), 2.0)
    # print("a")
    # cumsum_res = np.cumsum(parameter_results, axis=1).squeeze()
    # n_array = np.linspace(1, cumsum_res.shape[1], cumsum_res.shape[1]).reshape([1, -1])
    mean_res = cummean(parameter_results)
    error_res = np.abs(mean_res - crb_results) / crb_results
    n_array = np.linspace(1, ratio.shape[1], ratio.shape[1]).reshape([1, -1])
    error_bound = np.sqrt(ratio / (n_array * 0.1))
    # param_array_np = param_array.numpy()
    # for i, theta in enumerate(param_array):
    i = 0
    plt.plot(error_res[i, :],
             label=r"\frac{\abs{\mathrm{F}_{\mathrm{R}}-\overline{\mathrm{F}}_{\mathrm{G}}}}{\mathrm{F}_{\mathrm{R}}}")
    plt.plot(error_bound[i, :], "--", label=f"$\epsilon$")
    plt.ylabel(r"\frac{\abs{\mathrm{F}_{\mathrm{R}}-\overline{\mathrm{F}}_{\mathrm{G}}}}{\mathrm{F}_{\mathrm{R}}}")
    plt.xlabel("$")
    plt.grid()
    plt.show()
    # error = np.abs(np.asarray(results_per_batch) - crb_value)  # Relative Error
    # std_array = np.nanvar(np.asarray(error), axis=1)
    # mean_array = np.nanmean(np.asarray(error), axis=1)
    # parameter_results.append(np.stack([mean_array, std_array], axis=-1))
    # results = np.stack(parameter_results, axis=0)
    # print("a")
    # param_array_np = param_array.numpy()
    # ratio_array = results[:, -1, 1] / np.power(results[:, -1, 0], 2)
    # plt.plot(param_array_np, ratio_array)
    # plt.show()
    #
    # plt.subplot(1, 2, 1)
    # for i in range(len(param_array)):
    #     plt.errorbar(batch_size_list, results[i, :, 0], results[i, :, 1], label=r"$\theta=$" + f"{param_array[i]}")
    #
    # plt.grid()
    # plt.legend()
    # plt.xlabel("Batch-size")
    # plt.ylabel("|GCRB-CRB|")
    # plt.subplot(1, 2, 2)
    # for i in range(len(param_array)):
    #     plt.plot(batch_size_list, results[i, :, 1], label=r"$\theta=$" + f"{param_array[i]}")
    # plt.grid()
    # plt.legend()
    # plt.xlabel("Batch-size")
    # plt.ylabel("|GCRB-CRB|")
    # plt.show()
    # plt.plot(parameter_list, crb_list, label="CRB")
    # plt.plot(parameter_list, gcrb_opt_list, label="NF Optimal Simple Mean")
    # plt.plot(parameter_list, gcrb_list, label="NF Adaptive Iteration Mean")
    # plt.grid()
    # plt.legend()
    # plt.show()
