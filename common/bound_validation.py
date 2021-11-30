import time
import gcrb
import torch
import numpy as np
from matplotlib import pyplot as plt
import wandb


def norm(x_array):
    return np.sqrt(np.power(x_array, 2.0).sum(axis=(1, 2)))


def gcrb_empirical_error(in_gcrb, in_crb):
    if len(in_gcrb.shape) != len(in_crb.shape):
        raise Exception("Shpae mismatch")
    if len(in_gcrb.shape) == 2:
        in_gcrb = np.expand_dims(in_gcrb, axis=0)
        in_crb = np.expand_dims(in_crb, axis=0)
    return norm(in_gcrb - in_crb) / norm(in_crb)


def generate_gcrb_validation_function(current_data_model, in_regression_network, batch_size,
                                      logging=False, n_validation_point=20, optimal_model=None,
                                      return_full_results=False, eps=0.01):
    def check_example(in_flow_model):
        start_time = time.time()
        crb_list = []
        fim_list = []
        mse_regression_list = []
        parameter_list = []

        gcrb_flow_list = []
        gfim_flow_list = []
        gfim_optimal_flow_list = []
        gcrb_optimal_flow_list = []
        optimal_model_exists = optimal_model is not None
        if in_regression_network is not None:
            in_regression_network.eval()
        for theta in current_data_model.parameter_range(n_validation_point):

            if in_regression_network is not None:
                x = current_data_model.generate_data(512, theta)
                theta_hat = in_regression_network(x)
                mse_regression_list.append(torch.pow(theta_hat - theta, 2.0).mean().item())
            if optimal_model_exists:
                fim_optimal_back = gcrb.adaptive_sampling_gfim(optimal_model, theta.reshape([-1]),
                                                               batch_size=batch_size)
                grcb_optimal_flow = torch.linalg.inv(fim_optimal_back)
                gcrb_optimal_flow_list.append(grcb_optimal_flow.detach().cpu().numpy())
                gfim_optimal_flow_list.append(fim_optimal_back.detach().cpu().numpy())
            crb_list.append(current_data_model.crb(theta).detach().cpu().numpy())
            fim_list.append(np.linalg.inv(crb_list[-1]))

            fim_back = gcrb.adaptive_sampling_gfim(in_flow_model, theta.reshape([-1]),
                                                   batch_size=batch_size, eps=eps)
            grcb_flow = torch.linalg.inv(fim_back)

            parameter_list.append(theta.detach().cpu().numpy())
            gcrb_flow_list.append(grcb_flow.detach().cpu().numpy())
            gfim_flow_list.append(fim_back.detach().cpu().numpy())

        gcrb_flow_list = np.asarray(gcrb_flow_list)
        gfim_flow_list = np.asarray(gfim_flow_list)
        crb_list = np.asarray(crb_list)
        parameter_list = np.asarray(parameter_list)
        if optimal_model_exists:
            gcrb_optimal_flow_list = np.asarray(gcrb_optimal_flow_list)
            gfim_optimal_flow_list = np.asarray(gfim_optimal_flow_list)

        if return_full_results:
            results_dict = {"gfim_flow": gfim_flow_list,
                            "gcrb_flow": gcrb_flow_list,
                            "crb": crb_list,
                            "parameter": parameter_list}
            if optimal_model_exists:
                results_dict.update({"gfim_optimal_flow": gfim_optimal_flow_list,
                                     "gcrb_optimal_flow": gcrb_optimal_flow_list})
            return results_dict
        relative_delta_crb = np.abs(np.asarray(crb_list) - np.asarray(gcrb_flow_list)) / np.asarray(np.abs(crb_list))
        relative_delta_fim = np.abs(np.asarray(fim_list) - np.asarray(gfim_flow_list)) / np.asarray(np.abs(fim_list))
        relative_delta_crb_trace = np.abs(
            np.diagonal(np.asarray(crb_list), axis1=1, axis2=2).mean(axis=-1) - np.diagonal(np.asarray(gcrb_flow_list),
                                                                                            axis1=1, axis2=2).mean(
                axis=-1)) / np.abs(np.diagonal(np.asarray(crb_list), axis1=1, axis2=2).mean(axis=-1))

        gcrb_flow_dual_error = relative_delta_crb.mean()
        gfim_flow_dual_error = relative_delta_fim.mean()
        gcrb_flow_dual_max_error = relative_delta_crb.max()
        gcrb_flow_trace_max_error = relative_delta_crb_trace.max()
        gfim_flow_dual_max_error = relative_delta_fim.max()
        ene = gcrb_empirical_error(np.asarray(gcrb_flow_list), np.asarray(crb_list))

        error_dict = {"gcrb_nf_error": gcrb_flow_dual_error,
                      "gcrb_nf_max_error": gcrb_flow_dual_max_error,
                      "gcrb_nf_trace_max_error": gcrb_flow_trace_max_error,
                      "gfim_nf_max_error": gfim_flow_dual_max_error,
                      "gfim_nf_mean_error": gfim_flow_dual_error,
                      "gcrb_norm_error_max": ene.max(),
                      "gcrb_norm_error_mean": ene.mean()}

        if logging:
            plt.plot(parameter_list[:, 0], np.trace(gcrb_flow_list, axis1=1, axis2=2), label="GCRB NF")
            plt.plot(parameter_list[:, 0], np.trace(crb_list, axis1=1, axis2=2), label="CRB")
            plt.legend()
            plt.xlabel(r"$\theta$")
            plt.ylabel(r"$MSE(\theta)$")
            error_dict_final = {k + "_final": v for k, v in error_dict.items()}
            error_dict_final.update({"CRB Compare": wandb.Image(plt)})
            wandb.log(error_dict_final)
        print("Time End For Model Check")
        print(time.time() - start_time)
        return error_dict

    return check_example
