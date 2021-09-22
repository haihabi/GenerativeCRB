import os.path

import numpy as np
import torch
import data_model
import gcrb
from matplotlib import pyplot as plt
from main import generate_flow_model
import wandb
import os
import constants


def get_data_model(in_config):
    def generate_model_parameter_dict() -> dict:
        return {constants.DIM: in_config[constants.DIM],
                constants.THETA_MIN: in_config[constants.THETA_MIN],
                constants.SIGMA_N: in_config[constants.SIGMA_N],
                constants.THETA_MAX: in_config[constants.THETA_MAX]}

    model_type = data_model.ModelType[in_config["model_type"].split(".")[-1]]
    return data_model.get_model(model_type, generate_model_parameter_dict())


def load_wandb_run(run_name):
    api = wandb.Api()
    runs = api.runs(f"HVH/GenerativeCRB")
    for run in runs:
        print(run.name)
        if run.state == "finished" and run.name == run_name:
            if os.path.isfile("flow_best.pt"):
                os.remove("flow_best.pt")
            run.file("flow_best.pt").download()
            config = run.config
            model_flow = generate_flow_model(config['dim'], config['n_flow_blocks'], config['spline_flow'])
            model_flow.load_state_dict(torch.load(f"flow_best.pt"))
            dm = get_data_model(config)
            return model_flow, dm


if __name__ == '__main__':
    # cr = config()
    run_name = "brisk-sound-182"
    model, dm = load_wandb_run(run_name)
    model_opt = dm.get_optimal_model()
    # param = cr.get_user_arguments()
    # dim = 2
    # dm = data_model.Pow1Div3Gaussian(param.dim, 2.0, 10)
    # prior = MultivariateNormal(torch.zeros(param.dim), torch.eye(param.dim))
    # model_opt = nf.NormalizingFlowModel(prior, [dm._get_optimal_model()])
    # model = generate_flow_model(param)
    # model.load_state_dict(torch.load(f"/Users/haihabi/projects/GenerativeCRB/logs/21_09_2021_00_27_34/flow_best.pt"))

    mse_regression_list = []
    parameter_list = []
    gcrb_opt_list = []
    gcrb_list = []
    crb_list = []
    for theta in dm.parameter_range(20):
        # x = current_data_model.generate_data(512, theta)
        # theta_hat = in_regression_network(x)
        # theta_ml = current_data_model.ml_estimator(x)
        # mse_regression_list.append(torch.pow(theta_hat - theta, 2.0).mean().item())
        # ml_mse_list.append(torch.pow(theta_ml - theta, 2.0).mean().item())
        crb_list.append(dm.crb(theta).item())
        # theta.ze
        fim = gcrb.repeat_compute_fim(model_opt, theta.reshape([1]), batch_size=512)  # 2048
        grcb_opt = torch.linalg.inv(fim)

        fim = gcrb.repeat_compute_fim(model, theta.reshape([1]), batch_size=512)  # 2048
        grcb = torch.linalg.inv(fim)
        # fim = gcrb.compute_fim(in_flow_model, theta.reshape([1]), batch_size=512)
        # grcb_flow = torch.linalg.inv(fim)
        parameter_list.append(theta.item())
        gcrb_opt_list.append(grcb_opt.item())
        gcrb_list.append(grcb.item())
        print(theta, grcb.item(), crb_list[-1])
        print(100 * (gcrb_list[-1] - crb_list[-1]) / crb_list[-1])

    plt.plot(parameter_list, crb_list, label="CRB")
    plt.plot(parameter_list, gcrb_opt_list, label="NF Optimal")
    plt.plot(parameter_list, gcrb_list, label="NF Learned")
    plt.ylabel("MSE")
    plt.xlabel(r"$\theta$")
    plt.grid()
    plt.legend()
    plt.show()
