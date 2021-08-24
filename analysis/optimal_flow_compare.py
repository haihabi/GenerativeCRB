import numpy as np
import torch
import data_model
import normalizing_flow as nf
import gcrb
from matplotlib import pyplot as plt
from torch.distributions import MultivariateNormal
from main import generate_flow_model, config

if __name__ == '__main__':
    param = config()
    # dim = 2
    dm = data_model.MultiplicationModel(param.dim, 0.2, 10)
    prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
    model_opt = nf.NormalizingFlowModel(prior, [dm.get_optimal_model()])
    model = generate_flow_model(param, np.zeros(2).astype("float32"), np.zeros(2).astype("float32"))
    model.load_state_dict(torch.load('/Users/haihabi/projects/GenerativeCRB/logs/flow_best.pt'))
    model.eval()
    # model.eval()
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
        fim = gcrb.compute_fim(model_opt, theta.reshape([1]), batch_size=4096)  # 2048
        grcb_opt = torch.linalg.inv(fim)

        fim = gcrb.compute_fim(model, theta.reshape([1]), batch_size=4096)  # 2048
        grcb = torch.linalg.inv(fim)
        # fim = gcrb.compute_fim(in_flow_model, theta.reshape([1]), batch_size=512)
        # grcb_flow = torch.linalg.inv(fim)
        parameter_list.append(theta.item())
        gcrb_opt_list.append(grcb_opt.item())
        gcrb_list.append(grcb.item())
        # gcrb_flow_list.append(grcb_flow.item())

    plt.plot(parameter_list, crb_list, label="CRB")
    plt.plot(parameter_list, gcrb_opt_list, label="NF Optimal")
    plt.plot(parameter_list, gcrb_list, label="NF")
    plt.grid()
    plt.legend()
    plt.show()
    print("a")
