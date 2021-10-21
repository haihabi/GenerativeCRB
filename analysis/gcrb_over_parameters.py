import torch

import gcrb
from matplotlib import pyplot as plt
from analysis_helpers import load_wandb_run, db

if __name__ == '__main__':
    # run_name = "youthful-sweep-6"
    run_name = "young-sweep-9"
    model, dm, config = load_wandb_run(run_name)
    model_opt = dm.get_optimal_model()

    mse_regression_list = []
    parameter_list = []
    gcrb_opt_list = []
    gcrb_list = []
    crb_list = []
    for theta in dm.parameter_range(20):
        crb_list.append(dm.crb(theta).cpu().detach().numpy())
        fim = gcrb.adaptive_sampling_gfim(model_opt, theta.reshape([-1]), batch_size=512)  # 2048
        grcb_opt = torch.linalg.inv(fim).cpu().detach().numpy()

        fim = gcrb.adaptive_sampling_gfim(model, theta.reshape([-1]), batch_size=512)  # 2048
        grcb = torch.linalg.inv(fim).cpu().detach().numpy()

        parameter_list.append(theta.cpu().detach().numpy())
        gcrb_opt_list.append(grcb_opt)
        gcrb_list.append(grcb)
        print(theta, grcb.item(), crb_list[-1])
        print(100 * (gcrb_list[-1] - crb_list[-1]) / crb_list[-1])

    # plt.plot(parameter_list, crb_list, label="CRB")
    # plt.plot(parameter_list, gcrb_opt_list, label="NF Optimal")
    # plt.plot(parameter_list, gcrb_list, label="NF Learned")
    plt.plot(parameter_list, db(crb_list), label="CRB")
    plt.plot(parameter_list, db(gcrb_opt_list), "--x", label="GCRB - Optimal NF")
    plt.plot(parameter_list, db(gcrb_list), "--+", label="GCRB - Learned NF")
    plt.ylabel("MSE[dB]")
    plt.xlabel(r"$\theta$")
    plt.grid()
    plt.legend()
    plt.show()
