import numpy as np
import torch

import gcrb
from matplotlib import pyplot as plt
from analysis_helpers import load_wandb_run, db

if __name__ == '__main__':
    # cr = config()
    run_name = "resilient-water-193"

    mse_regression_list = []
    parameter_list = []
    gcrb_opt_list = []
    gcrb_list = []
    crb_list = []
    theta = torch.ones(1) * 5

    for run_name in ["resilient-water-193", "driven-silence-192", "vague-moon-194", "warm-breeze-195",
                     "genial-wave-196", "playful-sky-191", "brisk-sound-182"]:
        model, dm, config = load_wandb_run(run_name)
        model_opt = dm.get_optimal_model()

        crb_list.append(dm.crb(theta).item())

        fim = gcrb.repeat_compute_fim(model_opt, theta.reshape([1]), batch_size=512)  # 2048
        grcb_opt = torch.linalg.inv(fim)

        fim = gcrb.repeat_compute_fim(model, theta.reshape([1]), batch_size=512)  # 2048
        grcb = torch.linalg.inv(fim)

        parameter_list.append(config["sigma_n"])
        gcrb_opt_list.append(grcb_opt.item())
        gcrb_list.append(grcb.item())
        print(theta, grcb.item(), crb_list[-1])
        print(100 * (gcrb_list[-1] - crb_list[-1]) / crb_list[-1])

    parameter_list = np.asarray(parameter_list)
    gcrb_opt_list = np.asarray(gcrb_opt_list)
    gcrb_list = np.asarray(gcrb_list)
    crb_list = np.asarray(crb_list)
    i_sort = np.argsort(parameter_list)
    plt.plot(parameter_list[i_sort], db(crb_list[i_sort]), label="CRB")
    plt.plot(parameter_list[i_sort], db(gcrb_opt_list[i_sort]), label="NF Optimal")
    plt.plot(parameter_list[i_sort], db(gcrb_list[i_sort]), label="NF Learned")
    plt.ylabel("MSE[dB]")
    plt.xlabel(r"$\sigma_v$")
    plt.grid()
    plt.legend()
    plt.show()
