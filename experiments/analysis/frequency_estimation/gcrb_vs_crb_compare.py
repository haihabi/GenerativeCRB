import numpy as np
from experiments import common
from experiments.analysis.bound_validation import generate_gcrb_validation_function, gcrb_empirical_error
from matplotlib import pyplot as plt
from experiments.analysis.analysis_helpers import load_wandb_run, db
import gcrb
import torch

if __name__ == '__main__':
    run_name = "cool-armadillo-1713"  # Linear Model

    m = 64e3
    common.set_seed(0)
    model, dm, config = load_wandb_run(run_name)
    model_opt = dm.get_optimal_model()
    batch_size = 4096
    zoom = True
    eps = 0.01
    results_gcrb = []
    results_crb = []
    f_0_array = np.linspace(0.01, 0.49)
    for f_0 in np.linspace(0.01, 0.49):
        theta = [1, f_0, 0]
        theta = torch.tensor(theta).float()

        fim_optimal_back = gcrb.sampling_gfim(model, theta.reshape([-1]), m,
                                              batch_size=batch_size)
        egcrb = torch.linalg.inv(fim_optimal_back).detach().cpu().numpy()
        crb = dm.crb(theta).detach().cpu().numpy()

        results_crb.append(crb)
        results_gcrb.append(egcrb)
    results_crb = np.stack(results_crb)
    results_gcrb = np.stack(results_gcrb)
    for i in range(3):
        for j in range(3):
            plt.subplot(3, 3, 1 + i + 3 * j)
            plt.plot(f_0_array, results_gcrb[:, i, j], label="eGCRB")
            plt.plot(f_0_array, results_crb[:, i, j], label="CRB")
            plt.grid()
            plt.legend()
            plt.title(f"{i},{j}")
    plt.show()

    print("a")

    # bound_thm2 = eps
