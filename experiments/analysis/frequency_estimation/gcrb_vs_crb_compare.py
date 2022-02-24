import numpy as np
from experiments import common
from experiments.analysis.bound_validation import generate_gcrb_validation_function, gcrb_empirical_error
from matplotlib import pyplot as plt
from experiments.analysis.analysis_helpers import load_wandb_run, db
import gcrb
import torch

models_dict = {0: "easy-snowflake-1688"}


def build_parameter_vector(amp, freq, phase):
    theta = [amp, freq, phase]
    theta = torch.tensor(theta).float()
    return theta


def compare_gcrb_vs_crb_over_freq(amp, phase, freq_array):
    results_gcrb = []
    results_crb = []
    for f_0 in freq_array:
        theta = build_parameter_vector(amp, f_0, phase)
        fim_optimal_back = gcrb.sampling_gfim(model, theta.reshape([-1]), m,
                                              batch_size=batch_size)
        egcrb = torch.linalg.inv(fim_optimal_back).detach().cpu().numpy()
        crb = dm.crb(theta).detach().cpu().numpy()
        results_crb.append(crb)
        results_gcrb.append(egcrb)
    results_crb = np.stack(results_crb)
    results_gcrb = np.stack(results_gcrb)
    return results_crb, results_gcrb


if __name__ == '__main__':
    m = 64e3
    batch_size = 4096
    f_0_array = np.linspace(0.01, 0.49, num=50)
    common.set_seed(0)
    run_name = "bumbling-glade-1752"  # Linear Model
    run_name = "firm-snowball-1732"  # Linear Model
    run_name = "iconic-moon-1711"  # Linear Model
    run_name = "curious-lion-1757"  # Linear Model

    model, dm, config = load_wandb_run(run_name)
    model_opt = dm.get_optimal_model()
    results_crb, results_gcrb = compare_gcrb_vs_crb_over_freq(1, 0, f_0_array)
    re=gcrb_empirical_error(results_gcrb,results_crb)

    plt.plot(f_0_array,re)
    plt.grid()
    plt.show()

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
