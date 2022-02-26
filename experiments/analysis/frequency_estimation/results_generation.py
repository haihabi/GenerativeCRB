import math
import numpy as np
from experiments import common
from experiments.analysis.bound_validation import generate_gcrb_validation_function, gcrb_empirical_error
from matplotlib import pyplot as plt
from experiments.analysis.analysis_helpers import load_wandb_run, db
import gcrb
import torch
from scipy.special import erf

models_dict = {0: "easy-snowflake-1688"}


def kap(alpha, snr):
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)
    t1 = np.power(s_alpha, 2.0) * np.exp(-2 * np.power(snr * c_alpha, 2.0)) / (1 - np.power(erf(snr * c_alpha), 2.0))
    t2 = np.power(c_alpha, 2.0) * np.exp(-2 * np.power(snr * s_alpha, 2.0)) / (1 - np.power(erf(snr * s_alpha), 2.0))
    return t1 + t2


def plot_all_crb(crb_matrix, x_axis):
    for i in range(crb_matrix.shape[-2]):
        for j in range(crb_matrix.shape[-2]):
            plt.subplots(crb_matrix.shape[-1], crb_matrix.shape[-2], i + j * crb_matrix.shape[-2] + 1)
            plt.plot(x_axis, crb_matrix[:, i, j])
    plt.show()


def crb_1bit_quantization(dim, amp, sigma, phase, f_0):
    snr = amp / sigma
    k = np.linspace(0, dim - 1, dim)
    alpha = 2 * math.pi * f_0 * k + phase
    fim = np.zeros([2, 2])
    fim[0, 0] = (np.sum(np.power(k, 2.0) * kap(alpha, snr)) * np.power(snr, 2.0) * 2 / math.pi)
    fim[1, 0] = fim[0, 1] = (np.sum(np.power(k, 1.0) * kap(alpha, snr)) * np.power(snr, 2.0) * 2 / math.pi)
    fim[1, 1] = (np.sum(kap(alpha, snr)) * np.power(snr, 2.0) * 2 / math.pi)
    crb = np.linalg.inv(fim)

    return 1/fim[0, 0]


def build_parameter_vector(amp, freq, phase):
    theta = [amp, freq, phase]
    theta = torch.tensor(theta).float()
    return theta


def compare_gcrb_vs_crb_over_freq(in_model, in_dm, amp, phase, freq_array):
    results_gcrb = []
    results_crb = []
    for f_0 in freq_array:
        theta = build_parameter_vector(amp, f_0, phase)
        fim_optimal_back = gcrb.sampling_gfim(in_model, theta.reshape([-1]), m,
                                              batch_size=batch_size)
        egcrb = torch.linalg.inv(fim_optimal_back).detach().cpu().numpy()
        if in_dm.has_crb:
            crb = in_dm.crb(theta).detach().cpu().numpy()
            results_crb.append(crb)
        results_gcrb.append(egcrb)
    if in_dm.has_crb:
        results_crb = np.stack(results_crb)
    results_gcrb = np.stack(results_gcrb)
    return results_crb, results_gcrb


if __name__ == '__main__':
    m = 64e3
    batch_size = 4096
    f_0_array = np.linspace(0.01, 0.49, num=50)
    common.set_seed(0)

    one_bit_run = "sunny-mountain-1781"
    model, dm, config = load_wandb_run(one_bit_run)
    # crb1_array = [crb_1bit_quantization(config["dim"], 1, config["sigma_n"], 0, f_0) for f_0 in f_0_array]
    _, results_gcrb_one_bit = compare_gcrb_vs_crb_over_freq(model, dm, 1, 0, f_0_array)
    # # plot_all_crb(results_gcrb_one_bit, f_0_array)
    #
    # plt.plot(f_0_array, crb1_array, label="CRB-1Bit")
    # plt.plot(f_0_array, results_gcrb_one_bit[:, 1, 1], label="eGCRB")
    # plt.legend()
    # plt.show()
    print("a")
    run_name = "prime-grass-1783"  # Linear Model

    model, dm, config = load_wandb_run(run_name)
    model_opt = dm.get_optimal_model()

    results_crb, results_gcrb = compare_gcrb_vs_crb_over_freq(model, dm, 1, 0, f_0_array)
    re = gcrb_empirical_error(results_gcrb, results_crb)

    plt.plot(f_0_array, re)
    plt.xlabel(f"$f_0$")
    plt.ylabel(r"$\frac{||\mathrm{eGCRB}-\mathrm{CRB}||_2}{||\mathrm{CRB}||_2}$")
    plt.grid()
    plt.savefig("re_results.svg")
    plt.show()

    plt.plot(f_0_array, results_gcrb[:, 1, 1], label="eGCRB")
    plt.plot(f_0_array, results_crb[:, 1, 1], label="CRB")
    plt.xlabel(f"$f_0$")
    plt.ylabel(r"$\mathrm{Var}(\hat{f_0})$")
    plt.grid()
    plt.legend()
    plt.savefig("compare_egcrb_crb.svg")

    plt.show()

    plt.plot(f_0_array, results_gcrb[:, 1, 1], label="eGCRB")
    plt.plot(f_0_array, results_crb[:, 1, 1], label="CRB")
    plt.plot(f_0_array, results_gcrb_one_bit[:, 1, 1], label="eGCRB 1-Bit")
    plt.xlabel(f"$f_0$")
    plt.ylabel(r"$\mathrm{Var}(\hat{f_0})$")
    plt.legend()
    plt.grid()
    plt.show()

    print("a")

    # bound_thm2 = eps
