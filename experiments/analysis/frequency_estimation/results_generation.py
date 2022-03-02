import math
import numpy as np
from experiments import common
from experiments.analysis.bound_validation import generate_gcrb_validation_function, gcrb_empirical_error
from matplotlib import pyplot as plt
from experiments.analysis.analysis_helpers import load_wandb_run, db
import gcrb
import torch
from scipy.special import erf
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

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

    return crb


def build_parameter_vector(amp, freq, phase):
    theta = [amp, freq, phase]
    theta = torch.tensor(theta).float()
    return theta


def compare_gcrb_vs_crb_over_freq(in_model, in_dm, amp, phase, freq_array, in_m):
    results_gcrb = []
    results_crb = []
    for f_0 in freq_array:
        theta = build_parameter_vector(amp, f_0, phase)
        fim_optimal_back = gcrb.sampling_gfim(in_model, theta.reshape([-1]), in_m,
                                              batch_size=batch_size, temperature=1.0)
        egcrb = torch.linalg.inv(fim_optimal_back).detach().cpu().numpy()
        if in_dm.has_crb:
            crb = in_dm.crb(theta).detach().cpu().numpy()
            results_crb.append(crb)
        results_gcrb.append(egcrb)
    if in_dm.has_crb:
        results_crb = np.stack(results_crb)
    results_gcrb = np.stack(results_gcrb)
    return results_crb, results_gcrb


def compare_gcrb_vs_crb_over_phase(in_model, in_dm, amp, phase_array, f_0, in_m):
    results_gcrb = []
    results_crb = []
    for phase in phase_array:
        theta = build_parameter_vector(amp, f_0, phase)
        fim_optimal_back = gcrb.sampling_gfim(in_model, theta.reshape([-1]), in_m,
                                              batch_size=batch_size, temperature=1.0)
        egcrb = torch.linalg.inv(fim_optimal_back).detach().cpu().numpy()
        if in_dm.has_crb:
            crb = in_dm.crb(theta).detach().cpu().numpy()
            results_crb.append(crb)
        results_gcrb.append(egcrb)
    if in_dm.has_crb:
        results_crb = np.stack(results_crb)
    results_gcrb = np.stack(results_gcrb)
    return results_crb, results_gcrb


def compare_gcrb_vs_crb_over_amp(in_model, in_dm, amp_array, phase, f_0, in_m, in_trimming):
    results_gcrb = []
    results_crb = []
    for amp in amp_array:
        theta = build_parameter_vector(amp, f_0, phase)
        fim_optimal_back = gcrb.sampling_gfim(in_model, theta.reshape([-1]), in_m,
                                              batch_size=batch_size, trimming_step=in_trimming, temperature=1.0)
        egcrb = torch.linalg.inv(fim_optimal_back).detach().cpu().numpy()
        if in_dm.has_crb:
            crb = in_dm.crb(theta).detach().cpu().numpy()
            results_crb.append(crb)
        results_gcrb.append(egcrb)
    if in_dm.has_crb:
        results_crb = np.stack(results_crb)
    results_gcrb = np.stack(results_gcrb)
    return results_crb, results_gcrb


phase_noise_results_dict = {0.5: {
    2.5: "prime-snow-1931",
    2: "treasured-smoke-1934",
    1.8: "misty-cloud-1951",
    1.6: "lyric-violet-1950",
    1.4: "elated-sky-1948",
    1.2: "charmed-mountain-1943",
    1: "ancient-microwave-1935",
    0.8: "good-cherry-1936",
    0.6: "unique-sea-1937",
    0.4: "desert-deluge-1939",
    0.2: "balmy-voice-1941",
    0.1: "chocolate-silence-1827",
    0.08: "eager-pond-1859",
    0.06: "bumbling-voice-1829",
    0.04: "rare-monkey-1846",
    0.02: "efficient-glitter-1831",
    0.01: "misunderstood-blaze-1870"},
    1.0: {
        2.5: "visionary-salad-1970",
        2.4: "lucky-disco-1960",
        # 2.2: "laced-sunset-1954",
        2: "fragrant-planet-1938",
        1.8: "bumbling-hill-1965",
        1.6: "skilled-spaceship-1962",
        1.4: "feasible-salad-1957",
        1.2: "upbeat-firefly-1952",
        1: "azure-meadow-1940",
        0.8: "kind-mountain-1942",
        0.6: "legendary-voice-1945",
        0.4: "logical-fog-1947",
        0.2: "dutiful-durian-1949",
        0.1: "worldly-lake-1911",
        0.08: "desert-yogurt-1912",
        0.06: "unique-gorge-1913",
        0.04: "icy-dust-1914",
        0.02: "balmy-sea-1915",
        0.01: "misty-valley-1916"},
    0.25: {
        2.5: "kind-capybara-1971",
        2.4: "drawn-terrain-1967",
        # 2.2: "wild-surf-1964",
        # 2: "robust-pyramid-1953",
        1.8: "crisp-glitter-1969",
        1.6: "efficient-firefly-1968",
        1.4: "warm-meadow-1966",
        1.2: "quiet-donkey-1963",
        1: "fresh-dawn-1955",
        0.8: "magic-cherry-1956",
        0.6: "crimson-morning-1958",
        0.4: "spring-shadow-1959",
        0.2: "splendid-frog-1961",
        0.1: "zany-morning-1905",
        0.08: "volcanic-silence-1906",
        0.06: "crimson-firefly-1907",
        0.04: "legendary-wind-1908",
        0.02: "divine-universe-1909",
        0.01: "light-brook-1910"}
}
quantization_results_dict = {0.25: {1: "sparkling-firebrand-1917",
                                    2: "different-wave-1899",
                                    3: "summer-wave-1902",
                                    4: "skilled-cosmos-1900"},
                             0.5: {1: "super-darkness-1849",
                                   2: "stilted-sun-1845",
                                   3: "lyric-night-1852",
                                   4: "divine-darkness-1848"},
                             1.0: {2: "bumbling-water-1893",
                                   3: "dry-morning-1896",
                                   4: "sweet-yogurt-1894"},
                             }

snr_base_line = {0.25: "warm-serenity-1918",
                 0.5: "devout-water-1825",
                 1.0: "blooming-jazz-1919"}

ph_q_run_dict = {1: {0.1: "lemon-cherry-1853",
                     0.08: "lucky-pine-1858",
                     0.06: "fiery-surf-1861",
                     0.04: "twilight-armadillo-1865",
                     0.02: "bumbling-sea-1869"},
                 2: {0.1: "frosty-dream-1857",
                     0.08: "generous-terrain-1860",
                     0.06: "faithful-frog-1863",
                     0.04: "wild-terrain-1864",
                     0.02: "azure-water-1866", },
                 3: {0.1: "ancient-moon-1868",
                     0.08: "misunderstood-frog-1872",
                     0.06: "earthy-donkey-1873",
                     0.04: "dauntless-sunset-1874",
                     0.02: "good-elevator-1875"},
                 4: {0.1: "vague-violet-1876",
                     0.08: "trim-durian-1877",
                     0.06: "snowy-water-1878",
                     0.04: "efficient-leaf-1879",
                     0.02: "peach-spaceship-1880"}
                 }

snr_phase_noise = {
    0.01: "morning-planet-1924",
    0.25: "crimson-firefly-1907",
    0.5: "bumbling-voice-1829",
    1.0: "unique-gorge-1913"}
only_phase_noise = "neat-pyramid-1920"

quantization_full = {1: "sparkling-firebrand-1917",
                     2: "different-wave-1899",
                     3: "summer-wave-1902",
                     4: "skilled-cosmos-1900",
                     5: "effortless-bush-1901",
                     8: "exalted-pond-1903"}


def sweep_test(in_run_dict):
    results_list = []
    results_x_axis = []
    for phase_scale, run_name in in_run_dict.items():
        m, dm, _, _ = load_wandb_run(run_name)
        _, results_gcrb_phase = compare_gcrb_vs_crb_over_freq(m, dm, base_amp, base_phase,
                                                              [base_f_0], m_samples)

        results_list.append(results_gcrb_phase[0, 1, 1])
        results_x_axis.append(phase_scale)
    return results_x_axis, results_list


def dual_sweep(in_run_dict):
    results_list = []
    results_y_axis = []
    results_x_axis = []
    for k, v in in_run_dict.items():
        hat_results_x_axis, hat_results_list = sweep_test(v)
        results_list.append(hat_results_list)
        results_x_axis.append(hat_results_x_axis)
        results_y_axis.append([k for _ in range(len(hat_results_x_axis))])
    return results_x_axis, results_y_axis, results_list


if __name__ == '__main__':
    m_samples = 64e3
    batch_size = 4096

    common.set_seed(0)
    base_amp = 1
    base_phase = 1
    base_f_0 = 0.2

    run_quantization_full = False
    run_comapre2crb = False
    phase_noise_results = False
    quantization_results = True or run_quantization_full
    dual_results = False
    snr_results_phase = False
    base_line_run_name = snr_base_line[0.25]  # Linear Model

    base_model, dm, config, tp = load_wandb_run(base_line_run_name)
    trimming_model = gcrb.AdaptiveTrimming(tp, gcrb.TrimmingType.MAX)

    model_opt = dm.get_optimal_model()
    results_crb_phase, results_gcrb_phase = compare_gcrb_vs_crb_over_freq(base_model, dm, base_amp, base_phase,
                                                                          [base_f_0], m_samples)
    gcrb_float_base_line = results_gcrb_phase[0, 1, 1]
    crb_float_base_line = results_crb_phase[0, 1, 1]
    if snr_results_phase:
        phase_noise_only_model, dm, _, _ = load_wandb_run(only_phase_noise)
        # trimming_model = gcrb.AdaptiveTrimming(tp, gcrb.TrimmingType.MAX)

        # model_opt = dm.get_optimal_model()
        results_crb_phase, results_gcrb_phase = compare_gcrb_vs_crb_over_freq(phase_noise_only_model, dm, base_amp,
                                                                              base_phase,
                                                                              [base_f_0], m_samples)
        results_x_axis, results_list = sweep_test(snr_phase_noise)
        snr = -10 * np.log10(np.asarray(results_x_axis))

        plt.plot(snr, results_list)
        plt.plot(snr, np.ones(len(results_x_axis)) * results_gcrb_phase[0, 1, 1])

        plt.show()
        print("a")

    if dual_results:
        x_list, y_list, res_list = dual_sweep(ph_q_run_dict)
        y_array = np.asarray(y_list)
        x_array = np.asarray(x_list)
        res_arary = np.asarray(res_list)

        z = db(res_arary)
        nrows, ncols = z.shape

        x = x_array
        y = y_array

        # Set up plot
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        ls = LightSource(90, 45)
        # To use a custom hillshading mode, override the built-in shading and pass
        # in the rgb colors of the shaded surface calculated from "shade".
        rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False, shade=False,
                               label="eGCRB (Quantization+Phase Noise+AWGN)")
        _ = ax.plot_surface(x, y, np.ones(z.shape) * db(gcrb_float_base_line), rstride=1, cstride=1,
                            cmap=cm.coolwarm,
                            linewidth=0, antialiased=False, shade=False, label="eGCRB (AWGN)")
        # plt.legend()
        # Add a color bar which maps values to colors.
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$n_b$")
        ax.set_zlabel(r"$\mathrm{Var}(\hat{f_0})[dB]$")
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.view_init(30, 90 + 45 + 35)
        plt.savefig("quantization_phase_results.svg")
        plt.show()

        print("a")
    if run_comapre2crb:
        name = ["amp", "freq", "phase"]
        x_axis_name = [f"$A$", f"$f_0$", f"$\phi$"]
        y_axis_name = [r"$\mathrm{Var}(\hat{A})$", r"$\mathrm{Var}(\hat{f_0})$", r"$\mathrm{Var}(\hat{\phi})$"]
        for i in range(3):
            if i == 0:
                amp_array = np.linspace(0.81, 1.19, num=20)
                x_axis_array = amp_array
                results_crb, results_gcrb = compare_gcrb_vs_crb_over_amp(base_model, dm, amp_array, base_phase,
                                                                         base_f_0,
                                                                         m_samples, trimming_model)
            elif i == 1:
                f_0_array = np.linspace(0.01, 0.49, num=20)
                x_axis_array = f_0_array
                results_crb, results_gcrb = compare_gcrb_vs_crb_over_freq(base_model, dm, base_amp, 0,
                                                                          f_0_array,
                                                                          m_samples)
            elif i == 2:
                phase_array = np.linspace(0.01, 2 * math.pi - 0.01, num=20)
                x_axis_array = phase_array
                results_crb, results_gcrb = compare_gcrb_vs_crb_over_phase(base_model, dm, base_amp, phase_array,
                                                                           base_f_0,
                                                                           m_samples)
            else:
                raise NotImplemented
            re = gcrb_empirical_error(results_gcrb, results_crb)
            print(np.mean(re), np.max(re))

            plt.plot(x_axis_array, re)
            plt.xlabel(x_axis_name[i])
            plt.ylabel(r"$\frac{||\mathrm{GCRB}-\mathrm{CRB}||_2}{||\mathrm{CRB}||_2}$")
            plt.grid()
            plt.savefig(f"re_results_{name[i]}.svg")
            plt.show()

            plt.plot(x_axis_array, db(results_gcrb[:, i, i]), label="eGCRB")
            plt.plot(x_axis_array, db(results_crb[:, i, i]), label="CRB")
            plt.xlabel(x_axis_name[i])
            plt.ylabel(r"$\mathrm{Var}(\hat{f_0})[dB]$")
            plt.grid()
            plt.legend()
            plt.savefig(f"compare_egcrb_crb_{name[i]}.svg")
            plt.show()

    if quantization_results:
        if run_quantization_full:
            quantization_results_dict = {0.25: quantization_full}
        for sigma, v in quantization_results_dict.items():
            base_line_run_name = snr_base_line[sigma]  # Linear Model
            base_model, dm, config, tp = load_wandb_run(base_line_run_name)
            results_crb_phase, results_gcrb_phase_base = compare_gcrb_vs_crb_over_freq(base_model, dm, base_amp,
                                                                                       base_phase,
                                                                                       [base_f_0], m_samples)
            snr = max(-np.round(10 * np.log10(np.power(sigma, 2.0))), 0)
            gcrb_float_base_line = results_gcrb_phase_base[0, 1, 1]
            crb_float_base_line = results_crb_phase[0, 1, 1]
            base = min(crb_float_base_line, gcrb_float_base_line)

            results_x_axis, results_list = sweep_test(v)
            print(results_list)
            plt.plot(np.asarray(results_x_axis), db(results_list), "o--", label=f"AWGN+Q, SNR={snr}dB")
            plt.plot(results_x_axis, db(np.ones(len(results_x_axis)) * base), "--",
                     label=f"AWGN, SNR={snr}dB")
        plt.legend(loc='upper left')
        plt.xlabel(r"$n_b$")
        plt.ylabel(r"$\mathrm{Var}(\hat{f_0})[dB]$")
        plt.grid()
        plt.savefig("quantization_egcrb.svg")
        plt.show()
    if phase_noise_results:
        for sigma, v in phase_noise_results_dict.items():
            base_line_run_name = snr_base_line[sigma]  # Linear Model
            base_model, dm, config, tp = load_wandb_run(base_line_run_name)
            results_crb_phase, results_gcrb_phase_base = compare_gcrb_vs_crb_over_freq(base_model, dm, base_amp,
                                                                                       base_phase,
                                                                                       [base_f_0], m_samples)
            snr = max(-np.round(10 * np.log10(np.power(sigma, 2.0))), 0)
            gcrb_float_base_line = results_gcrb_phase_base[0, 1, 1]
            crb_float_base_line = results_crb_phase[0, 1, 1]
            base = min(crb_float_base_line, gcrb_float_base_line)

            results_x_axis, results_list = sweep_test(v)
            x_axis = np.array(results_x_axis) * 180 / np.pi
            plt.semilogy(results_x_axis, results_list, label=f"AWGN+PN, SNR={snr}dB")
            plt.semilogy(results_x_axis, np.ones(len(results_x_axis)) * base, "--", label=f"AWGN, SNR={snr}dB")
        plt.legend()
        plt.xlabel(r"$\alpha$")
        plt.ylabel(r"$\mathrm{Var}(\hat{f_0})[dB]$")
        plt.grid()
        plt.savefig("phase_noise_egcrb.svg")
        plt.show()
