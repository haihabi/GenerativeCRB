import numpy as np
import common
from matplotlib import pyplot as plt
from analysis.analysis_helpers import load_wandb_run

if __name__ == '__main__':
    # run_name = "decent-disco-350"  # Scale  Model
    # run_name = "toasty-sweep-79"  # Linear Model
    # run_name = "youthful-haze-1350"  # Linear Model
    run_name = "fancy-dream-1438"  # Linear Model
    run_name = "feasible-vortex-1480"  # Linear Model
    common.set_seed(0)
    # theta_scale_min = 0.9
    # theta_scale_max = 0.9
    model, dm, config = load_wandb_run(run_name)
    model_opt = dm.get_optimal_model()
    batch_size = 4096

    eps = 0.01
    check_func = common.generate_gcrb_validation_function(dm, None, batch_size, optimal_model=model_opt,
                                                          return_full_results=True,
                                                          n_validation_point=20, eps=eps)
    bound_thm2 = eps
    data_dict = check_func(model)
    ene_trained = common.gcrb_empirical_error(data_dict["gcrb_flow"], data_dict["crb"])
    ene_optimal = common.gcrb_empirical_error(data_dict["gcrb_optimal_flow"], data_dict["crb"])
    print(np.max(ene_trained), np.max(ene_optimal))
    parameter = data_dict["parameter"][:, 0]
    plt.plot(parameter, bound_thm2 * np.ones(parameter.shape[0]), label=r"SEB")
    plt.plot(parameter, ene_optimal, "--x", label="GCRB - Optimal NF")
    plt.plot(parameter, ene_trained, "--+", label="GCRB - Learned NF")
    plt.grid()
    plt.legend()
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\frac{||\overline{\mathrm{GCRB}}-\mathrm{CRB}||_2}{||\mathrm{CRB}||_2}$")
    plt.show()

    fig, ax = plt.subplots()
    max_value = np.diagonal(data_dict["crb"], axis1=1, axis2=2).mean(axis=-1).max() + 1.5
    min_value = np.diagonal(data_dict["crb"], axis1=1, axis2=2).mean(axis=-1).max() - 1.5
    ax.plot(parameter, np.diagonal(data_dict["crb"], axis1=1, axis2=2).mean(axis=-1), label="CRB")
    ax.plot(parameter, np.diagonal(data_dict["gcrb_optimal_flow"], axis1=1, axis2=2).mean(axis=-1), "--x",
            label="GCRB - Optimal NF")
    ax.plot(parameter, np.diagonal(data_dict["gcrb_flow"], axis1=1, axis2=2).mean(axis=-1), "--+",
            label="GCRB - Learned NF")
    plt.ylabel(r"$\frac{1}{k}\mathrm{Tr}(\overline{\mathrm{xCRB}})$")
    plt.xlabel(r"$\theta_1$")
    plt.grid()
    plt.legend()
    plt.ylim([min_value, max_value])
    axins = ax.inset_axes([0.4, 0.07, 0.4, 0.4])
    axins.plot(parameter, np.diagonal(data_dict["crb"], axis1=1, axis2=2).mean(axis=-1), label="CRB")
    axins.plot(parameter, np.diagonal(data_dict["gcrb_optimal_flow"], axis1=1, axis2=2).mean(axis=-1), "--x",
               label="GCRB - Optimal NF")
    axins.plot(parameter, np.diagonal(data_dict["gcrb_flow"], axis1=1, axis2=2).mean(axis=-1), "--+",
               label="GCRB - Learned NF")
    axins.grid()
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])

    ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.show()
