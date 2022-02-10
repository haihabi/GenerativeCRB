import numpy as np
from experiments import common
from experiments.analysis.bound_validation import generate_gcrb_validation_function, gcrb_empirical_error
from matplotlib import pyplot as plt
from experiments.analysis.analysis_helpers import load_wandb_run, db

if __name__ == '__main__':
    # run_name = "decent-disco-350"  # Scale  Model
    # run_name = "toasty-sweep-79"  # Linear Model
    # run_name = "youthful-haze-1350"  # Linear Model
    # run_name = "brisk-surf-1495"  # Linear Model
    run_name = "smart-pyramid-1492"  # Linear Model
    run_name = "whole-dust-1526"  # Linear Model
    run_name = "whole-lake-1535"  # Linear Model
    n_samples = 512e3
    common.set_seed(0)
    model, dm, config = load_wandb_run(run_name)
    model_opt = dm.get_optimal_model()
    batch_size = 4096
    zoom = False
    eps = 0.01

    check_func = generate_gcrb_validation_function(dm, None, batch_size, optimal_model=model_opt,
                                                   return_full_results=True,
                                                   n_validation_point=20, eps=eps, m=n_samples)
    # bound_thm2 = eps
    data_dict = check_func(model)
    ene_trained = gcrb_empirical_error(data_dict["gcrb_flow"], data_dict["crb"])
    ene_optimal = gcrb_empirical_error(data_dict["gcrb_optimal_flow"], data_dict["crb"])

    parameter = data_dict["parameter"][:, 0]

    plt.plot(parameter, ene_optimal, "--x", label="GCRB - Optimal NF")
    plt.plot(parameter, ene_trained, "--+", label="GCRB - Learned NF")
    plt.grid()
    plt.legend()
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\frac{||\overline{\mathrm{GCRB}}-\mathrm{CRB}||_2}{||\mathrm{CRB}||_2}$")
    plt.show()

    crb_db = db(np.diagonal(data_dict["crb"], axis1=1, axis2=2).mean(axis=-1))
    gcrb_optimal_db = db(np.diagonal(data_dict["gcrb_optimal_flow"], axis1=1, axis2=2).mean(axis=-1))
    gcrb_flow_db = db(np.diagonal(data_dict["gcrb_flow"], axis1=1, axis2=2).mean(axis=-1))
    fig, ax = plt.subplots()
    max_value = crb_db.max() + 1.5
    min_value = crb_db.min() - 1.5
    ax.plot(parameter, crb_db, label="CRB")
    ax.plot(parameter, gcrb_optimal_db, "--x",
            label="GCRB - Optimal NF")
    ax.plot(parameter, gcrb_flow_db, "--+",
            label="GCRB - Learned NF")
    plt.ylabel(r"$\frac{1}{k}\mathrm{Tr}(\overline{\mathrm{xCRB}})$")
    plt.xlabel(r"$\theta_1$")
    plt.grid()
    plt.legend()
    plt.ylim([min_value, max_value])
    if zoom:
        axins = ax.inset_axes([0.4, 0.07, 0.4, 0.4])
        axins.plot(parameter, crb_db, label="CRB")
        axins.plot(parameter, gcrb_optimal_db, "--x",
                   label="GCRB - Optimal NF")
        axins.plot(parameter, gcrb_flow_db, "--+",
                   label="GCRB - Learned NF")
        axins.grid()
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        #
        ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.show()
