import numpy as np
from experiments import common
from experiments.analysis.bound_validation import generate_gcrb_validation_function, gcrb_empirical_error
from matplotlib import pyplot as plt
from experiments.analysis.analysis_helpers import load_wandb_run, db
from experiments.data_model.linear_example import LinearModel

if __name__ == '__main__':
    run_name = "fiery-bush-1640"  # Linear Model
    # run_name = "silver-snow-1652"  # Linear Model
    # run_name = "electric-cherry-1891"  # Linear Model

    # model2dataset_size_dict = {"floral-gorge-1706": 1000,
    #                            "sparkling-sun-1703": 2000,
    #                            "magic-gorge-1700": 4000,
    #                            # "silver-darkness-1699": 6000,
    #                            "fancy-fog-1687": 8000,
    #                            "azure-sunset-1669": 10000,
    #                            "fragrant-glade-1668": 20000,
    #                            "dry-grass-1667": 30000,
    #                            "magic-yogurt-1666": 40000,
    #                            "apricot-glade-1660": 50000,
    #                            "unique-wind-1659": 60000,
    #                            "curious-wood-1658": 70000,
    #                            "drawn-dawn-1657": 80000,
    #                            "glowing-resonance-1656": 90000,
    #                            "different-microwave-1655": 100000,
    #                            # "resilient-sponge-1654": 120000,
    #                            # "laced-tree-1653": 140000,
    #                            # "silver-snow-1652": 160000,
    #                            # "astral-totem-1650": 180000,
    #                            # "fiery-bush-1640": 200000,
    #                            }
    n_samples = 64e3
    batch_size = 4096
    # results_mean = []
    # results_max = []
    # dataset_size = []
    # for n, size in model2dataset_size_dict.items():
    #     common.set_seed(0)
    #     model, dm, config, _ = load_wandb_run(n)
    #     model_opt = dm.get_optimal_model()
    #     check_func = generate_gcrb_validation_function(dm, None, batch_size, optimal_model=None,
    #                                                    return_full_results=True,
    #                                                    n_validation_point=20, m=n_samples)
    #     data_dict = check_func(model)
    #     ene_trained = gcrb_empirical_error(data_dict["gcrb_flow"], data_dict["crb"])
    #     results_mean.append(np.mean(ene_trained))
    #     results_max.append(np.max(ene_trained))
    #     dataset_size.append(size)
    # plt.plot(dataset_size, results_mean)
    # plt.plot(dataset_size, results_max)
    # plt.xlabel(r"Size Of $\mathcal{D}$")
    # plt.ylabel("xRE")
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(f"dataset_size_spec.svg")
    # plt.show()

    common.set_seed(0)
    model, dm, config, _ = load_wandb_run(run_name)
    model_opt = dm.get_optimal_model()

    model_name = str(type(dm)).split(".")[-1].split("'")[0]
    zoom = isinstance(dm, LinearModel)
    eps = 0.01

    check_func = generate_gcrb_validation_function(dm, None, batch_size, optimal_model=model_opt,
                                                   return_full_results=True,
                                                   n_validation_point=20, eps=eps, m=n_samples)
    # bound_thm2 = eps
    data_dict = check_func(model)
    ene_trained = gcrb_empirical_error(data_dict["gcrb_flow"], data_dict["crb"])
    ene_optimal = gcrb_empirical_error(data_dict["gcrb_optimal_flow"], data_dict["crb"])
    print("-" * 100)
    print(f"Mean RE:{np.mean(ene_trained)}")
    print(f"Max RE:{np.max(ene_trained)}")
    print("-" * 100)
    parameter = data_dict["parameter"][:, 0]

    plt.plot(parameter, ene_optimal, "--x", label="eGCRB - Optimal NF", color="green")
    plt.plot(parameter, ene_trained, "--+", label="eGCRB - Learned NF", color="red")
    plt.grid()
    plt.legend()
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$\frac{||\overline{\mathrm{GCRB}}-\mathrm{CRB}||}{||\mathrm{CRB}||}$")
    plt.tight_layout()
    # plt.gca().set_position([0, 0, 1, 1])
    plt.savefig(f"re_results_{model_name}_spectral.svg")

    plt.show()

    crb_db = db(np.diagonal(data_dict["crb"], axis1=1, axis2=2).mean(axis=-1))
    gcrb_optimal_db = db(np.diagonal(data_dict["gcrb_optimal_flow"], axis1=1, axis2=2).mean(axis=-1))
    gcrb_flow_db = db(np.diagonal(data_dict["gcrb_flow"], axis1=1, axis2=2).mean(axis=-1))
    fig, ax = plt.subplots()
    max_value = crb_db.max() + 1.5
    min_value = crb_db.min() - 1.5
    ax.plot(parameter, crb_db, label="CRB", color="black")
    ax.plot(parameter, gcrb_optimal_db, "--x",
            label="eGCRB - Optimal NF", color="green")
    ax.plot(parameter, gcrb_flow_db, "--+",
            label="eGCRB - Learned NF", color="red")
    plt.ylabel(r"$\frac{1}{k}\mathrm{Tr}(\mathrm{GCRB})$")
    # plt.xlabel(r"$\theta_1$")
    plt.xlabel(r"$\xi$")
    plt.grid()
    plt.legend()
    plt.ylim([min_value, max_value])
    if zoom:
        axins = ax.inset_axes([0.4, 0.07, 0.4, 0.4])
        axins.plot(parameter, crb_db, label="CRB", color="black")
        axins.plot(parameter, gcrb_optimal_db, "--x",
                   label="eGCRB - Optimal NF", color="green")
        axins.plot(parameter, gcrb_flow_db, "--+",
                   label="eGCRB - Learned NF", color="red")
        axins.grid()
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        #
        ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.tight_layout()
    plt.savefig(f"trace_results_{model_name}.svg")
    plt.show()
