import numpy as np
import common
from matplotlib import pyplot as plt
from analysis.analysis_helpers import load_wandb_run, db
# from main import generate_gcrb_validation_function

if __name__ == '__main__':
    # run_name = "youthful-sweep-6"
    run_name = "young-sweep-9"
    run_name = "zany-pyramid-702"
    model, dm, config = load_wandb_run(run_name)
    model_opt = dm.get_optimal_model()
    batch_size = 4096
    # mse_regression_list = []
    # parameter_list = []
    # gcrb_opt_list = []
    # gcrb_list = []
    # crb_list = []
    check_func = common.generate_gcrb_validation_function(dm, None, batch_size, optimal_model=model_opt,
                                                   return_full_results=True,
                                                   n_validation_point=20)
    data_dict = check_func(model)
    # print(data_dict)
    print((np.abs(data_dict["gcrb_flow"] - data_dict["crb"]) / np.abs(data_dict["crb"])).max() * 100)
    parameter = data_dict["parameter"][:, 0]
    plt.plot(parameter, np.diagonal(data_dict["crb"], axis1=1, axis2=2).sum(axis=-1), label="CRB")
    plt.plot(parameter, np.diagonal(data_dict["gcrb_optimal_flow"], axis1=1, axis2=2).sum(axis=-1), "--x",
             label="GCRB - Optimal NF")
    plt.plot(parameter, np.diagonal(data_dict["gcrb_flow"], axis1=1, axis2=2).sum(axis=-1), "--+",
             label="GCRB - Learned NF")
    plt.ylabel("MSE")
    plt.xlabel(r"$\theta$")
    plt.grid()
    plt.legend()
    plt.show()
