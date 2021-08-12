import torch
import common
import data_model
import neural_network
from matplotlib import pyplot as plt


def config():
    cr = common.ConfigReader()
    cr.add_parameter('dataset_size', default=50000, type=int)
    cr.add_parameter('batch_size', default=64, type=int)
    #############################################
    # Regression Network
    #############################################
    cr.add_parameter('n_epochs', default=5, type=int)
    cr.add_parameter('depth', default=4, type=int)
    cr.add_parameter('width', default=32, type=int)
    return cr.read_parameters()


def check_example(current_data_model, in_regression_network):
    crb_list = []
    mse_regression_list = []
    parameter_list = []
    ml_mse_list = []
    in_regression_network.eval()
    for theta in current_data_model.parameter_range(20):
        x = current_data_model.generate_data(512, theta)
        theta_hat = in_regression_network(x)
        theta_ml = current_data_model.ml_estimator(x)
        mse_regression_list.append(torch.pow(theta_hat - theta, 2.0).mean().item())
        ml_mse_list.append(torch.pow(theta_ml - theta, 2.0).mean().item())
        crb_list.append(current_data_model.crb(theta).item())
        parameter_list.append(theta.item())

    plt.plot(parameter_list, crb_list, label='CRB')
    plt.plot(parameter_list, mse_regression_list, label='Regression Network')
    plt.plot(parameter_list, ml_mse_list, label='ML Estimator Error')
    plt.grid()
    plt.legend()
    plt.xlabel(r"$\theta$")
    plt.show()


if __name__ == '__main__':
    run_parameters = config()
    dm = data_model.MultiplicationModel(1, 0.2, 10)
    training_data = dm.build_dataset(run_parameters.dataset_size)
    dataset_loader = torch.utils.data.DataLoader(training_data, batch_size=run_parameters.batch_size,
                                                 shuffle=True, num_workers=0)
    regression_network = neural_network.get_network(run_parameters, dm)
    optimizer = neural_network.SingleNetworkOptimization(regression_network)
    neural_network.regression_training(dataset_loader, regression_network, optimizer, torch.nn.MSELoss(),
                                       run_parameters.n_epochs)
    check_example(dm, regression_network)
