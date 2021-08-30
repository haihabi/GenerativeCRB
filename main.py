import torch
import common
import data_model
import neural_network
from matplotlib import pyplot as plt

import normalizing_flow as nf
from torch.distributions import MultivariateNormal
import gcrb
import itertools
import os
import constants


def config():
    cr = common.ConfigReader()
    cr.add_parameter('dataset_size', default=200000, type=int)
    cr.add_parameter('val_dataset_size', default=20000, type=int)
    cr.add_parameter('batch_size', default=64, type=int)
    cr.add_parameter('base_log_folder', default="/Users/haihabi/projects/GenerativeCRB/logs", type=str)

    #############################################
    # Model Config
    #############################################
    cr.add_parameter('model_type', default="Linear", type=str, enum=data_model.ModelType)
    cr.add_parameter('dim', default=2, type=int)
    cr.add_parameter('theta_min', default=0.2, type=float)
    cr.add_parameter('theta_max', default=10.0, type=float)
    cr.add_parameter('sigma_n', default=1.0, type=float)
    cr.add_parameter('load_model_data', type=str)
    #############################################
    # Regression Network
    #############################################
    cr.add_parameter('n_epochs', default=5, type=int)
    cr.add_parameter('depth', default=4, type=int)
    cr.add_parameter('width', default=32, type=int)
    #############################################
    # Regression Network - Flow
    #############################################
    cr.add_parameter('n_epochs_flow', default=50, type=int)
    return cr


def check_example(current_data_model, in_regression_network, optimal_model, in_flow_model, batch_size=4096):
    crb_list = []
    mse_regression_list = []
    parameter_list = []
    ml_mse_list = []
    gcrb_opt_list = []
    gcrb_flow_list = []
    in_regression_network.eval()
    for theta in current_data_model.parameter_range(20):
        x = current_data_model.generate_data(512, theta)
        theta_hat = in_regression_network(x)
        theta_ml = current_data_model.ml_estimator(x)
        mse_regression_list.append(torch.pow(theta_hat - theta, 2.0).mean().item())
        ml_mse_list.append(torch.pow(theta_ml - theta, 2.0).mean().item())
        crb_list.append(current_data_model.crb(theta).item())

        fim = gcrb.compute_fim(optimal_model, theta.reshape([1]), batch_size=batch_size)
        grcb_opt = torch.linalg.inv(fim)

        fim = gcrb.compute_fim(in_flow_model, theta.reshape([1]), batch_size=batch_size)
        grcb_flow = torch.linalg.inv(fim)

        parameter_list.append(theta.item())
        gcrb_opt_list.append(grcb_opt.item())
        gcrb_flow_list.append(grcb_flow.item())

    plt.plot(parameter_list, crb_list, label='CRB')
    plt.plot(parameter_list, gcrb_opt_list, label='GCRB Optimal NF')
    plt.plot(parameter_list, gcrb_flow_list, label='GCRB NF')
    # plt.plot(parameter_list, mse_regression_list, label='Regression Network')
    plt.plot(parameter_list, ml_mse_list, "--x", label='ML Estimator Error')
    plt.grid()
    plt.legend()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$MSE(\theta)$")
    plt.show()


def generate_flow_model(in_param, in_mu, in_std):
    nfs_flow = nf.NSF_CL if True else nf.NSF_AR
    flows = [nfs_flow(dim=in_param.dim, K=8, B=3, hidden_dim=16) for _ in range(3)]
    # flows = [MAF(dim=2, parity=i%2) for i in range(4)]
    convs = [nf.Invertible1x1Conv(dim=in_param.dim) for _ in flows]
    norms = [nf.ActNorm(dim=in_param.dim) for _ in flows]
    affine = [nf.AffineHalfFlow(dim=in_param.dim, parity=i % 2, scale=True) for i, _ in enumerate(flows)]

    flows = [nf.InputNorm(in_mu, in_std), *list(itertools.chain(*zip(convs, affine, norms, flows)))]
    return nf.NormalizingFlowModel(MultivariateNormal(torch.zeros(in_param.dim), torch.eye(in_param.dim)), flows).to(
        constants.DEVICE)


def generate_model_parameter_dict(in_param) -> dict:
    return {constants.DIM: in_param.dim,
            constants.THETA_MIN: in_param.theta_min,
            constants.SIGMA_N: in_param.sigma_n,
            constants.THETA_MAX: in_param.theta_max}


if __name__ == '__main__':
    # TODO: refactor the main code
    cr = config()

    run_parameters = cr.get_user_arguments()
    run_log_dir = common.generate_log_folder(run_parameters.base_log_folder)
    cr.save_config(run_log_dir)
    dm = data_model.get_model(run_parameters.model_type, generate_model_parameter_dict(run_parameters))
    if run_parameters.load_model_data is not None:
        dm.load_data_model(run_parameters.load_model_data)
    dm.save_data_model(run_log_dir)

    training_data = dm.build_dataset(run_parameters.dataset_size)
    mu, std = training_data.get_second_order_stat()
    print("")
    print(mu)
    print(std)
    validation_data = dm.build_dataset(run_parameters.val_dataset_size)
    training_dataset_loader = torch.utils.data.DataLoader(training_data, batch_size=run_parameters.batch_size,
                                                          shuffle=True, num_workers=0)
    validation_dataset_loader = torch.utils.data.DataLoader(validation_data, batch_size=run_parameters.batch_size,
                                                            shuffle=True, num_workers=0)
    prior = MultivariateNormal(torch.zeros(run_parameters.dim), torch.eye(run_parameters.dim))
    model_opt = nf.NormalizingFlowModel(prior, [dm.get_optimal_model()])
    regression_network = neural_network.get_network(run_parameters, dm)
    optimizer = neural_network.SingleNetworkOptimization(regression_network, run_parameters.n_epochs)
    neural_network.regression_training(training_dataset_loader, regression_network, optimizer, torch.nn.MSELoss())
    flow_model = generate_flow_model(run_parameters, mu, std)
    optimizer_flow = neural_network.SingleNetworkOptimization(flow_model, run_parameters.n_epochs_flow, lr=1e-3,
                                                              optimizer_type=neural_network.OptimizerType.Adam,
                                                              weight_decay=1e-5)
    flow_model = nf.normalizing_flow_training(flow_model, training_dataset_loader, validation_dataset_loader,
                                              optimizer_flow, run_parameters.n_epochs_flow)

    torch.save(flow_model.state_dict(), os.path.join(run_log_dir, "flow_best.pt"))
    d = flow_model.sample(1000, torch.tensor(2.0).repeat([1000]).reshape([-1, 1]))[-1][:, 0].detach().numpy()

    # z = flow_model.prior.sample((1000,))
    # xs, _ = flow_model.flow.backward(z, torch.tensor(5.0).repeat([1000]).reshape([-1, 1]))
    plt.hist(d)
    plt.show()
    # neural_network.flow_train(flow, dataset_loader, optimizer_flow)
    check_example(dm, regression_network, model_opt, flow_model)
