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
import pickle
import wandb


def config():
    cr = common.ConfigReader()
    cr.add_parameter('dataset_size', default=50000, type=int)
    cr.add_parameter('val_dataset_size', default=10000, type=int)
    cr.add_parameter('batch_size', default=64, type=int)
    main_path = os.getcwd()
    cr.add_parameter('base_log_folder', default=os.path.join(main_path, constants.LOGS), type=str)
    cr.add_parameter('base_dataset_folder', default=os.path.join(main_path, constants.DATASETS), type=str)

    #############################################
    # Model Config
    #############################################
    cr.add_parameter('model_type', default="Multiplication", type=str, enum=data_model.ModelType)
    cr.add_parameter('dim', default=2, type=int)
    cr.add_parameter('theta_min', default=0.3, type=float)
    cr.add_parameter('theta_max', default=10.0, type=float)
    cr.add_parameter('sigma_n', default=0.1, type=float)
    cr.add_parameter('load_model_data', type=str)
    #############################################
    # Regression Network
    #############################################
    cr.add_parameter('n_epochs', default=2, type=int)
    cr.add_parameter('depth', default=4, type=int)
    cr.add_parameter('width', default=32, type=int)
    #############################################
    # Regression Network - Flow
    #############################################
    cr.add_parameter('n_epochs_flow', default=240, type=int)
    cr.add_parameter('nf_weight_decay', default=0, type=float)
    cr.add_parameter('nf_lr', default=1e-4, type=float)
    return cr


def check_example(current_data_model, in_regression_network, optimal_model, in_flow_model, in_min_vector, in_max_vector,
                  batch_size=4096):
    crb_list = []
    mse_regression_list = []
    parameter_list = []
    ml_mse_list = []
    gcrb_opt_list = []
    gcrb_opt_back_list = []
    gcrb_flow_list = []
    if in_regression_network is not None:
        in_regression_network.eval()
    for theta in current_data_model.parameter_range(20):
        x = current_data_model.generate_data(512, theta)
        if in_regression_network is not None:
            theta_hat = in_regression_network(x)
            mse_regression_list.append(torch.pow(theta_hat - theta, 2.0).mean().item())

        theta_ml = current_data_model.ml_estimator(x)
        ml_mse_list.append(torch.pow(theta_ml - theta, 2.0).mean().item())

        crb_list.append(current_data_model.crb(theta).item())

        fim = gcrb.compute_fim(optimal_model, theta.reshape([1]), batch_size=batch_size)
        grcb_opt = torch.linalg.inv(fim)

        # fim_back = gcrb.compute_fim_backward(optimal_model, theta.reshape([1]),
        #                                      batch_size=batch_size)
        # grcb_opt_back = torch.linalg.inv(fim_back)

        fim = gcrb.compute_fim(in_flow_model, theta.reshape([1]), batch_size=batch_size)
        grcb_flow = torch.linalg.inv(fim)

        parameter_list.append(theta.item())
        gcrb_opt_list.append(grcb_opt.item())
        # gcrb_opt_back_list.append(grcb_opt_back.item())
        gcrb_flow_list.append(grcb_flow.item())

    plt.plot(parameter_list, crb_list, label='CRB')
    plt.plot(parameter_list, gcrb_opt_list, label='GCRB Optimal NF')
    # plt.plot(parameter_list, gcrb_opt_back_list, label='GCRB Optimal NF - Back')
    plt.plot(parameter_list, gcrb_flow_list, label='GCRB NF')
    # plt.plot(parameter_list, mse_regression_list, label='Regression Network')
    # plt.plot(parameter_list, ml_mse_list, "--x", label='ML Estimator Error')
    plt.grid()
    plt.legend()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$MSE(\theta)$")
    plt.show()


def generate_flow_model(in_param, in_mu, in_std, condition_embedding_size=1):
    nfs_flow = nf.NSF_CL if True else nf.NSF_AR
    flows = [nfs_flow(dim=in_param.dim, K=8, B=3, hidden_dim=16) for _ in range(4)]
    # flows = [MAF(dim=2, parity=i%2) for i in range(4)]
    convs = [nf.Invertible1x1Conv(dim=in_param.dim) for _ in flows]
    norms = [nf.ActNorm(in_param.dim) for _ in flows]
    affine = [
        nf.AffineHalfFlow(dim=in_param.dim, parity=i % 2, scale=True, condition_vector_size=0)
        for i, _ in
        enumerate(flows)]
    affine_inj = [nf.AffineInjector(dim=in_param.dim, scale=True, condition_vector_size=condition_embedding_size) for
                  i, _ in enumerate(flows)]

    flows = [nf.InputNorm(in_mu, in_std), *list(itertools.chain(*zip(affine, affine_inj, convs, norms, flows)))]
    # condition_network = nf.MLP(1, condition_embbeding_size, 24)
    return nf.NormalizingFlowModel(MultivariateNormal(torch.zeros(in_param.dim), torch.eye(in_param.dim)), flows,
                                   condition_network=None).to(
        constants.DEVICE)


def generate_model_parameter_dict(in_param) -> dict:
    return {constants.DIM: in_param.dim,
            constants.THETA_MIN: in_param.theta_min,
            constants.SIGMA_N: in_param.sigma_n,
            constants.THETA_MAX: in_param.theta_max}


def save_dataset2file(in_ds, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(in_ds, f)


def load_dataset2file(file_path):
    with open(file_path, "rb") as f:
        ds = pickle.load(f)
    return ds


if __name__ == '__main__':
    # TODO: refactor the main code
    cr = config()

    run_parameters = cr.get_user_arguments()
    wandb.init(project=constants.PROJECT)
    wandb.config.update(run_parameters)  # adds all of the arguments as config variables

    run_log_dir = common.generate_log_folder(run_parameters.base_log_folder)
    cr.save_config(run_log_dir)

    dm = data_model.get_model(run_parameters.model_type, generate_model_parameter_dict(run_parameters))
    if run_parameters.load_model_data is not None:
        dm.load_data_model(run_parameters.load_model_data)
    dm.save_data_model(run_log_dir)
    training_dataset_file_path = os.path.join(run_parameters.base_dataset_folder, "training_dataset.pickle")
    validation_dataset_file_path = os.path.join(run_parameters.base_dataset_folder, "validation_dataset.pickle")
    if os.path.isfile(training_dataset_file_path) and os.path.isfile(validation_dataset_file_path):
        training_data = load_dataset2file(training_dataset_file_path)
        validation_data = load_dataset2file(validation_dataset_file_path)
        print("Loading Dataset Files")
    else:
        training_data = dm.build_dataset(run_parameters.dataset_size)
        validation_data = dm.build_dataset(run_parameters.val_dataset_size)
        save_dataset2file(training_data, training_dataset_file_path)
        save_dataset2file(validation_data, validation_dataset_file_path)
        print("Saving Dataset Files")

    min_vector, max_vector = training_data.get_min_max_vector()
    min_vector = torch.tensor(min_vector, device=constants.DEVICE).reshape([1, -1])
    max_vector = torch.tensor(max_vector, device=constants.DEVICE).reshape([1, -1])
    mu = min_vector
    std = max_vector - min_vector
    # print("")
    # print(min_vector)
    # print(max_vector)

    training_dataset_loader = torch.utils.data.DataLoader(training_data, batch_size=run_parameters.batch_size,
                                                          shuffle=True, num_workers=0)
    validation_dataset_loader = torch.utils.data.DataLoader(validation_data, batch_size=run_parameters.batch_size,
                                                            shuffle=False, num_workers=0)

    prior = MultivariateNormal(torch.zeros(run_parameters.dim), torch.eye(run_parameters.dim))
    model_opt = nf.NormalizingFlowModel(prior, [dm.get_optimal_model()])
    # regression_network = neural_network.get_network(run_parameters, dm)
    # optimizer = neural_network.SingleNetworkOptimization(regression_network, run_parameters.n_epochs)
    # neural_network.regression_training(training_dataset_loader, regression_network, optimizer, torch.nn.MSELoss())
    flow_model = generate_flow_model(run_parameters, mu, std)
    optimizer_flow = neural_network.SingleNetworkOptimization(flow_model, run_parameters.n_epochs_flow, lr=run_parameters.nf_lr,
                                                              optimizer_type=neural_network.OptimizerType.Adam,
                                                              weight_decay=run_parameters.nf_weight_decay)
    best_flow_model, flow_model = nf.normalizing_flow_training(flow_model, training_dataset_loader,
                                                               validation_dataset_loader,
                                                               optimizer_flow, run_parameters.n_epochs_flow)
    # flow_model2check
    torch.save(best_flow_model.state_dict(), os.path.join(run_log_dir, "flow_best.pt"))
    data_list = []
    for dr, _ in training_dataset_loader:
        data_list.append(dr)
        if len(data_list) > 15:
            break
    data2plot = torch.cat(data_list, dim=0)
    d = best_flow_model.sample(1000, torch.tensor(2.0).repeat([1000]).reshape([-1, 1]))[-1][:, 0].detach().numpy()

    plt.hist(data2plot.numpy()[:, 0], density=True, label="Real Samples")
    plt.hist(d, density=True, label="NF Samples")
    plt.legend()
    plt.grid()
    plt.show()
    # neural_network.flow_train(flow, dataset_loader, optimizer_flow)
    check_example(dm, None, model_opt, best_flow_model, min_vector, max_vector)
