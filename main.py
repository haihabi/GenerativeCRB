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
    cr.add_parameter('dataset_size', default=100000, type=int)
    cr.add_parameter('val_dataset_size', default=20000, type=int)
    cr.add_parameter('batch_size', default=512, type=int)
    main_path = os.getcwd()
    cr.add_parameter('base_log_folder', default=os.path.join(main_path, constants.LOGS), type=str)
    cr.add_parameter('base_dataset_folder', default=os.path.join(main_path, constants.DATASETS), type=str)

    #############################################
    # Model Config
    #############################################
    cr.add_parameter('model_type', default="Pow1Div3Gaussian", type=str, enum=data_model.ModelType)
    cr.add_parameter('dim', default=2, type=int)
    cr.add_parameter('theta_min', default=0.3, type=float)
    cr.add_parameter('theta_max', default=10.0, type=float)
    cr.add_parameter('sigma_n', default=0.1, type=float)
    cr.add_parameter('load_model_data', type=str)
    ############################################
    # Regression Network
    #############################################
    cr.add_parameter('n_epochs', default=2, type=int)
    cr.add_parameter('depth', default=4, type=int)
    cr.add_parameter('width', default=32, type=int)
    #############################################
    # Regression Network - Flow
    #############################################
    cr.add_parameter('n_epochs_flow', default=2, type=int)
    cr.add_parameter('nf_weight_decay', default=1e-6, type=float)
    cr.add_parameter('nf_lr', default=1e-4, type=float)
    return cr


def check_example(current_data_model, in_regression_network, optimal_model, in_flow_model, in_min_vector, in_max_vector,
                  batch_size=4096):
    crb_list = []
    mse_regression_list = []
    parameter_list = []
    ml_mse_list = []
    gcrb_opt_list = []
    gcrb_back_list = []
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

        fim_back = gcrb.compute_fim_backward(in_flow_model, theta.reshape([1]),
                                             batch_size=batch_size)
        grcb_back = torch.linalg.inv(fim_back)

        fim = gcrb.compute_fim(in_flow_model, theta.reshape([1]), batch_size=batch_size)
        grcb_flow = torch.linalg.inv(fim)

        parameter_list.append(theta.item())
        gcrb_opt_list.append(grcb_opt.item())
        gcrb_back_list.append(grcb_back.item())
        gcrb_flow_list.append(grcb_flow.item())
    plt.clf()
    plt.cla()
    plt.plot(parameter_list, crb_list, label='CRB')
    plt.plot(parameter_list, gcrb_opt_list, label='GCRB Optimal NF')
    plt.plot(parameter_list, gcrb_flow_list, label='GCRB NF - DUAL')
    plt.plot(parameter_list, gcrb_back_list, label='GCRB NF - Backward')
    # plt.plot(parameter_list, mse_regression_list, label='Regression Network')
    # plt.plot(parameter_list, ml_mse_list, "--x", label='ML Estimator Error')
    plt.grid()
    plt.legend()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$MSE(\theta)$")
    wandb.log({"CRB Compare": wandb.Image(plt)})
    # plt.show()


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

    flows = [*list(itertools.chain(*zip(affine, affine_inj, convs, norms, flows)))]
    # condition_network = nf.MLP(1, condition_embbeding_size, 24)
    return nf.NormalizingFlowModel(MultivariateNormal(torch.zeros(in_param.dim, device=constants.DEVICE),
                                                      torch.eye(in_param.dim, device=constants.DEVICE)), flows,
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
    common.set_seed(0)

    run_parameters = cr.get_user_arguments()

    os.makedirs(run_parameters.base_log_folder, exist_ok=True)  # TODO:make a function
    wandb.init(project=constants.PROJECT, dir=run_parameters.base_log_folder)  # Set WandB Folder to log folder
    wandb.config.update(run_parameters)  # adds all of the arguments as config variables
    run_log_dir = common.generate_log_folder(run_parameters.base_log_folder)
    cr.save_config(run_log_dir)

    dm = data_model.get_model(run_parameters.model_type, generate_model_parameter_dict(run_parameters))

    os.makedirs(run_parameters.base_dataset_folder, exist_ok=True)  # TODO:make a function & change name to model names
    training_dataset_file_path = os.path.join(run_parameters.base_dataset_folder, f"training_{dm.name}_dataset.pickle")
    validation_dataset_file_path = os.path.join(run_parameters.base_dataset_folder,
                                                f"validation_{dm.name}_dataset.pickle")
    model_dataset_file_path = os.path.join(run_parameters.base_dataset_folder, "model")
    if os.path.isfile(training_dataset_file_path) and os.path.isfile(validation_dataset_file_path):
        training_data = load_dataset2file(training_dataset_file_path)
        validation_data = load_dataset2file(validation_dataset_file_path)
        dm.load_data_model(model_dataset_file_path)
        print("Loading Dataset Files")
    else:
        training_data = dm.build_dataset(run_parameters.dataset_size)
        validation_data = dm.build_dataset(run_parameters.val_dataset_size)
        save_dataset2file(training_data, training_dataset_file_path)
        save_dataset2file(validation_data, validation_dataset_file_path)
        dm.save_data_model(model_dataset_file_path)
        print("Saving Dataset Files")

    min_vector, max_vector = training_data.get_min_max_vector()  # TODO:make a function
    min_vector = torch.tensor(min_vector, device=constants.DEVICE).reshape([1, -1])
    max_vector = torch.tensor(max_vector, device=constants.DEVICE).reshape([1, -1])
    mu = min_vector
    std = max_vector - min_vector

    training_dataset_loader = torch.utils.data.DataLoader(training_data, batch_size=run_parameters.batch_size,
                                                          shuffle=True, num_workers=0)
    validation_dataset_loader = torch.utils.data.DataLoader(validation_data, batch_size=run_parameters.batch_size,
                                                            shuffle=False, num_workers=0)

    model_opt = dm.get_optimal_model()

    flow_model = generate_flow_model(run_parameters, mu, std)
    optimizer_flow = neural_network.SingleNetworkOptimization(flow_model, run_parameters.n_epochs_flow,
                                                              lr=run_parameters.nf_lr,
                                                              optimizer_type=neural_network.OptimizerType.Adam,
                                                              weight_decay=run_parameters.nf_weight_decay)
    best_flow_model, flow_model = nf.normalizing_flow_training(flow_model, training_dataset_loader,
                                                               validation_dataset_loader,
                                                               optimizer_flow, run_parameters.n_epochs_flow)
    # flow_model2check
    torch.save(best_flow_model.state_dict(), os.path.join(run_log_dir, "flow_best.pt"))

    d = best_flow_model.sample(1000, torch.tensor(2.0, device=constants.DEVICE).repeat([1000]).reshape([-1, 1]))[-1][:,
        0].detach().cpu().numpy()

    d_opt = model_opt.sample(1000, torch.tensor(2.0, device=constants.DEVICE).repeat([1000]).reshape([-1, 1]))[-1][:,
            0].detach().cpu().numpy()

    plt.hist(d_opt, density=True, label="Optimal NF Samples")
    plt.hist(d, density=True, label="NF Samples")
    plt.legend()
    plt.grid()
    wandb.log({"NF Histogram": wandb.Image(plt)})

    # plt.show()
    # neural_network.flow_train(flow, dataset_loader, optimizer_flow)
    check_example(dm, None, model_opt, best_flow_model, min_vector, max_vector)
