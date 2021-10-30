import torch
import common
import data_model
import neural_network

import normflowpy as nf
from torch.distributions import MultivariateNormal
from torch import nn
import os
import constants
import pickle
import wandb


def config():
    cr = common.ConfigReader()
    cr.add_parameter('dataset_size', default=200000, type=int)
    cr.add_parameter('val_dataset_size', default=20000, type=int)
    cr.add_parameter('batch_size', default=512, type=int)
    cr.add_parameter('n_validation_point', default=20, type=int)
    cr.add_parameter('batch_size_validation', default=4096, type=int)
    cr.add_parameter('group', default="", type=str)
    main_path = os.getcwd()
    cr.add_parameter('base_log_folder', default=os.path.join(main_path, constants.LOGS), type=str)
    cr.add_parameter('base_dataset_folder', default=os.path.join(main_path, constants.DATASETS), type=str)

    #############################################
    # Model Config
    #############################################
    cr.add_parameter('model_type', default="Pow1Div3Gaussian", type=str, enum=data_model.ModelType)
    cr.add_parameter('dim', default=2, type=int)
    cr.add_parameter('theta_min', default=1, type=float)
    cr.add_parameter('theta_max', default=3.0, type=float)
    cr.add_parameter('theta_dim', default=1, type=int)
    cr.add_parameter('sigma_n', default=0.1, type=float)
    ############################################
    # Regression Network
    #############################################
    cr.add_parameter('n_epochs', default=2, type=int)
    cr.add_parameter('depth', default=4, type=int)
    cr.add_parameter('width', default=32, type=int)
    #############################################
    # Regression Network - Flow
    #############################################
    cr.add_parameter('n_epochs_flow', default=250, type=int)
    cr.add_parameter('nf_weight_decay', default=0, type=float)
    cr.add_parameter('nf_lr', default=0.0005, type=float)
    cr.add_parameter('grad_norm_clipping', default=0.1, type=float)

    cr.add_parameter('n_flow_blocks', default=9, type=int)
    cr.add_parameter('n_layer_cond', default=4, type=int)
    cr.add_parameter('hidden_size_cond', default=32, type=int)
    cr.add_parameter('evaluation_every_step', type=str, default="false")
    cr.add_parameter('spline_flow', type=str, default="true")
    return cr


def generate_flow_model(dim, theta_dim, n_flow_blocks, spline_flow, n_layer_cond=4,
                        hidden_size_cond=24, spline_b=3,
                        spline_k=8):
    flows = []
    affine_coupling = False
    condition_embedding_size = theta_dim

    def generate_nl():
        return nn.PReLU(init=1.0)

    for i in range(n_flow_blocks):
        if affine_coupling:
            flows.append(
                nf.AffineHalfFlow(dim=dim, parity=i % 2, scale=True))
        flows.append(
            nf.AffineInjector(dim=dim, net_class=nf.generate_mlp_class(hidden_size_cond, n_layer=n_layer_cond,
                                                                       non_linear_function=generate_nl), scale=True,
                              condition_vector_size=condition_embedding_size))

        flows.append(
            nf.InvertibleFullyConnected(dim=dim))
        if spline_flow and i != (n_flow_blocks - 1):
            flows.append(nf.NSF_CL(dim=dim, K=spline_k, B=spline_b))

    return nf.NormalizingFlowModel(MultivariateNormal(torch.zeros(dim, device=constants.DEVICE),
                                                      torch.eye(dim, device=constants.DEVICE)), flows,
                                   condition_network=None).to(
        constants.DEVICE)


def generate_model_parameter_dict(in_param) -> dict:
    return {constants.DIM: in_param.dim,
            constants.THETA_MIN: in_param.theta_min,
            constants.THETA_DIM: in_param.theta_dim,
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
    training_dataset_file_path = os.path.join(run_parameters.base_dataset_folder,
                                              f"training_{dm.name}_{run_parameters.dataset_size}_dataset.pickle")
    validation_dataset_file_path = os.path.join(run_parameters.base_dataset_folder,
                                                f"validation_{dm.name}_{run_parameters.val_dataset_size}_dataset.pickle")
    model_dataset_file_path = os.path.join(run_parameters.base_dataset_folder, "models")
    os.makedirs(model_dataset_file_path, exist_ok=True)
    if dm.model_exist(model_dataset_file_path):
        dm.load_data_model(model_dataset_file_path)
        print("Load Model")
    else:
        dm.save_data_model(model_dataset_file_path)
        print("Save Model")
    dm.save_data_model(wandb.run.dir)

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
    common.set_seed(0)

    training_dataset_loader = torch.utils.data.DataLoader(training_data, batch_size=run_parameters.batch_size,
                                                          shuffle=True, num_workers=0)
    validation_dataset_loader = torch.utils.data.DataLoader(validation_data, batch_size=run_parameters.batch_size,
                                                            shuffle=False, num_workers=0)

    model_opt = dm.get_optimal_model()

    flow_model = generate_flow_model(run_parameters.dim, run_parameters.theta_dim, run_parameters.n_flow_blocks,
                                     run_parameters.spline_flow,
                                     n_layer_cond=run_parameters.n_layer_cond,
                                     hidden_size_cond=run_parameters.hidden_size_cond
                                     )
    optimizer_flow = neural_network.SingleNetworkOptimization(flow_model, run_parameters.n_epochs_flow,
                                                              lr=run_parameters.nf_lr,
                                                              optimizer_type=neural_network.OptimizerType.Adam,
                                                              weight_decay=run_parameters.nf_weight_decay,
                                                              grad_norm_clipping=run_parameters.grad_norm_clipping,
                                                              enable_lr_scheduler=True,
                                                              scheduler_steps=[int(run_parameters.n_epochs_flow / 2)])
    check_training = common.generate_gcrb_validation_function(dm, None, run_parameters.batch_size_validation,
                                                              logging=False)

    best_flow_model, flow_model = nf.normalizing_flow_training(flow_model, training_dataset_loader,
                                                               validation_dataset_loader,
                                                               optimizer_flow,
                                                               check_gcrb=check_training if run_parameters.evaluation_every_step else None)
    # Save Flow to Weights and Bias
    torch.save(best_flow_model.state_dict(), os.path.join(wandb.run.dir, "flow_best.pt"))
    torch.save(flow_model.state_dict(), os.path.join(wandb.run.dir, "flow_last.pt"))
    torch.save(best_flow_model.state_dict(), os.path.join(run_log_dir, "flow_best.pt"))
    torch.save(flow_model.state_dict(), os.path.join(run_log_dir, "flow_last.pt"))

    check_final = common.generate_gcrb_validation_function(dm, None, run_parameters.batch_size_validation,
                                                           logging=True,
                                                           n_validation_point=run_parameters.n_validation_point)
    check_final(best_flow_model)  # Check Best Flow
