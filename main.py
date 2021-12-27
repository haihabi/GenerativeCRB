import torch
import common
import data_model
import neural_network

import normflowpy as nfp
from torch.distributions import MultivariateNormal
from torch import nn
import os
import constants
import pickle
import wandb
from neural_network.nf_training import normalizing_flow_training


def config():
    cr = common.ConfigReader()
    cr.add_parameter('dataset_size', default=200000, type=int)
    cr.add_parameter('val_dataset_size', default=20000, type=int)
    cr.add_parameter('batch_size', default=64, type=int)
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
    cr.add_parameter('n_epochs_flow', default=40, type=int)
    cr.add_parameter('nf_weight_decay', default=0, type=float)
    cr.add_parameter('nf_lr', default=0.00001, type=float)
    cr.add_parameter('grad_norm_clipping', default=0.1, type=float)

    cr.add_parameter('n_flow_blocks', default=9, type=int)
    cr.add_parameter('n_layer_cond', default=4, type=int)
    cr.add_parameter('spline_k', default=8, type=int)
    cr.add_parameter('spline_b', default=3.0, type=float)
    cr.add_parameter('hidden_size_cond', default=32, type=int)
    cr.add_parameter('mlp_bias', type=str, default="false")
    cr.add_parameter('affine_scale', type=str, default="false")
    cr.add_parameter('evaluation_every_step', type=str, default="true")
    cr.add_parameter('spline_flow', type=str, default="false")
    cr.add_parameter('affine_coupling', type=str, default="false")
    cr.add_parameter('enable_lr_scheduler', type=str, default="false")
    return cr


def generate_flow_model(dim, theta_dim, n_flow_blocks, spline_flow, affine_coupling, n_layer_cond=4,
                        hidden_size_cond=24, spline_b=3,
                        spline_k=8, bias=True, affine_scale=True):
    flows = []
    condition_embedding_size = theta_dim

    def generate_nl():
        return nn.SiLU()

    for i in range(n_flow_blocks):
        flows.append(nfp.flows.ActNorm(dim=dim))
        flows.append(
            nfp.flows.InvertibleFullyConnected(dim=dim))

        flows.append(
            nfp.flows.AffineInjector(dim=dim,
                                     net_class=nfp.base_nets.generate_mlp_class(hidden_size_cond, n_layer=n_layer_cond,
                                                                                non_linear_function=generate_nl,
                                                                                bias=bias),
                                     scale=affine_scale,
                                     condition_vector_size=condition_embedding_size))
        if affine_coupling:
            flows.append(
                nfp.flows.AffineCouplingFlowVector(dim=dim, parity=i % 2, scale=affine_scale))
        if spline_flow:
            flows.append(nfp.flows.NSF_CL(dim=dim, K=spline_k, B=spline_b))

    return nfp.NormalizingFlowModel(MultivariateNormal(torch.zeros(dim, device=constants.DEVICE),
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
                                              f"training_{dm.name}_{run_parameters.dataset_size}_{run_parameters.theta_min}_{run_parameters.theta_max}_dataset.pickle")
    validation_dataset_file_path = os.path.join(run_parameters.base_dataset_folder,
                                                f"validation_{dm.name}_{run_parameters.val_dataset_size}_{run_parameters.theta_min}_{run_parameters.theta_max}_dataset.pickle")
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
                                                          shuffle=True, num_workers=4, pin_memory=True)
    validation_dataset_loader = torch.utils.data.DataLoader(validation_data, batch_size=run_parameters.batch_size,
                                                            shuffle=False, num_workers=4, pin_memory=True)

    model_opt = dm.get_optimal_model()

    flow_model = generate_flow_model(run_parameters.dim, run_parameters.theta_dim, run_parameters.n_flow_blocks,
                                     run_parameters.spline_flow, run_parameters.affine_coupling,
                                     n_layer_cond=run_parameters.n_layer_cond,
                                     hidden_size_cond=run_parameters.hidden_size_cond,
                                     bias=run_parameters.mlp_bias,
                                     affine_scale=run_parameters.affine_scale,
                                     spline_b=run_parameters.spline_b,
                                     spline_k=run_parameters.spline_k,
                                     )

    # ema_flow_model = generate_flow_model(run_parameters.dim, run_parameters.theta_dim, run_parameters.n_flow_blocks,
    #                                      run_parameters.spline_flow, run_parameters.affine_coupling,
    #                                      n_layer_cond=run_parameters.n_layer_cond,
    #                                      hidden_size_cond=run_parameters.hidden_size_cond,
    #                                      bias=run_parameters.mlp_bias,
    #                                      affine_scale=run_parameters.affine_scale
    #                                      )
    optimizer_flow = neural_network.SingleNetworkOptimization(flow_model, run_parameters.n_epochs_flow,
                                                              lr=run_parameters.nf_lr,
                                                              optimizer_type=neural_network.OptimizerType.Adam,
                                                              weight_decay=run_parameters.nf_weight_decay,
                                                              grad_norm_clipping=run_parameters.grad_norm_clipping,
                                                              enable_lr_scheduler=run_parameters.enable_lr_scheduler,
                                                              scheduler_steps=[int(run_parameters.n_epochs_flow / 2)])
    check_training = common.generate_gcrb_validation_function(dm, None, run_parameters.batch_size_validation,
                                                              logging=False)
    # ema = common.ExponentialMovingAverage(flow_model.parameters(), decay=0.9)
    # ema.copy_to(ema_flow_model.parameters())
    best_flow_model, flow_model = normalizing_flow_training(flow_model, training_dataset_loader,
                                                            validation_dataset_loader,
                                                            optimizer_flow,
                                                            check_gcrb=check_training if run_parameters.evaluation_every_step else None,
                                                            ema=None,
                                                            ema_flow=None)
    # Save Flow to Weights and Bias
    torch.save(best_flow_model.state_dict(), os.path.join(wandb.run.dir, "flow_best.pt"))
    torch.save(flow_model.state_dict(), os.path.join(wandb.run.dir, "flow_last.pt"))
    torch.save(best_flow_model.state_dict(), os.path.join(run_log_dir, "flow_best.pt"))
    torch.save(flow_model.state_dict(), os.path.join(run_log_dir, "flow_last.pt"))

    check_final = common.generate_gcrb_validation_function(dm, None, run_parameters.batch_size_validation,
                                                           logging=True,
                                                           n_validation_point=run_parameters.n_validation_point)
    check_final(best_flow_model)  # Check Best Flow
