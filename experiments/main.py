import torch
from torchvision import transforms

import gcrb
from experiments import common
from experiments import data_model
from experiments import constants

import os
import pickle
import wandb
import numpy as np
import normflowpy as nfp

from experiments.experiment_training.nf_training import normalizing_flow_training
from experiments.models_architecture.simple_normalzing_flow import generate_flow_model
from experiments.experiment_training.single_network_optimization import SingleNetworkOptimization, OptimizerType
from experiments.analysis.bound_validation import generate_gcrb_validation_function
from experiments.analysis.dataset_validation import dataset_vs_testset_checking


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
    cr.add_parameter('m', default=64000, type=int)
    cr.add_parameter('force_data_generation', type=str, default="false")
    #############################################
    # Model Config
    #############################################
    cr.add_parameter('model_type', default="Pow1Div3Gaussian", type=str, enum=data_model.ModelType)
    cr.add_parameter('dim', default=2, type=int)
    cr.add_parameter('theta_min', default=1, type=float)
    cr.add_parameter('theta_max', default=3.0, type=float)
    cr.add_parameter('theta_dim', default=1, type=int)
    cr.add_parameter('sigma_n', default=0.1, type=float)
    # FrequencyPhaseEstimation Model Parameters
    cr.add_parameter('dual_flow', type=str, default="false")
    cr.add_parameter('neighbor_splitting', type=str, default="false")
    cr.add_parameter('quantization', type=str, default="false")
    cr.add_parameter('q_bit_width', default=1, type=int)
    cr.add_parameter('q_threshold', default=1, type=int)
    cr.add_parameter('phase_noise', type=str, default="false")
    cr.add_parameter('phase_noise_scale', default=0.1, type=float)
    #############################################
    # Regression Network - Flow
    #############################################
    cr.add_parameter('n_epochs_flow', default=40, type=int)
    cr.add_parameter('nf_weight_decay', default=1e-5, type=float)
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
    cr.add_parameter('sine_flow', type=str, default="false")
    cr.add_parameter('affine_coupling', type=str, default="false")
    cr.add_parameter('enable_lr_scheduler', type=str, default="false")
    #############################################
    # Trimming
    #############################################
    cr.add_parameter("trimming_type", default="ALL", type=str, enum=gcrb.TrimmingType)
    cr.add_parameter("trimming_p", default=0.99, type=float)
    return cr


def generate_model_parameter_dict(in_param) -> dict:
    return {constants.DIM: in_param.dim,
            constants.THETA_MIN: in_param.theta_min,
            constants.THETA_DIM: in_param.theta_dim,
            constants.QUANTIZATION: in_param.quantization,
            constants.PHASENOISE: in_param.phase_noise,
            constants.PHASENOISESCALE: in_param.phase_noise_scale,
            constants.BITWIDTH: in_param.q_threshold,
            constants.THRESHOLD: in_param.theta_dim,
            constants.SIGMA_N: in_param.sigma_n,
            constants.THETA_MAX: in_param.theta_max}


def save_dataset2file(in_ds, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(in_ds, f)


def load_dataset2file(file_path):
    with open(file_path, "rb") as f:
        ds = pickle.load(f)
    return ds


def calculate_trimming_parameter(in_dataset, p):
    data_matrix = np.stack(in_dataset.data, axis=0)
    return gcrb.calculate_trimming_parameters(data_matrix, p=p)


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
    if dm.model_exist(model_dataset_file_path) and not run_parameters.force_data_generation:
        dm.load_data_model(model_dataset_file_path)
        print("Load Model")
    else:
        dm.save_data_model(model_dataset_file_path)
        print("Save Model")
    dm.save_data_model(wandb.run.dir)

    transform = None
    if dm.is_quantized:
        print("Running quantized model")

        transform = transforms.Compose([
            nfp.preprocess.NumPyArray2Tensor(),
            nfp.preprocess.Dequatization(bit_width=run_parameters.q_bit_width, threshold_max=run_parameters.q_threshold,
                                         threshold_min=-run_parameters.q_threshold)])

    if os.path.isfile(training_dataset_file_path) and os.path.isfile(
            validation_dataset_file_path) and not run_parameters.force_data_generation:
        training_data = load_dataset2file(training_dataset_file_path)
        validation_data = load_dataset2file(validation_dataset_file_path)
        print("Loading Dataset Files")
    else:
        training_data = dm.build_dataset(run_parameters.dataset_size, transform)
        validation_data = dm.build_dataset(run_parameters.val_dataset_size, transform)
        save_dataset2file(training_data, training_dataset_file_path)
        save_dataset2file(validation_data, validation_dataset_file_path)
        print("Saving Dataset Files")
    common.set_seed(0)
    dataset_vs_testset_checking(dm, training_data)
    dataset_vs_testset_checking(dm, validation_data)
    trimming_parameters = calculate_trimming_parameter(training_data, run_parameters.trimming_p)
    wandb.config.update(trimming_parameters.as_dict())  # Adding trimming parameters to config
    trimming_module = gcrb.AdaptiveTrimming(trimming_parameters, run_parameters.trimming_type).to(constants.DEVICE)

    training_dataset_loader = torch.utils.data.DataLoader(training_data, batch_size=run_parameters.batch_size,
                                                          shuffle=True, num_workers=4, pin_memory=True)
    validation_dataset_loader = torch.utils.data.DataLoader(validation_data, batch_size=run_parameters.batch_size,
                                                            shuffle=False, num_workers=4, pin_memory=True)

    model_opt = dm.get_optimal_model()

    flow_model = generate_flow_model(run_parameters.dim, dm.theta_dim, run_parameters.n_flow_blocks,
                                     run_parameters.spline_flow, run_parameters.affine_coupling,
                                     n_layer_cond=run_parameters.n_layer_cond,
                                     hidden_size_cond=run_parameters.hidden_size_cond,
                                     bias=run_parameters.mlp_bias,
                                     affine_scale=run_parameters.affine_scale,
                                     spline_b=run_parameters.spline_b,
                                     spline_k=run_parameters.spline_k,
                                     sine_layer=run_parameters.sine_flow,
                                     dual_flow=run_parameters.dual_flow,
                                     neighbor_splitting=run_parameters.neighbor_splitting,
                                     )

    optimizer_flow = SingleNetworkOptimization(flow_model, run_parameters.n_epochs_flow,
                                               lr=run_parameters.nf_lr,
                                               optimizer_type=OptimizerType.Adam,
                                               weight_decay=run_parameters.nf_weight_decay,
                                               grad_norm_clipping=run_parameters.grad_norm_clipping,
                                               enable_lr_scheduler=run_parameters.enable_lr_scheduler,
                                               scheduler_steps=[int(run_parameters.n_epochs_flow / 2)])
    check_training = generate_gcrb_validation_function(dm, None, run_parameters.batch_size_validation,
                                                       logging=False, m=run_parameters.m,
                                                       trimming_function=trimming_module)

    best_flow_model, flow_model = normalizing_flow_training(flow_model, training_dataset_loader,
                                                            validation_dataset_loader,
                                                            optimizer_flow,
                                                            check_gcrb=check_training if run_parameters.evaluation_every_step else None)
    # Save Flow to Weights and Bias
    torch.save(best_flow_model.state_dict(), os.path.join(wandb.run.dir, "flow_best.pt"))
    torch.save(flow_model.state_dict(), os.path.join(wandb.run.dir, "flow_last.pt"))
    torch.save(best_flow_model.state_dict(), os.path.join(run_log_dir, "flow_best.pt"))
    torch.save(flow_model.state_dict(), os.path.join(run_log_dir, "flow_last.pt"))

    check_final = generate_gcrb_validation_function(dm, None, run_parameters.batch_size_validation,
                                                    logging=True,
                                                    n_validation_point=run_parameters.n_validation_point,
                                                    m=run_parameters.m, trimming_function=trimming_module)
    check_final(best_flow_model)  # Check Best Flow
