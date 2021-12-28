import numpy as np
from main import generate_flow_model
import wandb
import os
import constants
import torch
import data_model
import normflowpy as nfp


def get_data_model(in_config):
    def generate_model_parameter_dict() -> dict:
        return {constants.DIM: in_config[constants.DIM],
                constants.THETA_MIN: in_config[constants.THETA_MIN],
                constants.THETA_DIM: in_config.get(constants.THETA_DIM, 1),
                constants.SIGMA_N: in_config[constants.SIGMA_N],
                constants.THETA_MAX: in_config[constants.THETA_MAX]}

    model_type = data_model.ModelType[in_config["model_type"].split(".")[-1]]
    return data_model.get_model(model_type, generate_model_parameter_dict()), model_type


def load_wandb_run(run_name):
    api = wandb.Api()
    runs = api.runs(f"HVH/GenerativeCRB")
    for run in runs:
        print(run.name, run.state)
        if run.name == run_name:
            if os.path.isfile("flow_best.pt"):
                os.remove("flow_best.pt")
            run.file("flow_best.pt").download()

            config = run.config
            model_flow = generate_flow_model(config['dim'], config.get("theta_dim", 1), config['n_flow_blocks'],
                                             config["spline_flow"], config.get("affine_coupling", False),
                                             n_layer_cond=config["n_layer_cond"],
                                             hidden_size_cond=config["hidden_size_cond"],
                                             bias=config["mlp_bias"],
                                             affine_scale=config["affine_scale"],
                                             spline_k=config.get("spline_k", 8),
                                             spline_b=config.get("spline_b", 3))
            model_flow.load_state_dict(torch.load(f"flow_best.pt", map_location=torch.device('cpu')))
            model_flow = model_flow.to(constants.DEVICE).eval()
            for flow in model_flow.flow.flows:
                if isinstance(flow, nfp.flows.ActNorm):
                    flow.data_dep_init_done = True

            dm, model_type = get_data_model(config)
            if model_type == data_model.ModelType.Linear:
                if os.path.isfile(f"{dm.model_name}_model.pt"):
                    os.remove(f"{dm.model_name}_model.pt")
                run.file(f"{dm.model_name}_model.pt").download()
                dm.load_data_model("")
            return model_flow, dm, config


def db(x):
    return 10 * np.log10(x)
