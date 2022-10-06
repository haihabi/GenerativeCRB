import numpy as np
import wandb
import os
from experiments import constants
import torch
from experiments import data_model
import pickle
import gcrb
from experiments.main import generate_flow_model

META_FILE = "metadata_edge.pickle"

with open(os.path.join(os.path.dirname(__file__), META_FILE), "rb") as f:
    bayer_2by2, wb, cst2 = pickle.load(f)


def unpack_raw(raw4ch):
    """Unpacks 4 channels to Bayer image (h/2, w/2, 4) --> (h, w)."""
    img_shape = raw4ch.shape
    h = img_shape[0]
    w = img_shape[1]
    # d = img_shape[2]
    bayer = np.zeros([h * 2, w * 2], dtype=np.float32)
    # bayer = raw4ch
    # bayer.reshape((h * 2, w * 2))
    bayer[0::2, 0::2] = raw4ch[:, :, 0]
    bayer[0::2, 1::2] = raw4ch[:, :, 1]
    bayer[1::2, 1::2] = raw4ch[:, :, 2]
    bayer[1::2, 0::2] = raw4ch[:, :, 3]
    return bayer


def image_channel_swipe_nhwc2nchw(in_image):
    return torch.permute(in_image, (0, 3, 1, 2))


def image_channel_swipe_nchw2nhwc(in_image):
    return torch.permute(in_image, (0, 2, 3, 1))


def image_shape(patch_size):
    return [4, patch_size, patch_size]


def rggb2rgb(in_image):
    try:
        from sidd.pipeline import process_sidd_image
    except:
        raise Exception("Please import NoiseFlow repo")
    image_srgb = process_sidd_image(unpack_raw(in_image), bayer_2by2, wb, cst2)
    return image_srgb.astype('int')


def get_data_model(in_config):
    def generate_model_parameter_dict() -> dict:
        return {constants.DIM: in_config[constants.DIM],
                constants.THETA_MIN: in_config[constants.THETA_MIN],
                constants.THETA_DIM: in_config.get(constants.THETA_DIM, 1),
                constants.QUANTIZATION: in_config.get(constants.QUANTIZATION, False),
                constants.PHASENOISE:in_config.get(constants.PHASENOISE,False),
                constants.BITWIDTH: in_config.get("q_bit_width", 8),
                constants.THRESHOLD: in_config.get("q_threshold", 1),
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
            dm, model_type = get_data_model(config)
            model_flow = generate_flow_model(config['dim'], dm.theta_dim, config['n_flow_blocks'],
                                             config["spline_flow"], config.get("affine_coupling", False),
                                             n_layer_cond=config["n_layer_cond"],
                                             hidden_size_cond=config["hidden_size_cond"],
                                             bias=config["mlp_bias"],
                                             affine_scale=config["affine_scale"],
                                             spline_k=config.get("spline_k", 8),
                                             spline_b=config.get("spline_b", 3),
                                             sine_layer=config.get("sine_flow", False),
                                             dual_flow=config.get("dual_flow", False),
                                             neighbor_splitting=config.get("neighbor_splitting", False))
            model_flow.load_state_dict(torch.load(f"flow_best.pt", map_location=torch.device('cpu')))
            model_flow = model_flow.to(constants.DEVICE).eval()
            # data_mean_temp = [t.split("\n")[0].split("]]")[0].split("[[")[-1] for t in
            #                   config["trimming_mean"].split(" ")]
            # data_mean = np.asarray([float(b) for b in data_mean_temp if len(b) > 0])
            # trimming_max = config["trimming_b_max"]
            # trimming_percentile = config["trimming_b_percentile"]
            if model_type == data_model.ModelType.Linear:
                if os.path.isfile(f"{dm.model_name}_model.pt"):
                    os.remove(f"{dm.model_name}_model.pt")
                run.file(f"{dm.model_name}_model.pt").download()
                dm.load_data_model("")
            return model_flow, dm, config, None #gcrb.TrimmingParameters(data_mean, trimming_max, trimming_percentile)


def db(x):
    return 10 * np.log10(x)
