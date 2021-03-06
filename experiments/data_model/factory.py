from experiments import constants
from experiments.data_model.base_mode import BaseModel
from experiments.data_model.pow_1_3_gaussian_variance import Pow1Div3Gaussian
from experiments.data_model.frequency_estimation_example import FrequencyModel
from experiments.data_model.linear_example import LinearModel
from enum import Enum


class ModelType(Enum):
    Pow1Div3Gaussian = 0
    Linear = 1
    FrequencyPhaseEstimation = 2


def get_model(model_type, model_parameter_dict) -> BaseModel:
    if model_type == ModelType.Pow1Div3Gaussian:
        return Pow1Div3Gaussian(model_parameter_dict[constants.DIM], model_parameter_dict[constants.THETA_MIN],
                                model_parameter_dict[constants.THETA_MAX])
    elif model_type == ModelType.FrequencyPhaseEstimation:
        return FrequencyModel(model_parameter_dict[constants.DIM], model_parameter_dict[constants.SIGMA_N],
                              quantization=model_parameter_dict[constants.QUANTIZATION],
                              bit_width=model_parameter_dict[constants.BITWIDTH],
                              threshold=model_parameter_dict[constants.THRESHOLD],
                              phase_noise=model_parameter_dict.get(constants.PHASENOISE,False),
                              phase_noise_scale=model_parameter_dict.get(constants.PHASENOISESCALE,0.0))
    elif model_type == ModelType.Linear:
        return LinearModel(model_parameter_dict[constants.DIM], model_parameter_dict[constants.THETA_DIM],
                           model_parameter_dict[constants.THETA_MIN],
                           model_parameter_dict[constants.THETA_MAX])

    else:
        raise NotImplemented
