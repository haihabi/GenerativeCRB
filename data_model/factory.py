import constants
from data_model.base_mode import BaseModel
from data_model.pow_1_3_gaussian_variance import Pow1Div3Gaussian
from data_model.pow_3_gaussian_variance import Pow3Gaussian
from data_model.linear_example import LinearModel
from data_model.mean_example import MeanModel
from data_model.gaussian_variance import GaussianVarianceDataModel
from data_model.doa import DOAModel
from enum import Enum


# TODO:clean up
class ModelType(Enum):
    Pow1Div3Gaussian = 0
    Linear = 1
    Mean = 4
    GaussianVariance = 2
    Pow3Gaussian = 3
    DOA = 5


def get_model(model_type, model_parameter_dict) -> BaseModel:
    if model_type == ModelType.Pow1Div3Gaussian:
        return Pow1Div3Gaussian(model_parameter_dict[constants.DIM], model_parameter_dict[constants.THETA_MIN],
                                model_parameter_dict[constants.THETA_MAX])
    elif model_type == ModelType.Pow3Gaussian:
        return Pow3Gaussian(model_parameter_dict[constants.DIM], model_parameter_dict[constants.THETA_MIN],
                            model_parameter_dict[constants.THETA_MAX])
    elif model_type == ModelType.GaussianVariance:
        return GaussianVarianceDataModel(model_parameter_dict[constants.DIM], model_parameter_dict[constants.THETA_MIN],
                                         model_parameter_dict[constants.THETA_MAX])
    elif model_type == ModelType.Mean:
        return MeanModel(model_parameter_dict[constants.DIM], model_parameter_dict[constants.THETA_MIN],
                         model_parameter_dict[constants.THETA_MAX], model_parameter_dict[constants.SIGMA_N])
    elif model_type == ModelType.Linear:
        return LinearModel(model_parameter_dict[constants.DIM], model_parameter_dict[constants.THETA_MIN],
                           model_parameter_dict[constants.THETA_MAX], model_parameter_dict[constants.SIGMA_N])
    elif model_type == ModelType.DOA:
        return DOAModel(sensors_arrangement, 16, 50, 0.1, model_parameter_dict[constants.SIGMA_N])
    else:
        raise NotImplemented
