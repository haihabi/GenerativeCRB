import constants
from data_model.base_mode import BaseModel
from data_model.multiplication_example import MultiplicationModel
from data_model.linear_example import LinearModel
from enum import Enum


class ModelType(Enum):
    Multiplication = 0
    Linear = 1


def get_model(model_type, model_parameter_dict) -> BaseModel:
    if model_type == ModelType.Multiplication:
        return MultiplicationModel(model_parameter_dict[constants.DIM], model_parameter_dict[constants.THETA_MIN],
                                   model_parameter_dict[constants.THETA_MAX])
    elif model_type == ModelType.Linear:
        return LinearModel(model_parameter_dict[constants.DIM], model_parameter_dict[constants.THETA_MIN],
                           model_parameter_dict[constants.THETA_MAX], model_parameter_dict[constants.SIGMA_N])
    else:
        raise NotImplemented
