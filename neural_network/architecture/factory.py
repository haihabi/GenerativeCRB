from neural_network.architecture.multiplication.multiplication_regression import MultiplicationRegression
import data_model
import constants


def get_network(parameters, current_data_model):
    if isinstance(current_data_model, data_model.MultiplicationModel):
        return MultiplicationRegression(current_data_model.dim, parameters.depth, parameters.width).to(constants.DEVICE)
    else:
        raise Exception('')
