from neural_network.architecture.multiplication.multiplication_regression import MultiplicationRegression
import data_model
import constants


def get_network(parameters, current_data_model):
    if isinstance(current_data_model, data_model.MultiplicationModel) or isinstance(current_data_model,
                                                                                    data_model.LinearModel):
        net_reg = MultiplicationRegression(current_data_model.dim, parameters.depth, parameters.width).to(
            constants.DEVICE)

        return net_reg
    else:
        raise Exception('')
