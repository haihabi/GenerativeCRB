from neural_network.architecture.multiplication.multiplication_regression import MultiplicationRegression
from neural_network.architecture import normalizing_flows as nf
from neural_network.architecture.layer.multilayer_perceptron import MultilayerPerceptron
import data_model
import constants


def get_network(parameters, current_data_model):
    if isinstance(current_data_model, data_model.MultiplicationModel):
        net_reg = MultiplicationRegression(current_data_model.dim, parameters.depth, parameters.width).to(
            constants.DEVICE)
        dim = parameters.dim
        dim_split = dim // 2

        flow_steps_list = [nf.BatchNormFlow(dim),
                           nf.AffineCoupling(dim,
                                             mlp_addition=MultilayerPerceptron(
                                                 [dim_split + current_data_model.parameter_vector_length, 2 * dim_split,
                                                  dim_split],
                                                 last_layer_act=False),
                                             mlp_scale=MultilayerPerceptron(
                                                 [dim_split + current_data_model.parameter_vector_length, 2 * dim_split,
                                                  dim_split],
                                                 last_layer_act=False)),
                           nf.Permute(dim, mode='swap'),
                           nf.BatchNormFlow(dim),
                           nf.AffineCoupling(dim,
                                             mlp_addition=MultilayerPerceptron(
                                                 [dim_split + current_data_model.parameter_vector_length, 2 * dim_split,
                                                  dim_split],
                                                 last_layer_act=False),
                                             mlp_scale=MultilayerPerceptron(
                                                 [dim_split + current_data_model.parameter_vector_length, 2 * dim_split,
                                                  dim_split],
                                                 last_layer_act=False))
                           # nf.Permute(dim, mode='swap'),
                           # nf.BatchNormFlow(dim),
                           # nf.AffineCoupling(dim,
                           #                   mlp_addition=MultilayerPerceptron(
                           #                       [dim_split + current_data_model.parameter_vector_length, 2 * dim_split,
                           #                        dim_split],
                           #                       last_layer_act=False),
                           #                   mlp_scale=MultilayerPerceptron(
                           #                       [dim_split + current_data_model.parameter_vector_length, 2 * dim_split,
                           #                        dim_split],
                           #                       last_layer_act=False)),
                           # nf.Permute(dim, mode='swap'),
                           # nf.BatchNormFlow(dim),
                           # nf.AffineCoupling(dim,
                           #                   mlp_addition=MultilayerPerceptron(
                           #                       [dim_split + current_data_model.parameter_vector_length, 2 * dim_split,
                           #                        dim_split],
                           #                       last_layer_act=False),
                           #                   mlp_scale=MultilayerPerceptron(
                           #                       [dim_split + current_data_model.parameter_vector_length, 2 * dim_split,
                           #                        dim_split],
                           #                       last_layer_act=False))
                           ]
        flow = nf.NormalizingFlow(dim, flow_steps_list=flow_steps_list).to(constants.DEVICE)
        return net_reg, flow
    else:
        raise Exception('')
