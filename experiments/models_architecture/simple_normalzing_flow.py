import torch
import normflowpy as nfp
from experiments import constants
from torch import nn
from torch.distributions import MultivariateNormal
from experiments.models_architecture.sine_flow_layer import SineFlowLayer


def generate_flow_model(dim, theta_dim, n_flow_blocks, spline_flow, affine_coupling, n_layer_cond=4,
                        hidden_size_cond=24, spline_b=3,
                        spline_k=8, bias=True, affine_scale=True, sine_layer=True):
    flows = []
    condition_embedding_size = theta_dim

    def generate_nl():
        return nn.SiLU()

    input_vector_shape = [dim]

    for i in range(n_flow_blocks):
        flows.append(nfp.flows.ActNorm(x_shape=input_vector_shape))
        flows.append(
            nfp.flows.InvertibleFullyConnected(dim=dim))
        flows.append(
            nfp.flows.AffineInjector(x_shape=input_vector_shape,
                                     condition_vector_size=condition_embedding_size, n_hidden=hidden_size_cond,
                                     net_class=nfp.base_nets.generate_mlp_class(n_layer=n_layer_cond,
                                                                                non_linear_function=generate_nl,
                                                                                bias=bias),
                                     scale=affine_scale))

        if affine_coupling:
            flows.append(
                nfp.flows.AffineCoupling(x_shape=input_vector_shape, parity=i % 2,
                                         net_class=nfp.base_nets.generate_mlp_class(non_linear_function=generate_nl)))
        if spline_flow:
            flows.append(nfp.flows.NSF_CL(dim=dim, K=spline_k, B=spline_b,
                                          base_network=nfp.base_nets.generate_mlp_class(
                                              non_linear_function=generate_nl)))
    if sine_layer:
        flows.append(SineFlowLayer(x_shape=input_vector_shape))
        for i in range(n_flow_blocks): # TODO: make a function
            flows.append(nfp.flows.ActNorm(x_shape=input_vector_shape))
            flows.append(
                nfp.flows.InvertibleFullyConnected(dim=dim))
            flows.append(
                nfp.flows.AffineInjector(x_shape=input_vector_shape,
                                         condition_vector_size=condition_embedding_size, n_hidden=hidden_size_cond,
                                         net_class=nfp.base_nets.generate_mlp_class(n_layer=n_layer_cond,
                                                                                    non_linear_function=generate_nl,
                                                                                    bias=bias),
                                         scale=affine_scale))

            if affine_coupling:
                flows.append(
                    nfp.flows.AffineCoupling(x_shape=input_vector_shape, parity=i % 2,
                                             net_class=nfp.base_nets.generate_mlp_class(
                                                 non_linear_function=generate_nl)))
            if spline_flow:
                flows.append(nfp.flows.NSF_CL(dim=dim, K=spline_k, B=spline_b,
                                              base_network=nfp.base_nets.generate_mlp_class(
                                                  non_linear_function=generate_nl)))
    return nfp.NormalizingFlowModel(MultivariateNormal(torch.zeros(dim, device=constants.DEVICE),
                                                       torch.eye(dim, device=constants.DEVICE)), flows,
                                    condition_network=None).to(
        constants.DEVICE)
