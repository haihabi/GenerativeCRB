import torch
import normflowpy as nfp
from experiments import constants
from torch import nn
from torch.distributions import MultivariateNormal
from experiments.models_architecture.sine_flow_layer import SineFlowLayer


def append_k_blocks(flows, n_flow_blocks, affine_coupling, spline_flow, generate_nl, input_vector_shape, dim,
                    condition_embedding_size, hidden_size_cond, n_layer_cond,
                    affine_inject_scale, bias,
                    spline_k, spline_b, act_norm=True, affine_inject=True, invertible_fully_connected=True,
                    neighbor_splitting=False, affine_coupling_scale=True):
    for i in range(n_flow_blocks):  # TODO: make this a function
        if act_norm:
            flows.append(nfp.flows.ActNorm(x_shape=input_vector_shape))
        if invertible_fully_connected:
            flows.append(
                nfp.flows.InvertibleFullyConnected(dim=dim))
        if affine_inject:
            flows.append(
                nfp.flows.AffineInjector(x_shape=input_vector_shape,
                                         condition_vector_size=condition_embedding_size, n_hidden=hidden_size_cond,
                                         net_class=nfp.base_nets.generate_mlp_class(n_layer=n_layer_cond,
                                                                                    non_linear_function=generate_nl,
                                                                                    bias=bias),
                                         scale=affine_inject_scale))

        if affine_coupling:
            flows.append(
                nfp.flows.AffineCoupling(x_shape=input_vector_shape, parity=i % 2,
                                         net_class=nfp.base_nets.generate_mlp_class(non_linear_function=generate_nl,
                                                                                    output_nl=nfp.base_nets.ScaledTanh),
                                         scale=affine_coupling_scale,
                                         neighbor_splitting=neighbor_splitting))
        if spline_flow:
            flows.append(nfp.flows.NSF_CL(dim=dim, K=spline_k, B=spline_b,
                                          base_network=nfp.base_nets.generate_mlp_class(
                                              non_linear_function=generate_nl)))


def generate_flow_model(dim, theta_dim, n_flow_blocks, spline_flow, affine_coupling, n_layer_cond=4,
                        hidden_size_cond=24, spline_b=3,
                        spline_k=8, bias=True, affine_scale=True, sine_layer=True, dual_flow=False,
                        neighbor_splitting=False):
    flows = []
    condition_embedding_size = theta_dim

    def generate_nl():
        return nn.SiLU()

    input_vector_shape = [dim]
    if dual_flow:
        append_k_blocks(flows, n_flow_blocks, affine_coupling, spline_flow, generate_nl, input_vector_shape, dim,
                        condition_embedding_size, hidden_size_cond, n_layer_cond,
                        affine_scale, bias,
                        spline_k, spline_b, act_norm=False, affine_inject=False, neighbor_splitting=neighbor_splitting,
                        affine_coupling_scale=False)
    if sine_layer:
        flows.append(SineFlowLayer(x_shape=input_vector_shape))
    append_k_blocks(flows, n_flow_blocks, affine_coupling, spline_flow, generate_nl, input_vector_shape, dim,
                    condition_embedding_size, hidden_size_cond, n_layer_cond,
                    affine_scale, bias,
                    spline_k, spline_b, neighbor_splitting=neighbor_splitting)

    return nfp.NormalizingFlowModel(MultivariateNormal(torch.zeros(dim, device=constants.DEVICE),
                                                       torch.eye(dim, device=constants.DEVICE)), flows,
                                    condition_network=None).to(
        constants.DEVICE)
