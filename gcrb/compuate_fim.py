import math

import torch
import torch.autograd as autograd
import constants
import normalizing_flow as nf
from tqdm import tqdm
import numpy as np


def jacobian_single(out_gen, z, create_graph=False):
    grad_list = []
    for i in range(out_gen.shape[1]):
        gradients = autograd.grad(outputs=out_gen[:, i], inputs=z,
                                  grad_outputs=torch.ones(out_gen[:, i].size(), requires_grad=True).to(
                                      constants.DEVICE),
                                  create_graph=create_graph, retain_graph=True, only_inputs=True, allow_unused=True)[0]
        if gradients is None:  # In case there is not gradients
            gradients = torch.zeros(z.shape, requires_grad=True).to(constants.DEVICE)
        grad_list.append(gradients)
    return torch.stack(grad_list, dim=-1).transpose(-1, -2)


def compute_fim_tensor(in_model, in_theta_tensor, batch_size=128):
    theta_tensor = in_theta_tensor * torch.ones([batch_size, in_theta_tensor.shape[0]], requires_grad=True,
                                                device=constants.DEVICE)
    nll_tensor = in_model.sample_nll(batch_size, cond=theta_tensor).reshape([-1, 1])
    j_matrix = jacobian_single(nll_tensor, theta_tensor)
    return torch.matmul(j_matrix.transpose(dim0=1, dim1=2),j_matrix)


def compute_fim(in_model, in_theta_tensor, batch_size=128):
    return compute_fim_tensor(in_model, in_theta_tensor, batch_size).mean(dim=0)


def adaptive_sampling_gfim(in_model, in_theta_tensor, batch_size=128, eps=0.01, p_min=0.1, n_max=1e7):
    fim_list = []
    status = True
    iteration_step = 1
    while status:
        for _ in tqdm(range(iteration_step)):
            fim_list.append(compute_fim_tensor(in_model, in_theta_tensor, batch_size=batch_size))
        fim_stack = torch.cat(fim_list, dim=0)

        fim_stack = fim_stack[torch.logical_not(torch.any(torch.any(fim_stack.isnan(), dim=2), dim=1)),
                    :]  # Clear non values

        var_mean_ratio = fim_stack.var(dim=0).max() / (torch.pow(fim_stack.mean(dim=0).max(), 2.0) + 1e-6)
        n_est = int(torch.ceil(var_mean_ratio / (p_min * (eps ** 2))).item())
        print(100 * fim_stack.shape[0] / n_est)
        if n_est > fim_stack.shape[0] and fim_stack.shape[0] < n_max:
            iteration_step = min(math.ceil((n_est - fim_stack.shape[0]) / batch_size), 1000)
        else:
            status = False

    print(f"Finished GFIM calculation after {fim_stack.shape[0]} Iteration")
    return fim_stack.mean(dim=0)


def compute_fim_tensors_backward(in_model: nf.NormalizingFlowModel, in_theta_tensor, batch_size=128):
    theta_tensor = in_theta_tensor * torch.ones([batch_size, in_theta_tensor.shape[0]], requires_grad=True,
                                                device=constants.DEVICE)
    z = in_model.prior.sample((batch_size,))
    z.requires_grad_()
    gamma = in_model.backward(z, theta_tensor)[0][-1]
    g_theta_matrix = jacobian_single(gamma, theta_tensor)
    j_g_matrix = jacobian_single(gamma, z, create_graph=True)

    j_gamma_z_theta_matrix = jacobian_single(j_g_matrix.reshape([batch_size, -1]), theta_tensor)
    j_gamma_z_theta_matrix = j_gamma_z_theta_matrix.reshape(
        [batch_size, *list(j_g_matrix.shape[1:]), theta_tensor.shape[1]])

    j_gamma_z_z_matrix = jacobian_single(j_g_matrix.reshape([batch_size, -1]), z)
    j_gamma_z_z_matrix = j_gamma_z_z_matrix.reshape([batch_size, *list(j_g_matrix.shape[1:]), z.shape[1]])

    j_g_matrix_inv = torch.linalg.inv(j_g_matrix)
    term1 = torch.matmul(
        torch.matmul(g_theta_matrix.transpose(dim0=1, dim1=2), j_g_matrix_inv.transpose(dim0=1, dim1=2)),
        z.unsqueeze(dim=-1)).squeeze(dim=-1)

    j_gamma_z_theta_matrix_reorder = j_gamma_z_theta_matrix.permute([0, 3, 1, 2])
    term2 = torch.matmul(j_g_matrix_inv.unsqueeze(dim=1), j_gamma_z_theta_matrix_reorder).diagonal(dim1=2, dim2=3).sum(
        dim=-1)

    term3 = -torch.matmul(j_gamma_z_z_matrix.unsqueeze(dim=-2),
                          torch.matmul(j_g_matrix_inv, g_theta_matrix).unsqueeze(dim=1).unsqueeze(dim=1)).squeeze(
        dim=-1)
    term3 = torch.matmul(j_g_matrix_inv.unsqueeze(dim=1), term3.permute([0, -1, 1, 2])).diagonal(dim1=2, dim2=3).sum(
        dim=-1)
    term_total = -term1 + term2 + term3

    term_total = term_total.unsqueeze(dim=-1)
    fim = torch.matmul(term_total.transpose(dim0=1, dim1=2), term_total)

    return fim


def compute_fim_backward(in_model: nf.NormalizingFlowModel, in_theta_tensor, batch_size=128):
    return compute_fim_tensors_backward(in_model, in_theta_tensor, batch_size).mean(dim=0)
