import torch
import torch.autograd as autograd
import constants
import normflowpy as nf


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


def compute_fim_tensor_model(in_model, in_theta_tensor, batch_size=128, score_vector=False):
    theta_tensor = in_theta_tensor * torch.ones([batch_size, in_theta_tensor.shape[0]], requires_grad=True,
                                                device=constants.DEVICE)

    def sample_func(in_batch_size,in_theta_tensor_hat):
        return in_model.sample_nll(in_batch_size, cond=in_theta_tensor_hat).reshape([-1, 1])

    return compute_fim_tensor_sample_function(sample_func, theta_tensor, batch_size, score_vector=score_vector)


def compute_fim_tensor_sample_function(sample_func, in_theta_tensor, batch_size=128, score_vector=False):
    nll_tensor = sample_func(batch_size, in_theta_tensor).reshape([-1, 1])
    j_matrix = jacobian_single(nll_tensor, in_theta_tensor)
    if score_vector:  # Output also score vector
        return torch.matmul(j_matrix.transpose(dim0=1, dim1=2), j_matrix), j_matrix
    return torch.matmul(j_matrix.transpose(dim0=1, dim1=2), j_matrix)


def compute_fim_tensor(model, in_theta_tensor, batch_size=128, score_vector=False):
    if isinstance(model, nf.NormalizingFlowModel):
        return compute_fim_tensor_model(model, in_theta_tensor, batch_size=batch_size, score_vector=score_vector)
    elif callable(model):
        return compute_fim_tensor_sample_function(model, in_theta_tensor, batch_size=batch_size,
                                                  score_vector=score_vector)
    else:
        raise Exception("")


def compute_fim(in_model, in_theta_tensor, batch_size=128):
    return compute_fim_tensor_model(in_model, in_theta_tensor, batch_size).mean(dim=0)

#
#
# def compute_fim_tensors_backward(in_model: nf.NormalizingFlowModel, in_theta_tensor, batch_size=128):
#     theta_tensor = in_theta_tensor * torch.ones([batch_size, in_theta_tensor.shape[0]], requires_grad=True,
#                                                 device=constants.DEVICE)
#     z = in_model.prior.sample((batch_size,))
#     z.requires_grad_()
#     gamma = in_model.backward(z, theta_tensor)[0][-1]
#     g_theta_matrix = jacobian_single(gamma, theta_tensor)
#     j_g_matrix = jacobian_single(gamma, z, create_graph=True)
#
#     j_gamma_z_theta_matrix = jacobian_single(j_g_matrix.reshape([batch_size, -1]), theta_tensor)
#     j_gamma_z_theta_matrix = j_gamma_z_theta_matrix.reshape(
#         [batch_size, *list(j_g_matrix.shape[1:]), theta_tensor.shape[1]])
#
#     j_gamma_z_z_matrix = jacobian_single(j_g_matrix.reshape([batch_size, -1]), z)
#     j_gamma_z_z_matrix = j_gamma_z_z_matrix.reshape([batch_size, *list(j_g_matrix.shape[1:]), z.shape[1]])
#
#     j_g_matrix_inv = torch.linalg.inv(j_g_matrix)
#     term1 = torch.matmul(
#         torch.matmul(g_theta_matrix.transpose(dim0=1, dim1=2), j_g_matrix_inv.transpose(dim0=1, dim1=2)),
#         z.unsqueeze(dim=-1)).squeeze(dim=-1)
#
#     j_gamma_z_theta_matrix_reorder = j_gamma_z_theta_matrix.permute([0, 3, 1, 2])
#     term2 = torch.matmul(j_g_matrix_inv.unsqueeze(dim=1), j_gamma_z_theta_matrix_reorder).diagonal(dim1=2, dim2=3).sum(
#         dim=-1)
#
#     term3 = -torch.matmul(j_gamma_z_z_matrix.unsqueeze(dim=-2),
#                           torch.matmul(j_g_matrix_inv, g_theta_matrix).unsqueeze(dim=1).unsqueeze(dim=1)).squeeze(
#         dim=-1)
#     term3 = torch.matmul(j_g_matrix_inv.unsqueeze(dim=1), term3.permute([0, -1, 1, 2])).diagonal(dim1=2, dim2=3).sum(
#         dim=-1)
#     term_total = -term1 + term2 + term3
#
#     term_total = term_total.unsqueeze(dim=-1)
#     fim = torch.matmul(term_total.transpose(dim0=1, dim1=2), term_total)
#
#     return fim
#
#
# def compute_fim_backward(in_model: nf.NormalizingFlowModel, in_theta_tensor, batch_size=128):
#     return compute_fim_tensors_backward(in_model, in_theta_tensor, batch_size).mean(dim=0)
