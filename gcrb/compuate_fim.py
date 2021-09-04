import torch
import torch.autograd as autograd
import constants
import normalizing_flow as nf


def jacobian_single(out_gen, z, create_graph=False):
    grad_list = []
    for i in range(out_gen.shape[1]):
        gradients = autograd.grad(outputs=out_gen[:, i], inputs=z,
                                  grad_outputs=torch.ones(out_gen[:, i].size(), requires_grad=True).to(
                                      constants.DEVICE),
                                  create_graph=create_graph, retain_graph=True, only_inputs=True, allow_unused=False)[0]
        grad_list.append(gradients)
    return torch.stack(grad_list, dim=-1).transpose(-1, -2)


def jacobian_two_single(out_gen, z, theta, create_graph=False):
    grad_list = []
    for i in range(out_gen.shape[0]):
        for j in range(out_gen.shape[1]):
            d_y_d_z = autograd.grad(outputs=out_gen[i, j], inputs=z, create_graph=True, retain_graph=True,
                                    only_inputs=False, allow_unused=False)[0]
            for k in range(d_y_d_z.shape[1]):
                d_y_d_z_theta = autograd.grad(outputs=d_y_d_z[i, k], inputs=theta,
                                          create_graph=create_graph, retain_graph=True, only_inputs=True,
                                          allow_unused=False)[0]
        grad_list.append(d_y_d_z_theta)
    return torch.stack(grad_list, dim=-1).transpose(-1, -2)


def compute_fim(in_model, in_theta_tensor, batch_size=128):
    theta_tensor = in_theta_tensor * torch.ones([batch_size, in_theta_tensor.shape[0]], requires_grad=True,
                                                device=constants.DEVICE)
    nll_tensor = in_model.sample_nll(batch_size, cond=theta_tensor).reshape([-1, 1])
    j_matrix = jacobian_single(nll_tensor, theta_tensor)
    return torch.matmul(j_matrix.transpose(dim0=1, dim1=2), j_matrix).mean(dim=0)


def compute_fim_forward(in_model, in_theta_tensor, min_value, max_value, batch_size=128):
    delta = (max_value - min_value).reshape([1, -1])
    gamma_unifrom = min_value.reshape([1, -1]) + delta * torch.rand([batch_size, min_value.shape[0]])

    theta_tensor = in_theta_tensor * torch.ones([batch_size, in_theta_tensor.shape[0]], requires_grad=True,
                                                device=constants.DEVICE)
    nll_tensor = in_model.nll(gamma_unifrom, cond=theta_tensor).reshape([-1, 1])
    j_matrix = jacobian_single(nll_tensor, theta_tensor)
    fim_per_sample = torch.matmul(j_matrix.transpose(dim0=1, dim1=2), j_matrix)
    return torch.mean(torch.prod(delta) * fim_per_sample * (torch.exp(-nll_tensor).reshape([-1, 1, 1])), dim=0)


def compute_fim_backward(in_model: nf.NormalizingFlowModel, in_theta_tensor, batch_size=128):
    theta_tensor = in_theta_tensor * torch.ones([batch_size, in_theta_tensor.shape[0]], requires_grad=True,
                                                device=constants.DEVICE)
    z = in_model.prior.sample((batch_size,))
    z.requires_grad_()
    gamma = in_model.backward(z, theta_tensor)[0][-1]
    g_theta_matrix = jacobian_single(gamma, theta_tensor)
    j_g_matrix = jacobian_single(gamma, z)

    j_g_theta_matrix = jacobian_two_single(gamma, z, theta_tensor)
    print("a")
