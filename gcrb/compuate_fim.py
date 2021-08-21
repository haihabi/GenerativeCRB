import torch
import torch.autograd as autograd
import constants


def jacobian_single(out_gen, z):
    grad_list = []
    for i in range(out_gen.shape[1]):
        gradients = autograd.grad(outputs=out_gen[:, i], inputs=z,
                                  grad_outputs=torch.ones(out_gen[:, i].size(), requires_grad=False).to(
                                      constants.DEVICE),
                                  create_graph=False, retain_graph=True, only_inputs=True)[0]
        grad_list.append(gradients)
    return torch.stack(grad_list, dim=-1).transpose(-1, -2)


def compute_fim(in_model, in_theta_tensor, batch_size=128):
    theta_tensor = in_theta_tensor * torch.ones([batch_size, in_theta_tensor.shape[0]], requires_grad=True,
                                                device=constants.DEVICE)
    nll_tensor = in_model.sample_nll(batch_size, cond=theta_tensor).reshape([-1, 1])
    j_matrix = jacobian_single(nll_tensor, theta_tensor)
    return torch.matmul(j_matrix.transpose(dim0=1, dim1=2), j_matrix).mean(dim=0)
