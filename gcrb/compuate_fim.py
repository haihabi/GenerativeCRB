import torch
import torch.autograd as autograd
from experiments import constants
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


def compute_fim_tensor_model(in_model, in_theta_tensor, batch_size=128, score_vector=False, trimming_step=None,
                             temperature=1.0):
    theta_tensor = in_theta_tensor * torch.ones([batch_size, in_theta_tensor.shape[0]], requires_grad=True,
                                                device=constants.DEVICE)

    def sample_func(in_batch_size, in_theta_tensor_hat: torch.Tensor):
        gamma = in_model.sample(in_batch_size, cond=in_theta_tensor_hat, temperature=temperature)
        gamma = gamma.detach()
        if trimming_step is not None:
            trimming_status = trimming_step(gamma)
        else:
            trimming_status = torch.ones(in_batch_size).bool().to(in_theta_tensor_hat.device)
        return in_model.nll(gamma, in_theta_tensor_hat).reshape([-1, 1]), trimming_status

    return compute_fim_tensor_sample_function(sample_func, theta_tensor, batch_size, score_vector=score_vector)


def compute_fim_tensor_sample_function(sample_func, in_theta_tensor, batch_size=128, score_vector=False):
    nll_tensor, status = sample_func(batch_size, in_theta_tensor)
    nll_tensor = nll_tensor.reshape([-1, 1])
    j_matrix = jacobian_single(nll_tensor, in_theta_tensor)
    j_matrix = j_matrix[status, ::]  # Filter score vector
    gfim = torch.matmul(j_matrix.transpose(dim0=1, dim1=2), j_matrix)
    if score_vector:  # Output also score vector
        return gfim, j_matrix
    return gfim


def compute_fim_tensor(model, in_theta_tensor, batch_size=128, score_vector=False, trimming_step=None, temperature=1.0):
    if isinstance(model, nf.NormalizingFlowModel):
        return compute_fim_tensor_model(model, in_theta_tensor, batch_size=batch_size, score_vector=score_vector,
                                        trimming_step=trimming_step, temperature=temperature)
    elif callable(model):
        return compute_fim_tensor_sample_function(model, in_theta_tensor, batch_size=batch_size,
                                                  score_vector=score_vector)
    else:
        raise Exception("")
