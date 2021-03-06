import torch

import torch.autograd as autograd


def batch_func_jacobain(func, input_tensor):
    output_tensor = func(input_tensor)
    output_tensor = torch.reshape(output_tensor, [output_tensor.shape[0], -1])
    return batch_jacobian(output_tensor, input_tensor, output_tensor.device)


def batch_jacobian(out_gen, z, input_working_device):
    if torch.is_complex(out_gen):
        raise Exception('Jacobian is not supported for complex number')
    else:
        return jacobian_single(out_gen, z, input_working_device)


def jacobian_single(out_gen, z, input_working_device):
    grad_list = []
    for i in range(out_gen.shape[1]):
        gradients = autograd.grad(outputs=out_gen[:, i], inputs=z,
                                  grad_outputs=torch.ones(out_gen[:, i].size(), requires_grad=False).to(
                                      input_working_device),
                                  create_graph=False, retain_graph=True, only_inputs=True)[0]
        grad_list.append(gradients)
    return torch.stack(grad_list, dim=-1).transpose(-1, -2)
