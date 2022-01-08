import math
import numpy as np
import torch
from experiments import constants


def mle_gaussian(noisy_image, in_p_h, in_p_l, in_edge_width):
    patch_size = noisy_image.shape[1]
    norm_image = (noisy_image - in_p_l) / (in_p_h - in_p_l)
    mean_image_wc = norm_image.mean(dim=(1, 3))
    index = torch.tensor(np.linspace(0, patch_size - 1, patch_size).astype("float32"),
                         device=constants.DEVICE).reshape([1, 1, -1, 1])
    w_exp = torch.exp(index / in_edge_width)
    s = (1 - mean_image_wc.mean(dim=-1)) / (w_exp.reshape([1, -1]) * mean_image_wc).mean(dim=-1)
    return -in_edge_width * torch.log(s)
    # pass


if __name__ == '__main__':
    from experiments.analysis.edge_bound.edge_image_generator import EdgeImageGenerator

    color_swip = False
    eig = EdgeImageGenerator(32)
    p_h, p_l = eig.get_pixel_color(color_swip=color_swip)
    edge_width = 8
    batch_size = 64
    n_iter = 4000
    image_func = eig.get_image_function(edge_width, color_swip=color_swip)
    m_iter_loop = math.ceil(n_iter / batch_size)
    results_list = []
    for theta in np.linspace(0, 31, 32):
        im = image_func(theta * torch.ones(batch_size))
        _res_list = []
        for _ in range(m_iter_loop):
            im_noise = im + 0.001 * torch.randn(im.shape, device=constants.DEVICE)
            im_hat = mle_gaussian(im_noise, p_h, p_l, edge_width)
        _res_list.append(np.power(theta - im_hat.detach().numpy(), 2.0))
        results_list.append(np.mean(np.concatenate(_res_list)))

    from matplotlib import pyplot as plt

    plt.plot(results_list)
    plt.show()
    # print(im.shape)
