import numpy as np
from experiments.analysis.edge_bound.edge_image_generator import EdgeImageGenerator


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def calculate_edge_crb_nlf(edge_position, edge_width, alpha, delta, in_p_h, in_p_l, patch_size, color_s=False):
    if color_s:
        p_h = in_p_l.reshape([1, 1, -1])
        p_l = in_p_h.reshape([1, 1, -1])
    else:
        p_h = in_p_h.reshape([1, 1, -1])
        p_l = in_p_l.reshape([1, 1, -1])
    x_array = np.linspace(0, patch_size - 1, patch_size).astype("float32").reshape([1, patch_size, 1])
    x_delta = (edge_position - x_array) / edge_width
    h = (p_h - p_l) * sigmoid(x_delta) + p_l
    h = np.tile(h, [patch_size, 1, 1])

    dh_dtheta = (p_h - p_l) * sigmoid_derivative(x_delta) / edge_width
    dh_dtheta = np.tile(dh_dtheta, [patch_size, 1, 1])

    b = np.power(alpha, 2) * h + np.power(delta, 2.0)
    a_new = np.power(alpha, 2) * dh_dtheta / b
    term_cov = 0.5 * np.power(a_new, 2.0).sum()
    term_mean = (np.power(dh_dtheta, 2.0) / b).sum()

    fim = term_mean + term_cov
    return 1 / fim


def get_crb_function(alpha, delta, edge_width, color_swip, eig: EdgeImageGenerator):
    p_h, p_l = eig.get_pixel_color(color_swip)

    def f(edge_position):
        return calculate_edge_crb_nlf(edge_position, edge_width, alpha, delta, p_h.detach().cpu().numpy(),
                                      p_l.detach().cpu().numpy(),
                                      eig.patch_size,
                                      color_s=color_swip)

    return f
