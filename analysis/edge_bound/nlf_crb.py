import numpy as np


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
    # dh_dtheta = dh_dtheta.flatten().reshape([-1, 1])

    b = np.power(alpha, 2) * h + np.power(delta, 2.0)
    a_new = np.power(alpha, 2) * dh_dtheta / b
    term_cov = 0.5 * np.power(a_new, 2.0).sum()
    term_mean = (np.power(dh_dtheta, 2.0) / b).sum()

    fim = term_mean + term_cov
    return 1 / fim


def get_crb_function(edge_width, patch_size=32):
    p_h_t = np.asarray([0.34292033, 0.22350113, 0.33940127, 0.13547006]).astype("float32")
    p_l_t = np.asarray([0.03029606, 0.02065253, 0.02987719, 0.01571027]).astype("float32")

    def f(edge_position, alpha, delta):
        return calculate_edge_crb_nlf(edge_position, edge_width, alpha, delta, p_h_t, p_l_t, patch_size, color_s=True)

    return f
