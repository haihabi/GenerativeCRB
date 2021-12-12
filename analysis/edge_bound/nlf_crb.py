import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def calculate_edge_crb_nlf(edge_position, edge_width, alpha, delta, in_p_h, in_p_l, patch_size, color_s=True):
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
    dh_dtheta = dh_dtheta.flatten().reshape([-1, 1])

    c = np.power(alpha, 2) * h.flatten() + np.power(delta, 2.0)
    c = np.diag(c)
    c_inv = np.linalg.inv(c)
    dc_dtheta = np.diag(np.power(alpha, 2) * dh_dtheta.flatten())
    a = np.matmul(c_inv, dc_dtheta)
    term_cov = 0.5 * np.trace(np.matmul(a.T, a))
    term_mean = np.matmul(np.matmul(dh_dtheta.T, c_inv), dh_dtheta)
    fim = term_mean + term_cov
    return 1 / fim


def get_crb_function(edge_width, patch_size=32):
    p_h_t = np.asarray([0.34292033, 0.22350113, 0.33940127, 0.13547006]).astype("float32")
    p_l_t = np.asarray([0.03029606, 0.02065253, 0.02987719, 0.01571027]).astype("float32")

    def f(edge_position, alpha, delta):
        return calculate_edge_crb_nlf(edge_position, edge_width, alpha, delta, p_h_t, p_l_t, patch_size, color_s=True)

    return f


if __name__ == '__main__':
    # from sidd.pipeline import process_sidd_image
    # from matplotlib import pyplot as plt
    # import pickle
    #
    #
    # def unpack_raw(raw4ch):
    #     """Unpacks 4 channels to Bayer image (h/2, w/2, 4) --> (h, w)."""
    #     img_shape = raw4ch.shape
    #     h = img_shape[0]
    #     w = img_shape[1]
    #     # d = img_shape[2]
    #     bayer = np.zeros([h * 2, w * 2], dtype=np.float32)
    #     # bayer = raw4ch
    #     # bayer.reshape((h * 2, w * 2))
    #     bayer[0::2, 0::2] = raw4ch[:, :, 0]
    #     bayer[0::2, 1::2] = raw4ch[:, :, 1]
    #     bayer[1::2, 1::2] = raw4ch[:, :, 2]
    #     bayer[1::2, 0::2] = raw4ch[:, :, 3]
    #     return bayer

    edge_position_array = np.linspace(1, 30, 20)
    print(edge_position_array)
    plt.plot(edge_position_array,
             10 * np.log10(
                 np.asarray([calculate_edge_crb_nlf(edge_position, 2, 2.0, 1.0, p_h_t, p_l_t, 32) for edge_position in
                             edge_position_array]).flatten()))
    plt.show()

    # with open("../metadata_edge.pickle", "rb") as f:
    #     bayer_2by2, wb, cst2 = pickle.load(f)
    # I = process_sidd_image(unpack_raw(edge_i), bayer_2by2, wb, cst2)
    # plt.imshow(I.astype('int'))
    # plt.axis('off')
    # plt.show()
