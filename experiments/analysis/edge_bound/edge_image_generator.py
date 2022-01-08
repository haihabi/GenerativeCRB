import numpy as np
import torch
from experiments import constants


class EdgeImageGenerator(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size
        self.c_b = torch.tensor(
            np.asarray([0.34292033, 0.22350113, 0.33940127, 0.13547006]).astype("float32").reshape(1, 1, 1, -1),
            device=constants.DEVICE)
        self.c_a = torch.tensor(
            np.asarray([0.03029606, 0.02065253, 0.02987719, 0.01571027]).astype("float32").reshape(1, 1, 1, -1),
            device=constants.DEVICE)

        self.x_array = torch.tensor(np.linspace(0, patch_size - 1, patch_size).astype("float32"),
                                    device=constants.DEVICE).reshape([1, 1, -1, 1])

    def get_image_function(self, edge_width, color_swip):
        def generate_image(in_theta):
            in_theta = in_theta.reshape([in_theta.shape[0], 1, 1, 1])
            p = torch.sigmoid((in_theta - self.x_array) / edge_width)
            p_h, p_l = self.get_pixel_color(color_swip)
            alpha = p_h - p_l
            i_x = alpha * p + p_l
            i_xy = i_x.repeat([1, self.patch_size, 1, 1])
            return i_xy

        return generate_image

    def get_pixel_color(self, color_swip):

        if color_swip:
            p_low = self.c_b
            p_high = self.c_a

        else:
            p_low = self.c_a
            p_high = self.c_b
        return p_high, p_low
