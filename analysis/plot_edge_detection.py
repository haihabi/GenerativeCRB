# from matplotlib import pyplot as plt
#
# width_array = [1, 1.1, 1.5, 2, 4, 6, 8, 16, 24, 31]
# width_results = [-41.4398, -47.0369, -45.4288, -44.9259, -44.4628, -43.4858, -42.6747, -40.2980, -37.9747, -36.4974]
#
# plt.plot(width_array, width_results)
# plt.xlabel("Edge width")
# plt.ylabel("MSE[dB]")
# plt.grid()
# plt.show()
#
# position_results = [-45.1646, -45.1061, -45.0113, -45.1221, -44.9504, -44.9909, -44.9866]
# position_results_cross = [-45.0214, -49.5257, -48.2796, -48.3650, -48.3871, -48.3560, -48.3051]
# cross_point_array = [1, 2, 4, 8, 16, 24, 31]
#
# position_results = [ -45.0113, -45.1221, -44.9504, -44.9909, -44.9866]
# position_results_cross = [ -48.2796, -48.3650, -48.3871, -48.3560, -48.3051]
# cross_point_array = [ 4, 8, 16, 24, 31]
# plt.subplot(1,2,1)
# plt.plot(cross_point_array,position_results)
# plt.xlabel("Edge position")
# plt.ylabel("MSE[dB]")
# plt.grid()
# plt.title("Original Colors")
# plt.subplot(1,2,2)
# plt.plot(cross_point_array,position_results_cross)
# plt.xlabel("Edge position")
# plt.ylabel("MSE[dB]")
# plt.title("Swap colors")
# plt.grid()
# plt.show()
import pickle

import torch
import constants
import numpy as np
import os
import glob
from sidd.raw_utils import read_metadata
from sidd.pipeline import process_sidd_image, flip_bayer, stack_rggb_channels
from matplotlib import pyplot as plt


def unpack_raw(raw4ch):
    """Unpacks 4 channels to Bayer image (h/2, w/2, 4) --> (h, w)."""
    img_shape = raw4ch.shape
    h = img_shape[0]
    w = img_shape[1]
    # d = img_shape[2]
    bayer = np.zeros([h * 2, w * 2], dtype=np.float32)
    # bayer = raw4ch
    # bayer.reshape((h * 2, w * 2))
    bayer[0::2, 0::2] = raw4ch[:, :, 0]
    bayer[0::2, 1::2] = raw4ch[:, :, 1]
    bayer[1::2, 1::2] = raw4ch[:, :, 2]
    bayer[1::2, 0::2] = raw4ch[:, :, 3]
    return bayer


dataset_folder = "/data/datasets/SIDD_Medium_Raw/Data/"
scene_number = "001"
iso_str = "00100"
illumination_code = "N"
# iso = 100
folder_name = f"*_{scene_number}_S6_{iso_str}_*_*_{illumination_code}"

folder_optins = glob.glob(os.path.join(dataset_folder, folder_name))
temp = [int(fo.split("_")[-2]) for fo in folder_optins]
i = np.argmax(temp)

folder_base = folder_optins[i]
print(f"illuminant temperature:{temp[i]}", folder_base)

metadata, bayer_2by2, wb, cst2, _, _ = read_metadata(
    glob.glob(f"{folder_base}/*_METADATA_RAW_010.MAT")[0])

import pickle
d = (bayer_2by2, wb, cst2)
with open("metadata_edge.pickle", "wb") as f:
    pickle.dump(d, f)


class GenerateEdgeFunction(object):
    def __init__(self, patch_size=32):
        self.patch_size = patch_size
        self.c_b = torch.tensor(
            np.asarray([0.34292033, 0.22350113, 0.33940127, 0.13547006]).astype("float32").reshape(1, 1, 1, -1),
            device=constants.DEVICE)

        self.c_a = torch.tensor(
            np.asarray([0.03029606, 0.02065253, 0.02987719, 0.01571027]).astype("float32").reshape(1, 1, 1, -1),
            device=constants.DEVICE)

        self.x_array = torch.tensor(np.linspace(0, patch_size - 1, patch_size).astype("float32"),
                                    device=constants.DEVICE).reshape([1, 1, -1, 1])

    def generate_image_function(self, edge_width: float = 2.0, color_swip=False):
        def f(in_theta):
            in_theta = in_theta.reshape([in_theta.shape[0], 1, 1, 1])
            p = torch.sigmoid((in_theta - self.x_array) / edge_width)
            if color_swip:
                alpha = self.c_a - self.c_b
                i_x = alpha * p + self.c_b
            else:
                alpha = self.c_b - self.c_a
                i_x = alpha * p + self.c_a
            i_xy = i_x.repeat([1, self.patch_size, 1, 1])
            return i_xy

        return f


batch_size = 1
gef = GenerateEdgeFunction(patch_size=32)
generate_image = gef.generate_image_function(2, color_swip=False)
cross_point = 16

theta_vector = cross_point * torch.ones(batch_size, requires_grad=True).to(constants.DEVICE)
I = generate_image(theta_vector)
I = I.cpu().detach().numpy()[0, :, :, :]
I = process_sidd_image(unpack_raw(I), bayer_2by2, wb, cst2)
plt.subplot(2, 2, 1)
plt.imshow(I.astype('int'))
plt.axis('off')
plt.title(f"Position:{cross_point}, Edge Width:{2}")
plt.subplot(2, 2, 2)
cross_point = 2
theta_vector = cross_point * torch.ones(batch_size, requires_grad=True).to(constants.DEVICE)
I = generate_image(theta_vector)
I = I.cpu().detach().numpy()[0, :, :, :]
I = process_sidd_image(unpack_raw(I), bayer_2by2, wb, cst2)
plt.imshow(I.astype('int'))
plt.axis('off')
plt.title(f"Position:{cross_point}, Edge Width:{2}")
plt.subplot(2, 2, 3)
cross_point = 16
theta_vector = cross_point * torch.ones(batch_size, requires_grad=True).to(constants.DEVICE)
generate_image = gef.generate_image_function(4, color_swip=False)
I = generate_image(theta_vector)
I = I.cpu().detach().numpy()[0, :, :, :]
I = process_sidd_image(unpack_raw(I), bayer_2by2, wb, cst2)
plt.imshow(I.astype('int'))
plt.axis('off')
plt.title(f"Position:{cross_point}, Edge Width:{4}")
plt.subplot(2, 2, 4)
cross_point = 16
theta_vector = cross_point * torch.ones(batch_size, requires_grad=True).to(constants.DEVICE)
generate_image = gef.generate_image_function(1, color_swip=False)
I = generate_image(theta_vector)
I = I.cpu().detach().numpy()[0, :, :, :]
I = process_sidd_image(unpack_raw(I), bayer_2by2, wb, cst2)
plt.imshow(I.astype('int'))
plt.axis('off')
plt.title(f"Position:{cross_point}, Edge Width:{1}")
plt.show()
print("a")
