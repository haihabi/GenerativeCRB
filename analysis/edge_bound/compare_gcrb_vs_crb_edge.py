import numpy as np
from matplotlib import pyplot as plt
import pickle
from analysis.edge_bound.nlf_crb import get_crb_function
from analysis.edge_bound.edge_image_generator import EdgeImageGenerator
import torch
from training_nlf.training_main import generate_nlf_flow
import gcrb

iso_list = [100, 400, 800, 1600, 3200]
cross_point_array = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 30]
cam_dict = {'Apple': 0, 'Google': 1, 'samsung': 2, 'motorola': 3, 'LGE': 4}
index2cam = {v: k for k, v in cam_dict.items()}
color_swip = True
with open("results_edge_swip.pickle", "rb") as file:
    data = pickle.load(file)
import constants

iso = 1600
iso_index = iso_list.index(iso)
cam = 2
edge_width = 16
print(f"Comparing CRB using Device:{index2cam[cam]} with ISO {iso} and Edge Width:{edge_width}")

gcrb_db = data[cam][iso][edge_width]
crb_func = get_crb_function(edge_width)
patch_size = 32
input_shape = [4, patch_size, patch_size]
trained_alpha = True
flow = generate_nlf_flow(input_shape, trained_alpha, noise_only=False)
parameter_nlf = torch.load("/data/projects/GenerativeCRB/analysis/edge_bound/training_nlf/flow_nlf_best.pt",
                           map_location="cpu")
flow.flow.flows[1].alpha.data = parameter_nlf["flow.flows.0.alpha"]
flow.flow.flows[1].delta.data = parameter_nlf["flow.flows.0.delta"]
eig = EdgeImageGenerator(patch_size)
generate_image = eig.get_image_function(edge_width, color_swip)


def image2vector(in_image):
    return torch.permute(in_image, (0, 3, 1, 2))


def sample_function(in_batch_size, in_theta):
    bayer_img = generate_image(in_theta)
    in_cond_vector = [image2vector(bayer_img), iso, cam]
    return flow.sample_nll(in_batch_size, cond=in_cond_vector).reshape([-1, 1])


batch_size = 32
_results_croos_points = []
for cross_point in cross_point_array:
    theta_vector = cross_point * torch.ones(batch_size, requires_grad=True).to(constants.DEVICE)
    gfim = gcrb.adaptive_sampling_gfim(sample_function, theta_vector.reshape([-1, 1]),
                                       batch_size=batch_size,
                                       n_max=64000)
    print("a")
# parameter_gaussian = torch.load("/data/projects/GenerativeCRB/analysis/edge_bound/training_nlf/flow_gaussian.pt",
#                                 map_location="cpu")
# sigma_var = parameter_gaussian["flow.flows.0.delta"].cpu().numpy()

# parameter_nlf = torch.load("/data/projects/GenerativeCRB/analysis/edge_bound/training_nlf/flow_nlf_best.pt",
#                            map_location="cpu")

# alpha_var = parameter_nlf["flow.flows.0.alpha"].cpu().numpy()
# delta_var = parameter_nlf["flow.flows.0.delta"].cpu().numpy()
#
# print("a")
#
# # delta = 0.1
#
#
# # def residual_function_nlf(in_array):
# #     alpha, delta = in_array[0], in_array[1]
# #     a = np.asarray(gcrb_db).flatten() - 10 * np.log10(np.asarray(
# #         [crb_func(edge_position, alpha, delta) for edge_position in cross_point_array]).flatten())
# #     return a
# #
# #
# # def residual_function_gaussian(in_array):
# #     delta = in_array
# #     a = np.asarray(gcrb_db).flatten() - 10 * np.log10(np.asarray(
# #         [crb_func(edge_position, 0, delta) for edge_position in cross_point_array]).flatten())
# #     return a
#
# import math
#
# # x0 = np.asarray([0.1, 1e-4])
# # res_nlf = least_squares(residual_function_nlf, x0, bounds=(1e-6, np.inf))
# # print(res_nlf)
res_nlf_alpha = alpha_var[iso_index, device]
# res_nlf_alpha = 2.25816115e-02
res_nlf_delta = delta_var[iso_index, device]
print(res_nlf_delta, res_nlf_alpha)
# res_nlf_delta = 5.55012599e-05
# x0 = 1e-4
# res_gaussian = least_squares(residual_function_gaussian, x0, bounds=(1e-6, np.inf))
# print(res_gaussian)
std_g = np.array([[0.04474904, 0.02932946, 0.03030572, 0.02605926, 0.],
                  [0.02963125, 0.02126121, 0.05687612, 0.06298191, 0.03050465],
                  [0.05892951, 0.03158211, 0.09212589, 0.10110627, 0.23274043],
                  [0.02439415, 0.02525071, 0.05441846, 0.06251305, 0.10548679],
                  [0.02644052, 0.03422556, 0.03171875, 0., 0.]])
res_gaussian_delta = std_g[device, iso_index]
print(res_gaussian_delta)
# res_gaussian_delta = 0.00829678

plt.plot(cross_point_array,
         10 * np.log10(
             np.asarray(
                 [crb_func(edge_position, res_nlf_alpha, res_nlf_delta) / 4096 for edge_position in
                  cross_point_array]).flatten()),
         label="NLF Noise")
# plt.plot(cross_point_array,
#          10 * np.log10(
#              np.asarray(
#                  [crb_func(edge_position, 0.0, res_gaussian_delta) for edge_position in cross_point_array]).flatten()),
#          label="Gaussian Noise")
plt.plot(cross_point_array, np.asarray(gcrb_db).flatten(), label="GCRB")
plt.grid()

plt.legend()
plt.xlabel("Edge Position")
plt.ylabel("MSE[dB]")
plt.show()
