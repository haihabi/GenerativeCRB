import numpy as np
from matplotlib import pyplot as plt
import pickle
from analysis.edge_bound.nlf_crb import get_crb_function
from scipy.optimize import least_squares

iso_list = [100, 400, 800, 1600, 3200]
cross_point_array = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 30]
cam_dict = {'Apple': 0, 'Google': 1, 'samsung': 2, 'motorola': 3, 'LGE': 4}
index2cam = {v: k for k, v in cam_dict.items()}
with open("results_edge_swip.pickle", "rb") as file:
    data = pickle.load(file)

iso = 3200
device = 2
edge_width = 8
print(f"Comparing CRB using Device:{index2cam[device]} with ISO {iso} and Edge Width:{edge_width}")

gcrb_db = data[device][iso][edge_width]
crb_func = get_crb_function(edge_width)


# delta = 0.1


def residual_function_nlf(in_array):
    alpha, delta = in_array[0], in_array[1]
    a = np.asarray(gcrb_db).flatten() - 10 * np.log10(np.asarray(
        [crb_func(edge_position, alpha, delta) for edge_position in cross_point_array]).flatten())
    return a


def residual_function_gaussian(in_array):
    delta = in_array
    a = np.asarray(gcrb_db).flatten() - 10 * np.log10(np.asarray(
        [crb_func(edge_position, 0, delta) for edge_position in cross_point_array]).flatten())
    return a


x0 = np.asarray([0.1, 1e-4])
res_nlf = least_squares(residual_function_nlf, x0, bounds=(1e-6, np.inf))
print(res_nlf)
res_nlf_alpha = res_nlf.x[0]
# res_nlf_alpha = 2.25816115e-02
res_nlf_delta = res_nlf.x[1]
# res_nlf_delta = 5.55012599e-05
x0 = 1e-4
res_gaussian = least_squares(residual_function_gaussian, x0, bounds=(1e-6, np.inf))
print(res_gaussian)
res_gaussian_delta = res_gaussian.x
# res_gaussian_delta = 0.00829678

plt.plot(cross_point_array,
         10 * np.log10(
             np.asarray(
                 [crb_func(edge_position, res_nlf_alpha, res_nlf_delta) for edge_position in
                  cross_point_array]).flatten()),
         label="NLF Noise")
plt.plot(cross_point_array,
         10 * np.log10(
             np.asarray(
                 [crb_func(edge_position, 0.0, res_gaussian_delta) for edge_position in cross_point_array]).flatten()),
         label="Gaussian Noise")
plt.plot(cross_point_array, np.asarray(gcrb_db).flatten(), label="GCRB")
plt.grid()

plt.legend()
plt.xlabel("Edge Position")
plt.ylabel("MSE[dB]")
plt.show()
