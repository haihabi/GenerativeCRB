import numpy as np
from matplotlib import pyplot as plt
import pickle

iso_list = [100, 400, 800, 1600, 3200]
cross_point_array = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 30]
cam_dict = {'Apple': 0, 'Google': 1, 'samsung': 2, 'motorola': 3, 'LGE': 4}
index2cam = {v: k for k, v in cam_dict.items()}
with open("new_results_edge.pickle", "rb") as file:
    data = pickle.load(file)

device = 0
iso = 100
ax = plt.subplot(111)

for k, v in data[device][iso].items():
    ax.plot(cross_point_array, 10*np.log10(v), label=f"Edge width {k}")
# plt.legend()
plt.grid()
# plt.title(f"Device {index2cam[device]} on ISO {iso}")
# plt.legend(loc="upper right")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, fancybox=True, shadow=True)
plt.xlabel("Edge Position")
plt.ylabel("MSE[dB]")
plt.savefig("edge_position_different_width.svg")
plt.show()
#
# for device in range(5):
#     results_iso = []
#     for k, iso in enumerate(iso_list):
#         results_iso.append(data[device][iso][2][2])
#     plt.plot(iso_list, results_iso, label=f"Device {index2cam[device]}")
# plt.grid()
# plt.legend()
# plt.xlabel("ISO Level")
# plt.ylabel("MSE[dB]")
# plt.show()
#
# for device in range(5):
#     for n, iso in enumerate(iso_list):
#         # plt.subplots(5,5,1 + device + n * 5)
#         plt.subplot(1, 5, 1 + n)
#         for k, v in data[device][iso].items():
#             plt.plot(cross_point_array, v, label=f"Edge width {k}")
#         if n == 0 and device == 0: plt.legend(loc='upper center', bbox_to_anchor=(1.1, 1.4),
#                                               ncol=3, fancybox=True, shadow=True)
#         if n == 4: plt.xlabel("Edge Position")
#         if device == 0: plt.ylabel("MSE[dB]")
#         plt.title(f"Device {index2cam[device]} on ISO {iso}")
#         plt.grid()
#     plt.show()
