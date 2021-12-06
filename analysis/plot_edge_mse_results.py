import numpy as np
from matplotlib import pyplot as plt
import pickle

iso_list = [100, 400, 800, 1600, 3200]
cam_dict = {'Apple': 0, 'Google': 1, 'samsung': 2, 'motorola': 3, 'LGE': 4}
index2cam = {v: k for k, v in cam_dict.items()}
cross_point_array = [1, 12, 16, 28, 31]
with open("./results_edge.pickle", "rb") as file:
    data = pickle.load(file)

print("a")

# for device in range(5):
#     for k, iso in enumerate(iso_list):
#         plt.subplot(5, 5, 1 + device + k * 5)
device = 0
iso = 100
for k, v in data[device][iso].items():
    plt.plot(cross_point_array, v, label=f"Edge width {k}")
# plt.legend()
plt.grid()
# plt.title(f"Device {index2cam[device]} on ISO {iso}")
plt.legend()
plt.xlabel("Edge Position")
plt.ylabel("MSE[dB]")
plt.show()

for device in range(5):
    results_iso = []
    for k, iso in enumerate(iso_list):
        results_iso.append(data[device][iso][2][2])
    plt.plot(iso_list, results_iso, label=f"Device {index2cam[device]}")
plt.grid()
plt.legend()
plt.xlabel("ISO Level")
plt.ylabel("MSE[dB]")
plt.show()

for device in range(5):
    for n, iso in enumerate(iso_list):
        plt.subplot(5, 5, 1 + device + n * 5)
        for k, v in data[device][iso].items():
            plt.plot(cross_point_array, v, label=f"Edge width {k}")
        if n == 0 and device == 0: plt.legend()
        if n == 4: plt.xlabel("Edge Position")
        if device == 0: plt.ylabel("MSE[dB]")
        plt.title(f"Device {index2cam[device]} on ISO {iso}")
        plt.grid()
plt.show()
