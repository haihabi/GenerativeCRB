import numpy as np
from matplotlib import pyplot as plt
import pickle

iso_list = [100, 400, 800, 1600, 3200]
cam_dict = {'Apple': 0, 'Google': 1, 'samsung': 2, 'motorola': 3, 'LGE': 4}
index2cam = {v: k for k, v in cam_dict.items()}

with open("results.pickle", "rb") as file:
    data = pickle.load(file)
print("a")
n_device = 4
scenraios_list = ["003", "007", "010"]
d = data['003']
for device in range(5):
    plt.plot(iso_list, [d[iso][device]['Relative_RMSE'] for iso in iso_list], label=f"Device {index2cam[device]}")
plt.grid()
plt.xlabel("ISO Level")
plt.ylabel("NMSE[dB]")
plt.legend()
plt.show()

results_type = ["Relative_RMSE_R", "Relative_RMSE_G1", "Relative_RMSE_G2", "Relative_RMSE_B"]
results_legened = ["Red", "Green 1", "Green 2", "Blue"]
color = ["red", "green", "green", "blue"]
line_type = ["-", "-", "--", "-"]
for i in range(n_device):
    for j, sen in enumerate(scenraios_list):
        print(i, j, 1 + i + n_device * j)
        plt.subplot(3, n_device, 1 + i + n_device * j)
        for k in range(4):
            iso_list = []
            results = []
            for iso, v in data[sen].items():
                iso_list.append(iso)
                results.append(v[i][results_type[k]])
            plt.plot(iso_list, results, line_type[k], label=f"{results_legened[k]}", color=color[k])

        plt.grid()
        plt.legend()

        if j == 2: plt.xlabel("ISO Level")
        if i == 0: plt.ylabel("NMSE[dB]")
        plt.title(f"Scenario {j + 1}, Device {index2cam[i]} ")
plt.show()
