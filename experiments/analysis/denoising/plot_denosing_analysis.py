import numpy as np
from matplotlib import pyplot as plt
import pickle

iso_list = [100, 400, 800, 1600, 3200]
cam_dict = {'Apple': 0, 'Google': 1, 'samsung': 2, 'motorola': 3, 'LGE': 4}
index2cam = {v: k for k, v in cam_dict.items()}

with open("results_fixed_metric_plug_trimming.pickle", "rb") as file:
    data = pickle.load(file)

for i in range(1):  # Device Loop
    for j, sen in enumerate(["003", "007", "010"]):
        iso_list = []
        results = []
        for iso, v in data[sen].items():
            iso_list.append(iso)
            results.append(v[i]['Relative_RMSE'])
        plt.plot(iso_list, results, label=f"Scene: {j + 1}")

        plt.grid()
        plt.legend()
        plt.xlabel("ISO Level")
        plt.ylabel("NRMSE[dB]")
plt.savefig("different_scene.svg")
plt.show()

sencrio_list = ["003", "007", "010"]
plt.figure(figsize=(20,5))

for i in range(5):
    for j, sen in enumerate(sencrio_list):
        plt.subplot(1, len(sencrio_list), 1 + j)
        iso_list = []
        results = []
        for iso, v in data[sen].items():
            iso_list.append(iso)
            results.append(v[i]['Relative_RMSE'])
        plt.plot(iso_list, results, label=f"Device: {index2cam[i]}")

        plt.grid()
        plt.legend()
        plt.xlabel("ISO Level")
        plt.ylabel("NRMSE[dB]")
        if len(sencrio_list) > 1:
            plt.title(f"Scene: {j + 1}")
plt.savefig("different_device_all_scene.svg")

plt.show()

sencrio_list = ["003"]
for i in range(5):
    for j, sen in enumerate(sencrio_list):
        plt.subplot(1, len(sencrio_list), 1 + j)
        iso_list = []
        results = []
        for iso, v in data[sen].items():
            iso_list.append(iso)
            results.append(v[i]['Relative_RMSE'])
        plt.plot(iso_list, results, label=f"Device: {index2cam[i]}")

        plt.grid()
        plt.legend()
        plt.xlabel("ISO Level")
        plt.ylabel("NRMSE[dB]")
        if len(sencrio_list) > 1:
            plt.title(f"Scene: {j + 1}")
plt.savefig("different_device_scene_one.svg")
plt.show()
