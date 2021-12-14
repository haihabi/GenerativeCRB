import numpy as np
from matplotlib import pyplot as plt
import pickle

iso_list = [100, 400, 800, 1600, 3200]
cam_dict = {'Apple': 0, 'Google': 1, 'samsung': 2, 'motorola': 3, 'LGE': 4}
index2cam = {v: k for k, v in cam_dict.items()}

with open("results.pickle", "rb") as file:
    data = pickle.load(file)
print("a")

for i in range(1):  # Device Loop
    plt.subplot(2, 1, 1)
    for j, sen in enumerate(["003", "007", "010"]):
        # plt.subplot(1, 3, 1 + j)
        iso_list = []
        results = []
        for iso, v in data[sen].items():
            iso_list.append(iso)
            # results.append(v[i]["MSE"])
            results.append(v[i]['Relative_RMSE'])
        plt.plot(iso_list, results, label=f"Scenario: {j + 1}")

        plt.grid()
        plt.legend()
        plt.xlabel("ISO Level")
        plt.ylabel("NMSE[dB]")
    plt.subplot(2, 1, 2)
    for j, sen in enumerate(["003", "007", "010"]):
        # plt.subplot(1, 3, 1 + j)
        iso_list = []
        results = []
        for iso, v in data[sen].items():
            iso_list.append(iso)
            # results.append(v[i]["MSE"])
            results.append(v[i]['MSE'])
        plt.plot(iso_list, results, label=f"Scenario: {j + 1}")
        plt.grid()
        plt.legend()
        plt.xlabel("ISO Level")
        plt.ylabel("MSE[dB]")
plt.show()

for i in range(5):  # Device Loop
    # plt.subplot(2,1,1)
    for j, sen in enumerate(["003", "007", "010"]):
        plt.subplot(1, 3, 1 + j)
        iso_list = []
        results = []
        for iso, v in data[sen].items():
            iso_list.append(iso)
            results.append(v[i]['Relative_RMSE'])
        plt.plot(iso_list, results, label=f"Device: {index2cam[i]}")

        plt.grid()
        plt.legend()
        plt.xlabel("ISO Level")
        plt.ylabel("NMSE[dB]")
        plt.title(f"Scenario: {j + 1}")
plt.show()

results_type = ["Relative_RMSE_R", "Relative_RMSE_G1", "Relative_RMSE_G2", "Relative_RMSE_B"]
results_legened = ["Red", "Green 1", "Green 2", "Blue"]
for i in range(5):
    for j, sen in enumerate(["003", "007", "010"]):
        print(i, j, 1 + i + 5 * j)
        plt.subplot(3, 5, 1 + i + 5 * j)
        for k in range(4):
            iso_list = []
            results = []
            for iso, v in data[sen].items():
                iso_list.append(iso)
                results.append(v[i][results_type[k]])
            plt.plot(iso_list, results, label=f"{results_legened[k]}")

        plt.grid()
        plt.legend()

        if j == 2: plt.xlabel("ISO Level")
        if i == 0: plt.ylabel("NMSE[dB]")
        plt.title(f"Scenario {j + 1}, Device {index2cam[i]} ")
plt.show()
