import numpy as np
from matplotlib import pyplot as plt
import pickle

iso_list = [100, 400, 800, 1600, 3200]
cam_dict = {'Apple': 0, 'Google': 1, 'samsung': 2, 'motorola': 3, 'LGE': 4}
index2cam = {v: k for k, v in cam_dict.items()}

with open("./results.pickle", "rb") as file:
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

# data = {0: {0: [-34.0670, -32.1120, -31.6062, -31.5051, -30.3647],
#             1: [-39.9364, -37.7417, -37.2609, -37.1844, -36.3955],
#             2: [-36.7895, -34.8797, -34.4486, -34.3716, -33.4148]},
#         1: {0: [-38.0502, -32.5503, -30.3972, -28.8526, -26.0768],
#             1: [-44.0648, -38.2402, -36.4129],
#             2: [-40.7811, -35.2982, -33.4459]},
#         2: {0: [-35.6636, -28.2672, -25.7941, -24.2900, -21.4030],
#             1: [],
#             2: []},
#         3: {0: [-36.3637, -30.0631, -27.5671, -26.0472, -23.6950],
#             1: [],
#             2: []},
#         4: {0: [-37.4331, -30.2514, -27.3804, -25.6725, -22.8974],
#             1: [-43.2245],
#             2: [-40.0900]},
#         }
# iso_array = np.asarray(iso_list)
#
# iso_data = data[0]
# for i in range(3):
#     plt.plot(iso_list, iso_data[i], label=f"Scenario {i + 1}")
# plt.grid()
# plt.xlabel("ISO Level")
# plt.legend()
# plt.ylabel("MSE[dB]")
# plt.show()
#
# for i in range(5):
#     plt.plot(iso_list, data[i][0], label=f"{index2cam[i]}")
# plt.grid()
# plt.xlabel("ISO Level")
# plt.legend()
# plt.ylabel("MSE[dB]")
# plt.show()
