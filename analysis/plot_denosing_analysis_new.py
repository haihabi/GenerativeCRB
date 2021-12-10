import numpy as np
from matplotlib import pyplot as plt
import pickle

iso_list = [100, 400, 800, 1600, 3200]
cam_dict = {'Apple': 0, 'Google': 1, 'samsung': 2, 'motorola': 3, 'LGE': 4}
index2cam = {v: k for k, v in cam_dict.items()}

with open("./results_new.pickle", "rb") as file:
    data = pickle.load(file)
print("a")
d = data['001']
for device in range(4):
    plt.plot(iso_list, [d[iso][device]['Relative_RMSE'] for iso in iso_list], label=f"Device {index2cam[device]}")
plt.grid()
plt.legend()
plt.show()
