import numpy as np
from matplotlib import pyplot as plt

dataset_size = [1000, 2000, 8000, 20000, 50000, 70000, 90000, 120000, 140000, 180000]
max_re = [0.1353, 0.06267, 0.06015, 0.03895, 0.02834, 0.02506, 0.01991, 0.01678, 0.01394, 0.008904]
mean_re = [0.09877, 0.05968, 0.05073, 0.03001, 0.02608, 0.0203, 0.01608, 0.01168, 0.01057, 0.006213]
# mean_re = [0.01897, 0.02517, 0.02298, 0.04076, 0.07129, 0.04336, 0.06298, 0.09723, 0.1309]
eps = 0.01
dataset_size = np.asarray(dataset_size)
max_re = np.asarray(max_re)
mean_re = np.asarray(mean_re)
i_sort = np.argsort(dataset_size)
dataset_size = dataset_size[i_sort]
max_re = max_re[i_sort]
mean_re = mean_re[i_sort]

plt.plot(dataset_size, max_re, label=r"$\mathrm{MRE}$")
plt.plot(dataset_size, mean_re, label=r"$\overline{\mathrm{MRE}}$")
# plt.plot(dataset_size, mean_re)
plt.plot(dataset_size, eps * np.ones(len(dataset_size)), "--", label=r"$\epsilon$ Value")
plt.xlabel(r"Size Of $\mathcal{D}$")
plt.ylabel("xRE")
plt.legend()
plt.grid()
plt.show()
