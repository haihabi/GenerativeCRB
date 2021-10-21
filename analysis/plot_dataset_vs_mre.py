import numpy as np
from matplotlib import pyplot as plt

dataset_size = [200000, 180000, 140000, 120000, 100000, 60000, 50000, 40000, 20000]
mre = [0.06792, 0.08276, 0.08069, 0.1211, 0.183, 0.2001, 0.2911, 0.34, 0.5716]
mean_re = [0.01897, 0.02517, 0.02298, 0.04076, 0.07129, 0.04336, 0.06298, 0.09723, 0.1309]
eps = 0.1
dataset_size = np.asarray(dataset_size)
mre = np.asarray(mre)
i_sort = np.argsort(dataset_size)
dataset_size = dataset_size[i_sort]
mre = mre[i_sort]
mean_re = mean_re[i_sort]

plt.plot(dataset_size, mre)
# plt.plot(dataset_size, mean_re)
plt.plot(dataset_size, eps * np.ones(len(dataset_size)), "--", label=r"$\epsilon$ Value")
plt.xlabel(r"Size Of $\mathcal{D}$")
plt.ylabel("GFIM Relative Error")
plt.legend()
plt.grid()
plt.show()
