import numpy as np
from matplotlib import pyplot as plt

dataset_size = [200000,1000,2000,4000,8000,180000]
mre = [0.0066527864,0.27885184,0.17859441,0.14280443,0.06721287,0.013518601]
# mean_re = [0.01897, 0.02517, 0.02298, 0.04076, 0.07129, 0.04336, 0.06298, 0.09723, 0.1309]
eps = 0.01
dataset_size = np.asarray(dataset_size)
mre = np.asarray(mre)
i_sort = np.argsort(dataset_size)
dataset_size = dataset_size[i_sort]
mre = mre[i_sort]


plt.plot(dataset_size, mre)
# plt.plot(dataset_size, mean_re)
plt.plot(dataset_size, eps * np.ones(len(dataset_size)), "--", label=r"$\epsilon$ Value")
plt.xlabel(r"Size Of $\mathcal{D}$")
plt.ylabel("MRE")
plt.legend()
plt.grid()
plt.show()
