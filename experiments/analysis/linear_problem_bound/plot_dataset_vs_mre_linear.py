import numpy as np
from matplotlib import pyplot as plt

# dataset_size = [1000, 2000, 8000, 20000, 50000, 70000, 90000, 120000, 140000, 180000]
# max_re = [0.1353, 0.06267, 0.06015, 0.03895, 0.02834, 0.02506, 0.01991, 0.01678, 0.01394, 0.008904]
# mean_re = [0.09877, 0.05968, 0.05073, 0.03001, 0.02608, 0.0203, 0.01608, 0.01168, 0.01057, 0.006213]


dataset_size = [1000, 2000, 4000, 6000, 8000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000,
                120000, 140000, 160000, 180000, 200000]
max_re = [0.09006, 0.06188, 0.04352, 0.06955, 0.05285, 0.06099, 0.03559, 0.03127, 0.02736, 0.03605, 0.03456, 0.02256,
          0.0167, 0.01983, 0.02161, 0.01823, 0.01494, 0.01145, 0.01255, 0.01239]
mean_re = [0.08697, 0.05799, 0.0403, 0.06643, 0.04696, 0.05322, 0.03003, 0.02419, 0.02282, 0.0261, 0.03005, 0.01636,
           0.01265, 0.01529, 0.01467, 0.01192, 0.008423, 0.008287, 0.005746, 0.008734]
print(len(max_re), len(mean_re))
# eps = 0.01
dataset_size = np.asarray(dataset_size)
max_re = np.asarray(max_re)
mean_re = np.asarray(mean_re)
i_sort = np.argsort(dataset_size)
dataset_size = dataset_size[i_sort]
max_re = max_re[i_sort]
mean_re = mean_re[i_sort]

plt.plot(dataset_size, max_re, label=r"$\mathrm{MRE}$")
plt.plot(dataset_size, mean_re,"--", label=r"$\overline{\mathrm{MRE}}$")
# plt.plot(dataset_size, mean_re)
# plt.plot(dataset_size, eps * np.ones(len(dataset_size)), "--", label=r"$\epsilon$ Value")
plt.xlabel(r"Size Of $\mathcal{D}$")
plt.ylabel("xRE")
plt.legend()
plt.grid()
plt.show()
