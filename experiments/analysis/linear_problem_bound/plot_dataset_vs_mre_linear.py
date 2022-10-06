import numpy as np
from matplotlib import pyplot as plt

dataset_size = [1000, 2000, 4000, 8000, 10000, 20000, 40000, 60000, 100000, 160000, 200000]
max_re = [0.03366, 0.02, 0.01918, 0.01603, 0.01804, 0.009486, 0.008615, 0.005721, 0.005793, 0.006804, 0.00616]
mean_re = [0.02945, 0.01684, 0.01716, 0.01313, 0.01429, 0.006746, 0.005648, 0.00353, 0.003181, 0.004529,
           0.003989]

max_re = [0.03330664336681366,
          0.021489934995770454,
          0.02297823503613472,
          0.020753704011440277,
          0.021568188443779945,
          0.013910066336393356,
          0.016595376655459404,
          0.014924240298569202, 0.011546915397047997,
          0.006804,
          0.00616]
mean_re = [0.02589404582977295,
           0.014404421672224998,
           0.014790629036724567,
           0.010978098027408123,
           0.012274573557078838,
           0.006738536059856415,
           0.007207322865724564,
           0.006075683049857616, 0.005070552695542574,
           0.004529,
           0.003989]

print(len(max_re), len(mean_re))
zoom = False
# eps = 0.01
dataset_size = np.asarray(dataset_size)
max_re = np.asarray(max_re)
mean_re = np.asarray(mean_re)
i_sort = np.argsort(dataset_size)
dataset_size = dataset_size[i_sort]
max_re = max_re[i_sort]
mean_re = mean_re[i_sort]

fig, ax = plt.subplots()

ax.plot(dataset_size, max_re, label=r"$\mathrm{MRE}$")
ax.plot(dataset_size, mean_re, "--", label=r"$\overline{\mathrm{MRE}}$")
if zoom:
    axins = ax.inset_axes([0.4, 0.3, 0.5, 0.5])
    axins.plot(dataset_size, max_re, label=r"$\mathrm{MRE}$")
    axins.plot(dataset_size, mean_re, "--", label=r"$\overline{\mathrm{MRE}}$")
    axins.grid()
    axins.set_ylim(0.002, 0.00810)
    axins.set_xlim(50000, 200000)
    axins.locator_params(axis="x", nbins=3)
    ax.indicate_inset_zoom(axins, edgecolor="black", label=None)
plt.xlabel(r"Size Of $\mathcal{D}$")
plt.ylabel("xRE")
plt.legend()
plt.grid()
plt.savefig("results_vs_dataset_size.svg")
plt.show()
