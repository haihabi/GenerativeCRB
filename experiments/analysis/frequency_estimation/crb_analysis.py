from experiments.data_model.frequency_estimation_example import FrequencyModel
import numpy as np
from matplotlib import pyplot as plt
import torch

if __name__ == '__main__':
    dm = FrequencyModel(20, 0.4)
    f_0_array = np.linspace(0.01, 0.49)
    results_crb = []
    for f_0 in f_0_array:
        theta = [1, f_0, 0]
        theta = torch.tensor(theta).float()
        crb = dm.crb(theta).detach().cpu().numpy()
        results_crb.append(crb)
    results_crb = np.stack(results_crb)
    for i in range(3):
        for j in range(3):
            plt.subplot(3, 3, 1 + i + 3 * j)
            plt.plot(f_0_array, results_crb[:, i, j], label="CRB")
            plt.grid()
            plt.legend()
            plt.title(f"{i},{j}")
    plt.show()
