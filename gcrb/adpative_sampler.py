from gcrb.collector_fim import FisherInformationMatrixCollector
from gcrb.compuate_fim import compute_fim_tensor
import math
from tqdm import tqdm
import torch
import numpy as np


def numric_stable_log_mean_exp(x):
    x_max = x.max()
    x_norm = x - x_max
    return torch.log(torch.mean(torch.exp(x_norm))) + x_max


def adaptive_sampling_gfim(model, in_theta_tensor, batch_size=128, eps=0.01, p_min=0.1,
                           n_max=1e7, iter_size=0.1):
    status = True
    iteration_step = 1
    fim_collector = None
    with tqdm(total=n_max) as pbar:

        while status:
            for _ in range(iteration_step):

                gfim, s_vector = compute_fim_tensor(model, in_theta_tensor, batch_size=batch_size, score_vector=True)
                if fim_collector is None:
                    fim_collector = FisherInformationMatrixCollector(m_parameters=gfim.shape[-1]).to(gfim.device)

                update_size = fim_collector.append_fim(gfim)
                fim_collector.append_score(s_vector)
                pbar.update(update_size)

            e_norm = torch.cat(fim_collector.score_norm_list).mean()  # L2 norm of the score vector
            u = -np.log(p_min)
            n_est = math.ceil((e_norm ** 2) * (u + 1) / (eps ** 2))
            print(n_est)

            if n_est > fim_collector.size and fim_collector.size < n_max:
                iteration_step = min(math.ceil((n_est - fim_collector.size) / batch_size),
                                     int(iter_size * (n_max - fim_collector.size) / batch_size))
            else:
                status = False

        print(f"Finished GFIM calculation after {fim_collector.size} Iteration")
    return fim_collector.mean