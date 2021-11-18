from gcrb.collector_fim import FisherInformationMatrixCollector
from gcrb.compuate_fim import compute_fim_tensor
import math
from tqdm import tqdm
import torch
import numpy as np


def adaptive_sampling_gfim(model, in_theta_tensor, batch_size=128, eps=0.01, p_min=0.01,
                           n_max=1e7, iter_size=0.05):
    u = -np.log(p_min)
    status = True
    iteration_step = 1
    fim_collector = None
    with tqdm(total=n_max) as pbar:

        while status:
            for _ in range(iteration_step):

                gfim, s_vector = compute_fim_tensor(model, in_theta_tensor, batch_size=batch_size, score_vector=True)
                if fim_collector is None:
                    fim_collector = FisherInformationMatrixCollector(m_parameters=gfim.shape[-1]).to(gfim.device)

                update_size = fim_collector.append_fim(gfim, s_vector)
                pbar.update(update_size)

            e_norm = fim_collector.calculate_score_norm()  # L2 norm of the score vector
            fim_inv_est = torch.linalg.inv(fim_collector.calculate_final_fim())
            fim_inv_norm = torch.sqrt(torch.pow(fim_inv_est, 2.0).sum())
            conv_inv_half = torch.linalg.inv(
                torch.linalg.cholesky(fim_collector.calculate_final_fim()))  # Calculating an estimation of Cov^{-0.5}

            n_est_must = int(torch.ceil(
                (1 + u) * torch.pow(torch.linalg.norm(conv_inv_half, ord=2), 4) * torch.pow(e_norm, 2)).item())
            n_est = math.ceil((fim_inv_norm ** 2) * (e_norm ** 4) * (u + 1) / (eps ** 2))
            n_est = max(n_est_must, n_est)
            pbar.set_postfix({'estimated_m_samples': n_est})
            if n_est > fim_collector.size and fim_collector.size < n_max:
                iteration_step = min(math.ceil((n_est - fim_collector.size) / batch_size),
                                     int(iter_size * n_max / batch_size))
            else:
                status = False

        print(f"Finished GFIM calculation after {fim_collector.size} Iteration")
    return fim_collector.calculate_final_fim()
