from gcrb.collector_fim import FisherInformationMatrixCollector
from gcrb.compuate_fim import compute_fim_tensor
import math
from tqdm import tqdm


def sampling_gfim(model, in_theta_tensor, m, batch_size=128, trimming_step=None):
    # iteration_step = int(math.ceil(m / batch_size))
    fim_collector = None
    with tqdm(total=m) as pbar:
        while fim_collector is None or fim_collector.i < m:
            gfim, s_vector = compute_fim_tensor(model, in_theta_tensor, batch_size=batch_size, score_vector=True,
                                                trimming_step=trimming_step)
            if fim_collector is None:
                fim_collector = FisherInformationMatrixCollector(m_parameters=gfim.shape[-1]).to(gfim.device)
            update_size = fim_collector.append_fim(gfim, s_vector)
            pbar.update(update_size)
    return fim_collector.calculate_final_fim()
