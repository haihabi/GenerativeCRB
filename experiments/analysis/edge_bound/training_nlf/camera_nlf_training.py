import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from experiments.data_model.image_denoising.noise_dataset import NoiseDataSet
from experiments.models_architecture.camera_nlf_flow import generate_nlf_flow


def train_step(in_noise, in_cond_vector):
    opt.zero_grad()
    loss = flow.nll_mean(in_noise, in_cond_vector)
    loss.backward()
    loss_list.append(loss.item())
    opt.step()


if __name__ == '__main__':

    lr = 1e-4
    patch_size = 32
    n_epochs = 5
    batch_size = 32
    n_iter_per_epoch = 1000
    input_shape = [4, patch_size, patch_size]
    trained_alpha = True

    flow = generate_nlf_flow(input_shape, trained_alpha)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    nds = NoiseDataSet("/data/datasets/SIDD_Medium_Raw/Data", n_pat_per_im=5000)

    nds_dl = DataLoader(nds, batch_size=batch_size, shuffle=True)
    loss_best = np.inf
    for n in range(n_epochs):
        loss_list = []
        for noise, clean, cam, iso in tqdm(nds_dl):
            noise, clean, cam, iso = noise.cuda(), clean.cuda(), cam.long().cuda(), iso.cuda()
            clean = torch.permute(clean, (0, 3, 1, 2)).float()
            noise = torch.permute(noise, (0, 3, 1, 2)).float()
            cond_vector = [clean, iso, cam]
            train_step(noise, cond_vector)

        loss_current = sum(loss_list) / len(loss_list)
        print(loss_current)
        if loss_current < loss_best:
            flow_name = "flow_nlf_best.pt" if trained_alpha else "flow_gaussian_best.pt"
            torch.save(flow.state_dict(), f"./{flow_name}")
            loss_best = loss_current
            print(f"Update Best To:{loss_current}")

    flow_name = "flow_nlf.pt" if trained_alpha else "flow_gaussian.pt"
    torch.save(flow.state_dict(), f"./{flow_name}")
