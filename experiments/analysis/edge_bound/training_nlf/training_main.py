from experiments.analysis.edge_bound.training_nlf.noise_level_function import NoiseLevelFunction
from experiments.analysis.edge_bound.edge_image_generator import EdgeImageGenerator
import normflowpy as nfp
from experiments import constants
import torch
from torch.distributions import MultivariateNormal
import numpy as np
from tqdm import tqdm
from pytorch_model.noise_flow import generate_noisy_image_flow, ImageFlowStep
from torch.utils.data import DataLoader
from experiments.analysis.edge_bound.training_nlf.noise_dataset import NoiseDataSet


def generate_nlf_flow(in_input_shape, in_trained_alpha, noise_only=True):
    dim = int(np.prod(in_input_shape))
    prior = MultivariateNormal(torch.zeros(dim, device=constants.DEVICE),
                               torch.eye(dim, device=constants.DEVICE))
    flows = []
    if not noise_only:
        flows.append(ImageFlowStep())
    flows.extend([NoiseLevelFunction(trained_alpha=in_trained_alpha), nfp.flows.Tensor2Vector(in_input_shape)])
    return nfp.NormalizingFlowModel(prior, flows).to(constants.DEVICE)


def sample_input_model():
    iso = np.random.choice(iso_list, size=1).repeat(32)
    cam = np.random.choice(device_list, size=1).repeat(32)
    edge_position = np.random.choice(edge_position_list, size=32)
    clean = image_func(torch.tensor(edge_position).to(constants.DEVICE).reshape([-1]))
    clean = torch.permute(clean, (0, 3, 1, 2))
    cond_vector = [clean, torch.tensor(iso).to(constants.DEVICE).reshape([-1]).long(),
                   torch.tensor(cam).to(constants.DEVICE).reshape([-1]).long()]
    noisy_image = noise_flow.sample(32, cond=cond_vector)
    noise = (noisy_image[-1] - clean).detach()
    return noise, cond_vector


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
    n_iter_per_epoch = 1000
    input_shape = [4, patch_size, patch_size]
    trained_alpha = True
    flow = generate_nlf_flow(input_shape, trained_alpha)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    train_dataset = True
    if train_dataset:
        nds = NoiseDataSet("/data/datasets/SIDD_Medium_Raw/Data", n_pat_per_im=5000)
        print(len(nds))
        nds_dl = DataLoader(nds, batch_size=32, shuffle=True)
    else:
        noise_flow = generate_noisy_image_flow([4, patch_size, patch_size], device=constants.DEVICE, load_model=True,
                                               noise_only=False).to(
            constants.DEVICE)
        eig = EdgeImageGenerator(patch_size)
        image_func = eig.get_image_function(16.0, color_swip=False)
        iso_list = [100, 400, 800, 1600, 3200]
        device_list = [0, 1, 2, 3, 4]
        edge_position_list = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 30]

    loss_best = np.inf

    for n in range(n_epochs):
        loss_list = []
        if train_dataset:
            for noise, clean, cam, iso in tqdm(nds_dl):
                noise, clean, cam, iso = noise.cuda(), clean.cuda(), cam.long().cuda(), iso.cuda()
                clean = torch.permute(clean, (0, 3, 1, 2)).float()
                noise = torch.permute(noise, (0, 3, 1, 2)).float()
                cond_vector = [clean, iso, cam]
                train_step(noise, cond_vector)
        else:
            for _ in tqdm(range(n_iter_per_epoch)):
                noise, cond_vector = sample_input_model()
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
