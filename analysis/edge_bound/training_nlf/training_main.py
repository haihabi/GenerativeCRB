from analysis.edge_bound.training_nlf.noise_level_function import NoiseLevelFunction
from analysis.edge_bound.edge_image_generator import EdgeImageGenerator
import normflowpy as nfp
import constants
import torch
from torch.distributions import MultivariateNormal
import numpy as np
from tqdm import tqdm
from pytorch_model.noise_flow import generate_noisy_image_flow, ImageFlowStep


def generate_nlf_flow(in_input_shape, in_trained_alpha, noise_only=True):
    dim = int(np.prod(in_input_shape))
    prior = MultivariateNormal(torch.zeros(dim, device=constants.DEVICE),
                               torch.eye(dim, device=constants.DEVICE))
    flows = []
    if not noise_only:
        flows.append(ImageFlowStep())
    flows.extend([NoiseLevelFunction(trained_alpha=in_trained_alpha), nfp.flows.Tensor2Vector(in_input_shape)])
    return nfp.NormalizingFlowModel(prior, flows).cuda()


if __name__ == '__main__':

    lr = 1e-4
    patch_size = 32
    n_epochs = 150
    n_iter_per_epoch = 1000
    input_shape = [4, patch_size, patch_size]
    trained_alpha = True
    noise_flow = generate_noisy_image_flow([4, patch_size, patch_size], device=constants.DEVICE, load_model=True,
                                           noise_only=False).to(
        constants.DEVICE)
    flow = generate_nlf_flow(input_shape, trained_alpha)

    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    eig = EdgeImageGenerator(patch_size)
    image_func = eig.get_image_function(8, color_swip=True)

    loss_best = np.inf
    iso_list = [100, 400, 800, 1600, 3200]
    device_list = [0, 1, 2, 3, 4]
    edge_position_list = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 30]

    for n in range(n_epochs):
        loss_list = []
        for _ in range(n_iter_per_epoch):
            iso = int(np.random.choice(iso_list))
            cam = int(np.random.choice(device_list))
            edge_position = np.random.choice(edge_position_list, size=32)
            clean = image_func(torch.tensor(edge_position).to(constants.DEVICE).reshape([-1]))
            clean = torch.permute(clean, (0, 3, 1, 2))
            in_cond_vector = [clean, iso, cam]
            noisy_image = noise_flow.sample(32, cond=in_cond_vector)
            noise = (noisy_image[-1] - clean).detach()
            # print(noise.mean(),noise.std())
            # noise, clean, cam, iso = noise.cuda(), clean.cuda(), cam.cuda(), iso.cuda()
            opt.zero_grad()

            loss = flow.nll(noise, in_cond_vector).mean()
            loss.backward()
            loss_list.append(loss.item())
            opt.step()
        loss_current = sum(loss_list) / len(loss_list)
        print(loss_current)
        if loss_current < loss_best:
            flow_name = "flow_nlf_best.pt" if trained_alpha else "flow_gaussian_best.pt"
            torch.save(flow.state_dict(), f"./{flow_name}")
            loss_best = loss_current
            print(f"Update Best To:{loss_current}")

    flow_name = "flow_nlf.pt" if trained_alpha else "flow_gaussian.pt"
    torch.save(flow.state_dict(), f"./{flow_name}")
