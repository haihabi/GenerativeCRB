import math

import matplotlib.pyplot as plt
import numpy as np
from experiments import constants
import torch
from experiments.analysis.analysis_helpers import image_channel_swipe_nhwc2nchw, image_shape, \
    image_channel_swipe_nchw2nhwc, \
    db
from pytorch_model.noise_flow import generate_noisy_image_flow
from experiments.data_model.edge_position.edge_image_generator import EdgeImageGenerator
from tqdm import tqdm
from experiments.analysis.edge_bound.estimators.mle_gaussian import mle_gaussian
from experiments.analysis.edge_bound.estimators.training_mvue import MVUENet
from experiments.analysis.analysis_helpers import rggb2rgb

if __name__ == '__main__':
    iso = 800
    cam = 2
    edge_width = 2
    patch_size = 32
    batch_size = 32
    n_samples = 2000

    color_swip = False
    flow = generate_noisy_image_flow(image_shape(patch_size), device=constants.DEVICE, load_model=True).to(
        constants.DEVICE)
    eig = EdgeImageGenerator(patch_size)
    p_h, p_l = eig.get_pixel_color(color_swip=color_swip)
    generate_image = eig.get_image_function(edge_width, color_swip)
    mvue = MVUENet(128).to(constants.DEVICE)
    state_dict = torch.load("mvue.pt")
    mvue.load_state_dict(state_dict)
    mvue = mvue.eval()


    def sample_function(in_batch_size, in_theta):
        bayer_img = generate_image(in_theta)
        in_cond_vector = [image_channel_swipe_nhwc2nchw(bayer_img), iso, cam]
        return flow.sample(in_batch_size, in_cond_vector, temperature=0.6)
        # img = image_channel_swipe_nhwc2nchw(bayer_img)
        # return img


    #

    # img = sample_function(1, 18 * torch.ones(1, device=constants.DEVICE))
    # im_rgb = rggb2rgb(image_channel_swipe_nchw2nhwc(img).cpu().detach().numpy()[0, :, :, :])
    # from matplotlib import  pyplot as plt
    # plt.imshow(im_rgb)
    # plt.show()

    # from sidd.pipeline import process_sidd_image, flip_bayer, stack_rggb_channels

    # clean_patch_srgb = process_sidd_image(unpack_raw(clean_patch), bayer_2by2, wb, cst2)
    n_iter = math.ceil(n_samples / batch_size)
    results = []
    results_mvue = []
    for edge_position in constants.CROSS_POINT:
        _results = []
        _results_mvue = []
        for _ in tqdm(range(n_iter)):
            data = sample_function(batch_size, edge_position * torch.ones(batch_size, device=constants.DEVICE))
            im_hat_mvue = mvue(data).detach().cpu().numpy()
            im_noise_data = image_channel_swipe_nchw2nhwc(data)
            im_hat = mle_gaussian(im_noise_data, p_h, p_l, edge_width).detach().cpu().numpy()
            _results.append(np.mean(np.power(im_hat - edge_position, 2.0)))
            _results_mvue.append(np.mean(np.power(im_hat_mvue - edge_position, 2.0)))
        results.append(db(np.mean(_results)))
        results_mvue.append(db(np.mean(_results_mvue)))
        print(edge_position, results[-1], results_mvue[-1])
