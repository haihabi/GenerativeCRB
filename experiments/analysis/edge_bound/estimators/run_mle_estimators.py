import math
import numpy as np
from experiments import constants
import torch
from experiments.analysis.analysis_helpers import image_channel_swipe_nhwc2nchw, image_shape, image_channel_swipe_nchw2nhwc, \
    db
from pytorch_model.noise_flow import generate_noisy_image_flow
from experiments.analysis.edge_bound.edge_image_generator import EdgeImageGenerator
from tqdm import tqdm
from experiments.analysis.edge_bound.estimators.mle_gaussian import mle_gaussian

if __name__ == '__main__':
    iso = 100
    cam = 2
    edge_width = 8
    patch_size = 32
    batch_size = 32
    n_samples = 2000

    color_swip = False
    flow = generate_noisy_image_flow(image_shape(patch_size), device=constants.DEVICE, load_model=True).to(
        constants.DEVICE)
    eig = EdgeImageGenerator(patch_size)
    p_h, p_l = eig.get_pixel_color(color_swip=color_swip)
    generate_image = eig.get_image_function(edge_width, color_swip)


    def sample_function(in_batch_size, in_theta):
        bayer_img = generate_image(in_theta)
        in_cond_vector = [image_channel_swipe_nhwc2nchw(bayer_img), iso, cam]
        return flow.sample(in_batch_size, in_cond_vector)[-1]


    n_iter = math.ceil(n_samples / batch_size)
    results = []
    for edge_position in constants.CROSS_POINT:
        _results = []
        for _ in tqdm(range(n_iter)):
            data = sample_function(batch_size, edge_position * torch.ones(batch_size, device=constants.DEVICE))
            im_noise_data = image_channel_swipe_nchw2nhwc(data)
            im_hat = mle_gaussian(im_noise_data, p_h, p_l, edge_width).detach().cpu().numpy()
            _results.append(np.mean(np.power(im_hat - edge_position, 2.0)))
        results.append(db(np.mean(_results)))
        print(edge_position,results[-1])
    # print(im_hat)

    # sdata = image_channel_swipe_nchw2nhwc(data)[0, :, :, :].cpu().detach().numpy()
    # rgb = rggb2rgb(sdata)
    # plt.imshow(rgb)
    # plt.axis('off')
    # plt.title(f"Position:{cross_point}, Edge Width:{1}")
    # plt.show()
