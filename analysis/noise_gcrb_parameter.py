from pytorch_model.noise_flow import generate_noisy_image_flow
import sidd.data_loader as loader
from sidd.raw_utils import read_metadata
from sidd.pipeline import process_sidd_image, flip_bayer, stack_rggb_channels
from matplotlib import pyplot as plt
import gcrb
import torch
import numpy as np
import constants
import os
import glob

if __name__ == '__main__':
    flow = generate_noisy_image_flow([4, 32, 32], device=constants.DEVICE, load_model=True).to(constants.DEVICE)
    dataset_folder = "/data/datasets/SIDD_Medium_Raw/Data/"
    scene_number = "001"
    iso_str = "00100"
    illumination_code = "N"
    # iso = 100
    folder_name = f"*_{scene_number}_S6_{iso_str}_*_*_{illumination_code}"

    folder_optins = glob.glob(os.path.join(dataset_folder, folder_name))
    temp = [int(fo.split("_")[-2]) for fo in folder_optins]
    i = np.argmax(temp)

    folder_base = folder_optins[i]
    print(f"illuminant temperature:{temp[i]}", folder_base)

    metadata, bayer_2by2, wb, cst2, _, _ = read_metadata(
        glob.glob(f"{folder_base}/*_METADATA_RAW_010.MAT")[0])
    batch_size = 32
    patch_size = 32
    plot_images = False

    c_b = torch.tensor(
        np.asarray([0.34292033, 0.22350113, 0.33940127, 0.13547006]).astype("float32").reshape(1, 1, 1, -1),
        device=constants.DEVICE)
    c_a = torch.tensor(
        np.asarray([0.03029606, 0.02065253, 0.02987719, 0.01571027]).astype("float32").reshape(1, 1, 1, -1),
        device=constants.DEVICE)

    x_array = torch.tensor(np.linspace(0, patch_size - 1, patch_size).astype("float32"),
                           device=constants.DEVICE).reshape([1, 1, -1, 1])


    def image2vector(in_image):
        return torch.permute(in_image, (0, 3, 1, 2))


    k = 0
    color_swip = False

    width_array = [2, 8, 16, 24, 30]
    cross_point_array = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 30]
    iso_array = [100, 400, 800, 1600, 3200]
    cam_array = [0, 1, 2, 3, 4]
    results = {}
    for cam in cam_array:
        results_iso = {}
        for iso in iso_array:
            results_edge_width = {}
            for edge_width in width_array:
                def generate_image(in_theta):
                    in_theta = in_theta.reshape([in_theta.shape[0], 1, 1, 1])
                    # p = torch.min(torch.abs(in_theta - x_array) / edge_width, torch.tensor(1.0, device=constants.DEVICE))
                    # p = torch.min(torch.relu(in_theta - x_array) / edge_width, torch.tensor(1.0, device=constants.DEVICE))
                    p = torch.sigmoid((in_theta - x_array) / edge_width)
                    if color_swip:
                        alpha = c_a - c_b
                        i_x = alpha * p + c_b
                    else:
                        alpha = c_b - c_a
                        i_x = alpha * p + c_a
                    i_xy = i_x.repeat([1, patch_size, 1, 1])
                    return i_xy


                def sample_function(in_batch_size, in_theta):
                    bayer_img = generate_image(in_theta)
                    in_cond_vector = [image2vector(bayer_img), iso, cam]
                    return flow.sample_nll(in_batch_size, cond=in_cond_vector).reshape([-1, 1])


                _results_croos_points = []
                for cross_point in cross_point_array:
                    theta_vector = cross_point * torch.ones(batch_size, requires_grad=True).to(constants.DEVICE)
                    gfim = gcrb.adaptive_sampling_gfim(sample_function, theta_vector.reshape([-1, 1]),
                                                       batch_size=batch_size,
                                                       n_max=64000)
                    psnr = 10 * torch.log(torch.linalg.inv(gfim).diagonal().mean()) / np.log(10)
                    _results_croos_points.append(psnr.item())
                    print(psnr)

                results_edge_width[edge_width] = _results_croos_points
            results_iso[iso] = results_edge_width
        results[cam] = results_iso

    import pickle

    file_name = "results_edge" if not color_swip else "results_edge_swip"
    with open(f"{file_name}.pickle", "wb") as file:
        pickle.dump(results, file)
