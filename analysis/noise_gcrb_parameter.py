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
from edge_bound.edge_image_generator import EdgeImageGenerator

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

    eig = EdgeImageGenerator(patch_size)


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
                generate_image = eig.get_image_function(edge_width, color_swip)


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
