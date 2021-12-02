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
import argparse

cam_dict = {'Apple': 0, 'Google': 1, 'samsung': 2, 'motorola': 3, 'LGE': 4}


def unpack_raw(raw4ch):
    """Unpacks 4 channels to Bayer image (h/2, w/2, 4) --> (h, w)."""
    img_shape = raw4ch.shape
    h = img_shape[0]
    w = img_shape[1]
    # d = img_shape[2]
    bayer = np.zeros([h * 2, w * 2], dtype=np.float32)
    # bayer = raw4ch
    # bayer.reshape((h * 2, w * 2))
    bayer[0::2, 0::2] = raw4ch[:, :, 0]
    bayer[0::2, 1::2] = raw4ch[:, :, 1]
    bayer[1::2, 1::2] = raw4ch[:, :, 2]
    bayer[1::2, 0::2] = raw4ch[:, :, 3]
    return bayer


def input_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--plot_images', action="store_true")
    parser.add_argument('--plot_images_iso', action="store_true")
    parser.add_argument('--plot_device', action="store_true")
    parser.add_argument('--sweep', action="store_true")
    parser.add_argument('--iso', default=100, type=int)
    parser.add_argument('--cam', default=0, type=int)
    parser.add_argument('--patch_size', default=34, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = input_arguments()
    dataset_folder = "/data/datasets/SIDD_Medium_Raw/Data/"
    scene_number2run = ["003", "007", "010"]  # "003", "007", "010"
    batch_size = 32
    n_max = 32000
    debug_plot = False

    sweep = args.sweep
    plot_images = args.plot_images and not sweep
    plot_images_iso = args.plot_images_iso and not sweep
    plot_device = args.plot_device and not sweep

    iso_array = [100, 400, 800, 1600, 3200] if plot_images_iso or sweep else [args.iso]
    cam_array = [0, 1, 2, 3, 4] if plot_device or sweep else [args.cam]
    patch_size = args.patch_size
    print(iso_array, cam_array, patch_size)
    title_list = ["R", "G", "G", "B"]
    flow = generate_noisy_image_flow([4, patch_size, patch_size], device=constants.DEVICE, load_model=True).to(
        constants.DEVICE)
    results_dict = {}
    for j, scene_number in enumerate(scene_number2run):
        if scene_number == "007":
            u, v = 1420, 1630
            folder_name = "0155_007_GP_00100_00100_5500_N"
        elif scene_number == "010":
            u, v = 790, 1840
            folder_name = "0199_010_GP_00800_01600_5500_N"
        elif scene_number == "003":
            u, v = 828, 950
            folder_name = "0054_003_N6_00100_00160_5500_N"
        else:
            raise NotImplemented

        folder_base = os.path.join(dataset_folder, folder_name)
        clean = loader.load_raw_image_packed(glob.glob(f"{folder_base}/*_GT_RAW_010.MAT")[0])
        metadata, bayer_2by2, wb, cst2, _, _ = read_metadata(
            glob.glob(f"{folder_base}/*_METADATA_RAW_010.MAT")[0])

        if patch_size == -1:
            clean_patch = clean[0, v:patch_size, u:patch_size, :]
        else:
            clean_patch = clean[0, v:v + patch_size, u:u + patch_size, :]
        clean_patch_srgb = process_sidd_image(unpack_raw(clean_patch), bayer_2by2, wb, cst2)
        if debug_plot:
            plt.imshow(clean_patch_srgb.astype('int'))
            plt.title("Clean Image")
            plt.show()

        clean_image_tensor = np.transpose(np.expand_dims(clean_patch, axis=0), (0, 3, 1, 2))
        theta_np = np.tile(clean_image_tensor.reshape([1, -1]), [batch_size, 1])
        theta_vector = torch.tensor(theta_np, requires_grad=True).to(constants.DEVICE)
        cam_dict_r = {v: k for k, v in cam_dict.items()}
        results_iso_dict = {}
        for k, iso in enumerate(iso_array):
            results_cam_dict = {}
            for m, cam in enumerate(cam_array):
                print(f"running:{cam},{iso},{scene_number}")


                def sample_function(in_batch_size, in_theta):
                    clean_im = torch.reshape(in_theta, [in_batch_size, 4, patch_size, patch_size])
                    in_cond_vector = [clean_im, iso, cam]
                    return flow.sample_nll(in_batch_size, cond=in_cond_vector).reshape([-1, 1])


                clean_im_np = torch.reshape(theta_vector, [batch_size, 4, patch_size, patch_size])[0, :, :,
                              :].cpu().detach().numpy()
                gfim = gcrb.adaptive_sampling_gfim(sample_function, theta_vector, batch_size=batch_size, n_max=n_max)
                gcrb_diag = torch.linalg.inv(gfim).detach().cpu().numpy().diagonal().reshape(
                    [4, patch_size, patch_size])
                gcrb_diag = np.transpose(gcrb_diag, (1, 2, 0))
                clean_im_np = np.transpose(clean_im_np, (1, 2, 0))
                relative_gcrb = np.sqrt(gcrb_diag) / clean_im_np
                from analysis.analysis_helpers import db

                psnr = db(gcrb_diag.mean())
                psnr_relative = db(relative_gcrb.mean())
                psnr_r = db(gcrb_diag[:, :, 0].mean())
                psnr_g1 = db(gcrb_diag[:, :, 1].mean())
                psnr_g2 = db(gcrb_diag[:, :, 2].mean())
                psnr_b = db(gcrb_diag[:, :, 3].mean())

                psnr_relative_r = db(relative_gcrb[:, :, 0].mean())
                psnr_relative_g1 = db(relative_gcrb[:, :, 1].mean())
                psnr_relative_g2 = db(relative_gcrb[:, :, 2].mean())
                psnr_relative_b = db(relative_gcrb[:, :, 3].mean())
                _res = {"MSE": psnr,
                        "Relative_RMSE": psnr_relative,
                        "MSE_R": psnr_r,
                        "Relative_RMSE_R": psnr_relative_r,
                        "MSE_G1": psnr_g1,
                        "Relative_RMSE_G1": psnr_relative_g1,
                        "MSE_G2": psnr_g2,
                        "Relative_RMSE_G2": psnr_relative_g2,
                        "MSE_B": psnr_b,
                        "Relative_RMSE_B": psnr_relative_b,
                        }
                results_cam_dict[cam] = _res
                if plot_images_iso or plot_device:
                    clean_im = torch.reshape(theta_vector, [batch_size, 4, patch_size, patch_size])
                    noise_image = flow.sample(batch_size, cond=[clean_im, iso, cam])
                    noise_image = noise_image[-1].cpu().detach().numpy()

                    noise_image = np.transpose(noise_image, (0, 2, 3, 1))[0, :, :, :]
                    noise_srgb = process_sidd_image(unpack_raw(noise_image), bayer_2by2, wb, cst2)
                    iter_index = m if plot_device else k
                    subplot_size = len(cam_array) if plot_device else len(iso_array)
                    if iter_index == 0:
                        plt.subplot(1, subplot_size + 1, 1)
                        plt.imshow(clean_patch_srgb.astype('int')[2:-2, 2:-2, :])
                        plt.title("Clean Image")
                        plt.axis('off')

                    plt.subplot(1, subplot_size + 1, 2 + iter_index)
                    plt.imshow(noise_srgb.astype('int')[2:-2, 2:-2, :])
                    if plot_device:
                        plt.title(f"{cam_dict_r[cam]} Camera" + " with MSE[dB]:{:.2f}".format(psnr.item()))
                    else:
                        plt.title(f"ISO {iso}" + " with MSE[dB]:{:.2f}".format(psnr.item()))
                    plt.axis('off')

                if plot_images:
                    gcrb_diag = torch.linalg.inv(gfim).detach().cpu().numpy().diagonal().reshape(
                        [4, patch_size, patch_size])
                    gcrb_diag_image = unpack_raw(np.transpose(gcrb_diag, (1, 2, 0)))
                    gcrb_diag_image = flip_bayer(gcrb_diag_image, bayer_2by2)
                    gcrb_diag_image = stack_rggb_channels(gcrb_diag_image)
                    clean_im = torch.reshape(theta_vector, [batch_size, 4, patch_size, patch_size])
                    noise_image = flow.sample(batch_size, cond=[clean_im, iso, cam])
                    noise_image = noise_image[-1].cpu().detach().numpy()

                    noise_image = np.transpose(noise_image, (0, 2, 3, 1))[0, :, :, :]
                    noise_srgb = process_sidd_image(unpack_raw(noise_image), bayer_2by2, wb, cst2)
                    m = len(scene_number2run)
                    plot_a = True
                    if plot_a:
                        clean_im_np = clean_im.cpu().detach().numpy()[0, :, :, :]
                        ngcrb = np.sqrt(gcrb_diag) / clean_im_np

                        ngcrb = unpack_raw(np.transpose(ngcrb, (1, 2, 0)))
                        ngcrb = flip_bayer(ngcrb, bayer_2by2)
                        ngcrb = stack_rggb_channels(ngcrb)
                        for i in range(2):
                            plt.subplot(2, len(scene_number2run), 1 + i * m + j)

                            gcrb_im = gcrb_diag_image[:, :, 0]
                            if i >= 1:
                                gcrb_im = ngcrb[:, :, 0]
                            plt.imshow(gcrb_im[1:-1, 1:-1])
                            ticks = [gcrb_im.min(), gcrb_im.max()]
                            print(ticks)
                            if i >= 1:
                                cbar = plt.colorbar(ticks=ticks, orientation="horizontal", fraction=0.046,
                                                    pad=0.04, format="%.2f")
                            else:
                                cbar = plt.colorbar(ticks=ticks, orientation="horizontal", fraction=0.046,
                                                    pad=0.04, format="%.4f")
                            plt.axis('off')
                    else:
                        plt.subplot(6, m, 1 + j)
                        plt.imshow(clean_patch_srgb.astype('int')[2:-2, 2:-2, :])

                        plt.axis('off')
                        plt.subplot(6, m, 1 + m + j)
                        plt.imshow(noise_srgb.astype('int')[2:-2, 2:-2, :])

                        plt.axis('off')
                        for i in range(4):
                            plt.subplot(6, len(scene_number2run), 1 + (2 + i) * m + j)

                            gcrb_im = gcrb_diag_image[:, :, i][1:-1, 1:-1]
                            plt.imshow(gcrb_diag_image[:, :, i][1:-1, 1:-1])
                            ticks = [gcrb_im.min(), gcrb_im.max()]
                            print(ticks)
                            cbar = plt.colorbar(ticks=ticks, orientation="horizontal", fraction=0.046,
                                                pad=0.04, format="%.4f")
                            plt.axis('off')

            results_iso_dict[iso] = results_cam_dict
        results_dict[scene_number] = results_iso_dict
    if plot_images_iso or plot_device or plot_images:
        plt.show()
    if sweep:
        import pickle

        with open("/data/projects/GenerativeCRB/analysis/results.pickle", "wb") as file:
            pickle.dump(results_dict, file)
    print("a")
