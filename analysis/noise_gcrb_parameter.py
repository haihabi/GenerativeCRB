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


if __name__ == '__main__':
    flow = generate_noisy_image_flow([4, 32, 32], device=constants.DEVICE, load_model=True).to(constants.DEVICE)
    dataset_folder = "/data/datasets/SIDD_Medium_Raw/Data/"
    scene_number = "001"
    iso_str = "00100"
    illumination_code = "N"
    iso = 100
    folder_name = f"*_{scene_number}_S6_{iso_str}_*_*_{illumination_code}"

    folder_optins = glob.glob(os.path.join(dataset_folder, folder_name))
    temp = [int(fo.split("_")[-2]) for fo in folder_optins]
    i = np.argmax(temp)

    folder_base = folder_optins[i]
    print(f"illuminant temperature:{temp[i]}", folder_base)

    # clean = loader.load_raw_image_packed(
    #     glob.glob(f"{folder_base}/*_GT_RAW_010.MAT")[0])
    metadata, bayer_2by2, wb, cst2, _, cam = read_metadata(
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
    width_array = [1, 1.1, 1.5, 2, 4, 6, 8, 16, 24, 31]
    width_array = [2.0]


    def image2vector(in_image):
        return torch.permute(in_image, (0, 3, 1, 2))


    k = 0
    plot_images = True
    color_swip = True
    cross_point_array = [1, 2, 4, 8, 16, 24, 31]
    cross_point_array = [24, 16]
    for edge_width in width_array:
        def generate_image(in_theta):
            in_theta = in_theta.reshape([in_theta.shape[0], 1, 1, 1])
            # p = torch.min(torch.abs(in_theta - x_array) / edge_width, torch.tensor(1.0, device=constants.DEVICE))
            p = torch.min(torch.relu(in_theta - x_array) / edge_width, torch.tensor(1.0, device=constants.DEVICE))
            if color_swip:
                alpha = c_a - c_b
                i_x = alpha * p + c_b
            else:
                alpha = c_b - c_a
                i_x = alpha * p + c_a
            i_xy = i_x.repeat([1, patch_size, 1, 1])
            return i_xy


        for cross_point in cross_point_array:
            theta_vector = cross_point * torch.ones(batch_size, requires_grad=True).to(constants.DEVICE)
            if plot_images and (cross_point == 24 or cross_point == 16):
                plt.subplot(1, 2, 1 + k)
                clean_im = generate_image(theta_vector)[0, :, :, :].detach().cpu().numpy()
                clean_im = process_sidd_image(unpack_raw(clean_im), bayer_2by2, wb, cst2)
                plt.imshow(clean_im.astype('int'))
                plt.title(f"Edge Position:{cross_point}")
                plt.axis('off')
                k += 1


            def sample_function(in_batch_size, in_theta):
                bayer_img = generate_image(in_theta)
                in_cond_vector = [image2vector(bayer_img), iso, cam]
                return flow.sample_nll(in_batch_size, cond=in_cond_vector).reshape([-1, 1])


            gfim = gcrb.adaptive_sampling_gfim(sample_function, theta_vector.reshape([-1, 1]), batch_size=batch_size,
                                               n_max=64000)
            psnr = 10 * torch.log(torch.linalg.inv(gfim).diagonal().mean()) / np.log(10)
            print(psnr)
    plt.show()
    # title_list = ["Red", "Green", "Green", "Blue"]
    # gcrb_diag = torch.linalg.inv(gfim).detach().cpu().numpy().diagonal().reshape([4, 32, 32])
    # gcrb_diag_image = unpack_raw(np.transpose(gcrb_diag, (1, 2, 0)))
    # gcrb_diag_image = flip_bayer(gcrb_diag_image, bayer_2by2)
    # gcrb_diag_image = stack_rggb_channels(gcrb_diag_image)

    #     noise_image = flow.sample(batch_size, cond=[clean_im, iso, cam])
    #     noise_image = noise_image[-1].cpu().detach().numpy()
    #
    #     noise_image = np.transpose(noise_image, (0, 2, 3, 1))[0, :, :, :]
    #     noise_srgb = process_sidd_image(unpack_raw(noise_image), bayer_2by2, wb, cst2)
    #     clean_patch_srgb = process_sidd_image(unpack_raw(clean_patch), bayer_2by2, wb, cst2)
    #
    #     plt.subplot(2, 2, 1)
    #     plt.imshow(clean_patch_srgb.astype('int'))
    #     plt.title("Clean Image")
    #     plt.subplot(2, 2, 2)
    #     plt.imshow(noise_srgb.astype('int'))
    #     plt.title("Noisy Image")
    #     for i in range(4):
    #         plt.subplot(2, 4, 5 + i)
    #         plt.imshow(gcrb_diag_image[:, :, i])
    #         plt.title(title_list[i])
    #         plt.colorbar()
    #     plt.show()
    #
    # # clean_patch =
    #
    # print("a")
