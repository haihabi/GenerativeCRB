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

    clean = loader.load_raw_image_packed(
        glob.glob(f"{folder_base}/*_GT_RAW_010.MAT")[0])
    metadata, bayer_2by2, wb, cst2, _, cam = read_metadata(
        glob.glob(f"{folder_base}/*_METADATA_RAW_010.MAT")[0])
    batch_size = 32
    patch_size = 32
    plot_images = True

    if illumination_code == "N":
        # u, v = 1418, 1979  # Two Color
        u, v = 1350, 1979  # Single Color
        # u, v = 1460, 1979  # Single Color (White)
    elif illumination_code == "L":
        u, v = 1375, 1979  # Two Color
    else:
        u, v = 1378, 1979  # Two Color
    clean_patch = clean[0, v:v + patch_size, u:u + patch_size, :]

    clean_image_tensor = np.transpose(np.expand_dims(clean_patch, axis=0), (0, 3, 1, 2))
    theta_np = np.tile(clean_image_tensor.reshape([1, -1]), [batch_size, 1])
    theta_vector = torch.tensor(theta_np, requires_grad=True).to(constants.DEVICE)


    # print(iso)

    def sample_function(in_batch_size, in_theta):
        clean_im = torch.reshape(in_theta, [in_batch_size, 4, patch_size, patch_size])
        in_cond_vector = [clean_im, iso, cam]
        return flow.sample_nll(in_batch_size, cond=in_cond_vector).reshape([-1, 1])


    gfim = gcrb.adaptive_sampling_gfim(sample_function, theta_vector, batch_size=batch_size, n_max=32000)
    psnr = 10 * torch.log(torch.linalg.inv(gfim).diagonal().mean()) / np.log(10)
    print(psnr)
    title_list = ["Red", "Green", "Green", "Blue"]
    gcrb_diag = torch.linalg.inv(gfim).detach().cpu().numpy().diagonal().reshape([4, 32, 32])
    gcrb_diag_image = unpack_raw(np.transpose(gcrb_diag, (1, 2, 0)))
    gcrb_diag_image = flip_bayer(gcrb_diag_image, bayer_2by2)
    gcrb_diag_image = stack_rggb_channels(gcrb_diag_image)
    if plot_images:
        clean_im = torch.reshape(theta_vector, [batch_size, 4, patch_size, patch_size])
        noise_image = flow.sample(batch_size, cond=[clean_im, iso, cam])
        noise_image = noise_image[-1].cpu().detach().numpy()

        noise_image = np.transpose(noise_image, (0, 2, 3, 1))[0, :, :, :]
        noise_srgb = process_sidd_image(unpack_raw(noise_image), bayer_2by2, wb, cst2)
        clean_patch_srgb = process_sidd_image(unpack_raw(clean_patch), bayer_2by2, wb, cst2)

        plt.subplot(2, 2, 1)
        plt.imshow(clean_patch_srgb.astype('int'))
        plt.title("Clean Image")
        plt.subplot(2, 2, 2)
        plt.imshow(noise_srgb.astype('int'))
        plt.title("Noisy Image")
        for i in range(4):
            plt.subplot(2, 4, 5 + i)
            plt.imshow(gcrb_diag_image[:, :, i])
            plt.title(title_list[i])
            plt.colorbar()
        plt.show()

    # clean_patch =

    print("a")
