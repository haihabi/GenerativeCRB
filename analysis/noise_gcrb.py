from pytorch_model.noise_flow import generate_noisy_image_flow
import sidd.data_loader as loader
from sidd.raw_utils import read_metadata
import gcrb
import torch
import numpy as np

if __name__ == '__main__':
    flow = generate_noisy_image_flow([4, 32, 32], load_model=True)

    clean = loader.load_raw_image_packed(
        "/Users/haihabi/projects/noise_flow/data/0001_001_S6_00100_00060_3200_L_X/0001_GT_RAW_010.MAT")
    metadata, bayer_2by2, wb, cst2, iso, cam = read_metadata(
        "/Users/haihabi/projects/noise_flow/data/0001_001_S6_00100_00060_3200_L_X/0001_METADATA_RAW_010.MAT")
    # clean = np.transpose(clean, (0, 2, 3, 1))
    batch_size = 4
    patch_size = 32
    v = np.random.randint(0, clean.shape[1] - patch_size)
    u = np.random.randint(0, clean.shape[2] - patch_size)

    clean_patch = clean[0, v:v + patch_size, u:u + patch_size, :]

    clean_image_tensor = np.transpose(np.expand_dims(clean_patch, axis=0), (0, 3, 1, 2))
    theta_np = np.tile(clean_image_tensor.reshape([1, -1]), [batch_size, 1])


    def sample_function(in_batch_size, in_theta):
        clean_im = torch.reshape(in_theta, [in_batch_size, 4, patch_size, patch_size])
        in_cond_vector = [clean_im, iso, cam]
        return flow.sample_nll(in_batch_size, cond=in_cond_vector).reshape([-1, 1])


    theta_vector = torch.tensor(theta_np, requires_grad=True)
    gfim = gcrb.adaptive_sampling_gfim_v2(sample_function, theta_vector, batch_size=batch_size)
    print("a")