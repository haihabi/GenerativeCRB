from torch.utils.data import Dataset
import numpy as np
import gc
import h5py
from tqdm import tqdm
import os


def sample_indices_random(h, w, ph, pw, n_p):
    """Randomly sample n_p patch indices from (0, 0) up to (h, w) with patch height and width (ph, pw) """
    ii = []
    jj = []
    for k in np.arange(0, n_p):
        i = np.random.randint(0, h - ph + 1)
        j = np.random.randint(0, w - pw + 1)
        ii.append(i)
        jj.append(j)
    return ii, jj


def pack_raw(raw_im):
    """Packs Bayer image to 4 channels (h, w) --> (h/2, w/2, 4)."""
    # pack Bayer image to 4 channels
    im = np.expand_dims(raw_im, axis=2)
    img_shape = im.shape
    # print('img_shape: ' + str(img_shape))
    h = img_shape[0]
    w = img_shape[1]
    out = np.concatenate((im[0:h:2, 0:w:2, :],
                          im[0:h:2, 1:w:2, :],
                          im[1:h:2, 1:w:2, :],
                          im[1:h:2, 0:w:2, :]), axis=2)

    del raw_im
    gc.collect()

    return out


class NoiseDataSet(Dataset):
    def __init__(self, image_folder, split="train", n_pat_per_im=5000):
        self.patch_height = 32
        self.patch_height = 32
        self.n_pat_per_im = n_pat_per_im
        self.noise_image_list = []
        self.clean_image_list = []
        self.iso_list = []
        self.cam_list = []
        self.count_array = np.zeros([5, 5])
        self.mean_array = np.zeros([5, 5])
        self.mean_p2_array = np.zeros([5, 5])
        if split == 'train':
            self.inst_idxs = [4, 11, 13, 17, 18, 20, 22, 23, 25, 27, 28, 29, 30, 34, 35, 39, 40, 42, 43, 44, 45, 47, 81,
                              86,
                              88,
                              90, 101, 102, 104, 105, 110, 111, 115, 116, 125, 126, 127, 129, 132, 135,
                              138, 140, 175, 177, 178, 179, 180, 181, 185, 186, 189, 192, 193, 194, 196, 197]
            # removed: 114, 134, 184, 136, 190, 188, 117, 137, 191
        else:
            self.inst_idxs = [54, 55, 57, 59, 60, 62, 63, 66, 150, 151, 152, 154, 155, 159, 160, 161, 163, 164, 165,
                              166,
                              198,
                              199]
        # self.inst_idxs = [4]
        for folder in tqdm(os.listdir(image_folder)):
            id_str = folder.split("_")[0]
            if int(id_str) in self.inst_idxs or True:
                image_folder_path = os.path.join(image_folder, folder)
                for i in range(2):
                    noisy_image = os.path.join(image_folder_path, id_str + f'_NOISY_RAW_01{i}.MAT')
                    clean_image = os.path.join(image_folder_path, id_str + f'_GT_RAW_01{i}.MAT')
                    meta_image = os.path.join(image_folder_path, id_str + f'_METADATA_RAW_01{i}.MAT')

                    with h5py.File(noisy_image, 'r') as f:  # (use this for .mat files with -v7.3 format)
                        raw = f[list(f.keys())[0]]  # use the first and only key
                        input_image = np.expand_dims(pack_raw(raw), axis=0)
                        input_image = np.nan_to_num(input_image)
                        input_image = np.clip(input_image, 0.0, 1.0)

                    with h5py.File(clean_image, 'r') as f:
                        gt_raw = f[list(f.keys())[0]]  # use the first and only key
                        gt_image = np.expand_dims(pack_raw(gt_raw), axis=0)
                        gt_image = np.nan_to_num(gt_image)
                        gt_image = np.clip(gt_image, 0.0, 1.0)

                    iso = float(folder[12:17])
                    iso_list = [100, 400, 800, 1600, 3200]
                    if iso in iso_list:
                        cam = float(['IP', 'GP', 'S6', 'N6', 'G4'].index(folder[9:11]))
                        noise_image = input_image - gt_image
                        image_mean = np.mean(noise_image, axis=(0, 1, 2, 3))
                        image_mean_p2 = np.mean(np.power(noise_image, 2.0), axis=(0, 1, 2, 3))
                        image_count = np.prod(noise_image.shape)
                        c = self.count_array[int(cam), iso_list.index(iso)]
                        old_scale = c / (c + image_count)
                        new_scale = image_count / (c + image_count)
                        self.mean_array[int(cam), iso_list.index(iso)] += old_scale * self.mean_array[
                            int(cam), iso_list.index(iso)] + new_scale * image_mean

                        self.mean_p2_array[int(cam), iso_list.index(iso)] += old_scale * self.mean_p2_array[
                            int(cam), iso_list.index(iso)] + new_scale * image_mean_p2

                        _, h, w, c = noise_image.shape

                        ii, jj = sample_indices_random(h, w, self.patch_height, self.patch_height, self.n_pat_per_im)

                        for (i, j) in zip(ii, jj):
                            in_patch = noise_image[:, i:i + self.patch_height, j:j + self.patch_height, :]
                            gt_patch = gt_image[:, i:i + self.patch_height, j:j + self.patch_height, :]
                            self.noise_image_list.append(np.squeeze(in_patch, axis=0))
                            self.clean_image_list.append(np.squeeze(gt_patch, axis=0))
                            self.cam_list.append(cam)
                            self.iso_list.append(iso)

    def __len__(self):
        return len(self.noise_image_list)

    def __getitem__(self, item):
        return self.noise_image_list[item], self.clean_image_list[item], self.cam_list[item], self.iso_list[item]


if __name__ == '__main__':
    nds = NoiseDataSet("/data/datasets/SIDD_Medium_Raw/Data", n_pat_per_im=2)
    print(len(nds))
