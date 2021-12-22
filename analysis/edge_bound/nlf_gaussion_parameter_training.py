import os
import glob
import sidd.data_loader as loader
from sidd.raw_utils import read_metadata
import numpy as np
import pickle

if __name__ == '__main__':
    iso_list = [100, 400, 800, 1600, 3200]
    results_dict = {i: {iso: [] for iso in iso_list} for i in range(5)}
    dataset_folder = "/data/datasets/SIDD_Medium_Raw/Data"
    folder_list = glob.glob(dataset_folder + "/*")
    for f in folder_list:
        print(f)
        folder_base = os.path.join(dataset_folder, f)
        clean = loader.load_raw_image_packed(glob.glob(f"{folder_base}/*_GT_RAW_010.MAT")[0])
        noisy = loader.load_raw_image_packed(glob.glob(f"{folder_base}/*_NOISY_RAW_010.MAT")[0])
        noise = noisy - clean
        metadata, bayer_2by2, wb, cst2, iso, cam = read_metadata(
            glob.glob(f"{folder_base}/*_METADATA_RAW_010.MAT")[0])
        if iso in iso_list:
            results_dict[cam][iso].append((np.power(noise, 2.0).mean(), noise.shape))
    # with open("varinace_gaussion_model.pickle", "wb") as file:
    #     pickle.dump(results_dict, file)
    # print("a")
