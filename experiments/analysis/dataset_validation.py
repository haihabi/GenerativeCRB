import numpy as np


def dataset_vs_testset_checking(data_model, in_dataset):
    print("-" * 100)
    print("Starting Dataset check")
    testing_set = data_model.parameter_range(20, theta_scale_min=None, theta_scale_max=None).cpu().detach().numpy()
    for theta in testing_set:
        label_array = np.stack(in_dataset.label)
        error = np.abs(theta.reshape([1, -1]) - label_array).sum(axis=-1)
        if np.any(error == 0):
            print(f"{theta} is in the training set")
    print("Finished dataset check")
    print("-" * 100)
